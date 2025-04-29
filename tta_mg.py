import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import dataloader_pth as dataloader
from sklearn.cluster import KMeans
from collections import Counter
from torchclustermetrics import silhouette
from torchmetrics.functional.multimodal import clip_score

from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import warnings
from functools import partial

gather_time_running = 0
optim_time_running = 0
concat_time_running = 0
batch_size_per_gpu = 32
image_embeds = None
text_embeds = None
scores = []
losses = []

class CustomDataset(Dataset):
    def __init__(self, prompts, target_images=None):
        self.prompts = prompts
        self.target_images = target_images
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        # Return prompt and target image if available
        if self.target_images:
            return self.prompts[idx], self.target_images[idx]
        return self.prompts[idx], None

def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_IB_DISABLE"] = "1"
    # os.environ["NCCL_P2P_DISABLE"] = "1"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

model_name = "openai/clip-vit-base-patch32"
clip_score_fn = partial(clip_score, model_name_or_path=model_name)

def algo(pipe, generated_images, prompts, rank, world_size, **kwargs):
    
    global gather_time_running
    global concat_time_running
    global image_embeds
    global text_embeds
    batch_size = generated_images.shape[0]
    
    latents = pipe.vae.encode(generated_images.to(rank)).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    latents = latents.view(batch_size, -1)
    text_input_ids = pipe.tokenizer(  
        prompts,
        padding="max_length",  
        max_length=pipe.tokenizer.model_max_length,  
        truncation=True,  
        return_tensors="pt"  
    ).input_ids
    prompt_embeds = pipe.text_encoder(text_input_ids.to(rank))[0].detach().flatten(start_dim=1)
    start_time = time.time()
    image_embeds[rank] = latents
    text_embeds[rank] = prompt_embeds.cpu()
    # print("rank{} data gathered".format(rank))
    dist.barrier()
    gather_time = time.time() - start_time
    gather_time_running += gather_time

    if rank == 0:  # Process on GPU 0
        # print("clustering")
        start_time = time.time()
        image_embeds = [item.cuda(0) for item in image_embeds]
        image_embeds_tensor = torch.cat(image_embeds, dim=0)
        text_embeds_tensor = torch.cat(text_embeds, dim=0)
        concat_time = time.time() - start_time
        concat_time_running += concat_time
        clustering = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(text_embeds_tensor)
        labels = clustering.labels_
        # print(labels)
        min_samples = 2
        label_counts = Counter(labels)
        satisfies_min_samples = all(count >= min_samples for count in label_counts.values())
        loss = 1 - silhouette.score(image_embeds_tensor, labels, True)
        # print(loss.item())
        if not satisfies_min_samples:   
            print("error: there are some one-element clusters")
            del loss
            return None
    else:
        loss = torch.tensor(0.0, device=rank, requires_grad=True)
    
    return loss

class GradientEnabledPipelineWrapper:
    """
    Wrapper for StableDiffusionPipeline that allows gradient computation
    during image generation.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def __call__(self, prompt, num_inference_steps=50, guidance_scale=7.5):
        pipeline = self.pipeline
        
        # Encode prompt
        text_embeddings = pipeline._encode_prompt(
            prompt,
            device=pipeline.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0
        )
        
        # Prepare scheduler
        pipeline.scheduler.set_timesteps(num_inference_steps, device=pipeline.device)
        timesteps = pipeline.scheduler.timesteps
        
        # Prepare latents
        latents = torch.randn(
            (len(prompt), pipeline.unet.config.in_channels, 12, 8),
            device=pipeline.device
        )
        latents = latents * pipeline.scheduler.init_noise_sigma
        
        # Denoising loop - keeping gradients enabled
        for i, t in enumerate(timesteps):
            # expand latents for classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # Scale latents according to timestep (no_grad not used here)
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            noise_pred = pipeline.unet(
                latent_model_input, 
                t,
                encoder_hidden_states=text_embeddings
            )
            
            # Apply guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.sample.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents to image tensor (keeping gradients)
        # Note: Usually VAE decode is in no_grad, but we're keeping gradients here
        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor).sample
        
        # Don't apply additional processing that would detach gradients
        # print(torch.cuda.memory_allocated() / (1024 * 1024 * 1024))
        
        return image

def save_checkpoint(rank, pipe, save_dir, epoch):
    """Save model checkpoint from rank 0 only"""
    if rank != 0:
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    # Get the underlying UNet model
    unet_model = pipe.unet.module
    
    # Save the UNet state dict
    torch.save(unet_model.state_dict(), f"{save_dir}/unet_epoch_{epoch}.pt")
    
    print(f"Saved checkpoint at epoch {epoch}")

def calculate_clip_score(images, prompts):
    images_int = (images * 255).to(torch.uint8)
    clip_score = clip_score_fn(images_int, prompts).detach()
    return round(float(clip_score), 4)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        return self.net(x)

def train(rank, world_size):
    """Main training function with distributed training and gradient-enabled generation"""
    # Initialize the distributed environment
    setup(rank, world_size)
    global optim_time_running
    global batch_size_per_gpu
    global scores
    global losses

    # device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    # model = SimpleModel().to(device)
    # model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    # print("rank{} model prepared".format(rank))

    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Load the Stable Diffusion pipeline
    model_id = "/data1/linziqing/hub/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06"  # or your custom checkpoint
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        # Use DDIM scheduler for more stable gradients
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    )
    pipe = pipe.to(device)
    print("rank{} pipeline prepared".format(rank))
    
    # Make VAE trainable if your algorithm needs it
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)  # Usually kept frozen
    pipe.unet.requires_grad_(True)
    
    # Wrap UNet with DDP
    pipe.unet = DDP(pipe.unet, device_ids=[rank])
    
    # Create gradient-enabled pipeline wrapper
    grad_pipe = GradientEnabledPipelineWrapper(pipe)
    
    # Set up optimizer - include VAE parameters if needed
    optimizer = optim.AdamW(
        list(pipe.unet.parameters()) + list(pipe.vae.parameters()),
        # list(pipe.unet.parameters()),
        lr=1e-5
    )
    
    # initialize dataset: 
    dataset = dataloader.CholecT50( 
        dataset_dir="/data1/linziqing/cholect50/CholecT50/", 
        dataset_variant="cholect45-crossval",
        test_fold=1,
        augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
        )
    # build dataset
    train_dataset, val_dataset,_ = dataset.build()
    
    # Set up distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,  # Process one prompt at a time due to memory constraints
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    
    # Training loop
    num_epochs = 10
    save_dir = "./sd_checkpoints"
    
    for epoch in range(num_epochs):
        # Set epoch for sampler to reshuffle data
        train_sampler.set_epoch(epoch)
        
        # Training
        pipe.unet.train()
        pipe.vae.train()  # Set VAE to train mode if it's part of your training
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, disable=rank != 0)
        for step, (_, prompt) in enumerate(progress_bar):
            tag = True

            if rank == 0:
                progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}")
            
            if rank == 0 and (step) % 50 == 0:
                # pipe.unet.eval()
                # pipe.vae.eval()
                score = 0
                score_random = 0
                score_real = 0
                original_unet = pipe.unet
                pipe.unet = original_unet.module
                for val_step, (real_imgs, val_prompts) in enumerate(val_dataloader):
                    val_imgs = pipe(
                        prompt=list(val_prompts),  # What to generate
                        height=480,
                        width=640,  # Specify the image size
                        guidance_scale=8,  # How strongly to follow the prompt
                        num_inference_steps=35,  # How many steps to take
                        output_type='pt',
                    ).images
                    sd_clip_score = calculate_clip_score(val_imgs, list(val_prompts))
                    # random_imgs = torch.rand(val_imgs.shape, dtype=val_imgs.dtype, device=val_imgs.device)
                    # sd_clip_score_random = calculate_clip_score(random_imgs, list(val_prompts))
                    # sd_clip_score_real = calculate_clip_score(real_imgs, list(val_prompts))
                    score += sd_clip_score
                    # score_random += sd_clip_score_random
                    # score_real += sd_clip_score_real
                    print(sd_clip_score)
                    # print((score-score_random)/(score_real-score_random))
                    if val_step == 5:
                        scores.append(score)
                        del val_imgs
                        pipe.unet = original_unet
                        pipe.unet.train()
                        pipe.vae.train()
                        break
            
            prompt = list(prompt)  # Get the prompt string
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Generate images with gradients enabled
            # Temporarily unwrap UNet from DDP for the gradient-enabled pipeline
            original_unet = pipe.unet
            pipe.unet = original_unet.module
            
            # Forward pass with gradients
            # generated_images = pipe(
            #     prompt=prompt,  # What to generate
            #     height=480,
            #     width=640,  # Specify the image size
            #     guidance_scale=8,  # How strongly to follow the prompt
            #     num_inference_steps=35,  # How many steps to take
            #     output_type='pt',
            # ).images
            generated_images = grad_pipe(prompt, num_inference_steps=1, guidance_scale=7.5)
            
            # Restore DDP-wrapped UNet
            pipe.unet = original_unet
            
            # Calculate loss
            loss = algo(pipe, generated_images, prompt, rank, world_size)

            if rank == 0 and loss != None:
                losses.append(loss.item())
            
            if loss == None:
                tag = False

            loss_tensor = torch.zeros(1, device=device)
            tag_tensor = torch.zeros(1, device=device)

            if rank == 0:
                tag_tensor[0] = tag

            dist.broadcast(tag_tensor, src=0)
            
            tag = tag_tensor.item()

            if tag == False:
                dist.barrier()
                continue

            
            # On rank 0, fill with actual loss value
            if rank == 0:
                loss_tensor[0] = loss.item()
                
                # Only backpropagate on rank 0
                loss.backward()
                
                # Convert model gradients to parameters for broadcasting
                # No need to modify code here as DDP will handle this
            else:
                # Zero out gradients on other devices
                # We'll override with gradients from rank 0 through DDP sync
                for param in pipe.unet.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                for param in pipe.vae.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
            
            # Broadcast loss from rank 0 to all processes for logging
            dist.broadcast(loss_tensor, src=0)
            running_loss += loss_tensor.item()
            
            # Optimize - this will apply the gradients from rank 0 to all devices
            # DDP automatically synchronizes gradients before this step
            start_time = time.time()
            optimizer.step()
            optim_time = time.time() - start_time
            optim_time_running += optim_time
            print("gather time on rank{}:{}".format(rank, gather_time_running / (step + 1)))
            print("optimize time:{}".format(optim_time_running / (step + 1)))
            print("concat time:{}".format(concat_time_running / (step + 1)))
            
            if rank == 0:
                progress_bar.set_postfix({"loss": running_loss / (step + 1)})
            
            dist.barrier()
                
            # Save intermediate generations and visualize progress
            #     # Save generated image for visualization
            #     # Convert tensor to image and save
            #     with torch.no_grad():
            #         # Process image tensor to PIL
            #         img_tensor = (generated_images[0].detach().cpu() + 1) / 2
            #         img_tensor = torch.clamp(img_tensor, 0, 1)
            #         img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            #       
            #   img_pil = Image.fromarray(img_np)
                    
            #         # Save image
            #         os.makedirs(f"{save_dir}/samples", exist_ok=True)
            #         img_pil.save(f"{save_dir}/samples/epoch{epoch}_step{step}.png")
        
        # Save checkpoint after each epoch
        save_checkpoint(rank, pipe, save_dir, epoch)
        
        # Log epoch results
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs} completed. Avg loss: {running_loss/len(train_loader):.4f}")
        
        # Generate and save end-of-epoch samples
        # if rank == 0:
        #     with torch.no_grad():
        #         # Store the DDP-wrapped UNet
        #         ddp_unet = pipe.unet
                
        #         # Replace with the underlying model for inference
        #         pipe.unet = ddp_unet.module
                
        #         # Set to eval mode for clean inference
        #         pipe.unet.eval()
        #         pipe.vae.eval()
                
        #         # Generate sample images with standard pipeline (no gradient needed here)
        #         test_prompts = [
        #             "a photograph of a cat in space",
        #             "an oil painting of a sunset over mountains"
        #         ]
                
        #         os.makedirs(f"{save_dir}/epoch_samples", exist_ok=True)
        #         for i, prompt in enumerate(test_prompts):
        #             image = pipe(prompt, num_inference_steps=30).images[0]
        #             image.save(f"{save_dir}/epoch_samples/epoch{epoch}_sample{i}.png")
                
        #         # Restore DDP-wrapped UNet and training mode
        #         pipe.unet = ddp_unet
        #         pipe.unet.train()
        #         pipe.vae.train()
        
        # Make sure all processes sync before next epoch
        dist.barrier()
    
    # Clean up
    cleanup()

def main():
    global image_embeds
    global text_embeds
    global batch_size_per_gpu
    # Number of GPUs available
    warnings.simplefilter(action='ignore', category=FutureWarning)
    world_size = torch.cuda.device_count()
    rank = int(os.environ.get("RANK", 0))
    print(f"Using {world_size} GPUs for training")
    
    # Spawn processes
    image_embeds = [torch.zeros(batch_size_per_gpu, 384) for _ in range(world_size)]
    text_embeds = [torch.zeros(batch_size_per_gpu, 78848) for _ in range(world_size)]
    train(rank, world_size)

if __name__ == "__main__":
    main()