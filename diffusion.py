import torch
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
import tqdm
from sklearn.cluster import KMeans
from collections import Counter
from torchvision import transforms
from datasets import load_dataset
import ivtmetrics # install using: pip install ivtmetrics
import dataloader_pth as dataloader
from torch.utils.data import DataLoader
from torchclustermetrics import silhouette
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import json
import torchvision.utils as vutils
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from peft import LoraConfig, get_peft_model

# We'll be exploring a number of pipelines today!
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionDepth2ImgPipeline,
)

# Set device
device = "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
# val_device = "cuda:1"

# Load the pipeline
model_id = "stabilityai/stable-diffusion-2-1-base"
model_path = "/data1/linziqing/cholect50/sd"
pipe = StableDiffusionPipeline.from_pretrained(model_path).to(device)

# Load val model CLIP
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    clip_score = clip_score_fn(images, prompts)
    return round(float(clip_score), 4)

# sd_clip_score = calculate_clip_score(images, prompts)

num_epochs = 2  # @param
lr = 5e-4  # 2param

optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr)

losses = []
clip_scores = []
fip_scores = []
mse_scores = []

# initialize dataset: 
dataset = dataloader.CholecT50( 
          dataset_dir="/data1/linziqing/cholect50/CholecT50/", 
          dataset_variant="cholect45-crossval",
          test_fold=1,
          augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
          )

# build dataset
train_dataset, val_dataset, test_dataset = dataset.build()

# train and val data loaders
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
seed = 42
torch.manual_seed(seed)

# test data set is built per video, so load differently
test_dataloaders = []
for video_dataset in test_dataset:
 test_dataloader = DataLoader(video_dataset, batch_size=32, shuffle=False)
 test_dataloaders.append(test_dataloader)

# image_size = 256  # @param
# batch_size = 4  # @param
# preprocess = transforms.Compose(
#     [
#         transforms.Resize((image_size, image_size)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]),
#     ]
# )
# def transform(examples):
#     images = [preprocess(image.convert("RGB")) for image in examples["image"]]
#     return {"images": images}
# dataset.set_transform(transform)
# train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

loss_type = "silhouette"
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["to_q", "to_v", "query", "value"],
    lora_dropout=0.1,
)
pipe.unet = get_peft_model(pipe.unet, lora_config)

for epoch in range(num_epochs):
    for step, (imgs, prompts) in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
    # for step, (imgs, prompts) in enumerate(train_dataloader):
        if loss_type == "ft":
            latents = pipe.vae.encode(imgs.to(device)).latent_dist.sample()
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],)).to(device)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            text_input_ids = pipe.tokenizer(  
                prompts,
                padding="max_length",  
                max_length=pipe.tokenizer.model_max_length,  
                truncation=True,  
                return_tensors="pt"  
            ).input_ids.to(device)
            prompt_embeds = pipe.text_encoder(text_input_ids)[0]
            noise_pred = pipe.unet(noisy_latents, timesteps, prompt_embeds).sample
            loss = F.mse_loss(
                noise_pred, noise
            )  # NB - trying to predict noise (eps) not (noisy_ims-clean_ims) or just (clean_ims)

        else:
            imgs = torch.tensor([]).to(device)
            for prompt in prompts:
                img = pipe(
                    prompt=prompt,  # What to generate
                    height=480,
                    width=640,  # Specify the image size
                    guidance_scale=8,  # How strongly to follow the prompt
                    num_inference_steps=35,  # How many steps to take
                    output_type='pt',
                ).images[0]
                vutils.save_image(img, './test.png')
                imgs = torch.cat((imgs, img.unsqueeze(0)), dim=0)
            if loss_type == "silhouette":
                latents = pipe.vae.encode(imgs).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                latents = latents.view(batch_size, 4*60*80)
                text_input_ids = pipe.tokenizer(  
                    prompts,
                    padding="max_length",  
                    max_length=pipe.tokenizer.model_max_length,  
                    truncation=True,  
                    return_tensors="pt"  
                ).input_ids.to(device)
                prompt_embeds = pipe.text_encoder(text_input_ids)[0].detach().cpu().flatten(start_dim=1)
                clustering = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(prompt_embeds)
                labels = clustering.labels_
                min_samples = 1
                label_counts = Counter(labels)
                satisfies_min_samples = all(count >= min_samples for count in label_counts.values())
                loss = 1 - silhouette.score(latents, labels, True)
                if not satisfies_min_samples:
                    loss.detach()
                    print("error: there are some one-element clusters")
            if loss_type == "clip":
                loss = clip_score_fn(imgs, list(prompts))
        
        print(loss.item())
        losses.append(loss.item())

        # Update the model parameters with the optimizer based on this loss
        loss.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    
        if step % 3 == 0:
            # pipe.save_pretrained("/data1/linziqing/cholect50/sd")
            print(losses)
            json.dump(losses, open('./losses.json', 'w'))
            # eval with clip
            for step, (real_imgs, prompts) in enumerate(val_dataloader):
                mse_sum = 0
                imgs = torch.tensor([]).to(device)
                for prompt in prompts:
                    img = pipe(
                        prompt=prompt,  # What to generate
                        height=480,
                        width=640,  # Specify the image size
                        guidance_scale=8,  # How strongly to follow the prompt
                        num_inference_steps=35,  # How many steps to take
                        output_type='pt',
                    ).images[0]
                    imgs = torch.cat((imgs, img.unsqueeze(0)), dim=0)
                
                latents = pipe.vae.encode(imgs.to(device)).latent_dist.sample()
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],)).to(device)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                text_input_ids = pipe.tokenizer(  
                    prompts,
                    padding="max_length",  
                    max_length=pipe.tokenizer.model_max_length,  
                    truncation=True,  
                    return_tensors="pt"  
                ).input_ids.to(device)
                prompt_embeds = pipe.text_encoder(text_input_ids)[0]
                noise_pred = pipe.unet(noisy_latents, timesteps, prompt_embeds).sample
                mse = F.mse_loss(
                    noise_pred, noise
                )  # NB - trying to predict noise (eps) not (noisy_ims-clean_ims) or just (clean_ims)
                mse_sum += mse
                mse_scores.append(mse/3)
                
                # #clip
                # sd_clip_score = calculate_clip_score(imgs, list(prompts))
                # print(f"CLIP: {sd_clip_score}")
                # sd_clip_score_sum += sd_clip_score
                # if step == 3:
                #     clip_scores.append(sd_clip_score_sum/3)
                #     break

                # #fid
                # fid = FrechetInceptionDistance(normalize=True)
                # fid.update(real_imgs, real=True)
                # fid.update(imgs.cpu(), real=False)
                # fid_score = fid.compute()
                # fid_scores.append(float(fid_score))
                # print(f"FID: {float(fid_score)}")
            
            json.dump(mse_scores, open('./mse_scores.json', 'w'))

    print(f"Epoch {epoch} average loss: {sum(losses[-len(train_dataloader):])/len(train_dataloader)}")

# Plot the loss curve:
plt.plot(losses)