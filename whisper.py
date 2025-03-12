from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor
from datasets import Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration
from functools import partial
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import csv

from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    "united-we-care/United-Syn-Med", split="train[:4000]+validation[:4000]"
)
common_voice["test"] = load_dataset(
    "united-we-care/United-Syn-Med", split="test[:500]"
)

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="english", task="transcribe"
)

sampling_rate = processor.feature_extractor.sampling_rate
common_voice = common_voice.cast_column("mp3", Audio(sampling_rate=sampling_rate))

test_csv = csv.reader(open('/home/linziqing/.cache/huggingface/hub/datasets--united-we-care--United-Syn-Med/snapshots/54b992a26c1b2b00eeace87aea61c3596e2e0c88/data/test.csv', 'r'))
train_csv = csv.reader(open('/home/linziqing/.cache/huggingface/hub/datasets--united-we-care--United-Syn-Med/snapshots/54b992a26c1b2b00eeace87aea61c3596e2e0c88/data/train.csv', 'r'))
validation_csv = csv.reader(open('/home/linziqing/.cache/huggingface/hub/datasets--united-we-care--United-Syn-Med/snapshots/54b992a26c1b2b00eeace87aea61c3596e2e0c88/data/validation.csv', 'r'))

test_dict = {row[0][:-4]: row[1] for row in test_csv}
train_dict = {row[0][:-4]: row[1] for row in train_csv}
validation_dict = {row[0][:-4]: row[1] for row in validation_csv}

trans = {'test': test_dict, 'train': train_dict, 'validation': validation_dict}

def prepare_dataset(example):
    audio = example["mp3"]
    key = example["__key__"].split('/')
    text = trans[key[0]][key[1]]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=text,
    )

    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example

common_voice = common_voice.map(
    prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1
)

max_input_length = 30.0

def is_audio_in_length_range(length):
    return length < max_input_length

common_voice["train"] = common_voice["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

normalizer = BasicTextNormalizer()

class Scorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.3f" % (method, score))
                total_scores[method] = score

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))
        return total_scores

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    # print(pred_str[0])
    # print(label_str[0])

    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# model.freeze_encoder()

model.config.use_cache = False

model.generate = partial(
    model.generate, language='english', task="transcribe", use_cache=True
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./wsm_tta",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=400,
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=50,
    logging_steps=5,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train()