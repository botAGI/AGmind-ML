#!/usr/bin/env python3
"""STRIZH v2 co-train: донастройка v1 (strizh-embed-4L) на RU+EN+mixed парах.
MNRL in-batch negatives (cross-lingual negatives естественны — разные языки в батче).
Init = v1 (сильный старт: RU 0.71, EN 0.30, mixed 0.52). Цель: поднять EN/mixed, держать RU.
Первая итерация — простой MNRL; bge-m3 similarity-KL + RU-anchor = итерация 2 если RU просядет."""
import os, json, random, glob
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
H=lambda p: os.path.expanduser("~/strizh/"+p)
rows=[{"anchor":r["anchor"],"positive":r["positive"]} for r in json.load(open(H("train_v2.json")))]
random.seed(42); random.shuffle(rows)
print(f"rows={len(rows)}",flush=True)
ds=Dataset.from_list(rows)
model=SentenceTransformer(H("strizh-embed-4L"), device="cuda")  # init = v1
model.max_seq_length=256
loss=MultipleNegativesRankingLoss(model)
args=SentenceTransformerTrainingArguments(
    output_dir=H("v2_ckpt"),
    num_train_epochs=2, per_device_train_batch_size=256, learning_rate=3e-5,
    warmup_ratio=0.05, bf16=True, logging_steps=200, save_steps=1500, save_total_limit=2,
    report_to="none", dataloader_num_workers=2, gradient_checkpointing=True, dataloader_pin_memory=False)
trainer=SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
ck=glob.glob(H("v2_ckpt/checkpoint-*"))
trainer.train(resume_from_checkpoint=bool(ck))
model.save(H("strizh-embed-4L-v2"))
print("V2_DONE",flush=True)
