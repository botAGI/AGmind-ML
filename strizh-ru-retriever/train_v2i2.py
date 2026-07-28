#!/usr/bin/env python3
"""v2 iter-2: MNRL с bge-m3 hard-negatives поверх v2-iter1. Cross-lingual негативы (bge-m3
знает оба языка). Тот приём, что вытянул v1 в чемпионы (FRIDA-негативы), но двуязычный."""
import os, json, random, glob
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
H=lambda p: os.path.expanduser("~/strizh/"+p)
rows=[]
for d in json.load(open(H("train_v2_neg.json"))):
    for neg in d["negs"]:
        rows.append({"anchor":d["anchor"],"positive":d["positive"],"negative":neg})
random.seed(42); random.shuffle(rows)
print(f"rows={len(rows)}",flush=True)
ds=Dataset.from_list(rows)
model=SentenceTransformer(H("strizh-embed-4L-v2"), device="cuda")  # init = v2-iter1
model.max_seq_length=256
loss=MultipleNegativesRankingLoss(model)
args=SentenceTransformerTrainingArguments(
    output_dir=H("v2i2_ckpt"), num_train_epochs=1, per_device_train_batch_size=256,
    learning_rate=2e-5, warmup_ratio=0.05, bf16=True, logging_steps=200, save_steps=2000,
    save_total_limit=2, report_to="none", dataloader_num_workers=2,
    gradient_checkpointing=True, dataloader_pin_memory=False)
trainer=SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
ck=glob.glob(H("v2i2_ckpt/checkpoint-*"))
trainer.train(resume_from_checkpoint=bool(ck))
model.save(H("strizh-embed-4L-v2i2"))
print("V2I2_DONE",flush=True)
