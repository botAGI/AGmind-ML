#!/usr/bin/env python3
"""STRIZH-student: дистилляция 4L/384 из warm-start ws_late [0,5,9,11] обрезки s6.
Stage B напрямую (warm-start = anti-collapse, минует отдельный Stage A):
MNRL на FRIDA-hard-negatives (s6_triplets, наш выигравший рецепт s6) + MIRACL-train.
Init = ws_late_bb (backbone 4L), одна ModernBERT-загрузка → без мутации config.
dev-qrels СВЯЩЕННЫ — не в трейне."""
import os, json, random, glob
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from datasets import Dataset
from sentence_transformers import SentenceTransformer, models, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss

H=lambda p: os.path.expanduser("~/strizh/"+p)
rows=[]
# FRIDA hard-negative триплеты (глазами учителя) — тот же сигнал что вытянул s6 до чемпиона
for l in open(H("s6_triplets.jsonl")):
    d=json.loads(l)
    for neg in d["negs"]:
        rows.append({"anchor":d["q"],"positive":d["pos"],"negative":neg})
# MIRACL-train (dev — священный холд-аут, СЮДА НЕ входит)
topics={}
for l in open(H("miracl_topics.tsv")):
    p=l.strip().split("\t")
    if len(p)>=2: topics[p[0]]=p[1]
pas={}
for l in open(H("miracl_ru_passages.jsonl")):
    d=json.loads(l); pas[d["docid"]]=(d.get("title","")+" "+d["text"]).strip()
pasv=list(pas.values()); mir=0
random.seed(42)
for l in open(H("miracl_qrels_train.tsv")):
    p=l.strip().split("\t")
    if len(p)>=4 and p[3]=="1" and p[0] in topics and p[2] in pas:
        rows.append({"anchor":topics[p[0]],"positive":pas[p[2]],"negative":random.choice(pasv)}); mir+=1
random.shuffle(rows)
print(f"rows={len(rows)} (frida_triplets*6 + miracl={mir})",flush=True)
ds=Dataset.from_list(rows)

# student = warm-start late backbone + mean + normalize (одна ModernBERT-загрузка)
w=models.Transformer(H("ws_late_bb"), max_seq_length=256)
pool=models.Pooling(w.get_word_embedding_dimension(), pooling_mode="mean")
model=SentenceTransformer(modules=[w,pool,models.Normalize()], device="cuda")
loss=MultipleNegativesRankingLoss(model)
args=SentenceTransformerTrainingArguments(
    output_dir=H("student_4L_ckpt"),
    num_train_epochs=3, per_device_train_batch_size=256, learning_rate=8e-5,
    warmup_ratio=0.05, bf16=True, logging_steps=200, save_steps=1500, save_total_limit=2,
    report_to="none", dataloader_num_workers=2, gradient_checkpointing=True, dataloader_pin_memory=False)
trainer=SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
ck=glob.glob(H("student_4L_ckpt/checkpoint-*"))
trainer.train(resume_from_checkpoint=bool(ck))
model.save(H("strizh-embed-4L"))
print("STUDENT_DONE",flush=True)
