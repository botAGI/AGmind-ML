#!/usr/bin/env python3
"""Eval RU-splitter checkpoint on holdout: JSON-valid %, boundary-F1 (exact + windowed ±1), exact-set-match.
Env: MODEL (path: merged dir OR checkpoint dir with adapter), HOLD, N. Run on winbox 5090."""
import os, sys, json, re
os.environ.setdefault("HF_HOME","/home/gamer/ru-splitter/hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset

MODEL=os.environ.get("MODEL","out_v1_merged")
HOLD=os.environ.get("HOLD","train_v1_holdout.jsonl")
N=int(os.environ.get("N","300"))
print(f"=== EVAL MODEL={MODEL} N={N} ===", flush=True)

model,tok=FastLanguageModel.from_pretrained(MODEL, max_seq_length=4096, dtype=torch.bfloat16, load_in_4bit=False)
FastLanguageModel.for_inference(model)
tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token

ds=load_dataset("json",data_files=HOLD,split="train")
N=min(N,len(ds)); ds=ds.select(range(N))

def parse_splits(txt):
    m=re.search(r'\{.*?\}', txt, re.S)
    if not m: return None
    try:
        o=json.loads(m.group(0)); return set(int(x) for x in o.get("splits",[]) if isinstance(x,(int,float)))
    except Exception: return None

def f1(pred,gold,win=0):
    if not pred and not gold: return 1.0
    if not pred or not gold: return 0.0
    def hit(a,B): return any(abs(a-b)<=win for b in B)
    tp_p=sum(1 for a in pred if hit(a,gold)); tp_g=sum(1 for b in gold if hit(b,pred))
    P=tp_p/len(pred); R=tp_g/len(gold)
    return 2*P*R/(P+R) if (P+R) else 0.0

prompts=[f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n" for ex in ds]
golds=[set(int(x) for x in json.loads(ex["output"])["splits"]) for ex in ds]
BATCH=16; valid=0; exact=0; f1e=[]; f1w=[]
for i in range(0,len(prompts),BATCH):
    bp=prompts[i:i+BATCH]
    inp=tok(bp,return_tensors="pt",padding=True,truncation=True,max_length=4096).to("cuda")
    with torch.no_grad():
        out=model.generate(**inp,max_new_tokens=220,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
    for j in range(len(bp)):
        gen=tok.decode(out[j][inp["input_ids"].shape[1]:],skip_special_tokens=True)
        ps=parse_splits(gen); g=golds[i+j]
        if ps is None: continue
        valid+=1
        if ps==g: exact+=1
        f1e.append(f1(ps,g,0)); f1w.append(f1(ps,g,1))
    print(f"  {i+len(bp)}/{N}",flush=True)
n=len(prompts)
print(f"\nRESULT {MODEL}")
print(f"JSON-valid:      {100*valid/n:.1f}%")
print(f"boundary-F1@0:   {sum(f1e)/len(f1e):.3f}" if f1e else "n/a")
print(f"boundary-F1@±1:  {sum(f1w)/len(f1w):.3f}" if f1w else "n/a")
print(f"exact-set-match: {100*exact/n:.1f}%")
