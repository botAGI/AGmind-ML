#!/usr/bin/env python3
"""Warm-start обрезка s6 (12 слоёв) → 4 слоя разными наборами + dev recall КАЖДОГО БЕЗ обучения.
Сохраняет ST-обёртку каждого варианта. Лучший старт → база дистилляции.
Ключевое: сохранять layer_types (local/global rope theta) каждого выбранного слоя — иначе rope mismatch."""
import os, json, numpy as np, torch, shutil
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from transformers import AutoModel, AutoConfig, AutoTokenizer
from sentence_transformers import SentenceTransformer, models

H=lambda p: os.path.expanduser("~/strizh/"+p)
S6=H("strizh-embed-30m-s6")

# --- dev данные (священный холд-аут) ---
qrels={}
for l in open(H("miracl_qrels_dev.tsv")):
    p=l.strip().split("\t")
    if len(p)>=4 and p[3]=="1": qrels.setdefault(p[0],set()).add(p[2])
topics={}
for l in open(H("miracl_ru_dev.tsv")):
    p=l.strip().split("\t")
    if len(p)>=2: topics[p[0]]=p[1]
docids=[]; texts=[]
for l in open(H("miracl_dev_passages.jsonl")):
    d=json.loads(l); docids.append(d["docid"]); texts.append((d.get("title","")+" "+d["text"]).strip()[:2000])
did2i={d:i for i,d in enumerate(docids)}
qids=[q for q in qrels if q in topics][:1000]

def dev_recall(st_path):
    m=SentenceTransformer(st_path, device="cuda"); m.max_seq_length=512
    D=m.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    Q=m.encode([topics[q] for q in qids], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    S=Q@D.T; top=np.argsort(-S,axis=1)[:,:10]
    r10=mrr=0
    for i,q in enumerate(qids):
        gold={did2i[d] for d in qrels[q] if d in did2i}
        hits=[j for j,idx in enumerate(top[i]) if idx in gold]
        if hits: r10+=1; mrr+=1/(hits[0]+1)
    del m; torch.cuda.empty_cache()
    return r10/len(qids), mrr/len(qids)

# --- s6 веса + config ---
s6=AutoModel.from_pretrained(S6); sd=s6.state_dict()
cfg0=AutoConfig.from_pretrained(S6)
LT=cfg0.layer_types if getattr(cfg0,"layer_types",None) else ["global_attention" if i%3==0 else "local_attention" for i in range(12)]
print(f"s6 layer_types: {LT}",flush=True)

from safetensors.torch import save_file
def make_warmstart(selected, tag):
    # state_dict студента = чистое ПЕРЕИМЕНОВАНИЕ весов s6 (без создания модели → без ModernBertConfig-бага)
    nsd={}
    for k,v in sd.items():
        if ".layers." not in k: nsd[k]=v.contiguous().clone()  # embeddings, final_norm
    for ni,oi in enumerate(selected):
        for k,v in sd.items():
            if f"layers.{oi}." in k:
                nsd[k.replace(f"layers.{oi}.", f"layers.{ni}.")]=v.contiguous().clone()
    bb=H(f"ws_{tag}_bb"); os.makedirs(bb, exist_ok=True)
    save_file(nsd, os.path.join(bb,"model.safetensors"), metadata={"format":"pt"})
    cj=json.load(open(os.path.join(S6,"config.json")))
    cj["num_hidden_layers"]=len(selected)
    cj["layer_types"]=[LT[i] for i in selected]
    rp=cj.get("rope_parameters")
    if isinstance(rp,dict): cj["rope_parameters"]={k:v for k,v in rp.items() if k in ("full_attention","sliding_attention")}
    json.dump(cj, open(os.path.join(bb,"config.json"),"w"), ensure_ascii=False, indent=1)
    for f in ["tokenizer.json","tokenizer_config.json"]: shutil.copy(os.path.join(S6,f), os.path.join(bb,f))
    return bb

VARIANTS={
    "last4":   [8,9,10,11],
    "strided": [2,5,8,11],
    "late":    [0,5,9,11],
    "spread":  [0,4,8,11],
}
for tag,sel in VARIANTS.items():
    make_warmstart(sel, tag)
    print(f"CREATED ws_{tag} {sel}",flush=True)
print("ALL_CREATED",flush=True)
