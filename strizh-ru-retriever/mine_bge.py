#!/usr/bin/env python3
"""iter-2: bge-m3 hard-negatives на train_v2 (двуязычный учитель → cross-lingual негативы).
Пул = уникальные позитивы; top-50 по bge-m3, негативы ранги 4-40 (скип top-3 и gold), 6/q."""
import os, json, numpy as np, torch, random
os.environ.setdefault("HF_HOME","/home/gamer/ru-splitter/hf"); os.environ["HF_HUB_DISABLE_XET"]="1"
from sentence_transformers import SentenceTransformer
H=lambda p: os.path.expanduser("~/strizh/"+p)
random.seed(42)
rows=json.load(open(H("train_v2.json")))
# уникальные позитивы = пул кандидатов
pool=[]; pkey={}
for r in rows:
    k=r["positive"][:200]
    if k not in pkey: pkey[k]=len(pool); pool.append(r["positive"])
print(f"queries={len(rows)} pool={len(pool)}",flush=True)
m=SentenceTransformer("BAAI/bge-m3", device="cuda"); m.max_seq_length=512
P=m.encode(pool, batch_size=128, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float16)
print("pool encoded",flush=True)
Pt=torch.from_numpy(P).cuda()
K=50; out=[]
BQ=2048
for a in range(0,len(rows),BQ):
    chunk=rows[a:a+BQ]
    Q=m.encode([r["anchor"] for r in chunk], batch_size=128, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float16)
    S=torch.from_numpy(Q).cuda()@Pt.T
    v,idx=S.topk(K,dim=1)
    idx=idx.cpu().numpy()
    for j,r in enumerate(chunk):
        gi=pkey.get(r["positive"][:200])
        cand=[int(i) for i in idx[j][3:] if int(i)!=gi][:37]
        if len(cand)<6: continue
        step=max(1,len(cand)//6)
        negs=[pool[cand[t]] for t in range(0,len(cand),step)][:6]
        out.append({"anchor":r["anchor"],"positive":r["positive"],"negs":negs})
    if a% (BQ*10)==0: print(f"  {a}/{len(rows)}",flush=True)
json.dump(out, open(H("train_v2_neg.json"),"w"), ensure_ascii=False)
print(f"MINE_DONE triplets={len(out)}",flush=True)
