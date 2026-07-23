#!/usr/bin/env python3
"""GGUF-качество через llama-server /v1/embeddings (llama-server Vulkan) → recall@10/MRR@10
на MIRACL-ru dev. L2-норм на клиенте (не зависим от серверной настройки). urllib (без requests)."""
import sys, json, time, numpy as np, urllib.request
URL=sys.argv[1]; TAG=sys.argv[2] if len(sys.argv)>2 else "?"
H=lambda p: __import__("os").path.expanduser("~/strizh/"+p)

def embed(texts, bs=32):
    out=[None]*len(texts)
    for a in range(0,len(texts),bs):
        chunk=texts[a:a+bs]
        for attempt in range(4):
            try:
                req=urllib.request.Request(URL+"/v1/embeddings",
                    data=json.dumps({"input":chunk}).encode(),
                    headers={"Content-Type":"application/json"})
                d=json.loads(urllib.request.urlopen(req,timeout=300).read())["data"]
                for e in d: out[a+e["index"]]=e["embedding"]
                break
            except Exception as ex:
                if attempt==3: raise
                time.sleep(3)
    v=np.array(out,dtype=np.float32)
    v/=np.linalg.norm(v,axis=1,keepdims=True)+1e-9
    return v

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
print(f"{TAG}: corpus={len(texts)} queries={len(qids)}",flush=True)

t0=time.time()
D=embed(texts); Q=embed([topics[q] for q in qids])
S=Q@D.T; top=np.argsort(-S,axis=1)[:,:10]
r10=mrr=0
for i,q in enumerate(qids):
    gold={did2i[d] for d in qrels[q] if d in did2i}
    hits=[j for j,idx in enumerate(top[i]) if idx in gold]
    if hits: r10+=1; mrr+=1/(hits[0]+1)
print(f"GGUF_RESULT {TAG} recall@10={r10/len(qids):.4f} MRR@10={mrr/len(qids):.4f} ({time.time()-t0:.0f}s)",flush=True)
