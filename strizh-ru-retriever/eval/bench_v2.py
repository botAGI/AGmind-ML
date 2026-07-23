import os, json, numpy as np
os.environ["HF_HUB_DISABLE_XET"]="1"
from sentence_transformers import SentenceTransformer
H=lambda p: os.path.expanduser("~/strizh/"+p)
m=SentenceTransformer(H("strizh-embed-4L-v2"), device="cuda"); m.max_seq_length=512
# RU: MIRACL-ru dev
qrels={}
for l in open(H("miracl_qrels_dev.tsv")):
    p=l.strip().split("\t")
    if len(p)>=4 and p[3]=="1": qrels.setdefault(p[0],set()).add(p[2])
topics={}
for l in open(H("miracl_ru_dev.tsv")):
    p=l.strip().split("\t")
    if len(p)>=2: topics[p[0]]=p[1]
docids=[];texts=[]
for l in open(H("miracl_dev_passages.jsonl")):
    d=json.loads(l); docids.append(d["docid"]); texts.append((d.get("title","")+" "+d["text"]).strip()[:2000])
did2i={d:i for i,d in enumerate(docids)}
qids=[q for q in qrels if q in topics][:1000]
D=m.encode(texts,batch_size=256,normalize_embeddings=True,show_progress_bar=False)
Q=m.encode([topics[q] for q in qids],batch_size=256,normalize_embeddings=True,show_progress_bar=False)
S=Q@D.T; top=np.argsort(-S,axis=1)[:,:10]
r=sum(1 for i,q in enumerate(qids) if any(did2i.get(d) in top[i] for d in qrels[q] if d in did2i))/len(qids)
print("RU  MIRACL-ru recall@10=%.3f (v1 было 0.71)"%r, flush=True)
# mixed
it=json.load(open(H("mixed_dev.json")))[:2000]
qs=[x["q"] for x in it]; cp=[x["pos"] for x in it]
Dc=m.encode(cp,batch_size=128,normalize_embeddings=True,show_progress_bar=False)
Qc=m.encode(qs,batch_size=128,normalize_embeddings=True,show_progress_bar=False)
Sc=Qc@Dc.T; tc=np.argsort(-Sc,axis=1)[:,:10]
rm=sum(1 for i in range(len(qs)) if i in tc[i])/len(qs)
print("MIX ru_stackoverflow recall@10=%.3f (v1 было 0.52)"%rm, flush=True)
print("BENCH_V2_DONE", flush=True)
