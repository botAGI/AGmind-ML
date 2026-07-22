import os, json, sys, numpy as np, torch
os.environ.setdefault("HF_HOME","/home/gamer/ru-splitter/hf")
from sentence_transformers import SentenceTransformer, models
H=lambda p: os.path.expanduser("~/strizh/"+p)
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
w=models.Transformer(sys.argv[1], max_seq_length=512)
pool=models.Pooling(w.get_word_embedding_dimension(), pooling_mode="mean")
m=SentenceTransformer(modules=[w,pool,models.Normalize()], device="cuda")
D=m.encode(texts,batch_size=256,normalize_embeddings=True,show_progress_bar=False)
Q=m.encode([topics[q] for q in qids],batch_size=256,normalize_embeddings=True,show_progress_bar=False)
S=Q@D.T; top=np.argsort(-S,axis=1)[:,:10]
r10=mrr=0
for i,q in enumerate(qids):
    gold={did2i[d] for d in qrels[q] if d in did2i}
    hits=[j for j,idx in enumerate(top[i]) if idx in gold]
    if hits: r10+=1; mrr+=1/(hits[0]+1)
print(f"{os.path.basename(sys.argv[1])}: recall@10={r10/len(qids):.4f} MRR@10={mrr/len(qids):.4f}")
