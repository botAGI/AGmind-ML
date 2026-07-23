import os, json, numpy as np
os.environ["HF_HUB_DISABLE_XET"]="1"
from sentence_transformers import SentenceTransformer
H=lambda p: os.path.expanduser("~/strizh/"+p)
# RU данные
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
qa=[q for q in qrels if q in topics][:1000]
# mixed
mit=json.load(open(H("mixed_dev.json")))[:2000]
mq=[x["q"] for x in mit]; mc=[x["pos"] for x in mit]
def ru_recall(m,qp,dp):
    D=m.encode([dp+t for t in texts],batch_size=256,normalize_embeddings=True,show_progress_bar=False)
    Q=m.encode([qp+topics[q] for q in qa],batch_size=256,normalize_embeddings=True,show_progress_bar=False)
    top=np.argsort(-(Q@D.T),axis=1)[:,:10]
    return sum(1 for i,q in enumerate(qa) if any(did2i.get(d) in top[i] for d in qrels[q] if d in did2i))/len(qa)
def mix_recall(m,qp,dp):
    D=m.encode([dp+t for t in mc],batch_size=128,normalize_embeddings=True,show_progress_bar=False)
    Q=m.encode([qp+t for t in mq],batch_size=128,normalize_embeddings=True,show_progress_bar=False)
    top=np.argsort(-(Q@D.T),axis=1)[:,:10]
    return sum(1 for i in range(len(mq)) if i in top[i])/len(mq)
CFG=[("USER2-small","deepvk/USER2-small","search_query: ","search_document: "),
     ("mE5-small","intfloat/multilingual-e5-small","query: ","passage: ")]
for tag,path,qp,dp in CFG:
    m=SentenceTransformer(path,device="cuda"); m.max_seq_length=512
    print("%-13s RU=%.3f mixed=%.3f"%(tag, ru_recall(m,qp,dp), mix_recall(m,qp,dp)), flush=True)
    del m; import torch; torch.cuda.empty_cache()
print("EXT_DONE", flush=True)
