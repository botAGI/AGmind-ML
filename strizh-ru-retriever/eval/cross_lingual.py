import os, numpy as np
os.environ["HF_HUB_DISABLE_XET"]="1"
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
ds=load_dataset("Helsinki-NLP/opus-100","en-ru",split="test",streaming=True)
ru=[];en=[]
for r in ds:
    t=r["translation"]; e=t.get("en","").strip(); u=t.get("ru","").strip()
    if len(e)>20 and len(u)>20: en.append(e); ru.append(u)
    if len(ru)>=2000: break
n=len(ru); print(f"opus-100 en-ru pairs: {n}",flush=True)
def xling(m,qp,dp):
    RU=m.encode([qp+x for x in ru],batch_size=128,normalize_embeddings=True,show_progress_bar=False)
    EN=m.encode([dp+x for x in en],batch_size=128,normalize_embeddings=True,show_progress_bar=False)
    r2e=np.argsort(-(RU@EN.T),axis=1)[:,:10]; a=sum(1 for i in range(n) if i in r2e[i])/n
    e2r=np.argsort(-(EN@RU.T),axis=1)[:,:10]; b=sum(1 for i in range(n) if i in e2r[i])/n
    return a,b
CFG=[("strizh-v2",os.path.expanduser("~/strizh/strizh-embed-4L-v2i2"),"",""),
     ("USER2-small","deepvk/USER2-small","search_query: ","search_document: "),
     ("mE5-small","intfloat/multilingual-e5-small","query: ","passage: "),
     ("bge-m3","BAAI/bge-m3","","")]
print("модель        RU->EN  EN->RU",flush=True)
for tag,path,qp,dp in CFG:
    m=SentenceTransformer(path,device="cuda"); m.max_seq_length=512
    a,b=xling(m,qp,dp); print("%-13s %.3f  %.3f"%(tag,a,b),flush=True)
    del m; import torch; torch.cuda.empty_cache()
print("XLING_DONE",flush=True)
