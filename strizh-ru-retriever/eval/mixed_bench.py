import os, json, numpy as np

os.environ["HF_HUB_DISABLE_XET"]="1"
from sentence_transformers import SentenceTransformer
items=json.load(open(os.path.expanduser("~/strizh/mixed_dev.json")))
items=items[:2000]
queries=[it["q"] for it in items]; corpus=[it["pos"] for it in items]
def bench(tag, path):
    m=SentenceTransformer(path, device="cuda"); m.max_seq_length=512
    D=m.encode(corpus, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    Q=m.encode(queries, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    S=Q@D.T; top=np.argsort(-S,axis=1)[:,:10]
    r10=sum(1 for i in range(len(queries)) if i in top[i])/len(queries)
    mrr=sum(1/(list(top[i]).index(i)+1) for i in range(len(queries)) if i in top[i])/len(queries)
    print("%-12s recall@10=%.3f MRR@10=%.3f" % (tag, r10, mrr), flush=True)
    del m; import torch; torch.cuda.empty_cache()
bench("strizh-v1", os.path.expanduser("~/strizh/strizh-embed-4L"))
bench("tiny2", "cointegrated/rubert-tiny2")
bench("bge-m3", "BAAI/bge-m3")
print("MIXED_BENCH_DONE", flush=True)
