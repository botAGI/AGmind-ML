#!/usr/bin/env python3
"""Батчевый throughput (реальный indexing-паттерн): B текстов в ОДНОМ /v1/embeddings.
args: port batch_size n_batches texts.json label"""
import sys, json, time, urllib.request
port=int(sys.argv[1]); B=int(sys.argv[2]); NB=int(sys.argv[3])
TEXTS=json.load(open(sys.argv[4])); LABEL=sys.argv[5]
def batch(i):
    chunk=[TEXTS[(i*B+j)%len(TEXTS)] for j in range(B)]
    t=time.perf_counter()
    req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings",
        data=json.dumps({"input":chunk}).encode(),headers={"Content-Type":"application/json"})
    urllib.request.urlopen(req,timeout=300).read()
    return time.perf_counter()-t
batch(0); batch(1)  # warmup
t0=time.perf_counter(); [batch(i) for i in range(NB)]; wall=time.perf_counter()-t0
tot=B*NB
print(f"{LABEL} batch={B:<4d} thr={tot/wall:8.1f}emb/s ({tot} emb / {wall:.1f}s)",flush=True)
