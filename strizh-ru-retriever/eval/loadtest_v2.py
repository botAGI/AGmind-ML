#!/usr/bin/env python3
"""Расширенный loadtest (loopback, stdlib). Режимы:
  n:<N>  — N запросов (online/indexing throughput+latency)
  t:<S>  — S секунд непрерывной нагрузки (sustained)
args: ports(csv) concurrency mode texts.json label"""
import sys, json, time, urllib.request, statistics, threading
from concurrent.futures import ThreadPoolExecutor
ports=[int(p) for p in sys.argv[1].split(",")]; C=int(sys.argv[2]); mode=sys.argv[3]
TEXTS=json.load(open(sys.argv[4])); LABEL=sys.argv[5] if len(sys.argv)>5 else ""
def one(i):
    text=TEXTS[i%len(TEXTS)]; port=ports[i%len(ports)]
    t=time.perf_counter()
    req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings",
        data=json.dumps({"input":[text]}).encode(),headers={"Content-Type":"application/json"})
    urllib.request.urlopen(req,timeout=120).read()
    return (time.perf_counter()-t)*1000
# warmup
with ThreadPoolExecutor(C) as ex: list(ex.map(one, range(C*3)))
lat=[]
if mode.startswith("n:"):
    N=int(mode[2:]); t0=time.perf_counter()
    with ThreadPoolExecutor(C) as ex: lat=list(ex.map(one, range(N)))
    wall=time.perf_counter()-t0
else:  # t:<seconds> sustained
    S=float(mode[2:]); stop=time.perf_counter()+S; cnt=[0]; lock=threading.Lock()
    def worker():
        while time.perf_counter()<stop:
            l=one(cnt[0])
            with lock: lat.append(l); cnt[0]+=1
    t0=time.perf_counter()
    ths=[threading.Thread(target=worker) for _ in range(C)]
    [t.start() for t in ths]; [t.join() for t in ths]
    wall=time.perf_counter()-t0
lat.sort(); p=lambda q: lat[min(len(lat)-1,int(len(lat)*q))]
print(f"{LABEL} inst={len(ports)} conc={C:<3d} n={len(lat):<5d} thr={len(lat)/wall:7.1f}emb/s "
      f"p50={p(.50):6.1f} p95={p(.95):6.1f} p99={p(.99):6.1f}ms",flush=True)
