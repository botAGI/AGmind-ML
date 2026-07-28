#!/usr/bin/env python3
"""Батчевая индексация ПОД КОНКУРЕНЦИЕЙ — тот режим, в котором работает -ub.

batch_thr.py последователен (concurrency 1): при батче 4 в полёте ~1400 токенов,
что влезает в один ubatch при любой из сравниваемых конфигураций, то есть флаг
там не может проявиться. Здесь W воркеров closed-loop, как в index_loop
pipeline_bench: суммарно в полёте W x batch чанков.

args: port workers batch duration_s texts.json
"""
import json, sys, time, threading, urllib.request

PORT, W, B, DUR = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
TEXTS = json.load(open(sys.argv[5], encoding="utf-8"))
BATCHES = [TEXTS[i:i+B] for i in range(0, len(TEXTS), B)]
cur = [0]; done = [0]; err = [0]; lat = []
L = threading.Lock()
deadline = None

def worker():
    while time.perf_counter() < deadline:
        with L:
            batch = BATCHES[cur[0] % len(BATCHES)]; cur[0] += 1
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/embeddings",
                json.dumps({"input": batch}).encode(), {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=120).read()
            t1 = time.perf_counter()
            with L:
                if t1 <= deadline: done[0] += 1; lat.append((t1-t0)*1000)
        except Exception:
            with L: err[0] += 1
            time.sleep(0.3)

ts = [threading.Thread(target=worker) for _ in range(W)]
deadline = time.perf_counter() + DUR
[t.start() for t in ts]; [t.join() for t in ts]
lat.sort()
p95 = lat[min(len(lat)-1, int(len(lat)*0.95))] if lat else None
print(json.dumps({"port": PORT, "workers": W, "batch": B, "window_s": DUR,
                  "batches_per_s": round(done[0]/DUR, 2), "texts_per_s": round(done[0]*B/DUR, 1),
                  "p95_ms": round(p95, 1) if p95 else None, "errors": err[0],
                  "tokens_in_flight_approx": W*B*350}, ensure_ascii=False))
