#!/usr/bin/env python3
"""Прод-паттерн RAG под мультиюзером на одном iGPU.

Полная цепочка на КАЖДУЮ транзакцию:
  embed(запрос) → косинусный поиск по векторному индексу (top-8)
  → rerank(запрос, top-8) → top-4 в prompt → generation 128 токенов (stream, TTFT)

Индекс строится ОДИН РАЗ на старте тем же эмбеддером, что и запросы, из чанков
переданного корпуса. Опционально: --index-port P — фоновая переиндексация
(3 воркера closed-loop, батчи по 4 чанка) на тот же embedding-порт.

Окно измерения: жёсткий дедлайн. После дедлайна новые транзакции не стартуют,
уже запущенные дожидаются (join), в throughput идут ТОЛЬКО завершённые внутри окна.

args: users emb_port rerank_port llm_port duration_s corpus.json [--index-port P]
Вывод: одна JSON-строка.
"""
import json, sys, time, threading, statistics, math
import urllib.request

USERS = int(sys.argv[1]); EMB = int(sys.argv[2]); RER = int(sys.argv[3]); LLM = int(sys.argv[4])
DUR = float(sys.argv[5]); CORPUS = sys.argv[6]
IDX_PORT = int(sys.argv[sys.argv.index("--index-port") + 1]) if "--index-port" in sys.argv else None

QUERIES = ["как настроить pooling у эмбеддера в llama-server", "почему падает индексация в векторном хранилище",
           "как мониторить gpu на strix halo", "чем reranking отличается от dense retrieval",
           "как ускорить векторный поиск в проде", "настройка healthcheck в docker compose",
           "как проверить деградацию модели после квантизации", "что делать при превышении контекста",
           "как устроен offline-режим установки", "какие сервисы входят в стек"]

CHUNKS = json.load(open(CORPUS, encoding="utf-8"))
IDX_TEXTS = CHUNKS[:4]

stop = threading.Event()
deadline = None
S = {"emb": [], "search": [], "rer": [], "ttft": [], "gen_tps": [], "e2e": [],
     "err": 0, "started": 0, "in_window": 0, "after_window": 0,
     "idx_in_window": 0, "idx_lat": []}
L = threading.Lock()

def post(port, path, payload, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def embed(texts):
    r = json.load(post(EMB, "/v1/embeddings", {"input": texts}))
    return [d["embedding"] for d in r["data"]]

def build_index():
    """Индекс строится тем же эмбеддером, что обслуживает запросы."""
    vecs = []
    for i in range(0, len(CHUNKS), 16):
        vecs.extend(embed(CHUNKS[i:i+16]))
    norms = [math.sqrt(sum(x*x for x in v)) or 1.0 for v in vecs]
    return vecs, norms

INDEX, INORM = None, None

def search(qv, k=8):
    qn = math.sqrt(sum(x*x for x in qv)) or 1.0
    sims = [(sum(a*b for a, b in zip(qv, v)) / (qn * n), i) for i, (v, n) in enumerate(zip(INDEX, INORM))]
    sims.sort(reverse=True)
    return [i for _, i in sims[:k]]

def one_transaction(i):
    q = QUERIES[i % len(QUERIES)]
    t0 = time.perf_counter()
    qv = embed([q])[0]                                   # 1) эмбеддинг запроса
    t1 = time.perf_counter()
    cand = search(qv, 8)                                 # 2) поиск по индексу
    t2 = time.perf_counter()
    docs = [CHUNKS[i] for i in cand]
    rr = json.load(post(RER, "/v1/rerank", {"query": q, "documents": docs, "top_n": 4}))
    top = [docs[x["index"]] for x in rr["results"][:4]]   # 3) реранк
    t3 = time.perf_counter()
    ctx = "\n\n".join(f"[Фрагмент {k+1}] {d}" for k, d in enumerate(top))
    body = {"prompt": f"Контекст:\n{ctx}\n\nВопрос: {q}\nКраткий ответ по контексту:",
            "n_predict": 128, "temperature": 0, "stream": True, "cache_prompt": False}
    ttft = None; tps = None
    t4 = time.perf_counter()                             # 4) генерация; TTFT от отправки
    with post(LLM, "/completion", body) as r:
        for line in r:
            if not line.startswith(b"data: "): continue
            if ttft is None: ttft = (time.perf_counter() - t4) * 1000
            try: d = json.loads(line[6:])
            except Exception: continue
            if d.get("stop"): tps = d.get("timings", {}).get("predicted_per_second")
    t5 = time.perf_counter()
    inside = t5 <= deadline
    with L:
        if inside:
            S["in_window"] += 1
            S["emb"].append((t1-t0)*1000); S["search"].append((t2-t1)*1000)
            S["rer"].append((t3-t2)*1000); S["e2e"].append((t5-t0)*1000)
            if ttft is not None: S["ttft"].append(ttft)
            if tps: S["gen_tps"].append(tps)
        else:
            S["after_window"] += 1

def user_loop(uid):
    i = uid
    while time.perf_counter() < deadline:                # новые транзакции только до дедлайна
        with L: S["started"] += 1
        try: one_transaction(i)
        except Exception:
            with L: S["err"] += 1
            time.sleep(0.5)
        i += USERS

def index_loop():
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        try:
            post(IDX_PORT, "/v1/embeddings", {"input": IDX_TEXTS}).read()
            t1 = time.perf_counter()
            with L:
                if t1 <= deadline:
                    S["idx_in_window"] += 1; S["idx_lat"].append((t1-t0)*1000)
        except Exception:
            with L: S["err"] += 1
            time.sleep(0.5)

def main():
    global INDEX, INORM, deadline
    INDEX, INORM = build_index()                          # индекс до старта окна
    threads = [threading.Thread(target=user_loop, args=(u,)) for u in range(USERS)]
    if IDX_PORT: threads += [threading.Thread(target=index_loop) for _ in range(3)]
    t_start = time.perf_counter()
    deadline = t_start + DUR
    [t.start() for t in threads]
    [t.join() for t in threads]                           # дожидаемся in-flight, без daemon-обрезки
    def pct(a, q):
        a = sorted(a); return round(a[min(len(a)-1, int(len(a)*q))], 1) if a else None
    out = {
        "users": USERS, "emb_port": EMB, "index_bg": bool(IDX_PORT),
        "window_s": DUR, "corpus_chunks": len(CHUNKS),
        "started": S["started"], "completed_in_window": S["in_window"],
        "completed_after_window": S["after_window"], "errors": S["err"],
        "tx_per_min": round(S["in_window"] / DUR * 60, 1),
        "emb_ms": {"p50": pct(S["emb"], .5), "p95": pct(S["emb"], .95)},
        "vector_search_ms": {"p50": pct(S["search"], .5), "p95": pct(S["search"], .95)},
        "rerank_ms": {"p50": pct(S["rer"], .5), "p95": pct(S["rer"], .95)},
        "llm_ttft_ms": {"p50": pct(S["ttft"], .5), "p95": pct(S["ttft"], .95)},
        "llm_gen_tps_per_stream_med": round(statistics.median(S["gen_tps"]), 1) if S["gen_tps"] else None,
        "e2e_ms": {"p50": pct(S["e2e"], .5), "p95": pct(S["e2e"], .95)},
        "index_batches_per_s_in_window": round(S["idx_in_window"] / DUR, 1) if IDX_PORT else None,
        "index_p95_ms": pct(S["idx_lat"], .95) if IDX_PORT else None,
    }
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
