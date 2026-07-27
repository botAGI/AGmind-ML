#!/usr/bin/env python3
"""Прод-паттерн RAG под мультиюзером на одном iGPU: N параллельных юзеров,
каждый в цикле гонит ПОЛНУЮ цепочку: embed(запрос) → rerank(запрос, 8 доков) →
LLM completion (128 ток., stream, TTFT по первому чанку).

Опционально: --index-port P — фоновая переиндексация (3 воркера, батчи 4×~350 ток.)
на указанный embedding-порт одновременно с юзерами.

args: users emb_port rerank_port llm_port duration_s [--index-port P]
Вывод: одна JSON-строка (стадийные латентности p50/p95, e2e, счётчики, LLM tok/s).
"""
import json, sys, time, threading, statistics
import urllib.request

USERS = int(sys.argv[1]); EMB = int(sys.argv[2]); RER = int(sys.argv[3]); LLM = int(sys.argv[4])
DUR = float(sys.argv[5])
IDX_PORT = int(sys.argv[sys.argv.index("--index-port") + 1]) if "--index-port" in sys.argv else None

QUERIES = ["как настроить pooling у эмбеддера в llama-server", "почему падает индексация в milvus",
           "как мониторить gpu на strix halo", "чем rerank отличается от retrieval",
           "как ускорить векторный поиск в проде", "настройка healthcheck в docker compose",
           "как проверить деградацию после квантизации", "что делать при exceed context size"]
DOCS = [("Документ %d. Настройка и эксплуатация сервиса в продовом стеке: контекст, пулинг, нормализация, "
         "лимиты, мониторинг и переиндексация. Практические детали конфигурации и типовые ошибки внедрения. " % i) * 2
        for i in range(8)]
IDX_TEXTS = [("Фрагмент индексации. Развёртывание RAG-стека: векторное хранилище, чанкинг, embedding-сервис, "
              "реранкер, мониторинг, лимиты контекста и политика переиндексации корпуса. " * 5)[:1300]] * 4

stop = threading.Event()
S = {"emb": [], "rer": [], "ttft": [], "gen_tps": [], "e2e": [], "err": 0, "done": 0, "idx_reqs": 0, "idx_lat": []}
L = threading.Lock()

def post(port, path, payload, timeout=180):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def one_request(i):
    q = QUERIES[i % len(QUERIES)]
    t0 = time.perf_counter()
    # 1) embed запроса
    post(EMB, "/v1/embeddings", {"input": q}).read()
    t1 = time.perf_counter()
    # 2) rerank топ-8 кандидатов → берём top-4
    rr = json.load(post(RER, "/v1/rerank", {"query": q, "documents": DOCS, "top_n": 4}))
    top_idx = [x["index"] for x in rr["results"][:4]]
    t2 = time.perf_counter()
    # 3) генерация ответа ПО ОТРАНЖИРОВАННОМУ КОНТЕКСТУ (stream, TTFT от отправки запроса)
    ctx = "\n\n".join(f"[Фрагмент {k+1}] {DOCS[i]}" for k, i in enumerate(top_idx))
    body = {"prompt": f"Контекст:\n{ctx}\n\nВопрос: {q}\nКраткий ответ по контексту:",
            "n_predict": 128, "temperature": 0, "stream": True, "cache_prompt": False}
    ttft = None; ntok = 0; tps = None
    t3 = time.perf_counter()          # ДО отправки: очередь и prefill входят в TTFT
    with post(LLM, "/completion", body) as r:
        for line in r:
            if not line.startswith(b"data: "): continue
            if ttft is None: ttft = (time.perf_counter() - t3) * 1000
            try: d = json.loads(line[6:])
            except Exception: continue
            ntok += 1
            if d.get("stop"): tps = d.get("timings", {}).get("predicted_per_second")
    t4 = time.perf_counter()
    with L:
        S["emb"].append((t1 - t0) * 1000); S["rer"].append((t2 - t1) * 1000)
        if ttft is not None: S["ttft"].append(ttft)
        if tps: S["gen_tps"].append(tps)
        S["e2e"].append((t4 - t0) * 1000); S["done"] += 1

def user_loop(uid):
    i = uid
    while not stop.is_set():
        try: one_request(i)
        except Exception:
            with L: S["err"] += 1
            time.sleep(0.5)
        i += USERS

def index_loop():
    while not stop.is_set():
        t0 = time.perf_counter()
        try:
            post(IDX_PORT, "/v1/embeddings", {"input": IDX_TEXTS}).read()
            with L: S["idx_reqs"] += 1; S["idx_lat"].append((time.perf_counter() - t0) * 1000)
        except Exception:
            with L: S["err"] += 1
            time.sleep(0.5)

def main():
    threads = [threading.Thread(target=user_loop, args=(u,), daemon=True) for u in range(USERS)]
    if IDX_PORT:
        threads += [threading.Thread(target=index_loop, daemon=True) for _ in range(3)]
    t_start = time.perf_counter()
    [t.start() for t in threads]
    time.sleep(DUR)
    stop.set(); time.sleep(2)
    wall = time.perf_counter() - t_start
    def pct(a, q):
        a = sorted(a); return round(a[min(len(a)-1, int(len(a)*q))], 1) if a else None
    out = {
        "users": USERS, "emb_port": EMB, "index_bg": bool(IDX_PORT), "wall_s": round(wall, 1),
        "req_done": S["done"], "req_per_min": round(S["done"] / wall * 60, 1), "errors": S["err"],
        "emb_ms": {"p50": pct(S["emb"], .5), "p95": pct(S["emb"], .95)},
        "rerank_ms": {"p50": pct(S["rer"], .5), "p95": pct(S["rer"], .95)},
        "llm_ttft_ms": {"p50": pct(S["ttft"], .5), "p95": pct(S["ttft"], .95)},
        "llm_gen_tps_per_stream_med": round(statistics.median(S["gen_tps"]), 1) if S["gen_tps"] else None,
        "e2e_ms": {"p50": pct(S["e2e"], .5), "p95": pct(S["e2e"], .95)},
        "index_batches_per_s": round(S["idx_reqs"] / wall, 1) if IDX_PORT else None,
        "index_p95_ms": pct(S["idx_lat"], .95) if IDX_PORT else None,
    }
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
