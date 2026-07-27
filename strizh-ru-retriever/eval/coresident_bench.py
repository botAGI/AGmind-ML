#!/usr/bin/env python3
"""Co-resident benchmark: влияние embedding-нагрузки на генерацию LLM (один iGPU).

Условия (каждое: 1 warmup-промпт + N измеренных, медианы):
  baseline          — LLM одна
  +<emb>@QPS        — LLM + open-loop embedding-нагрузка фиксированного QPS (короткие тексты)
  +<emb> indexing   — LLM + closed-loop батчевая индексация на полной тяге (длинные тексты)

Метрики LLM (клиентские + серверные): TTFT (первый SSE-чанк), wall-время,
tok/s генерации (server timings.predicted_per_second + client tokens/wall).
Метрики эмбеддера: достигнутый QPS, p50/p95, ошибки.

Использование: python3 coresident_bench.py <llm_port> <emb_port|-> <mode> [qps]
  mode: baseline | online (нужен qps) | indexing
Печатает одну JSON-строку результата — собирается внешним скриптом в лог.
"""
import json, sys, time, threading, statistics
import urllib.request

LLM_PORT = int(sys.argv[1])
EMB_PORT = None if sys.argv[2] == '-' else int(sys.argv[2])
MODE = sys.argv[3]
QPS = float(sys.argv[4]) if len(sys.argv) > 4 else 0
N_PROMPTS = 5

PROMPTS = [
    "Объясни разницу между dense retrieval и BM25 для поиска по русской технической документации.",
    "Напиши краткую инструкцию по настройке systemd-сервиса для docker compose стека.",
    "Какие подводные камни при квантизации embedding-моделей в GGUF? Ответь развёрнуто.",
    "Опиши, как устроен reranking в RAG-конвейере и когда он не помогает.",
    "Сравни подходы к чанкингу длинных документов для векторного поиска.",
    "Что такое gradient checkpointing и когда он ускоряет обучение?",
]
EMB_SHORT = ["как настроить pooling у эмбеддера", "ошибка индексации в milvus", "мониторинг gpu на strix halo",
             "как ускорить векторный поиск", "чем отличается rerank от retrieval", "настройка docker compose healthcheck",
             "почему падает llama-server", "как проверить квантизацию модели"]
EMB_LONG = [("Эксплуатация embedding-сервиса в продовом RAG-стеке требует внимания к деталям контекста, пулинга и нормализации. " * 12)[:1400]] * 4

stop_flag = threading.Event()
emb_stats = {"lat": [], "err": 0, "sent": 0}
emb_lock = threading.Lock()

def emb_call(texts):
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{EMB_PORT}/v1/embeddings",
            json.dumps({"input": texts}).encode(), {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=60).read()
        with emb_lock: emb_stats["lat"].append((time.perf_counter() - t0) * 1000)
    except Exception:
        with emb_lock: emb_stats["err"] += 1

def open_loop_driver():
    """Фиксированный QPS по таймеру — нагрузка не подстраивается под скорость сервера."""
    interval = 1.0 / QPS
    i = 0
    next_t = time.perf_counter()
    while not stop_flag.is_set():
        now = time.perf_counter()
        if now >= next_t:
            threading.Thread(target=emb_call, args=([EMB_SHORT[i % len(EMB_SHORT)]],), daemon=True).start()
            with emb_lock: emb_stats["sent"] += 1
            i += 1
            next_t += interval
        else:
            time.sleep(min(0.001, next_t - now))

def indexing_driver():
    """Closed-loop полная тяга: 6 воркеров, батчи по 4 длинных текста."""
    def worker():
        while not stop_flag.is_set():
            emb_call(EMB_LONG)
            with emb_lock: emb_stats["sent"] += 1
    for _ in range(6):
        threading.Thread(target=worker, daemon=True).start()

def llm_call(prompt):
    """Streaming completion: TTFT по первому чанку, tok/s из server timings."""
    body = json.dumps({"prompt": prompt, "n_predict": 200, "temperature": 0, "stream": True,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{LLM_PORT}/completion", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n_tok = 0
    srv_tps = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.startswith(b"data: "): continue
            if ttft is None: ttft = (time.perf_counter() - t0) * 1000
            try: d = json.loads(line[6:])
            except Exception: continue
            n_tok += 1
            if d.get("stop"):
                tm = d.get("timings", {})
                srv_tps = tm.get("predicted_per_second")
                n_tok = tm.get("predicted_n", n_tok)
    wall = time.perf_counter() - t0
    return {"ttft_ms": ttft, "wall_s": wall, "tokens": n_tok,
            "srv_tps": srv_tps, "cli_tps": n_tok / wall if wall else 0}

def main():
    if MODE == "online": open_loop_driver_t = threading.Thread(target=open_loop_driver, daemon=True); open_loop_driver_t.start()
    elif MODE == "indexing": indexing_driver()
    if MODE != "baseline": time.sleep(3)  # нагрузка выходит на режим до первого промпта

    llm_call(PROMPTS[-1])  # warmup, не считаем
    runs = [llm_call(PROMPTS[i % len(PROMPTS)]) for i in range(N_PROMPTS)]
    stop_flag.set(); time.sleep(1)

    lat = sorted(emb_stats["lat"])
    p = lambda q: lat[min(len(lat)-1, int(len(lat)*q))] if lat else None
    med = lambda k: statistics.median(r[k] for r in runs if r[k] is not None)
    out = {
        "mode": MODE, "emb_port": EMB_PORT, "target_qps": QPS or None,
        "llm": {"ttft_ms_med": round(med("ttft_ms"), 1),
                "srv_tps_med": round(med("srv_tps"), 2),
                "cli_tps_med": round(med("cli_tps"), 2),
                "runs": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in runs]},
        "emb": {"achieved_rps": round(len(lat) / max(1e-9, sum(r["wall_s"] for r in runs)), 1) if lat else None,
                "done": len(lat), "sent": emb_stats["sent"], "errors": emb_stats["err"],
                "p50_ms": round(p(0.5), 1) if lat else None, "p95_ms": round(p(0.95), 1) if lat else None},
    }
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
