#!/usr/bin/env python3
"""Прямая проверка: объясняет ли длина промпта разницу TTFT между эмбеддерами.

Символьный прокси (context_len_check.py) не годится: prompt processing считается
в токенах, а корпус смешанный (русская проза + код + YAML), где символ-на-токен
различается. Здесь берётся серверная метрика llama-server timings.prompt_n —
фактическое число токенов промпта, — и TTFT одиночного потока без очереди.

Цепочка на запрос та же, что в pipeline_bench: embed -> cosine top-8 -> rerank top-4
-> тот же шаблон prompt -> generation. Запросы идут ПОСЛЕДОВАТЕЛЬНО (конкуренции нет),
поэтому TTFT здесь = чистый prefill, без вклада очереди.

args: emb_port rerank_port llm_port corpus.json [repeats]
"""
import json, sys, time, math, statistics
import urllib.request

EMB, RER, LLM = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
CORPUS = sys.argv[4]
REPEATS = int(sys.argv[5]) if len(sys.argv) > 5 else 3

QUERIES = ["как настроить pooling у эмбеддера в llama-server", "почему падает индексация в векторном хранилище",
           "как мониторить gpu на strix halo", "чем reranking отличается от dense retrieval",
           "как ускорить векторный поиск в проде", "настройка healthcheck в docker compose",
           "как проверить деградацию модели после квантизации", "что делать при превышении контекста",
           "как устроен offline-режим установки", "какие сервисы входят в стек"]

CHUNKS = json.load(open(CORPUS, encoding="utf-8"))

def post(port, path, payload, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def embed(texts):
    return [d["embedding"] for d in json.load(post(EMB, "/v1/embeddings", {"input": texts}))["data"]]

def build_index():
    vecs = []
    for i in range(0, len(CHUNKS), 8):
        vecs.extend(embed(CHUNKS[i:i+8]))
    return vecs, [math.sqrt(sum(x*x for x in v)) or 1.0 for v in vecs]

INDEX, INORM = build_index()

def search(qv, k=8):
    qn = math.sqrt(sum(x*x for x in qv)) or 1.0
    s = sorted(((sum(a*b for a, b in zip(qv, v))/(qn*n), i)
                for i, (v, n) in enumerate(zip(INDEX, INORM))), reverse=True)
    return [i for _, i in s[:k]]

rows = []
for q in QUERIES:
    qv = embed([q])[0]
    docs = [CHUNKS[i] for i in search(qv, 8)]
    rr = json.load(post(RER, "/v1/rerank", {"query": q, "documents": docs, "top_n": 4}))
    top = [docs[x["index"]] for x in rr["results"][:4]]
    ctx = "\n\n".join(f"[Фрагмент {k+1}] {d}" for k, d in enumerate(top))
    prompt = f"Контекст:\n{ctx}\n\nВопрос: {q}\nКраткий ответ по контексту:"
    for rep in range(REPEATS):
        body = {"prompt": prompt, "n_predict": 32, "temperature": 0, "stream": True, "cache_prompt": False}
        ttft = None; pn = None; pms = None
        t0 = time.perf_counter()
        with post(LLM, "/completion", body) as r:
            for line in r:
                if not line.startswith(b"data: "): continue
                if ttft is None: ttft = (time.perf_counter() - t0) * 1000
                try: d = json.loads(line[6:])
                except Exception: continue
                if d.get("stop"):
                    tm = d.get("timings", {})
                    pn, pms = tm.get("prompt_n"), tm.get("prompt_ms")
        rows.append({"q": q, "rep": rep, "chars": len(prompt), "prompt_n": pn,
                     "prompt_ms": round(pms, 1) if pms else None, "ttft_ms": round(ttft, 1)})

med = lambda k: statistics.median(r[k] for r in rows if r[k] is not None)
mean = lambda k: statistics.mean(r[k] for r in rows if r[k] is not None)
out = {
    "emb_port": EMB, "queries": len(QUERIES), "repeats": REPEATS, "n_rows": len(rows),
    "prompt_tokens": {"median": med("prompt_n"), "mean": round(mean("prompt_n"), 1),
                      "min": min(r["prompt_n"] for r in rows), "max": max(r["prompt_n"] for r in rows)},
    "prompt_chars": {"median": med("chars"), "mean": round(mean("chars"), 1)},
    "chars_per_token": round(mean("chars") / mean("prompt_n"), 2),
    "ttft_ms_single_stream": {"median": med("ttft_ms"), "mean": round(mean("ttft_ms"), 1)},
    "server_prompt_ms": {"median": med("prompt_ms"), "mean": round(mean("prompt_ms"), 1)},
    "rows": rows,
}
print(json.dumps(out, ensure_ascii=False))
