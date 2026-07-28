#!/usr/bin/env python3
"""Доказательство идентичности промптов: SHA-256 итогового prompt на каждый запрос.

results-prompt-tokens-* фиксирует только число токенов и символов, а равенство размера
идентичности не доказывает. Здесь пересобирается ТОТ ЖЕ промпт (та же цепочка и тот же
шаблон, что в prompt_tokens_check.py и pipeline_bench.py) и печатается его SHA-256,
а также идентификаторы отобранных чанков. LLM не вызывается: retrieval и rerank
детерминированы, поэтому промпт восстанавливается точно.

args: emb_port rerank_port corpus.json
"""
import hashlib, json, math, sys, urllib.request

EMB, RER, CORPUS = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
QUERIES = ["как настроить pooling у эмбеддера в llama-server", "почему падает индексация в векторном хранилище",
           "как мониторить gpu на strix halo", "чем reranking отличается от dense retrieval",
           "как ускорить векторный поиск в проде", "настройка healthcheck в docker compose",
           "как проверить деградацию модели после квантизации", "что делать при превышении контекста",
           "как устроен offline-режим установки", "какие сервисы входят в стек"]
CHUNKS = json.load(open(CORPUS, encoding="utf-8"))

def post(port, path, pl, t=180):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        json.dumps(pl).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))

def embed(texts): return [d["embedding"] for d in post(EMB, "/v1/embeddings", {"input": texts})["data"]]

V = []
for i in range(0, len(CHUNKS), 8): V.extend(embed(CHUNKS[i:i+8]))
N = [math.sqrt(sum(x*x for x in v)) or 1.0 for v in V]

out = []
for q in QUERIES:
    qv = embed([q])[0]; qn = math.sqrt(sum(x*x for x in qv)) or 1.0
    top8 = sorted(((sum(a*b for a, b in zip(qv, v))/(qn*n), i)
                   for i, (v, n) in enumerate(zip(V, N))), reverse=True)[:8]
    ids = [i for _, i in top8]
    docs = [CHUNKS[i] for i in ids]
    rr = post(RER, "/v1/rerank", {"query": q, "documents": docs, "top_n": 4})
    order = [x["index"] for x in rr["results"][:4]]
    top_ids = [ids[j] for j in order]
    ctx = "\n\n".join(f"[Фрагмент {k+1}] {docs[j]}" for k, j in enumerate(order))
    prompt = f"Контекст:\n{ctx}\n\nВопрос: {q}\nКраткий ответ по контексту:"
    out.append({"q": q, "emb_port": EMB, "top4_chunk_ids": top_ids, "chars": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
print(json.dumps(out, ensure_ascii=False))
