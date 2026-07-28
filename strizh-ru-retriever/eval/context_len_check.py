#!/usr/bin/env python3
"""Контроль confound'а: даёт ли выбор эмбеддера разную ДЛИНУ контекста в prompt.

В прод-паттерне TTFT LLM включает prompt processing извлечённого контекста.
Разные эмбеддеры выбирают разные чанки → разная длина prompt → разный TTFT,
что не имеет отношения к скорости самого эмбеддера. Скрипт меряет эту разницу.
Порты — те же серверы, что участвовали в замере pipeline_bench.
"""
import json, math, urllib.request, statistics

CH = json.load(open("corpus_ru.json", encoding="utf-8"))
Q = ["как настроить pooling у эмбеддера в llama-server", "почему падает индексация в векторном хранилище",
     "как мониторить gpu на strix halo", "чем reranking отличается от dense retrieval",
     "как ускорить векторный поиск в проде", "настройка healthcheck в docker compose",
     "как проверить деградацию модели после квантизации", "что делать при превышении контекста",
     "как устроен offline-режим установки", "какие сервисы входят в стек"]

def post(p, path, pl, t=180):
    r = urllib.request.Request(f"http://127.0.0.1:{p}{path}",
        json.dumps(pl).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))

def emb(p, texts): return [d["embedding"] for d in post(p, "/v1/embeddings", {"input": texts})["data"]]

res = {}
for port, name in ((8084, "strizh"), (8085, "bge-m3")):
    V = []
    for i in range(0, len(CH), 8): V.extend(emb(port, CH[i:i+8]))
    N = [math.sqrt(sum(x*x for x in v)) or 1.0 for v in V]
    tot = []
    for q in Q:
        qv = emb(port, [q])[0]; qn = math.sqrt(sum(x*x for x in qv)) or 1.0
        s = sorted(((sum(a*b for a, b in zip(qv, v))/(qn*n), i)
                    for i, (v, n) in enumerate(zip(V, N))), reverse=True)[:8]
        docs = [CH[i] for _, i in s]
        rr = post(8087, "/v1/rerank", {"query": q, "documents": docs, "top_n": 4})
        tot.append(sum(len(docs[x["index"]]) for x in rr["results"][:4]))
    res[name] = tot
    print(f"{name}: символов в top-4 контексте — медиана {int(statistics.median(tot))}, "
          f"среднее {sum(tot)//len(tot)}, min {min(tot)}, max {max(tot)}", flush=True)
    json.dump(res, open("context_len_check.json", "w"), ensure_ascii=False)

a, b = res["strizh"], res["bge-m3"]
print(f"дельта: strizh − bge = {sum(a)//len(a)-sum(b)//len(b):+d} симв. среднего "
      f"({(sum(a)/sum(b)-1)*100:+.1f}%) — при равной длине контекста TTFT сравним")
