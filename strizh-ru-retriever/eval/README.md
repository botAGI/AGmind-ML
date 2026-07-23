# eval — скрипты замеров strizh

Все цифры из карточки и README воспроизводятся этими скриптами. Пути к данным (`~/strizh/miracl_*`, `mixed_dev.json`) — под нашу среду; датасеты (MIRACL-ru, ru_stackoverflow, GooAQ, opus-100) тянутся из HF или готовятся отдельно.

## Качество (retrieval)

| скрипт | что мерит | датасет |
|---|---|---|
| `dev_one.py <model>` | recall@10 / MRR@10 на русском | MIRACL-ru dev (holdout) |
| `bench_v2.py` | RU + mixed для одной модели | MIRACL-ru + ru_stackoverflow |
| `mixed_bench.py` | mixed RU+код, 3 модели | ru_stackoverflow (RU-вопрос + EN-код) |
| `extended_bench.py` | RU + mixed для USER2/mE5 с их префиксами | MIRACL-ru + ru_stackoverflow |
| `cross_lingual.py` | RU↔EN retrieval, 4 модели | opus-100 en-ru (параллельные пары) |
| `en_nanobeir.py <model>` | английский nDCG@10 | NanoBEIR (mteb) |
| `gguf_eval.py <url> <tag>` | качество GGUF через llama-server `/v1/embeddings` | MIRACL-ru dev |

## Скорость (llama.cpp Vulkan)

Клиенты бьют по локальному `llama-server` (эмбеддинги), чистый stdlib:

| скрипт | что мерит |
|---|---|
| `loadtest_v2.py <ports> <conc> <mode> <texts.json>` | throughput + p50/p95/p99; режимы `n:<N>` (запросов) или `t:<сек>` (sustained) |
| `batch_thr.py <port> <batch> <n_batches> <texts.json>` | батчевый throughput (indexing) |

Сервер поднимается как `llama-server -m strizh-ru-retriever.Q8_0.gguf --embeddings --pooling mean -c 8192 -np 8`; клиенты — на loopback для чистой латентности.

## Метод

Один харнесс на все модели (mean-pool + L2-норма на клиенте), MIRACL-ru dev — священный holdout (в обучение не попадает). Baseline-модели (USER2, mE5) меряются с их родными префиксами. Первый прогон loadtest — cold (прогрев); берутся прогретые прогоны.
