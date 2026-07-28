# eval — скрипты замеров strizh

Скрипты замеров качества и скорости. Пути к данным (`~/strizh/miracl_*`, `mixed_dev.json`) —
под нашу среду; датасеты (MIRACL-ru, ru_stackoverflow, MIRACL-en, opus-100) тянутся из HF или
готовятся отдельно. Полный rerun требует воспроизвести подготовку данных и окружение.

## Качество (retrieval)

| скрипт | что мерит | датасет |
|---|---|---|
| `dev_one.py <model>` | recall@10 / MRR@10 на русском | MIRACL-ru dev (см. passage-exposure ниже) |
| `dev_clean_ru.py <model> [qpref] [dpref]` | recall@10 full vs **отфильтрованная** подвыборка (без прямых совпадений gold с train-позитивами) | MIRACL-ru dev, 758/1000 |
| [`results-ru-filtered-2026-07-23.txt`](results-ru-filtered-2026-07-23.txt) | сохранённый прогон всех 5 моделей (env, prefix-политика, результаты) | — |
| [`results-speed-strix-2026-07-26.txt`](results-speed-strix-2026-07-26.txt) | контрольный скоростной прогон strizh vs bge-m3 (env, sha256, флаги, conc16 + sustained 60s) | — |
| `coresident_bench.py` + [`results-coresident-strix-2026-07-27.txt`](results-coresident-strix-2026-07-27.txt) | co-resident замер: дельта tok/s и TTFT LLM (Qwen3.6-35B) под open-loop QPS и indexing-нагрузкой каждого эмбеддера на одном iGPU | — |
| `pipeline_bench.py` + `corpus_ru.json` + [`results-rag-pipeline-strix-2026-07-27.txt`](results-rag-pipeline-strix-2026-07-27.txt) | RAG-конвейер под мультиюзером: embed → векторный поиск (99-чанковый индекс) → rerank → top-4 в prompt → generation; жёсткое окно 90с, сценарий с фоновой индексацией | — |
| `pipeline_bench.py` + [`results-rag-final-strix-2026-07-27.txt`](results-rag-final-strix-2026-07-27.txt) | **основной прогон RAG-конвейера**: 4 юзера, окно 420с, каждая точка снята дважды с обратным порядком во втором раунде | — |
| `context_len_check.py` | контроль confound'а: длина извлечённого контекста у strizh vs bge-m3 (TTFT включает prompt processing контекста) | — |
| [`results-embed-curves-strix-2026-07-27.txt`](results-embed-curves-strix-2026-07-27.txt) | изолированные кривые эмбеддеров: конкуренция 1–64 и размер батча 1–128 (длинные чанки) | — |
| `pipeline_bench_v1.py` | первая версия RAG-харнесса; ею снят лог на 90с (индексация гоняла первые 4 чанка, латентности с right-censoring). Для новых замеров не использовать | — |
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

Один харнесс на все модели (native pooling + префиксы каждой модели, 1000 запросов ×
9274 passage-кандидата — closed-candidate, не полный MIRACL leaderboard). **MIRACL-ru dev
НЕ является чистым holdout:** донор strizh (s-линия) видел часть dev-gold пассажей как
train-позитивы к другим запросам, поэтому честное RU-число снимается на отфильтрованной
подвыборке (`dev_clean_*.py`, 758/1000). Baseline-модели (USER2, mE5) меряются с их родными
префиксами. Первый прогон loadtest — cold (прогрев); берутся прогретые прогоны.

## Воспроизведение прогонов

Каждый `results-*.txt` во второй строке шапки называет **точный харнесс с sha256** и
оркестратор, которым он снят. Оркестраторы лежат в [`runners/`](runners/): в них
зафиксированы длительность окна, порядок точек, свипы и команды запуска серверов.

Входные данные замеров скорости: `texts_short_ru.json` (64 коротких запроса,
медиана 38 символов) и `texts_long_ru.json` (64 чанка, медиана ~1050 символов,
≈350 токенов) — их ждут `loadtest_v2.py` и `batch_thr.py` последним аргументом.
Корпус RAG-конвейера: `corpus_ru.json` (99 чанков документации AGmind).

Харнесс менялся между прогонами, поэтому лог на 90с пинуется на `pipeline_bench_v1.py`,
а финальный длинный прогон — на текущий `pipeline_bench.py`. Скрипт под тем же именем
не переписывается задним числом.
