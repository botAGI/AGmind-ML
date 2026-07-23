# strizh-ru-retriever

**RU-first** dense-эмбеддер для поиска в RAG (с остаточной поддержкой английского).
4 слоя, 24.4M параметров, вектор 384, mean-pooling, без префиксов; архитектура принимает
8192 токена (контрастив обучался на 256).

🤗 [AGmind/strizh-ru-retriever](https://huggingface.co/AGmind/strizh-ru-retriever) · [GGUF](https://huggingface.co/AGmind/strizh-ru-retriever-GGUF)

## Задача

Компактная модель для поиска релевантных документов по русскому запросу, быстрая на
`llama.cpp`/Vulkan (AMD Strix Halo), где эмбеддер делит один GPU с LLM и reranker'ом —
маленькая быстрая модель возвращает GPU-время генерации. Drop-in без префиксов, работает
как первая стадия в гибридном (вектор+BM25) поиске под reranker.

## Метрики

Один харнесс, MIRACL-ru dev, recall@10 на **passage-exposure-clean подвыборке** (758 из
1000 запросов, у которых ни один gold-пассаж не встречался в train-позитивах). Донор
видел часть dev-пассажей, поэтому на полном наборе strizh = 0.80 — завышено на ~5пп.
EN = NanoBEIR nDCG@10 (не recall), cross-lingual = opus-100 recall@10. **Метрики разных
типов — не усреднять.**

| ось | strizh (24M) | rubert-tiny2 | USER2-small (34M) | bge-m3 (568M) |
|---|---|---|---|---|
| Русский recall@10 (clean) | 0.75 | 0.63 | 0.82 | 0.83 |
| Английский nDCG@10 | 0.34 | 0.17 | 0.53 | 0.60 |
| Cross-lingual recall@10 | 0.76 | не измерял | 0.95 | 0.96 |

strizh обходит `rubert-tiny2` на русском (0.75 vs 0.63) и на Strix Halo по скорости
(1.2–1.5×, Q8_0), но **на CPU `rubert-tiny2` быстрее** (357 vs 265 эмб/с, ONNX-пути пока
нет). Более крупные `USER2-small` и `bge-m3` качественнее — особенно на английском и
cross-lingual. Это **RU-first** модель, а не равносильный двуязычный retriever. Замер на
`ru_stackoverflow` (mixed) пересекается с обучением → это диагностика, не бенчмарк.

## Метод

1. **Warm-start.** 12-слойный русский retrieval-донор на базе `deepvk/RuModernBERT-small`
   обрезается до 4 слоёв `[0, 5, 9, 11]` (`warmstart_sweep.py`).
2. **Co-train.** Один перемешанный поток русских (GPL), английских (**MIRACL-en**, 40k) и
   смешанных (`ru_stackoverflow`) пар — без последовательных этапов, чтобы не терять
   русский (`data_v2.py`, `train_v2.py`).
3. **Hard-negatives от BGE-M3.** Учитель держит оба языка в одном пространстве →
   cross-lingual негативы; финальный контрастивный проход (`mine_bge.py`, `train_v2i2.py`).
4. **GGUF.** Конвертация ModernBERT-тракта; `inject_epsilon.py` под llama.cpp
   server-vulkan; сервинг с `--pooling mean` (иначе CLS роняет recall до ~0.02).

Русский оценивается на MIRACL-ru dev; из-за passage-exposure донора headline-число снято
на clean-подвыборке (см. «Метрики»), а не на полном наборе.

Токенизатор RuModernBERT обучен на смеси RU+EN — английская fertility на уровне
мультиязычных баз (xlm-r/bge-m3), поэтому база не менялась, а компактный словарь (50k)
держит модель маленькой.

## Файлы

| файл | назначение |
|---|---|
| `warmstart_sweep.py` | обрезка донора до 4 слоёв |
| `data_v2.py` | сборка RU+EN(MIRACL-en)+mixed обучающих пар |
| `train_v2.py` | co-train (MNRL) на смеси языков |
| `mine_bge.py` | cross-lingual hard-negatives от BGE-M3 |
| `train_v2i2.py` | финальный контрастивный проход |
| `dev_one.py` | замер recall@10 на MIRACL-ru dev |
| `inject_epsilon.py` | правка GGUF под llama.cpp server-vulkan |

Пути в скриптах — под нашу среду; данные готовятся отдельно.

Замеры (качество RU/EN/cross-lingual + скорость + end-to-end RAG-обкатка) — воспроизводимы
скриптами в [`eval/`](eval/).
