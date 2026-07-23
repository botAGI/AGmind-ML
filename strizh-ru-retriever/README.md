# strizh-ru-retriever

Двуязычный (русский + английский) эмбеддер для поиска в RAG. 4 слоя, 24M параметров, вектор 384, mean-pooling, контекст 8192, без префиксов.

🤗 [AGmind/strizh-ru-retriever](https://huggingface.co/AGmind/strizh-ru-retriever) · [GGUF](https://huggingface.co/AGmind/strizh-ru-retriever-GGUF)

## Задача

Компактная модель, которая ищет релевантные документы по запросу на русском, английском или в смешанном тексте (русский документ с английским кодом/терминами — типичная техдокументация), быстро и на скромном железе (включая AMD Strix Halo через llama.cpp Vulkan).

## Метрики

Один харнесс, recall@10 (RU/mixed) и nDCG@10 (EN):

| ось | strizh (24M) | rubert-tiny2 | bge-m3 (568M) |
|---|---|---|---|
| Русский (MIRACL-ru) | 0.80 | 0.63 | 0.85 |
| Английский (NanoBEIR) | 0.34 | 0.17 | 0.60 |
| Смешанный RU+код | 0.62 | 0.27 | 0.85 |

Обходит rubert-tiny2 на всех осях при том же размере/скорости, подбирается к bge-m3 (в ~20× крупнее) на русском. Слабее в STS/парафразах. Скорость: в 1.2–1.5× быстрее rubert-tiny2 на Strix Halo (Q8_0).

## Метод

1. **Warm-start.** 12-слойный русский retrieval-донор на базе `deepvk/RuModernBERT-small` обрезается до 4 слоёв (`warmstart_sweep.py`).
2. **Co-train.** Один перемешанный поток русских (GPL), английских (GooAQ) и смешанных (ru_stackoverflow) пар — без последовательных этапов, чтобы не терять русский (`data_v2.py`, `train_v2.py`).
3. **Hard-negatives от BGE-M3.** Учитель держит оба языка в одном пространстве → cross-lingual негативы; финальный контрастивный проход (`mine_bge.py`, `train_v2i2.py`).
4. **GGUF.** Конвертация ModernBERT-тракта; `inject_epsilon.py` под llama.cpp server-vulkan; `--pooling mean`.

Русский оценивается на MIRACL-ru dev (holdout).

Токенизатор RuModernBERT обучен на смеси RU+EN — английская fertility на уровне мультиязычных баз (xlm-r/bge-m3), поэтому база не менялась, а компактный словарь (50k) держит модель маленькой и быстрой.

## Файлы

| файл | назначение |
|---|---|
| `warmstart_sweep.py` | обрезка донора до 4 слоёв |
| `data_v2.py` | сборка RU+EN+mixed обучающих пар |
| `train_v2.py` | co-train (MNRL) на смеси языков |
| `mine_bge.py` | cross-lingual hard-negatives от BGE-M3 |
| `train_v2i2.py` | финальный контрастивный проход |
| `dev_one.py` | замер recall@10 на MIRACL-ru dev |
| `inject_epsilon.py` | правка GGUF под llama.cpp server-vulkan |

Пути в скриптах — под нашу среду; данные готовятся отдельно.

Замеры (качество RU/EN/mixed/cross-lingual + скорость) — воспроизводимы скриптами в [`eval/`](eval/).
