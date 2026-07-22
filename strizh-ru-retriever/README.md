# strizh-ru-retriever

Русский эмбеддер для поиска (retrieval) в RAG. 4 слоя, 24M параметров, вектор 384, mean-pooling, контекст 8192, без префиксов.

🤗 [AGmind/strizh-ru-retriever](https://huggingface.co/AGmind/strizh-ru-retriever) · [GGUF](https://huggingface.co/AGmind/strizh-ru-retriever-GGUF)

## Задача

Компактная модель, которая ищет релевантные документы по запросу быстро и на скромном железе (включая AMD Strix Halo через llama.cpp Vulkan). Ниша — между быстрым, но слабым в поиске rubert-tiny2 и тяжёлыми bge-m3 / USER2.

## Метрики

ruMTEB (MTEB rus v1), 23 задачи, один харнесс:

| задача | strizh | rubert-tiny2 |
|---|---|---|
| Retrieval | 0.280 | 0.089 |
| Reranking | 0.453 | 0.310 |
| Среднее (23) | 0.449 | 0.422 |

MIRACL-ru dev (holdout): recall@10 0.71 против 0.63. Сильнее в поиске и переранжировании, слабее в STS/парафразах.

Скорость (Strix Halo, Vulkan, Q8_0): в 1.2–1.5× быстрее rubert-tiny2 на одиночных запросах и индексации, паритет на пиковой конкурентности. Sustained 60с — 4771 эмб/с.

## Метод

1. **Warm-start.** 12-слойный retrieval-вариант на базе `deepvk/RuModernBERT-small` обрезается до 4 слоёв. Отбор слоёв — свипом по dev recall (`warmstart_sweep.py`); лучший набор [0,5,9,11]: ранние + поздние слои, где живёт retrieval-геометрия.
2. **Контрастив.** MNRL на hard-negatives, размеченных учителем FRIDA (recall@10 0.87), поверх 12M русских пассажей + MIRACL-ru train (`train_student.py`). Дистилляция из сильного учителя даёт 4-слойной модели качество выше самого 12-слойного донора.
3. **GGUF.** Конвертация через llama.cpp; `inject_epsilon.py` дописывает ключ `layer_norm_epsilon` (конвертер ModernBERT пишет RMS-вариант, а server-vulkan ждёт обычный). Пулинг — `--pooling mean`.

Оценка — `dev_one.py` на MIRACL-ru dev, который в обучение не попадал.

## Файлы

| файл | назначение |
|---|---|
| `warmstart_sweep.py` | обрезка донора до 4 слоёв, свип наборов слоёв по dev recall |
| `train_student.py` | контрастивная дистилляция (MNRL на FRIDA-негативах) |
| `dev_one.py` | замер recall@10 / MRR@10 на MIRACL-ru dev |
| `inject_epsilon.py` | пост-хок правка GGUF под llama.cpp server-vulkan |

Пути в скриптах — под нашу среду (`~/strizh/…`); данные (12M пассажей, FRIDA-негативы, MIRACL) готовятся отдельно.
