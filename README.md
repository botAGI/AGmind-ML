# AGmind-ML

**Locally fine-tuned Russian models for the self-hosted [AGmind](https://github.com/botAGI/AGmind) RAG stack.** Teacher distillation, quantization to GGUF, inference on **AMD (Vulkan, no CUDA)** via `llama.cpp`. Commercial-OK licenses, fully local. Three models shipped so far — a document splitter, query expansion, and a retrieval embedder (see the table below).

_Полное описание на русском — ниже._

---

Дообученные русские модели для self-hosted RAG-стека **AGmind**. Учим на потребительском железе (RTX 5090), деплоим на **AMD Strix Halo (Vulkan, без CUDA)** через `llama.cpp` — полностью локально, лицензии commercial-OK.

Каждая модель живёт в своей папке: данные → обучение → оценка → инференс, со своим `README.md`.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Serve](https://img.shields.io/badge/inference-AMD%20Vulkan-red)
![Lang](https://img.shields.io/badge/lang-RU-informational)

## Модели

| Проект | Что делает | База | Ключевые цифры | Статус |
|---|---|---|---|---|
| [**agmind-rag-splitter-ru**](agmind-rag-splitter-ru/) · [🤗](https://huggingface.co/AGmind/agmind-rag-splitter-ru) | Context-aware сплиттер документов для RAG: смысловые чанки, таблицы и код целиком, lossless-реконструкция | `t-tech/T-lite-it-2.1` (Qwen3-8B) | валидный JSON 100%, boundary-F1@±1 0.821 | ✅ на HF |
| [**qmd-query-expansion-ru**](qmd-query-expansion-ru/) · [🤗](https://huggingface.co/AGmind/qmd-query-expansion-ru) | Расширение поисковых запросов для [qmd](https://github.com/tobi/qmd) (hyde/lex/vec, словоформы для BM25); лечит EN-галлюцинации стоковой модели на русском | `Qwen/Qwen3-1.7B` | reward-score 94.8; формат 100% | ✅ на HF |
| [**strizh-ru-retriever**](strizh-ru-retriever/) · [🤗](https://huggingface.co/AGmind/strizh-ru-retriever) · [GGUF](https://huggingface.co/AGmind/strizh-ru-retriever-GGUF) | Двуязычный (RU+EN) эмбеддер для поиска в RAG: 4 слоя, дистилляция обрезкой + BGE-M3; самый компактный и быстрый в своём классе, drop-in без префиксов | `deepvk/RuModernBERT-small` | recall@10 RU 0.80 / EN 0.34 / mixed 0.62; 3747 эмб/с (1.4× vs USER2-small) | ✅ на HF |
| _следующая…_ | _(150M-эмбеддер, reranker, grounded-QA)_ | | | в планах |

## Общий метод

- **Дистилляция от учителя.** Сильная модель размечает задачу или задаёт целевое пространство; без ручной разметки, с жёсткими гейтами валидации, дедупом и ретраями.
- **Обучение на одной RTX 5090 (32 ГБ).** Генеративные модели — bf16 LoRA (+rsLoRA), response-only loss. Эмбеддеры — обрезка слоёв учителя (warm-start) + контрастив (MNRL) на hard-negatives.
- **Кириллические базы.** Токенайзеры под русский экономнее ванильных.
- **Деплой где угодно.** GGUF → `llama.cpp` Vulkan на AMD Strix Halo, без CUDA.

## Структура

```
AGmind-ML/
├── agmind-rag-splitter-ru/   # сплиттер документов
├── qmd-query-expansion-ru/   # расширение запросов
└── strizh-ru-retriever/      # retrieval-эмбеддер
```

Детали каждого проекта — в его `README.md`: задача, метод, метрики, воспроизведение.
