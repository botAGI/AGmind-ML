# AGmind-ML

**Fine-tuned local models for the self-hosted [AGmind](https://github.com/botAGI/AGmind) stack — the full lifecycle by hand:** teacher distillation → PEFT/LoRA → GGUF quantization → `llama.cpp` inference on **AMD (Vulkan, no CUDA)**. Commercial-OK licenses, fully local.

**Shipped — [`agmind-rag-splitter-ru`](agmind-rag-splitter-ru/):** a Russian context-aware RAG document splitter — semantic chunks, tables and code kept whole, lossless reconstruction. Base `t-tech/T-lite-it-2.1` (Qwen3-8B); bf16 LoRA + rsLoRA, response-only loss; Cyrillic-reworked tokenizer (~1.6× fewer tokens). **100% valid JSON · boundary-F1@±1 = 0.821.**

- 🤗 **Model:** https://huggingface.co/AGmind/agmind-rag-splitter-ru
- 🤗 **Dataset:** https://huggingface.co/datasets/AGmind/agmind-rag-splitter-ru-data
- 📝 **Write-up (RU, Habr):** https://habr.com/ru/articles/1055628/

_Полное описание на русском — ниже._

---

Дообученные модели для self-hosted AI-стека **AGmind**. Каждая модель: дистилляция из сильной модели-учителя, PEFT-обучение на потребительском железе (RTX 5090 / Blackwell), квантизация в GGUF и инференс на **AMD (Vulkan, без CUDA)** через `llama.cpp` — полностью локально, лицензии commercial-OK.

Каждый проект — в своей папке, самодостаточный (данные → обучение → оценка → инференс → доки).

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![PEFT](https://img.shields.io/badge/PEFT-bf16%20LoRA-orange)
![Serve](https://img.shields.io/badge/inference-AMD%20Vulkan-red)
![Lang](https://img.shields.io/badge/lang-RU-informational)

## Модели

| Проект | Что делает | База | Ключевые цифры | Статус |
|---|---|---|---|---|
| [**agmind-rag-splitter-ru**](agmind-rag-splitter-ru/) · [🤗 HF](https://huggingface.co/AGmind/agmind-rag-splitter-ru) | Русский context-aware сплиттер документов для RAG: смысловые чанки, таблицы и код целиком, lossless-реконструкция | `t-tech/T-lite-it-2.1` (Qwen3-8B) | валидный JSON **100%**, boundary-F1@±1 **0.821** | ✅ обучена + развёрнута |
| [**qmd-query-expansion-ru**](qmd-query-expansion-ru/) · [🤗 HF](https://huggingface.co/AGmind/qmd-query-expansion-ru) | Русское расширение поисковых запросов для [qmd](https://github.com/tobi/qmd) (hyde/lex/vec, словоформы для BM25); лечит EN-галлюцинации стоковой модели на русских запросах | `Qwen/Qwen3-1.7B` | reward-score датасета 94.8; формат 100% | ✅ обучена + на HF |
| [**strizh-ru-retriever**](strizh-ru-retriever/) · [🤗 HF](https://huggingface.co/AGmind/strizh-ru-retriever) · [GGUF](https://huggingface.co/AGmind/strizh-ru-retriever-GGUF) | Русский эмбеддер для поиска в RAG: 4 слоя, дистилляция обрезкой + FRIDA-негативы; в поиске и реранкинге сильнее rubert-tiny2, в 1.2–1.5× быстрее на Strix Halo | `deepvk/RuModernBERT-small` | ruMTEB retrieval **0.280** vs 0.089; recall@10 **0.71** vs 0.63 | ✅ обучена + на HF |
| _следующая…_ | _(150M-эмбеддер, reranker, grounded-QA)_ | | | в планах |

## Общий метод (для всех проектов)
- **Дистилляция от учителя** — сильная модель размечает задачу; без ручной разметки (жёсткие гейты валидации + дедуп + ретраи).
- **PEFT** — bf16 LoRA (+rsLoRA), response-only loss, на одной RTX 5090 (32 ГБ).
- **Кириллические базы** — токенайзеры переработаны под русский (≈в 1.6× меньше токенов, чем у ванильных).
- **Деплой где угодно** — merge → GGUF (Q5_K_M) → `llama.cpp` Vulkan на AMD Strix Halo.

## Структура
```
AGmind-ML/
├── agmind-rag-splitter-ru/   # модель 1 — см. её README
│   ├── data/ training/ eval/ inference/ docs/
└── <будущие модели>/
```

Подробности каждого проекта — в его `README.md`: постановка задачи, методология, метрики, шаги воспроизведения.
