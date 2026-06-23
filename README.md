# AGmind-ML

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
| _следующая…_ | _(guardian / grounded-RAG генератор и т.д.)_ | | | в планах |

## Общий метод (для всех проектов)
- **Дистилляция от учителя** — сильная модель размечает задачу; без ручной разметки (self-consistency + жёсткие гейты валидации).
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
