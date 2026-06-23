# AGmind-ML

Fine-tuned models for the **AGmind** self-hosted AI stack. Each model is teacher-distilled, trained with parameter-efficient fine-tuning on consumer hardware (RTX 5090 / Blackwell), quantized to GGUF, and served on **AMD (Vulkan, no CUDA)** via `llama.cpp` — fully local, commercial-OK licenses.

Each project lives in its own folder, self-contained (data → training → eval → inference → docs).

## Models

| Project | What it does | Base | Headline results | Status |
|---|---|---|---|---|
| [**agmind-rag-splitter-ru**](agmind-rag-splitter-ru/) | Russian context-aware document splitter for RAG — semantic chunks, tables/code kept whole, lossless reconstruction | `t-tech/T-lite-it-2.1` (Qwen3-8B) | JSON-valid **100%**, boundary-F1@±1 **0.821** | ✅ trained + deployed |
| _next…_ | _(e.g. guardian / grounded-RAG generator)_ | | | planned |

## Method (shared across projects)
- **Teacher distillation** — a frontier model labels the task; no human annotation (self-consistency + hard validation gates).
- **PEFT** — bf16 LoRA (+rsLoRA), response-only loss, on a single RTX 5090 (32 GB).
- **Cyrillic-native bases** — tokenizers re-worked for Russian (≈1.6× fewer tokens than vanilla).
- **Deploy anywhere** — merge → GGUF (Q5_K_M) → `llama.cpp` Vulkan on AMD Strix Halo.

## Repo layout
```
AGmind-ML/
├── agmind-rag-splitter-ru/   # model 1 — see its README
│   ├── data/ training/ eval/ inference/ docs/
└── <future models>/
```

See each project's `README.md` for problem statement, methodology, metrics, and reproduction steps.
