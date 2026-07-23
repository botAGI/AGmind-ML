# AGmind-ML

**Locally fine-tuned Russian models for the self-hosted [AGmind](https://github.com/botAGI/AGmind) RAG stack.**
Teacher distillation, quantization to GGUF, inference on **AMD (Vulkan, no CUDA)** via
`llama.cpp`. Commercial-OK licenses, fully local. Trained on a single consumer GPU
(RTX 5090), deployed on AMD Strix Halo. Three models shipped so far — a document
splitter, query expansion, and a retrieval embedder.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Serve](https://img.shields.io/badge/inference-AMD%20Vulkan-red)
![Lang](https://img.shields.io/badge/lang-RU-informational)

## Models

| Project | What it does | Base | Key numbers | Status |
|---|---|---|---|---|
| [**agmind-rag-splitter-ru**](agmind-rag-splitter-ru/) · [🤗](https://huggingface.co/AGmind/agmind-rag-splitter-ru) | Context-aware document splitter for RAG: semantic chunks, tables and code kept whole, lossless reconstruction | `t-tech/T-lite-it-2.1` (Qwen3-8B) | 100% valid JSON, boundary-F1@±1 0.821 | ✅ on HF |
| [**qmd-query-expansion-ru**](qmd-query-expansion-ru/) · [🤗](https://huggingface.co/AGmind/qmd-query-expansion-ru) | Search-query expansion for [qmd](https://github.com/tobi/qmd) (hyde/lex/vec, word forms for BM25); fixes the stock model's English hallucinations on Russian | `Qwen/Qwen3-1.7B` | reward-score 94.8; 100% format | ✅ on HF |
| [**strizh-ru-retriever**](strizh-ru-retriever/) · [🤗](https://huggingface.co/AGmind/strizh-ru-retriever) · [GGUF](https://huggingface.co/AGmind/strizh-ru-retriever-GGUF) | Drop-in dense retriever for Russian RAG (RU+EN): 4 layers, distilled by layer-pruning + BGE-M3 hard negatives. A size/speed specialist — smallest & fastest drop-in in its tier, not a quality leader | `deepvk/RuModernBERT-small` | recall@10 RU 0.80 / EN 0.34 / mixed 0.62; 24M, 3747 emb/s (1.4× USER2-small) | ✅ on HF |
| _next…_ | _(150M embedder, reranker, grounded-QA)_ | | | planned |

## Method

- **Teacher distillation.** A strong model labels the task or defines the target space —
  no manual annotation, with strict validation gates, dedup and retries.
- **One RTX 5090 (32 GB).** Generative models: bf16 LoRA (+rsLoRA), response-only loss.
  Embedders: teacher layer-pruning (warm-start) + contrastive (MNRL) on hard negatives.
- **Cyrillic bases.** Russian-tuned tokenizers are more economical than vanilla ones.
- **Deploy anywhere.** GGUF → `llama.cpp` Vulkan on AMD Strix Halo, no CUDA.

## Layout

```
AGmind-ML/
├── agmind-rag-splitter-ru/   # document splitter
├── qmd-query-expansion-ru/   # query expansion
└── strizh-ru-retriever/      # retrieval embedder
```

Each project has its own `README.md`: task, method, metrics, reproduction.
