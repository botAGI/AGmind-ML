---
language:
- ru
license: apache-2.0
base_model: t-tech/T-lite-it-2.1
pipeline_tag: text-generation
tags:
- rag
- chunking
- text-segmentation
- russian
- lora
---

# RU Context-Aware Document Splitter (T-lite-it-2.1 LoRA)

Fine-tune of **`t-tech/T-lite-it-2.1`** (Qwen3-8B) that segments Russian documents into self-contained semantic chunks for RAG, keeping tables/code atomic. Given text pre-split into numbered units, it returns the boundary indices + a topic as JSON.

## Usage
The model is a **completion** model trained on a raw Alpaca prompt (no chat template). Pre-segment the document into numbered units first; reconstruct chunks host-side from the returned indices.

**Prompt:**
```
### Instruction:
Раздели документ на смысловые части для системы поиска (RAG). Каждая часть читается независимо, не разрывая предложений, таблиц и кода. Верни ТОЛЬКО номера единиц, после которых проходит граница, в формате JSON.

### Input:
[1] Первое предложение. [2] Второе. [3] | таблица |...|
### Response:
```
**Output:** `{"splits": [2], "topic": "..."}` — `splits` = unit indices after which a chunk boundary falls (1-indexed). Slice the original text at those points.

Full pre/post-processing + a llama.cpp serving recipe: see the [GitHub repo](.).

## Results (1,500 held-out, agreement with teacher labels)
| Valid JSON | F1@0 | F1@±1 | exact-set |
|---|---|---|---|
| 100% | 0.656 | 0.821 | 29% |

GGUF Q5_K_M matches FP16 within quantization noise; runs on AMD Vulkan via llama.cpp.

## Training
bf16 LoRA (r32, rsLoRA, all-linear, response-only) on RTX 5090; ~17k teacher-distilled examples (DeepSeek-V4-Flash). See repo `docs/methodology.md`.

## Files
- LoRA adapter / merged FP16 weights
- `*-Q5_K_M.gguf` (llama.cpp, Vulkan/CPU)

## Limitations
Metrics are teacher-agreement, not human ground truth. Slight over-segmentation. For very large tables use a dedicated table-handling strategy, not this boundary model.

## License
Apache-2.0 (inherits the T-lite-it-2.1 base license).
