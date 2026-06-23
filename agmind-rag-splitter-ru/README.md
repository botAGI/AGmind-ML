# RU Context-Aware Document Splitter for RAG

A Russian-language **context-aware document chunker** for Retrieval-Augmented Generation: it segments documents into self-contained semantic chunks, **keeps tables and code blocks intact**, and emits structured boundaries that reconstruct **byte-identically** from the source.

Fine-tuned from **`t-tech/T-lite-it-2.1`** (Qwen3-8B, Apache-2.0) via teacher-distillation from a frontier model, trained with bf16 LoRA on a single RTX 5090, quantized to GGUF, and served on **AMD (Vulkan)** with `llama.cpp`.

> Inspired by [`mhenrichsen/context-aware-splitter`](https://huggingface.co/mhenrichsen/context-aware-splitter-1b) (Danish) — re-designed for Russian with a Cyrillic-native tokenizer and a lossless index-based output.

---

## Results

Evaluated on a 1,500-example held-out set (boundary agreement with the teacher labels), greedy decoding:

| Metric | HF model (RTX 5090) | GGUF Q5_K_M (AMD Vulkan) |
|---|---|---|
| Valid JSON output | **100%** | **100%** |
| boundary-F1 @0 (exact position) | **0.656** | 0.639 |
| boundary-F1 @±1 (±1 sentence) | **0.821** | 0.817 |
| exact-set-match | 29% | 25% |

- **100% parseable structured output** — the model always returns `{"splits":[...], "topic":"..."}`.
- GGUF/Vulkan matches the FP16 model within quantization noise → **tokenizer + Q5 quantization preserve quality**.
- Tables stay **atomic** (verified on live documents — see `inference/demo.py`).

*(These measure agreement with the distillation teacher, not human ground truth — see [Limitations](#limitations).)*

---

## Approach

```
                        ┌──────────────────────── TRAIN (one-time) ────────────────────────┐
 Russian corpora  ──►  razdel sentence-split + table/code as atomic units  ──►  numbered units
 (wiki/habr/synth)                                                                     │
                                                                                       ▼
                              Teacher (DeepSeek-V4-Flash)  labels boundary indices + topic
                                                                                       │
                                                                                       ▼
                        bf16 LoRA (r32, rsLoRA, all-linear, response-only)  on  T-lite-it-2.1
                                                                                       │
                                                              merge ──► GGUF (Q5_K_M) ──► AMD Vulkan
 ────────────────────────────────────────────────────────────────────────────────────────────────
                        ┌──────────────────────── INFER ───────────────────────────────────┐
 document ─► [host] split into numbered units ─► [model] {"splits":[i,...],"topic":...}
          ─► [host] slice ORIGINAL text by indices ─► chunks (byte-identical, tables whole)
```

**Key design choices**

- **Cyrillic-native base.** `T-lite-it-2.1` (Qwen3-8B continued-pretrain by T-Bank) carries a re-worked tokenizer (~2.4 tok/word on Russian vs ~3.9 vanilla). A Llama-2 tokenizer (as in the Danish original) is Cyrillic-inefficient.
- **Index output, not text re-emission.** The model returns boundary **indices**, and the host slices the original text. This is **lossless** (no paraphrasing of ё/quotes), **~10× cheaper** (≈40 output tokens vs re-emitting the whole document), and **guarantees tables stay intact**.
- **Teacher distillation.** A strong model labels where to cut; no human annotation. Self-consistency + hard validation gates filter the labels.
- **Response-only loss.** Only the short JSON output contributes to the loss — the model isn't wasted learning to reproduce the input.

---

## Repo layout

```
data/
  generate.py          # teacher-distillation data generator (corpora → numbered units → teacher labels → Alpaca JSONL)
  sample_train.jsonl   # 600-example sample (full ~17k set: see data/README.md)
  sample_holdout.jsonl
training/
  train.py             # bf16 LoRA (Unsloth + TRL), response-only, rsLoRA
  patch_tokenizer_hash.py  # registers T-lite's BPE pre-tokenizer hash for GGUF conversion
eval/
  eval_hf.py           # boundary-F1 / JSON-valid / exact-match on the HF model
  eval_gguf.py         # same metrics against the deployed GGUF endpoint
inference/
  demo.py              # live chunking of a document (shows table stays whole)
  wrapper_openai.py    # optional OpenAI-compatible wrapper (text → chunks)
  ru-chunker.dify.yml  # importable Dify workflow (Start → preprocess → model → reconstruct → End)
docs/
  methodology.md       # base/PEFT/data/deploy decisions with rationale
  model_card.md        # HuggingFace model card (ready to publish)
```

---

## Quickstart

**1. Generate the dataset** (needs an OpenAI-compatible teacher endpoint):
```bash
pip install -r requirements.txt
# edit the teacher URL/model in data/generate.py
python data/generate.py 17000 train.jsonl 48      # target, output, workers
```

**2. Train** (RTX 5090 / Blackwell, WSL2; exact pins in `training/config.md`):
```bash
DATA=train.jsonl HOLD=train_holdout.jsonl OUTDIR=out EPOCHS=2 MAXLEN=4096 BS=2 GA=8 \
  python training/train.py
```

**3. Convert to GGUF** (patch the tokenizer hash first):
```bash
python training/patch_tokenizer_hash.py          # registers T-lite pre-tokenizer
python llama.cpp/convert_hf_to_gguf.py out_merged --outfile model-f16.gguf --outtype f16
./llama.cpp/build/bin/llama-quantize model-f16.gguf model-Q5_K_M.gguf Q5_K_M
```

**4. Serve** (AMD Vulkan):
```bash
llama-server -m model-Q5_K_M.gguf -ngl 99 -c 8192 --host 0.0.0.0 --port 8085
```

**5. Evaluate / demo:**
```bash
MODEL=out_merged N=300 python eval/eval_hf.py
python inference/demo.py
```

---

## Hardware

- **Training:** RTX 5090 (Blackwell, 32 GB) under WSL2 — bf16 LoRA, ~3.5 h, peak 25.4 GB VRAM, 2,122 steps (2 epochs over ~17k examples).
- **Inference:** AMD Strix Halo (gfx1151) via `llama.cpp` **Vulkan** (no CUDA) — Q5_K_M, ~5.9 GB.
- **Teacher / data-gen:** any OpenAI-compatible endpoint (we used a self-hosted DeepSeek-V4-Flash).

---

## Limitations

- Metrics are **boundary agreement with the teacher's labels**, not human-validated ground truth. A downstream RAG-retrieval eval (hit-rate / faithfulness) is the next step.
- The model **slightly over-segments** (occasional extra boundary) — a tendency inherited from the teacher's granularity; mitigable with a post-merge step or teacher-prompt tuning.
- Designed for **prose + small/medium tables**. **Very large tables** (exceeding the embedder budget) need a separate table-handling strategy (header-repetition row-chunking + whole-table retrieval), not a boundary splitter.

## License

Apache-2.0 (matches the `T-lite-it-2.1` base and the permissive corpora). Training data derived from publicly available Russian corpora — see `data/README.md` for per-source licenses.

## Credits

- Base model: [`t-tech/T-lite-it-2.1`](https://huggingface.co/t-tech/T-lite-it-2.1) (T-Bank)
- Concept inspiration: [`mhenrichsen/context-aware-splitter`](https://huggingface.co/mhenrichsen/context-aware-splitter-1b)
- Trained with [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl), served with [llama.cpp](https://github.com/ggml-org/llama.cpp)
