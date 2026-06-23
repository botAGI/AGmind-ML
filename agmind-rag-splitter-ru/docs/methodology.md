# Methodology & decisions

End-to-end notes on building a Russian context-aware RAG splitter — base selection, PEFT, data, deployment — with the reasoning behind each choice.

## 1. Base model — `t-tech/T-lite-it-2.1`
Requirements that drove the choice (intersection, not one factor):
1. **Runs on llama.cpp Vulkan (AMD, no CUDA)** — this disqualified the *newest* options: Qwen3.5-9B and Granite-4 are SSM/linear-attention hybrids with no Vulkan kernels (garbage output on the target box). Newer ≠ better when the architecture breaks the runtime.
2. **Cyrillic-efficient tokenizer** — T-lite re-worked the vocab (~2.4 tok/word RU vs ~3.9 vanilla). The Danish original's Llama-2 tokenizer is Cyrillic-inefficient.
3. **Apache-2.0** (commercial-OK) — ruled out YandexGPT-5 (custom NC) and Mistral MRL.
4. **Official GGUF exists** → the custom-tokenizer → GGUF round-trip is already proven by the vendor.

Among dense + Vulkan-safe + Cyrillic + Apache models, T-lite-it-2.1 is the freshest. Fallback: `RefalMachine/RuadaptQwen3-8B-Hybrid` (slightly better Cyrillic fertility, GGUF round-trip needs checking).

## 2. PEFT — bf16 LoRA (r32) + rsLoRA, response-only
- **bf16, not QLoRA:** the 8B fits in 32 GB bf16, so 4-bit buys nothing — and bitsandbytes-4bit on Blackwell sm_120 is fragile + throughput-penalized (~58% of FP16). QLoRA is the OOM fallback only.
- **rsLoRA on:** strictly-better scaling (`alpha/√r`); free, and unlocks safe r=64 if needed.
- **Not DoRA / full-FT:** DoRA adds 13–49% time + memory (pushes 8B out of bf16) with no benefit on a narrow task; full-FT ≈ LoRA on small instruction sets ("LoRA Without Regret") but adds catastrophic-forgetting risk.
- **Response-only loss:** the output (boundary JSON) is tiny vs the input; without masking, ~95% of the loss would be spent reproducing the input.

## 3. Data — teacher distillation with an index output
- **Index output, not text re-emission.** The reference Danish model outputs the full text of each chunk; that risks byte-mismatch (the model "fixes" ё/quotes → chunks ≠ source → breaks RAG citation) and is ~10× slower. Emitting **indices** + host-side slicing is lossless, fast, and keeps tables atomic by construction.
- **Teacher:** a frontier model labels boundaries; self-consistency + hard gates substitute for human annotation.
- **Mix:** real prose (web/edu/technical) + synthetic tables/code (12 domains) to guarantee table/code-atomicity coverage. A dynamic proportion guard keeps synthetic from dominating.

## 4. Deployment — GGUF on AMD Vulkan
- Convert merged FP16 → GGUF → quantize Q5_K_M (~5.9 GB).
- **Tokenizer gotcha:** T-lite's BPE pre-tokenizer hash isn't in upstream llama.cpp → `convert_hf_to_gguf.py` errors. Fix: register the hash as `qwen2` (`training/patch_tokenizer_hash.py`).
- Verify the GGUF on the target hardware: boundary-F1 of the Q5/Vulkan model matched the FP16 model within quantization noise (0.639 vs 0.656 @0) — tokenizer + quantization round-trip is clean.

## 5. What this is NOT for
A boundary splitter handles prose + small/medium tables. **Huge tables** (beyond the embedder budget) need a different strategy — header-repetition row-chunking + table summaries + whole-table retrieval (parent-document by `table_id`) — driven by the document parser + retrieval architecture, not an LLM boundary model.

## Lessons learned (the hard way)
- Pin Blackwell deps precisely (torch cu129 not cu130; align torchaudio to torch; raise `ulimit -n`).
- Use a gamer-owned HF cache (a root-owned `~/.cache/huggingface` silently fails as `PermissionError`).
- Select checkpoints by the **task metric**, not eval_loss.
- Verify the deployed GGUF on the real hardware, not just the trainer.
