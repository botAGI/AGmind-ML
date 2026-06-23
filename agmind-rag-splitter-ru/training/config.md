# Training configuration

## Hardware / environment (verified working)
- **GPU:** RTX 5090 (NVIDIA Blackwell, compute capability `sm_120`, 32 GB)
- **OS:** WSL2 (Ubuntu 24.04) on Windows — native Windows is unreliable for the Blackwell training stack
- **Peak VRAM:** ~25.4 GB · **wall time:** ~3.5 h (2 epochs, ~17k examples)

### Version pins (Blackwell sm_120 — the load-bearing part)
| package | pin | note |
|---|---|---|
| torch | `2.11.0` (cu129 index) | **never cu130** — ABI clash with bitsandbytes; `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129` |
| triton | `>=3.3.1` | Blackwell minimum |
| unsloth + unsloth_zoo | latest | `pip install unsloth unsloth_zoo` |
| transformers / trl / peft / accelerate | latest stable | |
| python | 3.12 | |
| NVIDIA driver | R570+ (CUDA ≥12.8) | |

Sanity check before training: `python -c "import torch; print('sm_120' in torch.cuda.get_arch_list())"` must print `True`.
Raise the file-descriptor limit (`ulimit -n 1048576`) — torch.compile/inductor opens many files.

## Recipe (PEFT)
- **Method:** bf16 LoRA (NOT QLoRA — 8B fits in 32 GB bf16, and bitsandbytes-4bit on Blackwell is fragile/throughput-penalized).
- **LoRA:** `r=32`, `alpha=32`, `dropout=0.05`, `use_rslora=True`, all linear targets (`q,k,v,o,gate,up,down`).
- **Loss:** response-only (mask instruction+input; only the JSON output is in the loss).
- **Optim:** `adamw_8bit`, `lr=2e-4` cosine, warmup 5%, weight_decay 0.01.
- **Batch:** micro-batch 2 × grad-accum 8 (effective 16). **Epochs:** 2. **max_seq_len:** 4096.
- **Grad checkpointing:** `"unsloth"`. **attn:** SDPA (flash-attn has no sm_120 build).

## Notes
- **Pick the checkpoint by boundary-F1, not eval_loss** — eval_loss plateaus at epoch 1 while the task metric keeps improving into epoch 2 (measured). Save checkpoints and evaluate each.
- More data > more epochs for a better model; >2 epochs on a fixed set overfits (train loss <0.2 is the overfit line).
- **GGUF tokenizer gotcha:** T-lite's BPE pre-tokenizer hash isn't registered in upstream `llama.cpp`. Run `training/patch_tokenizer_hash.py` (maps it to `qwen2`) before `convert_hf_to_gguf.py`, or conversion raises "BPE pre-tokenizer was not recognized".
