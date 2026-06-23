# Конфигурация обучения

## Железо / окружение (проверено рабочим)
- **GPU:** RTX 5090 (NVIDIA Blackwell, compute capability `sm_120`, 32 ГБ)
- **ОС:** WSL2 (Ubuntu 24.04) на Windows — нативный Windows ненадёжен для Blackwell-стека обучения
- **Пик VRAM:** ~25.4 ГБ · **время:** ~3.5 ч (2 эпохи, ~17k примеров)

### Пины версий (Blackwell sm_120 — load-bearing часть)
| пакет | пин | примечание |
|---|---|---|
| torch | `2.11.0` (индекс cu129) | **никогда cu130** — ABI-конфликт с bitsandbytes; `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129` |
| triton | `>=3.3.1` | минимум для Blackwell |
| unsloth + unsloth_zoo | latest | `pip install unsloth unsloth_zoo` |
| transformers / trl / peft / accelerate | latest stable | |
| python | 3.12 | |
| драйвер NVIDIA | R570+ (CUDA ≥12.8) | |

Перед обучением: `python -c "import torch; print('sm_120' in torch.cuda.get_arch_list())"` должно вернуть `True`.
Поднять лимит дескрипторов (`ulimit -n 1048576`) — torch.compile/inductor открывает много файлов.

## Рецепт (PEFT)
- **Метод:** bf16 LoRA (НЕ QLoRA — 8B влезает в 32 ГБ bf16, а bitsandbytes-4bit на Blackwell хрупок/штраф по скорости).
- **LoRA:** `r=32`, `alpha=32`, `dropout=0.05`, `use_rslora=True`, все линейные таргеты (`q,k,v,o,gate,up,down`).
- **Loss:** response-only (маскировать instruction+input; в лоссе только JSON-вывод).
- **Optim:** `adamw_8bit`, `lr=2e-4` cosine, warmup 5%, weight_decay 0.01.
- **Batch:** micro-batch 2 × grad-accum 8 (эффективный 16). **Эпохи:** 2. **max_seq_len:** 4096.
- **Grad checkpointing:** `"unsloth"`. **attn:** SDPA (flash-attn не имеет сборки под sm_120).

## Заметки
- **Выбирай чекпоинт по boundary-F1, не по eval_loss** — eval_loss выходит на плато к эпохе 1, пока task-метрика растёт во 2-й эпохе (замерено). Сохраняй чекпоинты и оценивай каждый.
- Больше данных > больше эпох; >2 эпох на фиксированном сете переобучают (train loss <0.2 — линия оверфита).
- **GGUF-граблина токенайзера:** хэш BPE-pre-tokenizer'а T-lite не зарегистрирован в upstream llama.cpp. Запусти `training/patch_tokenizer_hash.py` (маппит на `qwen2`) перед `convert_hf_to_gguf.py`, иначе конверсия падает с «BPE pre-tokenizer was not recognized».
