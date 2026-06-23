#!/usr/bin/env python3
"""Fine-tune T-lite-it-2.1 -> RU context-aware splitter. bf16 LoRA r32 + rsLoRA + all-linear,
response-only loss (output=boundary JSON is tiny vs input => mask instruction/input).
Env: DATA, HOLD, OUTDIR, SMOKE_STEPS (0=full epochs), EPOCHS, MAXLEN.
Run on RTX 5090 / WSL2. HF_HOME pinned to gamer-owned cache."""
import os, sys
os.environ.setdefault("HF_HOME","/home/gamer/ru-splitter/hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

DATA   = os.environ.get("DATA","train_v1.jsonl")
HOLD   = os.environ.get("HOLD","train_v1_holdout.jsonl")
OUTDIR = os.environ.get("OUTDIR","out_v1")
SMOKE  = int(os.environ.get("SMOKE_STEPS","0"))
EPOCHS = float(os.environ.get("EPOCHS","2"))
MAXLEN = int(os.environ.get("MAXLEN","4096"))

print(f"DATA={DATA} OUT={OUTDIR} smoke_steps={SMOKE} epochs={EPOCHS} maxlen={MAXLEN}", flush=True)
model, tok = FastLanguageModel.from_pretrained("t-tech/T-lite-it-2.1",
    max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=False)
model = FastLanguageModel.get_peft_model(model, r=32, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_rslora=True, use_gradient_checkpointing="unsloth", random_state=42)
assert model.peft_config["default"].use_rslora is True
print("rsLoRA + all-linear OK", flush=True)

EOS = tok.eos_token or "</s>"
def fmt(ex):
    return {"text": f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}{EOS}"}
train_ds = load_dataset("json", data_files=DATA, split="train").map(fmt)
eval_ds  = load_dataset("json", data_files=HOLD, split="train").map(fmt) if os.path.exists(HOLD) else None
print(f"train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}", flush=True)

cfg = dict(per_device_train_batch_size=int(os.environ.get("BS","2")), gradient_accumulation_steps=int(os.environ.get("GA","8")),
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01,
    optim="adamw_8bit", bf16=True, max_length=MAXLEN, packing=False, padding_free=False,
    dataset_text_field="text", logging_steps=10, save_steps=200, output_dir=OUTDIR, report_to="none")
if SMOKE>0: cfg["max_steps"]=SMOKE
else: cfg["num_train_epochs"]=EPOCHS
if eval_ds is not None and SMOKE==0:
    cfg.update(eval_strategy="steps", eval_steps=100)

trainer = SFTTrainer(model=model, train_dataset=train_ds,
    eval_dataset=(eval_ds if (eval_ds is not None and SMOKE==0) else None), args=SFTConfig(**cfg))
trainer = train_on_responses_only(trainer,
    instruction_part="### Instruction:\n", response_part="### Response:\n")
print("== training ==", flush=True)
trainer.train()
print("PEAK_VRAM_GB", round(torch.cuda.max_memory_allocated()/1e9,2), flush=True)
if SMOKE==0:
    model.save_pretrained_merged(OUTDIR+"_merged", tok, save_method="merged_16bit")
    print("MERGED_SAVED", OUTDIR+"_merged", flush=True)
print("TRAIN_DONE", flush=True)
