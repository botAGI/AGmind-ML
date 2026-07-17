#!/usr/bin/env python3
"""SFT qmd-query-expansion-ru: Qwen3-1.7B + LoRA (рецепт апстрима tobi/qmd finetune/configs/sft.yaml:
r16/alpha32 all-proj, 5 эпох, lr 2e-4 cosine, eff.batch 16, max_len 512).
Данные: наш qmd_ru JSONL {"query":..., "output":[["hyde",..],["lex",..],["vec",..]]} (hyde первой).
Формат текста = формат апстрима: Qwen3 chat template, user='/no_think Expand this search query: {q}',
assistant=построчно 'type: content', пустой think-блок вырезается.
Env: DATA, HOLD, OUTDIR, EPOCHS. Запуск на 5090/WSL (ulimit -n 1048576, HF_HOME gamer-owned)."""
import os, sys, glob
os.environ.setdefault("HF_HOME","/home/gamer/ru-splitter/hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")
import torch, json
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

DATA   = os.environ.get("DATA","qmd_ru_train.jsonl")
HOLD   = os.environ.get("HOLD","qmd_ru_holdout.jsonl")
OUTDIR = os.environ.get("OUTDIR","out_qmd_ru")
EPOCHS = float(os.environ.get("EPOCHS","5"))
MAXLEN = 512

assert os.path.exists(DATA), DATA
assert os.path.exists(HOLD), HOLD
print(f"DATA={DATA} OUT={OUTDIR} epochs={EPOCHS}", flush=True)

model, tok = FastLanguageModel.from_pretrained("Qwen/Qwen3-1.7B",
    max_seq_length=MAXLEN, dtype=torch.bfloat16, load_in_4bit=False)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)

def build_text(ex):
    # формат апстрима (prepare_data.py): chat template + вырезка пустого think
    out_lines="\n".join(f"{t}: {c}" for t,c in ex["output"])
    msgs=[{"role":"user","content":f"/no_think Expand this search query: {ex['query']}"},
          {"role":"assistant","content":out_lines}]
    text=tok.apply_chat_template(msgs, tokenize=False)
    return {"text": text.replace("<think>\n\n</think>\n\n","")}

train_ds = load_dataset("json", data_files=DATA, split="train").map(build_text)
eval_ds  = load_dataset("json", data_files=HOLD, split="train").map(build_text)
print(f"train={len(train_ds)} eval={len(eval_ds)}", flush=True)
print("SAMPLE:\n"+train_ds[0]["text"][:400], flush=True)

cfg = SFTConfig(per_device_train_batch_size=4, gradient_accumulation_steps=4,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01,
    optim="adamw_8bit", bf16=True, max_length=MAXLEN, packing=False, padding_free=False,
    num_train_epochs=EPOCHS, dataset_text_field="text", logging_steps=20, save_steps=200,
    eval_strategy="steps", eval_steps=100, output_dir=OUTDIR, report_to="none")
trainer = SFTTrainer(model=model, train_dataset=train_ds, eval_dataset=eval_ds, args=cfg)
_ck=glob.glob(os.path.join(OUTDIR,"checkpoint-*"))
trainer.train(resume_from_checkpoint=bool(_ck))
print("PEAK_VRAM_GB", round(torch.cuda.max_memory_allocated()/1e9,2), flush=True)

lora_dir=os.path.abspath(OUTDIR+"_lora")
model.save_pretrained(lora_dir); tok.save_pretrained(lora_dir)
assert os.path.getsize(os.path.join(lora_dir,"adapter_model.safetensors"))>5e6, "LORA SAVE FAILED"
print("LORA_SAVED", lora_dir, flush=True)
mdir=os.path.abspath(OUTDIR+"_merged")
try:
    model.save_pretrained_merged(mdir, tok, save_method="merged_16bit")
    _sz=sum(os.path.getsize(p) for p in glob.glob(mdir+"/*.safetensors"))
    ok=os.path.isfile(mdir+"/config.json") and _sz>2e9
    print("MERGED_SAVED" if ok else f"MERGED_INVALID sz={_sz/1e9:.1f}GB", mdir, flush=True)
except Exception as e:
    print("MERGED_FAILED", repr(e)[:200], flush=True)
print("TRAIN_DONE", flush=True)
