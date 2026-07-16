#!/usr/bin/env python3
"""План Б: merge LoRA -> merged bf16 чистым transformers+peft (без unsloth), на GPU.
Usage: merge_lora.py <lora_or_checkpoint_dir> <out_dir>"""
import os, sys, glob
os.environ.setdefault("HF_HOME","/home/gamer/ru-splitter/hf")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE="t-tech/T-lite-it-2.1"
lora, out = sys.argv[1], sys.argv[2]
m=AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"":0}, low_cpu_mem_usage=True)
m=PeftModel.from_pretrained(m, lora)
m=m.merge_and_unload()
m.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
shards=glob.glob(out+"/*.safetensors"); total=sum(map(os.path.getsize,shards))
assert os.path.isfile(out+"/config.json") and os.path.isfile(out+"/tokenizer.json") and total>10e9, f"MERGE INVALID {total/1e9:.1f}GB"
print(f"MERGE_OK {out} {total/1e9:.1f}GB shards={len(shards)}")
