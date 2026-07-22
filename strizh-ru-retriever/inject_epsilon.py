#!/usr/bin/env python3
"""Блокер 3: конвертер пишет modern-bert.attention.layer_norm_RMS_epsilon, а
server-vulkan требует ...layer_norm_epsilon → дописываем ОДИН FLOAT32-ключ,
переиспользуя штатный copy_with_new_metadata (корректно копирует все KV+тензоры)."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/ru-splitter/llama.cpp/gguf-py"))
import gguf
from gguf.scripts.gguf_new_metadata import copy_with_new_metadata, get_field_data, MetadataDetails

src, dst = sys.argv[1], sys.argv[2]
KEY="modern-bert.attention.layer_norm_epsilon"; VAL=1e-05
r=gguf.GGUFReader(src)
arch=get_field_data(r, gguf.Keys.General.ARCHITECTURE)
print(f"arch={arch} existing_epsilon={get_field_data(r,KEY)}",flush=True)
assert arch=="modern-bert", f"неожиданный arch: {arch}"
w=gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
copy_with_new_metadata(r, w, {KEY: MetadataDetails(gguf.GGUFValueType.FLOAT32, VAL)}, [])
print("INJECT_DONE",flush=True)
