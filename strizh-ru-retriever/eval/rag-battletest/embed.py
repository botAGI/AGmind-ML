#!/usr/bin/env python3
"""Embed corpus + queries with one model, saving normalized vectors as .npy.

Each model uses its OWN native pooling and prefixes — strizh / rubert-tiny2 / bge-m3 are
drop-in (no prefixes); USER2-small needs search_query:/search_document:. Omitting a
required prefix silently under-measures a model, so registered sentence-transformers
prompts are auto-detected and preferred over manual prefixes.

Usage:  python embed.py <model_key> [out_key]     (out_key overrides the output name)
Env:    DATA_DIR (default ./data), EVAL_MAXLEN (default 1024 — a common cap for a fair
        cross-model comparison; models truncate longer inputs to this).
"""
import os, sys, time, json
import numpy as np
from sentence_transformers import SentenceTransformer

DATA = os.environ.get("DATA_DIR", "./data")
EVAL_MAXLEN = int(os.environ.get("EVAL_MAXLEN", "1024"))
CHUNKS = [json.loads(l) for l in open(f"{DATA}/chunks.jsonl", encoding="utf-8")]
GOLD = [json.loads(l) for l in open(f"{DATA}/gold.jsonl", encoding="utf-8")]

# key -> (model id/path, query_prefix, doc_prefix)  (empty prefix = drop-in)
MODELS = {
    "strizh": ("AGmind/strizh-ru-retriever", "", ""),
    "tiny2":  ("cointegrated/rubert-tiny2", "", ""),
    "user2":  ("deepvk/USER2-small", "search_query: ", "search_document: "),
    "bgem3":  ("BAAI/bge-m3", "", ""),
}

def pick_prompt(model, want):
    prompts = getattr(model, "prompts", None) or {}
    # a retrieval prompt beats a generic alias: search_query over an empty 'query'
    for name in ([f"search_{want}", want] if want in ("query", "document") else [want]):
        if name in prompts and prompts.get(name):
            return name
    return None

def embed(model, texts, prefix, prompt_name, bs=64):
    kw = dict(batch_size=bs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    if prompt_name:
        return model.encode(texts, prompt_name=prompt_name, **kw).astype("float32")
    if prefix:
        texts = [prefix + t for t in texts]
    return model.encode(texts, **kw).astype("float32")

def main():
    key = sys.argv[1]
    out_key = sys.argv[2] if len(sys.argv) > 2 else key
    path, qpref, dpref = MODELS[key]
    m = SentenceTransformer(path, trust_remote_code=True)
    m.max_seq_length = EVAL_MAXLEN
    qn = pick_prompt(m, "query") if qpref else None
    dn = pick_prompt(m, "document") if dpref else None
    print(f"[{key}] dim={m.get_sentence_embedding_dimension()} max_seq={m.max_seq_length} "
          f"query={'prompt:'+qn if qn else ('prefix:'+repr(qpref) if qpref else 'DROP-IN')}", flush=True)

    docs = [c["text"] for c in CHUNKS]
    qs = [g["question"] for g in GOLD]
    t0 = time.time()
    dvec = embed(m, docs, "" if dn else dpref, dn)
    t_doc = time.time() - t0
    qvec = embed(m, qs, "" if qn else qpref, qn)
    np.save(f"{DATA}/emb_{out_key}_docs.npy", dvec)
    np.save(f"{DATA}/emb_{out_key}_queries.npy", qvec)
    print(f"[{key}] docs={dvec.shape} queries={qvec.shape} ||v0||={float(np.linalg.norm(dvec[0])):.3f} "
          f"index={t_doc:.1f}s ({len(docs)/t_doc:.0f} emb/s)", flush=True)

if __name__ == "__main__":
    main()
