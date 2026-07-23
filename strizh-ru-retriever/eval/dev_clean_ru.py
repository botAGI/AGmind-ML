#!/usr/bin/env python3
"""RU recall@10 on MIRACL-ru dev, full vs passage-exposure-FILTERED subset.

strizh's donor line trained on some MIRACL passages, so the full-dev score is optimistic.
This script also reports a filtered subset: dev queries whose gold passages never appeared
(by exact first-200-char match) among the donor's training positives. It is a filter for
KNOWN direct overlaps with strizh's own training positives — NOT a full near-duplicate audit,
and it does not control the training/pretraining corpora of the baseline models. The filtered
subset is also simply harder (all models score lower on it), so `full − filtered` is not a
clean "cost of leakage".

Usage:  python dev_clean_ru.py <model> [query_prefix] [doc_prefix]
Inputs (paths via env or edit H()):  DATA_DIR/{miracl_qrels_dev.tsv, miracl_ru_dev.tsv,
  miracl_dev_passages.jsonl}  and  DATA_DIR/train_positives.jsonl  (one JSON/line with a
  "pos" field — the donor's training-positive passages).
"""
import os, json, sys
import numpy as np
from sentence_transformers import SentenceTransformer

DATA = os.environ.get("DATA_DIR", ".")
H = lambda p: os.path.join(DATA, p)
qpref = sys.argv[2] if len(sys.argv) > 2 else ""
dpref = sys.argv[3] if len(sys.argv) > 3 else ""

train_pos = set()
tp_file = H("train_positives.jsonl")
if os.path.exists(tp_file):
    for l in open(tp_file):
        train_pos.add(json.loads(l)["pos"][:200])

qrels = {}
for l in open(H("miracl_qrels_dev.tsv")):
    p = l.strip().split("\t")
    if len(p) >= 4 and p[3] == "1":
        qrels.setdefault(p[0], set()).add(p[2])
topics = {}
for l in open(H("miracl_ru_dev.tsv")):
    p = l.strip().split("\t")
    if len(p) >= 2:
        topics[p[0]] = p[1]
docids, texts, id2full = [], [], {}
for l in open(H("miracl_dev_passages.jsonl")):
    d = json.loads(l)
    t = (d.get("title", "") + " " + d["text"]).strip()
    docids.append(d["docid"]); texts.append(t[:2000]); id2full[d["docid"]] = t
did2i = {d: i for i, d in enumerate(docids)}

qids_all = [q for q in qrels if q in topics][:1000]
qids_filtered = [q for q in qids_all
                 if not any(id2full.get(g, "")[:200] in train_pos for g in qrels[q])]

m = SentenceTransformer(sys.argv[1], device="cuda", trust_remote_code=True)
m.max_seq_length = 512
D = m.encode([dpref + t for t in texts], batch_size=256, normalize_embeddings=True, show_progress_bar=False)

def score(qids):
    Q = m.encode([qpref + topics[q] for q in qids], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    top = np.argsort(-(Q @ D.T), axis=1)[:, :10]
    hits = 0
    for i, q in enumerate(qids):
        gold = {did2i[d] for d in qrels[q] if d in did2i}
        if any(idx in gold for idx in top[i]):
            hits += 1
    return hits / len(qids)

name = os.path.basename(sys.argv[1].rstrip("/"))
print(f"{name}: full({len(qids_all)})={score(qids_all):.4f} "
      f"filtered({len(qids_filtered)})={score(qids_filtered):.4f}")
