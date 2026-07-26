# End-to-end RAG battle-test

Retrieval-quality numbers (MIRACL recall@10) answer "does the embedder rank the right
passage?" — not "does the retrieved context let the LLM answer?". This harness closes
that gap: a full **retrieve → rerank → answer → judge** pipeline on a separate real
corpus, swapping **only** the embedding model and holding corpus, chunker, questions,
reranker, answering-LLM and prompt constant.

We ran it over the AGmind repository itself (Russian docs + Python + Ansible/service
config + English identifiers) — a corpus not used in the model's own retrieval
training (MIRACL/FRIDA); no full provenance audit of base/teacher models is claimed, i.e. genuine out-of-distribution generalization to the workload the model is
actually sold for.

## Pipeline

| step | script | what it does |
|---|---|---|
| 1 | `build_corpus.py` | chunk a source repo into passages (2200–2600 chars target, ~630-tok median) with provenance + stratum |
| 2 | `gen_gold.py` | LLM generates stratified questions (pure-RU / RU→code / long / cross-lingual) with a gold chunk + reference answer |
| 3 | `embed.py` | embed corpus + queries per model (native pooling/prefixes), save `.npy` |
| 4 | `eval_retrieve.py` | recall@k / MRR / nDCG per stratum, retrieval-only **vs** +reranker |
| 5 | `eval_answer.py` | feed top-k to an LLM, judge answer correctness vs the reference |

## Run

```bash
export DATA_DIR=./data
export LLM_URL=http://localhost:8080/v1/chat/completions   # instruct LLM (generator + judge)
export LLM_MODEL=<your-model-name>
export RERANK_URL=http://localhost:8082/v1/rerank          # e.g. bge-reranker-v2-m3

python build_corpus.py /path/to/repo
python gen_gold.py
for m in strizh tiny2 user2 bgem3; do python embed.py $m; done
python eval_retrieve.py
python eval_answer.py
```

Compared models: `strizh` (`AGmind/strizh-ru-retriever`), `tiny2`
(`cointegrated/rubert-tiny2`), `user2` (`deepvk/USER2-small`, prefix-based), `bgem3`
(`BAAI/bge-m3`). Each is served with its own native pooling and prefixes — omitting a
required prefix silently under-measures a model.

## What we found (1699 chunks, 163 questions, bge-reranker-v2-m3)

recall@10, retrieval → +rerank:

| stratum | strizh | tiny2 | USER2-small | bge-m3 |
|---|:--:|:--:|:--:|:--:|
| pure-RU | 0.67 → 0.67 | 0.49 → 0.60 | 0.65 → 0.75 | 0.78 → 0.82 |
| RU → code | 0.75 → 0.75 | 0.71 → 0.79 | 0.79 → 0.75 | 0.86 → 0.86 |
| long passage | 0.60 → 0.70 | 0.57 → 0.60 | 0.57 → 0.72 | 0.82 → 0.80 |
| **cross-lingual** | **0.35 → 0.38** | 0.12 → 0.17 | 0.62 → 0.62 | 0.72 → 0.80 |
| **overall** | 0.589 → 0.620 | 0.460 → 0.528 | 0.650 → 0.712 | 0.791 → 0.816 |

- **strizh beats `rubert-tiny2` on every stratum** end-to-end — the one clean win.
- It stays competitive with `USER2-small` on Russian strata but sits below `USER2-small`
  and `bge-m3` overall (quality trade-off for size/speed/drop-in).
- **Cross-lingual (English query → Russian doc) is the disqualifier: 0.38, and the
  reranker cannot rescue it** — if the right chunk is not in the candidate set, no
  reranker recovers it. Don't use strizh for cross-lingual retrieval.

## Notes

- The gold set is **LLM-generated (synthetic)**, clearly labeled — good for a first
  end-to-end read; pair with an operator-authored set before publishing headline claims.
- `embed.py` applies a common `EVAL_MAXLEN` (default 1024) to all models for a fair
  comparison; the shipped strizh context is 8192.
- The single-gold assumption undercounts recall — not necessarily uniformly across models (a question may be answerable
  from more than one chunk) — the cross-model *comparison* still holds.
