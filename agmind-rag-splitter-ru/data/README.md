# Dataset card — RU context-aware split

Training data for the Russian context-aware document splitter. Each example teaches the model **where to cut a document** into self-contained semantic chunks.

## Format (Alpaca JSONL)
```json
{
  "instruction": "Раздели документ на смысловые части для системы поиска (RAG)...",
  "input": "[1] Первое предложение. [2] Второе. [3] | таблица | ... |\n[4] ...",
  "output": "{\"splits\": [3, 7], \"topic\": \"о чём документ\"}"
}
```
- **input** = the document pre-segmented into numbered *units*: prose split into sentences (via `razdel`), while **tables and code blocks are kept as single atomic units**.
- **output** = a JSON object: `splits` = the unit indices **after which** a chunk boundary falls (1-indexed), `topic` = a one-sentence summary.
- At inference the host slices the **original** text at these indices → chunks are **byte-identical** to the source and tables are never broken.

## How it was built (teacher distillation)
1. Pull documents from Russian corpora (below).
2. Segment into numbered units (`razdel` + markdown-table/code detection).
3. A strong teacher (**DeepSeek-V4-Flash**, OpenAI-compatible endpoint, grammar/`temperature=0`) labels the boundary indices + topic.
4. **Quality gates** (every example must pass): valid JSON; 1–60 splits; cyrillic-ratio > 0.6; no non-final chunk < 8 words; topic is Russian ≤ 200 chars; MD5/near-dup filtered.
   Reproduce with `data/generate.py`.

## Composition (~17k examples)
| source | share | license | role |
|---|---|---|---|
| `deepvk/cultura_ru_edu` | ~47% | apache-2.0 | web/edu prose |
| `IlyaGusev/habr` | ~34% | (unspec) — training-only | technical text + code blocks |
| synthetic tables/code | ~19% | generated | guaranteed table/code atomicity (12 domains: finance, configs, schedules, metrics, prices, specs…) |

A 12k synthetic-only top-up was generated separately for table/code emphasis (a v2 dataset). Full sets (~17k + 12k) live outside git (too large) — publish on HuggingFace Datasets. This repo ships a **600-example sample** (`sample_train.jsonl`) + **120-example holdout** (`sample_holdout.jsonl`).

## Licensing note
The trained weights learn a *labeling policy* over boundary indices, not the source prose, so copyleft on a source corpus attaches to that prose, not to the weights or the JSON output. For a cleanly-redistributable dataset, prefer the apache/CC0/CC-BY sources and treat unspecified-license corpora as training-only.
