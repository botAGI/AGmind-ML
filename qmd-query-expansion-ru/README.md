---
language:
- ru
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
pipeline_tag: text-generation
tags:
- search
- query-expansion
- rag
- russian
- qmd
- gguf
---

# qmd-query-expansion-ru

**Russian query-expansion model for [qmd](https://github.com/tobi/qmd)** — a drop-in replacement for the stock `tobil/qmd-query-expansion-1.7B` that actually speaks Russian.

The stock qmd expansion model is trained on 100% English data: on non-English queries it produces English boilerplate hallucinations and broken structure ([qmd issue #454](https://github.com/tobi/qmd/issues/454)). This model fixes that for Russian.

**Code, dataset generator, training & benchmark scripts:** [github.com/botAGI/AGmind-ML → qmd-query-expansion-ru](https://github.com/botAGI/AGmind-ML/tree/main/qmd-query-expansion-ru)

## Benchmark: 450 held-out Russian queries, ours vs stock

Mechanical metrics (no LLM judges), same sampling as qmd runtime (temp 0.7, top-k 20, top-p 0.8). Script: `bench_qmd.py` in the repo.

| metric | **ours** | stock `tobil/qmd-query-expansion-1.7B` |
|---|---|---|
| valid qmd contract (exactly 1 `hyde:` + 3 `lex:` + 2 `vec:`, no junk lines) | **99.8%** | 0.0% |
| Russian output (≥70% Cyrillic letters) | **99.3%** | 26.2% |
| EN boilerplate hallucination ("…is an important concept…") | **0.0%** | 22.9% |
| `hyde` within length contract (50–250 chars) | **96.0%** | 19.8% |
| verbatim query echo instead of expansion | 0.4% | 1.3% |

The stock model fails the upstream train-format contract on **every one of 450 Russian queries** and emits English template hallucinations in ~23% of them — the model simply was never shown Russian. Ours holds the contract at 99.8% while staying Russian.

## Использование в qmd / Usage

One line in `~/.config/qmd/index.yml` (or project `.qmd/index.yml`):

```yaml
models:
  generate: "hf:AGmind/qmd-query-expansion-ru/qmd-query-expansion-ru-q4_k_m.gguf"
```

No re-indexing needed. Same prompt contract as upstream: `/no_think Expand this search query: {query}` → line-oriented `hyde:` / `lex:` / `vec:` output (GBNF-compatible).

## Пример / Example (RU query, ours vs stock)

Query: **«как оформить налоговый вычет за лечение»**

Ours:
```
hyde: Для оформления налогового вычета за лечение необходимо подтвердить расходы на медицинские услуги...
lex: налоговый вычет лечение оформление декларация
lex: расходы медицинские услуги справки врачей
lex: вычет налога подача декларация
vec: Как получить налоговый вычет за медицинские расходы
vec: Подача заявки на налоговый вычет за лечение и справки
```

Stock (`tobil/qmd-query-expansion-1.7B`):
```
hyde: Как оформить налоговый вычет за лечение is an important concept that relates to...
     It provides functionality for various use cases in software development.   ← boilerplate hallucination
```

Key design point: **`lex:` lines cover Russian word forms and synonyms** («получить получение вернуть возврат») — qmd's BM25 uses a Porter stemmer that does not stem Russian, so lexical recall must come from the expansion itself.

## Training

- Base: Qwen/Qwen3-1.7B (same as upstream), LoRA r16/α32 all-proj, 5 epochs — upstream's exact SFT recipe (`tobi/qmd/finetune`).
- Data: 5 075 Russian queries (MIRACL-ru apache-2.0, Mr.TyDi-ru apache-2.0, Yandex.Q CC0) → teacher distillation (DeepSeek-V4-Flash, n=2 sampling + rule-based reward filter ≥70, avg score 94.8).
- Format identical to upstream train schema (`{"query", "output": [["hyde",…],["lex",…],["vec",…]]}`, hyde-first).

## Limitations

- 1.7B model: `hyde:` passages may hallucinate specifics (wrong form numbers, ingredients). For search expansion this is tolerable — terms stay topical — but don't treat hyde output as facts.
- Trained for Russian; English queries → use the stock model.

## Files

- `qmd-query-expansion-ru-q4_k_m.gguf` — for qmd / llama.cpp (recommended)
- safetensors — merged fp16 for transformers

Part of [AGmind-ML](https://github.com/botAGI/AGmind-ML). Related: [AGmind/agmind-rag-splitter-ru](https://huggingface.co/AGmind/agmind-rag-splitter-ru).
