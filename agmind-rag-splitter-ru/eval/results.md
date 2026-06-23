# Evaluation results

## Metrics (1,500-example held-out set, greedy decoding)

| Metric | HF model (RTX 5090) | GGUF Q5_K_M (AMD Vulkan) |
|---|---|---|
| Valid JSON output | 100.0% | 100.0% |
| boundary-F1 @0 | 0.656 | 0.639 |
| boundary-F1 @±1 | 0.821 | 0.817 |
| exact-set-match | 29.0% | 25.0% |

**Checkpoint selection** — epoch 2 (final) beat epoch 1 on the task metric despite a flat `eval_loss`:

| | boundary-F1@0 | boundary-F1@±1 | exact-set |
|---|---|---|---|
| epoch 1 (step 1000) | 0.610 | 0.800 | 23% |
| **epoch 2 (final)** | **0.656** | **0.821** | **29%** |

→ cross-entropy is a weak proxy for this structured task; always select on the task metric.

## Definitions
- **boundary-F1@k**: F1 over the set of predicted vs gold boundary indices, allowing ±k sentences of tolerance. `@0` = exact position, `@±1` = off-by-one allowed (usually harmless for chunking).
- **exact-set-match**: fraction of documents where the predicted boundary set equals the gold set exactly (a harsh metric — segmentation has many valid answers).
- **Valid JSON**: fraction of outputs parseable as `{"splits":[...], "topic":"..."}`.

## What "gold" means here
Labels come from the distillation **teacher** (DeepSeek-V4-Flash), filtered by self-consistency and hard validation gates. These metrics measure **agreement with the teacher**, i.e. how well the student learned the labeling policy — not human ground truth. A downstream retrieval eval (hit-rate@k / faithfulness on a Russian QA set) is the right next measurement and is future work.

## Reproduce
```bash
MODEL=out_merged       N=1500 python eval/eval_hf.py     # HF model
# against the deployed GGUF endpoint (edit URL in eval_gguf.py):
python eval/eval_gguf.py 1500
```
