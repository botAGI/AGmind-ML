"""Консенсус двух независимых разметчиков эталона (DeepSeek-voted × Claude).

Правило: метки согласны, если множества границ совпадают при допуске ±1
(каждая граница A имеет пару в B в пределах 1 юнита и наоборот).
Согласны → консенсус-gold: точное совпадение берётся как есть, при сдвиге ±1
берётся позиция Claude (разметчик с обоснованиями). Не согласны → disputes
на арбитраж (агент читает документ и обе метки, выносит финал).
"""
import json, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))


def match_pm1(a, b):
    """Паросочетание границ с допуском ±1: каждая точка A ↔ уникальная точка B."""
    a, b = sorted(a), sorted(b)
    used = set()
    pairs = []
    for x in a:
        cand = [y for y in b if abs(y - x) <= 1 and y not in used]
        if not cand:
            return None
        y = min(cand, key=lambda y: abs(y - x))
        used.add(y)
        pairs.append((x, y))
    if len(used) != len(b):
        return None
    return pairs


def main():
    units = {json.loads(l)["doc_id"]: json.loads(l) for l in open(f"{BASE}/units.jsonl")}
    claude = {json.loads(l)["doc_id"]: json.loads(l) for l in open(f"{BASE}/claude_labels.jsonl")}
    ds = {}
    for l in open(f"{BASE}/ds_labels.jsonl"):
        r = json.loads(l)
        if "err" not in r:
            ds[r["i"]] = r

    gold, disputes = [], []
    for did in sorted(units):
        u = units[did]
        c = claude.get(did)
        d = ds.get(did)
        if not c or not d:
            disputes.append({"doc_id": did, "reason": "missing_labeler",
                             "claude": c and c["splits"], "ds": d and d["ds_splits"]})
            continue
        pairs = match_pm1(c["splits"], d["ds_splits"])
        if pairs is not None:
            gold.append({"doc_id": did, "src": u["src"], "genre": u["genre"], "units": u["units"],
                         "gold_splits": sorted(c["splits"]), "provenance": "consensus",
                         "claude_confidence": c.get("confidence", "high"),
                         "ds_splits": d["ds_splits"], "note": c.get("note", "")})
        else:
            disputes.append({"doc_id": did, "src": u["src"], "genre": u["genre"],
                             "claude": sorted(c["splits"]), "claude_note": c.get("note", ""),
                             "claude_confidence": c.get("confidence"), "ds": d["ds_splits"],
                             "nunits": len(u["units"])})

    with open(f"{BASE}/gold_consensus.jsonl", "w") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with open(f"{BASE}/disputes.jsonl", "w") as f:
        for x in disputes:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    n = len(units)
    print(f"консенсус ±1: {len(gold)}/{n} ({100 * len(gold) / n:.0f}%) | disputes: {len(disputes)}")
    from collections import Counter
    print("disputes по жанрам:", dict(Counter((x.get("src"), x.get("genre", "")[:20]) for x in disputes)))


if __name__ == "__main__":
    main()
