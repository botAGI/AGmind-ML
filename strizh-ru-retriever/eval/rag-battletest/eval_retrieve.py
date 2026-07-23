#!/usr/bin/env python3
"""Layer-1 retrieval metrics + reranker ablation.

Cosine over the normalized .npy vectors -> recall@k / MRR / nDCG, overall and per
stratum, retrieval-only vs +cross-encoder-rerank. Dumps top-k chunk ids per query/model
for the Layer-2 answer stage.

Env:  DATA_DIR (default ./data), RERANK_URL (default http://localhost:8082/v1/rerank),
      MODELS (comma-separated keys, default strizh,tiny2,user2,bgem3).
The reranker endpoint (e.g. llama.cpp serving bge-reranker-v2-m3) may cap the physical
batch at 512 tokens, so candidate docs are truncated with retry-on-error.
"""
import os, json, math
import numpy as np
import concurrent.futures as cf
import requests

DATA = os.environ.get("DATA_DIR", "./data")
RERANK_URL = os.environ.get("RERANK_URL", "http://localhost:8082/v1/rerank")
MODELS = os.environ.get("MODELS", "strizh,tiny2,user2,bgem3").split(",")
CHUNKS = [json.loads(l) for l in open(f"{DATA}/chunks.jsonl", encoding="utf-8")]
IDS = [c["id"] for c in CHUNKS]
ID2IDX = {c["id"]: i for i, c in enumerate(CHUNKS)}
TEXT = {c["id"]: c["text"] for c in CHUNKS}
GOLD = [json.loads(l) for l in open(f"{DATA}/gold.jsonl", encoding="utf-8")]
KS = [1, 5, 10, 20]
STRATA = ["pure_ru", "ru_code", "long", "xling"]

def rank_of_gold(order, gold_id):
    gi = ID2IDX[gold_id]
    for r, idx in enumerate(order, 1):
        if idx == gi: return r
    return 10**9

def metrics(ranks):
    n = len(ranks)
    out = {f"r@{k}": sum(1 for r in ranks if r <= k) / n for k in KS}
    out["mrr@10"] = sum(1.0/r if r <= 10 else 0.0 for r in ranks) / n
    out["ndcg@10"] = sum(1.0/math.log2(r+1) if r <= 10 else 0.0 for r in ranks) / n
    return out

def rerank_call(query, cand_idxs, cut=1200):
    for c in (cut, 700, 400):   # shrink docs until they fit the reranker's batch
        docs = [TEXT[IDS[i]][:c] for i in cand_idxs]
        try:
            r = requests.post(RERANK_URL, json={"query": query, "documents": docs,
                                                "top_n": len(docs)}, timeout=60)
            if r.status_code != 200: continue
            return [cand_idxs[res["index"]] for res in r.json()["results"]]
        except Exception:
            continue
    return cand_idxs

def eval_model(key):
    dvec = np.load(f"{DATA}/emb_{key}_docs.npy")
    qvec = np.load(f"{DATA}/emb_{key}_queries.npy")
    topk = np.argsort(-(qvec @ dvec.T), axis=1)[:, :50]
    ret_ranks, rr_ranks = [], []
    per = {s: {"ret": [], "rr": []} for s in STRATA}
    dump = []
    def do_q(qi):
        g = GOLD[qi]; order = list(topk[qi])
        rr_order = rerank_call(g["question"], order[:20]) + order[20:]
        return qi, rank_of_gold(order, g["gold_id"]), rank_of_gold(rr_order, g["gold_id"]), order[:10], rr_order[:10]
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for qi, rr, rrr, t10, rt10 in ex.map(do_q, range(len(GOLD))):
            g = GOLD[qi]
            ret_ranks.append(rr); rr_ranks.append(rrr)
            per[g["stratum"]]["ret"].append(rr); per[g["stratum"]]["rr"].append(rrr)
            dump.append({"qid": g["qid"], "stratum": g["stratum"], "lang": g["lang"],
                         "gold_id": g["gold_id"], "ret_top10": [IDS[i] for i in t10],
                         "rr_top10": [IDS[i] for i in rt10]})
    json.dump(dump, open(f"{DATA}/topk_{key}.json", "w"))
    rate = lambda v, k: (sum(1 for r in v if r <= k) / len(v) if v else 0.0)
    return {"overall_ret": metrics(ret_ranks), "overall_rr": metrics(rr_ranks),
            "per_stratum": {s: {"n": len(v["ret"]), "ret_r@10": rate(v["ret"], 10), "rr_r@10": rate(v["rr"], 10)}
                            for s, v in per.items()}}

def main():
    print(f"corpus={len(CHUNKS)} queries={len(GOLD)}\n")
    res = {}
    for key in MODELS:
        if not os.path.exists(f"{DATA}/emb_{key}_docs.npy"):
            print(f"[{key}] SKIP (no embeddings)"); continue
        res[key] = eval_model(key)
        print(f"=== {key} ===")
        print("  retrieval: " + "  ".join(f"{k}={v:.3f}" for k, v in res[key]["overall_ret"].items()))
        print("  +rerank  : " + "  ".join(f"{k}={v:.3f}" for k, v in res[key]["overall_rr"].items()))
    json.dump(res, open(f"{DATA}/retrieve_results.json", "w"), indent=2)
    print("\n=== per-stratum recall@10 (retrieval -> +rerank) ===")
    print("stratum".ljust(10) + "n".rjust(4) + "".join(k.rjust(16) for k in MODELS if k in res))
    for s in STRATA:
        n = next((res[k]["per_stratum"][s]["n"] for k in res), 0)
        row = s.ljust(10) + str(n).rjust(4)
        for k in MODELS:
            if k in res:
                st = res[k]["per_stratum"][s]
                row += f"{st['ret_r@10']:.2f}->{st['rr_r@10']:.2f}".rjust(16)
        print(row)

if __name__ == "__main__":
    main()
