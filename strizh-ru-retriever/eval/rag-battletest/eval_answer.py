#!/usr/bin/env python3
"""Layer-2 end-to-end answer correctness.

Feed each query's top-k retrieved (and reranked) chunks to an LLM, then have the LLM
judge the answer against the reference. The decisive metric: % of questions answered
correctly given model X's retrieved context, with the LLM + prompt + reranker held
constant. Reported per stratum. Anchor the judge with a human spot-check.

Env:  DATA_DIR (default ./data), LLM_URL (default http://localhost:8080/v1/chat/completions),
      LLM_MODEL, CONDS (default: every model with rerank, plus strizh retrieval-only).
"""
import os, json, re
import concurrent.futures as cf
import requests

DATA = os.environ.get("DATA_DIR", "./data")
URL = os.environ.get("LLM_URL", "http://localhost:8080/v1/chat/completions")
MODEL = os.environ.get("LLM_MODEL", "local-model")
CHUNKS = {json.loads(l)["id"]: json.loads(l) for l in open(f"{DATA}/chunks.jsonl", encoding="utf-8")}
GOLD = {g["qid"]: g for g in (json.loads(l) for l in open(f"{DATA}/gold.jsonl", encoding="utf-8"))}
K = 5
STRATA = ["pure_ru", "ru_code", "long", "xling"]
# "<model>:<ret|rr>" pairs
CONDS = os.environ.get("CONDS", "strizh:rr,tiny2:rr,user2:rr,bgem3:rr,strizh:ret").split(",")

def qwen(system, user, max_tokens=400):
    last = None
    for _ in range(3):
        try:
            r = requests.post(URL, json={"model": MODEL,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}, timeout=240)
            return r.json()["choices"][0]["message"].get("content", "") or ""
        except Exception as e:
            last = e
    raise last

ANS_SYS = ("Ответь на вопрос СТРОГО по приведённым фрагментам контекста. Если ответа в контексте нет — "
           "напиши ровно 'НЕ НАЙДЕНО'. Отвечай кратко, тем же языком, что и вопрос.")
JUDGE_SYS = ("Ты судья фактической корректности. Дан вопрос, эталонный ответ и ответ системы. "
             "Верни СТРОГО JSON: {\"correct\": true|false} — true, если ответ системы фактически "
             "совпадает с эталоном по сути (парафраз допустим), иначе false.")

def build_ctx(ids):
    return "\n\n".join(f"[Фрагмент {i}] ({CHUNKS[cid]['source']})\n{CHUNKS[cid]['text'][:1500]}"
                       for i, cid in enumerate(ids[:K], 1))

def run_cond(model_key, cond):
    topk = {d["qid"]: d for d in json.load(open(f"{DATA}/topk_{model_key}.json"))}
    def one(qid):
        g = GOLD[qid]; d = topk[qid]
        ids = d["rr_top10"] if cond == "rr" else d["ret_top10"]
        try:
            ans = qwen(ANS_SYS, f"Контекст:\n{build_ctx(ids)}\n\nВопрос: {g['question']}")
            jr = qwen(JUDGE_SYS, f"Вопрос: {g['question']}\nЭталон: {g['answer']}\nОтвет системы: {ans}", 60)
        except Exception:
            return g["stratum"], None
        m = re.search(r'\{.*\}', jr, re.S)
        correct = bool(json.loads(m.group(0)).get("correct")) if m else False
        return g["stratum"], correct
    rows, errors = [], 0
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        for stratum, correct in ex.map(lambda q: one(q), list(GOLD)):
            if correct is None: errors += 1; continue
            rows.append((stratum, correct))
    n = len(rows)
    per = {s: (sum(1 for st, c in rows if st == s and c) / max(1, sum(1 for st, _ in rows if st == s)),
               sum(1 for st, _ in rows if st == s)) for s in STRATA}
    return {"acc": sum(1 for _, c in rows if c) / n if n else 0.0, "n": n, "errors": errors, "per": per}

def main():
    res = {}
    for cond in CONDS:
        model_key, mode = cond.split(":")
        try:
            res[cond] = run_cond(model_key, mode)
            r = res[cond]
            print(f"{cond:14s} acc={r['acc']:.3f}  " +
                  "  ".join(f"{s}={r['per'][s][0]:.2f}(n{r['per'][s][1]})" for s in STRATA), flush=True)
        except FileNotFoundError:
            print(f"{cond:14s} SKIP (no topk)")
    json.dump(res, open(f"{DATA}/answer_results.json", "w"), indent=2)

if __name__ == "__main__":
    main()
