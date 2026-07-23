#!/usr/bin/env python3
"""Generate a stratified synthetic gold set (question -> relevant chunk).

For sampled chunks, an instruct LLM writes one specific question answerable from that
chunk. Stratified to probe the model's weak axes: pure-RU, RU-query->code/config,
long-passage, and cross-lingual (English question over a Russian chunk).

These are SYNTHETIC questions (LLM-authored), clearly not human-curated — good for a
first end-to-end read; pair with an operator-authored set before publishing headline
claims.

Config via env:  LLM_URL (default http://localhost:8080/v1/chat/completions),
                 LLM_MODEL (model name for the endpoint), DATA_DIR (default ./data)
`chat_template_kwargs.enable_thinking=false` disables "thinking" models (e.g. Qwen3).
"""
import os, json, random, re
import concurrent.futures as cf
import requests

DATA = os.environ.get("DATA_DIR", "./data")
URL = os.environ.get("LLM_URL", "http://localhost:8080/v1/chat/completions")
MODEL = os.environ.get("LLM_MODEL", "local-model")
CHUNKS = [json.loads(l) for l in open(f"{DATA}/chunks.jsonl", encoding="utf-8")]
random.seed(42)

def ask(system, user, max_tokens=400):
    r = requests.post(URL, json={"model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
    return r.json()["choices"][0]["message"].get("content", "") or ""

def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

SYS_RU = ("Ты составляешь вопросы для оценки поисковой RAG-системы по технической документации "
          "и коду проекта. По ДАННОМУ фрагменту сформулируй ОДИН конкретный, специфичный вопрос на "
          "русском языке, ответ на который содержится ИМЕННО в этом фрагменте и который отличает его "
          "от десятков похожих фрагментов (упомяни конкретное имя сервиса/флага/функции/значения). "
          "Не пиши общих вопросов вроде 'о чём этот код'. Верни СТРОГО JSON: "
          '{"question": "...", "answer": "краткий ответ строго из фрагмента"}.')
SYS_EN = ("You write questions to evaluate a search/RAG system over a project's technical docs and code. "
          "Given the fragment (which may be in Russian), formulate ONE specific question IN ENGLISH whose "
          "answer is contained EXACTLY in this fragment and which distinguishes it from dozens of similar "
          "fragments (mention a concrete service/flag/function/value name). No generic questions. "
          'Return STRICT JSON: {"question": "...(English)...", "answer": "short answer from the fragment"}.')

def gen_one(chunk, lang):
    sysmsg = SYS_EN if lang == "en" else SYS_RU
    user = f"Фрагмент (источник {chunk['source']}):\n\n{chunk['text'][:2400]}"
    try:
        d = parse_json(ask(sysmsg, user))
    except Exception:
        return None
    if not d or not d.get("question") or not d.get("answer"): return None
    q = d["question"].strip(); a = str(d["answer"]).strip()
    if len(q) < 15 or len(q) > 300: return None
    if re.search(r"(о чём|о чем|what is this|about this (code|fragment|file))", q, re.I): return None
    return {"question": q, "answer": a[:400], "gold_id": chunk["id"],
            "source": chunk["source"], "stratum": None, "lang": lang}

def pool(pred): return [c for c in CHUNKS if pred(c)]

def main():
    plan = [
        ("pure_ru", pool(lambda c: c["src_type"] in ("readme", "docs") and c["approx_tok"] >= 50), "ru", 55),
        ("ru_code", pool(lambda c: c["src_type"] in ("py_code", "svc_yaml", "ansible") and c["cyr_frac"] >= 0.12), "ru", 50),
        ("long",    pool(lambda c: c["is_long"] and c["cyr_frac"] >= 0.2), "ru", 40),
        ("xling",   pool(lambda c: c["cyr_frac"] >= 0.3 and c["approx_tok"] >= 50), "en", 60),
    ]
    tasks = []
    for name, p, lang, n in plan:
        random.shuffle(p)
        print(f"  pool[{name}]={len(p)} -> take {min(n, len(p))}", flush=True)
        for c in p[:n]:
            tasks.append((name, c, lang))
    print(f"generating {len(tasks)} candidates...", flush=True)
    out = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(gen_one, c, lang): name for name, c, lang in tasks}
        for fut in cf.as_completed(futs):
            name = futs[fut]
            try: r = fut.result()
            except Exception: r = None
            if r:
                r["stratum"] = name; r["qid"] = f"q{len(out)}"; out.append(r)
    with open(f"{DATA}/gold.jsonl", "w", encoding="utf-8") as fh:
        for r in out: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"\nGOLD kept: {len(out)} -> {DATA}/gold.jsonl")
    print("  by stratum:", dict(Counter(r["stratum"] for r in out)))

if __name__ == "__main__":
    main()
