import os, json, re, random
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HUB_DISABLE_XET"]="1"
from datasets import load_dataset
H=lambda p: os.path.expanduser("~/strizh/"+p)
random.seed(42)
rows=[]
# --- RU: GPL пары (наши, дедуп по p_md5) ---
seen=set(); nru=0
for l in open(H("gpl_pairs.jsonl")):
    d=json.loads(l)
    if d["p_md5"] in seen: continue
    seen.add(d["p_md5"]); rows.append({"anchor":d["q"],"positive":d["pos"],"lang":"ru"}); nru+=1
    if nru>=120000: break
# --- MIXED: ru_stackoverflow (RU-вопрос + EN-код) ---
def cyr(s): return sum(1 for c in s if "а"<=c.lower()<="я")
def lat(s): return sum(1 for c in s if "a"<=c.lower()<="z")
nmix=0
ds=load_dataset("IlyaGusev/ru_stackoverflow", split="train", streaming=True, revision="refs/convert/parquet")
for r in ds:
    t=(r.get("title") or "").strip(); b=(r.get("text_markdown") or "").strip()
    if not t or len(b)<200: continue
    c,l=cyr(b),lat(b)
    if c>50 and l>30 and l/(c+l)>0.15:
        rows.append({"anchor":t[:200],"positive":b[:2000],"lang":"mix"}); nmix+=1
    if nmix>=60000: break
# --- EN: MIRACL-en train (query -> positive passage) ---
nen=0
try:
    topics=load_dataset("miracl/miracl","en",split="train",streaming=True,revision="refs/convert/parquet")
    for r in topics:
        q=r.get("query","")
        pos=[p["text"] for p in r.get("positive_passages",[])]
        if q and pos:
            rows.append({"anchor":q,"positive":pos[0][:2000],"lang":"en"}); nen+=1
        if nen>=40000: break
except Exception as e:
    print("MIRACL-en fail:", str(e)[:100], flush=True)
random.shuffle(rows)
json.dump(rows, open(H("train_v2.json"),"w"), ensure_ascii=False)
from collections import Counter
print("train_v2:", len(rows), dict(Counter(r["lang"] for r in rows)), flush=True)
