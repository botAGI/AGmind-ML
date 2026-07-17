#!/usr/bin/env python3
"""Датасет-генератор qmd-query-expansion-ru: RU-запросы → hyde/lex/vec (формат апстрима tobi/qmd).
Дистилляция DeepSeek (n=2 сэмпла → русифицированный reward-лайт → лучший, если ≥ порога).
Ключевое отличие от EN: lex-строки обязаны покрывать СЛОВОФОРМЫ (qmd BM25 = porter, русский не стеммит).
Usage: gen_qmd.py <N> <out.jsonl> [workers]"""
import json, sys, os, re, random, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

DS_URL="http://192.168.1.45:8000/v1"; MODEL="dspark"
BASE=os.path.dirname(os.path.abspath(__file__))
_client=None
def cli():
    global _client
    if _client is None: _client=OpenAI(base_url=DS_URL,api_key="x",timeout=180)
    return _client

SCHEMA={"type":"object","properties":{
    "hyde":{"type":"string"},
    "lex":{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":3},
    "vec":{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":2}},
    "required":["hyde","lex","vec"]}

SYS=("Ты — генератор расширений поисковых запросов для локального поискового движка (BM25 + векторный поиск) "
    "на РУССКОМ языке. По запросу пользователя верни JSON с тремя видами расширений:\n"
    "1. hyde — ОДИН гипотетический фрагмент документа-ответа (50-200 символов, одна строка, без переносов): "
    "как выглядел бы абзац идеального документа, отвечающего на запрос.\n"
    "2. lex — РОВНО 3 строки коротких ключевых слов для лексического поиска BM25. ВАЖНО: движок НЕ понимает "
    "русскую морфологию, поэтому покрывай СЛОВОФОРМЫ и СИНОНИМЫ ключевых терминов (например: «настройка настроить "
    "конфигурация», «оплата платёж оплатить»). 3-6 слов на строку, без стоп-слов.\n"
    "3. vec — РОВНО 2 переформулировки запроса естественным языком для векторного поиска (полные фразы, "
    "разными словами, сохраняя смысл).\n"
    "Правила: СОХРАНЯЙ имена собственные, аббревиатуры, числа и коды ТОЧНО как в запросе; не повторяй запрос "
    "дословно; никаких английских слов, если их не было в запросе. Только JSON.")

def teacher(query,n=2,temp=0.7):
    r=cli().chat.completions.create(model=MODEL,
        messages=[{"role":"system","content":SYS},
                  {"role":"user","content":f"Запрос: {query}"}],
        temperature=temp,max_tokens=400,n=n,extra_body={"guided_json":SCHEMA})
    outs=[]
    for ch in r.choices:
        try:
            o=json.loads((ch.message.content or "").strip())
            if "hyde" in o and "lex" in o and "vec" in o: outs.append(o)
        except Exception: pass
    return outs

# ---------- reward-лайт (русифицированный порт идей tobi/qmd finetune/reward.py) ----------
_STOP={"как","что","где","когда","почему","зачем","какой","какая","какие","это","для","при","про",
       "или","если","есть","ли","не","на","в","и","с","по","у","о","из","до","от","за"}
def _words(s): return [w for w in re.findall(r"[а-яёa-z0-9]+",s.lower()) if w not in _STOP]
def _ents(q):
    """сущности: Капитализированные не в начале, ЦИФРЫ, АББРЕВИАТУРЫ, латиница в ру-запросе"""
    ents=set(re.findall(r"\b[A-ZА-ЯЁ]{2,}\b",q))|set(re.findall(r"\d[\d.,:-]*",q))|set(re.findall(r"\b[a-zA-Z][a-z]+\b",q))
    for m in re.finditer(r"(?<!^)(?<![.!?]\s)\b[А-ЯЁ][а-яё]+",q): ents.add(m.group(0))
    return {e.lower() for e in ents if len(e)>1}

def score(query,o):
    s=0
    hyde=(o.get("hyde") or "").strip()
    lex=[x.strip() for x in o.get("lex",[])]; vec=[x.strip() for x in o.get("vec",[])]
    # формат (0-30)
    if hyde and len(lex)==3 and len(vec)==2: s+=30
    # hyde длина/однострочность (0-20)
    if 50<=len(hyde)<=200 and "\n" not in hyde: s+=20
    elif 30<=len(hyde)<=280: s+=8
    # разнообразие (0-20): нет дублей, lex != эхо запроса
    all_lines=lex+vec
    if len({l.lower() for l in all_lines})==len(all_lines): s+=10
    q_ws=set(_words(query))
    if all(set(_words(l))!=q_ws for l in all_lines if l): s+=10
    # словоформы в lex (0-15): суммарно lex должен дать ≥2 слов вне точных словоформ запроса
    lex_ws=set(w for l in lex for w in _words(l))
    if len(lex_ws-q_ws)>=2: s+=15
    # сущности (−20..+15)
    ents=_ents(query)
    if ents:
        cover=" ".join(all_lines+[hyde]).lower()
        hit=sum(1 for e in ents if e in cover)
        s += 15 if hit==len(ents) else (5 if hit>0 else -20)
    else: s+=10
    # язык: латиница, которой не было в запросе (штраф)
    lat_q=set(re.findall(r"[a-zA-Z]{3,}",query.lower()))
    lat_o=set(re.findall(r"[a-zA-Z]{3,}"," ".join(all_lines+[hyde]).lower()))
    if lat_o-lat_q: s-=15
    return s  # максимум 100

THRESHOLD=70

def to_upstream(query,o):
    """→ формат апстрима: {"query":..., "output":[["hyde",..],["lex",..]x3,["vec",..]x2]} (hyde первой)"""
    out=[["hyde",o["hyde"].strip()]]
    out+= [["lex",x.strip()] for x in o["lex"]]
    out+= [["vec",x.strip()] for x in o["vec"]]
    return {"query":query,"output":out}

def load_seeds():
    seeds=[]
    p=f"{BASE}/seeds"
    for ln in open(f"{p}/miracl_ru_train.tsv"):
        parts=ln.rstrip("\n").split("\t")
        if len(parts)>=2: seeds.append(("miracl",parts[1].strip()))
    for ln in open(f"{p}/mrtydi_ru_train.txt"):
        parts=ln.rstrip("\n").split("\t")
        if len(parts)>=2: seeds.append(("tydi",parts[1].strip()))
    for q in json.load(open(f"{p}/yandexq_ru.json")): seeds.append(("yandexq",q))
    # апстрим-категории: перевод запросов НЕ делаем тут (отдельный трек) — базу дают нативные RU
    random.seed(42); random.shuffle(seeds)
    return seeds

def process(item):
    src,q=item
    try: outs=teacher(q)
    except Exception: return None
    best=None; bs=-999
    for o in outs:
        sc=score(q,o)
        if sc>bs: bs=sc; best=o
    if best is None or bs<THRESHOLD: return None
    row=to_upstream(q,best); row["_src"]=src; row["_score"]=bs
    return row

def main():
    target=int(sys.argv[1]); out=sys.argv[2]; workers=int(sys.argv[3]) if len(sys.argv)>3 else 14
    seeds=load_seeds()
    print(f"сидов: {len(seeds)} | target={target} | workers={workers}",flush=True)
    rawf=open(out,"w"); seen=set(); ok=0; done=0; fail=0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        it=iter(seeds); futs={}
        def more(k):
            added=0
            while added<k:                      # фикс: дубль НЕ съедает слот подачи
                try: item=next(it)
                except StopIteration: return
                h=hashlib.md5(item[1].lower().encode()).hexdigest()
                if h in seen: continue
                seen.add(h); futs[ex.submit(process,item)]=1; added+=1
        more(workers*3)
        while futs and ok<target:
            for fut in as_completed(list(futs)):
                del futs[fut]; r=fut.result(); done+=1
                if r: ok+=1; rawf.write(json.dumps(r,ensure_ascii=False)+"\n"); rawf.flush()
                else: fail+=1
                if done%50==0: print(f"  done={done} ok={ok} fail={fail}",flush=True)
                if ok>=target: break
                more(1)
            if not futs: break
    print(f"QMD_GEN_DONE ok={ok} fail={fail}",flush=True)

if __name__=="__main__": main()
