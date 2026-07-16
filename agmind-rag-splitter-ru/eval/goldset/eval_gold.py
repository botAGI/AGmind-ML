"""Оценка модели-сплиттера на эталоне gold.jsonl (boundary-F1 vs арбитражный gold)."""
import json, re, sys, requests

URL=sys.argv[1] if len(sys.argv)>1 else "http://192.168.1.73:8085/completion"
GOLD=[json.loads(l) for l in open(__file__.rsplit("/",1)[0]+"/gold.jsonl")]
INSTR=("Раздели документ на смысловые части для системы поиска (RAG). Каждая часть читается "
       "независимо, не разрывая предложений, таблиц и кода. Верни ТОЛЬКО номера единиц, после "
       "которых проходит граница, в формате JSON.")

def predict(units):
    numbered="\n".join(f"[{k+1}] {u}" for k,u in enumerate(units))
    prompt=f"### Instruction:\n{INSTR}\n\n### Input:\n{numbered}\n\n### Response:\n"
    r=requests.post(URL,json={"prompt":prompt,"n_predict":256,"temperature":0,"cache_prompt":False},timeout=180).json()
    out=r.get("content","")
    m=re.search(r'\{.*?\}',out,re.S)
    if not m: return None
    try: return sorted({int(x) for x in json.loads(m.group(0)).get("splits",[]) if 0<int(x)<len(units)})
    except Exception: return None

def f1(pred,gold,tol):
    if not pred and not gold: return 1.0
    if not pred or not gold: return 0.0
    used=set(); tp=0
    for p in pred:
        c=[g for g in gold if abs(g-p)<=tol and g not in used]
        if c: used.add(min(c,key=lambda g:abs(g-p))); tp+=1
    prec=tp/len(pred); rec=tp/len(gold)
    return 2*prec*rec/(prec+rec) if prec+rec else 0.0

jok=0; f10=[]; f11=[]; per_src={}
for i,g in enumerate(GOLD):
    p=predict(g["units"])
    if p is None:
        f10.append(0); f11.append(0); continue
    jok+=1
    a,b=f1(p,g["gold_splits"],0),f1(p,g["gold_splits"],1)
    f10.append(a); f11.append(b)
    s=per_src.setdefault(g["src"],[0,0,0]); s[0]+=a; s[1]+=b; s[2]+=1
    if (i+1)%35==0: print(f"  {i+1}/{len(GOLD)}...",flush=True)
n=len(GOLD)
print(f"\nМОДЕЛЬ: {sys.argv[2] if len(sys.argv)>2 else URL}")
print(f"валидный JSON: {100*jok/n:.1f}%  |  boundary-F1@0: {sum(f10)/n:.3f}  |  boundary-F1@±1: {sum(f11)/n:.3f}")
for src,(a,b,c) in per_src.items(): print(f"  {src}: F1@0={a/c:.3f} F1@±1={b/c:.3f} (n={c})")
