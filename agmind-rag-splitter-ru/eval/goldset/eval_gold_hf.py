"""Eval HF-модели на эталоне gold.jsonl: boundary-F1 vs консенсус/арбитраж gold."""
import json, re, sys, torch
from unsloth import FastLanguageModel

MODEL_DIR=sys.argv[1]; GOLD=sys.argv[2]
model,tok=FastLanguageModel.from_pretrained(MODEL_DIR,max_seq_length=8192,dtype=torch.bfloat16,load_in_4bit=False)
FastLanguageModel.for_inference(model)
INSTR=("Раздели документ на смысловые части для системы поиска (RAG). Каждая часть читается "
       "независимо, не разрывая предложений, таблиц и кода. Верни ТОЛЬКО номера единиц, после "
       "которых проходит граница, в формате JSON.")
def predict(units):
    numbered="\n".join(f"[{k+1}] {u}" for k,u in enumerate(units))
    prompt=f"### Instruction:\n{INSTR}\n\n### Input:\n{numbered}\n\n### Response:\n"
    ids=tok(prompt,return_tensors="pt").to("cuda")
    out=model.generate(**ids,max_new_tokens=256,do_sample=False,temperature=None,top_p=None,pad_token_id=tok.eos_token_id)
    txt=tok.decode(out[0][ids["input_ids"].shape[1]:],skip_special_tokens=True)
    m=re.search(r'\{.*?\}',txt,re.S)
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
    pr=tp/len(pred); rc=tp/len(gold)
    return 2*pr*rc/(pr+rc) if pr+rc else 0.0
rows=[json.loads(l) for l in open(GOLD)]
jok=0; f10=[]; f11=[]; per={}
for i,g in enumerate(rows):
    p=predict(g["units"])
    if p is None: f10.append(0); f11.append(0); continue
    jok+=1; a,b=f1(p,g["gold_splits"],0),f1(p,g["gold_splits"],1)
    f10.append(a); f11.append(b)
    s=per.setdefault(g["src"],[0,0,0]); s[0]+=a; s[1]+=b; s[2]+=1
    if (i+1)%35==0: print(f"  {i+1}/{len(rows)}",flush=True)
n=len(rows)
print(f"\nМОДЕЛЬ {MODEL_DIR}: JSON {100*jok/n:.1f}% | F1@0 {sum(f10)/n:.3f} | F1@±1 {sum(f11)/n:.3f}")
for src,(a,b,c) in per.items(): print(f"  {src}: F1@0={a/c:.3f} F1@±1={b/c:.3f} (n={c})")
