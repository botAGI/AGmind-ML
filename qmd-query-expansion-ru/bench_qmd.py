"""Бенч qmd-ru vs сток на holdout 450: формат / русскость / EN-галлюцинации / reward. Механика, без судей."""
import os, json, re, torch
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface")); os.environ["HF_HUB_DISABLE_XET"]="1"
from transformers import AutoModelForCausalLM, AutoTokenizer

HOLD=[json.loads(l) for l in open("qmd_ru_holdout_full.jsonl")]
QS=[r["query"] for r in HOLD]
print(f"queries: {len(QS)}",flush=True)

def parse(txt):
    lines=[l.strip() for l in txt.strip().splitlines() if l.strip()]
    out={"hyde":[],"lex":[],"vec":[],"junk":0}
    for l in lines:
        m=re.match(r"^(hyde|lex|vec):\s*(.+)$",l)
        if m: out[m.group(1)].append(m.group(2))
        else: out["junk"]+=1
    return out

def cyr_ratio(s):
    a=[c for c in s if c.isalpha()]
    return sum(1 for c in a if "а"<=c.lower()<="я" or c.lower()=="ё")/max(len(a),1)

BOILER=re.compile(r"is an important concept|provides functionality|use cases in software|relates to",re.I)

def metrics(q,p):
    all_txt=" ".join(p["hyde"]+p["lex"]+p["vec"])
    fmt = (len(p["hyde"])==1 and len(p["lex"])==3 and len(p["vec"])==2 and p["junk"]==0)
    ru  = cyr_ratio(all_txt)>=0.7 if all_txt else False
    boiler = bool(BOILER.search(all_txt))
    hyde_ok = bool(p["hyde"]) and 50<=len(p["hyde"][0])<=250
    echo = any(l.lower().strip()==q.lower().strip() for l in p["lex"]+p["vec"])
    return fmt, ru, boiler, hyde_ok, echo

def run(name,path):
    tok=AutoTokenizer.from_pretrained(path)
    m=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"":0})
    m.eval()
    F=R=B=H=E=0
    for i in range(0,len(QS),16):
        batch=QS[i:i+16]
        msgs=[[{"role":"user","content":f"/no_think Expand this search query: {q}"}] for q in batch]
        enc=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                    return_dict=True, padding=True).to("cuda")
        with torch.no_grad():
            out=m.generate(**enc, max_new_tokens=220, do_sample=True, temperature=0.7,
                           top_k=20, top_p=0.8, pad_token_id=tok.eos_token_id)
        for j,q in enumerate(batch):
            txt=tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            txt=re.sub(r"<think>.*?</think>","",txt,flags=re.S).strip()
            f,r,b,h,e=metrics(q,parse(txt))
            F+=f; R+=r; B+=b; H+=h; E+=e
        if (i//16)%5==0: print(f"  {name} {i+len(batch)}/{len(QS)}",flush=True)
    n=len(QS)
    print(f"RESULT {name}: format={100*F/n:.1f}% russian={100*R/n:.1f}% en_boilerplate={100*B/n:.1f}% hyde_ok={100*H/n:.1f}% echo={100*E/n:.1f}%",flush=True)
    del m; torch.cuda.empty_cache()

tok_pad_fix=True
run("OURS","out_qmd_full_merged")
run("STOCK","tobil/qmd-query-expansion-1.7B")
print("BENCH_DONE",flush=True)
