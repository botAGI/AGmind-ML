import json, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
URL="http://192.168.1.73:8085/completion"
HOLD="/home/beelinknode/ru-splitter-data/train_v1_holdout.jsonl"
N=int(sys.argv[1]) if len(sys.argv)>1 else 100
rows=[json.loads(l) for l in open(HOLD)][:N]
def gen(prompt):
    data=json.dumps({"prompt":prompt,"n_predict":256,"temperature":0,"cache_prompt":False}).encode()
    req=urllib.request.Request(URL,data=data,headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req,timeout=180).read())["content"]
def parse(txt):
    m=re.search(r'\{.*?\}',txt,re.S)
    if not m: return None
    try: return set(int(x) for x in json.loads(m.group(0)).get("splits",[]))
    except: return None
def f1(p,g,w=0):
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    hp=sum(1 for a in p if any(abs(a-b)<=w for b in g)); hg=sum(1 for b in g if any(abs(a-b)<=w for a in p))
    P=hp/len(p); R=hg/len(g); return 2*P*R/(P+R) if P+R else 0.0
def work(i_ex):
    i,ex=i_ex
    pr=f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n"
    try: out=gen(pr)
    except Exception as e: return (i,"err",str(e)[:60],None,None)
    ps=parse(out); g=set(int(x) for x in json.loads(ex["output"])["splits"])
    return (i,ps,out,g,ex)
res=list(ThreadPoolExecutor(max_workers=4).map(work, list(enumerate(rows))))
valid=0;f0=[];fw=[];exm=0
for r in res:
    i,ps,out,g,ex=r
    if ps=="err" or ps is None: continue
    valid+=1
    if ps==g: exm+=1
    f0.append(f1(ps,g,0)); fw.append(f1(ps,g,1))
n=len(rows)
print(f"=== GGUF deployed (beelink2 Vulkan, Q5_K_M) N={n} ===")
print(f"JSON-valid:     {100*valid/n:.1f}%")
print(f"boundary-F1@0:  {sum(f0)/len(f0):.3f}" if f0 else "n/a")
print(f"boundary-F1@±1: {sum(fw)/len(fw):.3f}" if fw else "n/a")
print(f"exact-set:      {100*exm/n:.1f}%")
print("(HF-модель была: F1@0 0.656 / F1@±1 0.821 / exact 29%)")
print("\n--- 2 примера живьём ---")
shown=0
for r in res:
    i,ps,out,g,ex=r
    if ps in ("err",None) or shown>=2: continue
    shown+=1
    print(f"\n[{ex.get('_src','?')}] предсказано splits={sorted(ps) if ps else ps}, gold={sorted(g)}")
    print("вывод модели:", out.strip()[:200])
