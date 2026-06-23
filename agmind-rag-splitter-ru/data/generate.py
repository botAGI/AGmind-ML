#!/usr/bin/env python3
"""Production data-gen for RU context-aware document splitter.
Teacher = DeepSeek V4 Flash (cluster). Units = razdel sentences + atomic tables/code/headings.
Output = Alpaca JSONL: model predicts boundary indices over numbered units (lossless host-side cut).
Corpus mix (research wu6w6tvsx): cultura_ru_edu(apache) + habr_qna(CC0) + habr-tables(training-only) + synthetic tables.
Concurrency + retries + hard validation + dedup. Usage: gen.py <N> <out.jsonl> [workers]"""
import re, json, sys, os, hashlib, random, itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from razdel import sentenize

DS_URL="http://192.168.1.45:8000/v1"; MODEL="deepseek-v4-flash-spark"
SCHEMA={"type":"object","properties":{"splits":{"type":"array","items":{"type":"integer"}},
        "topic":{"type":"string"}},"required":["splits","topic"]}
WORD_COUNTS=[120,160,200,250,300]
_client=None
def cli():
    global _client
    if _client is None: _client=OpenAI(base_url=DS_URL,api_key="x",timeout=180)
    return _client

# ---------- segmentation (tables/code atomic) ----------
def habr_to_md(t):
    t=re.sub(r"\[code[^\]]*\]","\n```\n",t); return re.sub(r"\[/code\]","\n```\n",t)

# segment_units вынесён в общий модуль inference/segmenter.py — ОДИН источник
# правды для data-gen и инференса (нет train/serve skew; оба фикса таблиц там же).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inference"))
from segmenter import segment_units  # noqa: E402

# ---------- teacher labeling (retries) ----------
def label(units,wc,temp=0.2):
    numbered="\n".join(f"[{k+1}] {u[1]}" for k,u in enumerate(units))
    sysp=("Ты сегментируешь русские документы для системы поиска (RAG). Разбивай на самостоятельные "
          "смысловые фрагменты. Правила: режь ТОЛЬКО на границах пронумерованных единиц; НИКОГДА не "
          f"разрывай таблицы и блоки кода (цельные единицы); фрагмент ~{wc} слов, связный и понятный "
          "без остальных. Возвращай только JSON.")
    usrp=("Ниже документ по пронумерованным единицам (предложения, заголовки, [целые] таблицы и код). "
          "Укажи номера единиц, ПОСЛЕ которых граница смысловых фрагментов. Верни JSON "
          "{\"splits\":[номера],\"topic\":\"одно предложение: о чём документ\"}.\n\n"+numbered)
    last=""
    for attempt in range(3):
        try:
            r=cli().chat.completions.create(model=MODEL,
                messages=[{"role":"system","content":sysp},{"role":"user","content":usrp}],
                temperature=temp,max_tokens=700,extra_body={"guided_json":SCHEMA})
            c=(r.choices[0].message.content or "").strip()
            if not c: last="empty"; continue
            return json.loads(c)
        except Exception as e:
            last=str(e)[:80]
    raise RuntimeError(f"label failed x3: {last}")

def valid_splits(splits,nunits):
    return sorted({int(x) for x in splits if isinstance(x,(int,float)) and 1<=int(x)<nunits})

def gate(units,res):
    sp=valid_splits(res.get("splits",[]),len(units))
    if not (1<=len(sp)<=60): return None
    topic=(res.get("topic") or "").strip()
    if not topic or len(topic)>200 or sum(c.isalpha() and c.lower()>="а" and c.lower()<="я" for c in topic)<5:
        return None
    # chunk sanity: build chunk word counts, reject if any non-final chunk <8 words
    bounds=[0]+sp+[len(units)]
    for a,b in zip(bounds[:-1],bounds[1:]):
        wc=sum(len(u[1].split()) for u in units[a:b] if u[0]=="sent")
        ntab=sum(1 for u in units[a:b] if u[0] in ("table","code"))
        if b!=len(units) and ntab==0 and wc<8: return None
    return {"splits":sp,"topic":topic}

INSTR=("Раздели документ на смысловые части для системы поиска (RAG). Каждая часть читается "
       "независимо, не разрывая предложений, таблиц и кода. Верни ТОЛЬКО номера единиц, после "
       "которых проходит граница, в формате JSON.")
def to_alpaca(units,gated):
    numbered="\n".join(f"[{k+1}] {u[1]}" for k,u in enumerate(units))
    return {"instruction":INSTR,"input":numbered,
            "output":json.dumps({"splits":gated["splits"],"topic":gated["topic"]},ensure_ascii=False)}

# ---------- corpora ----------
def _money(): return f"{random.randint(5,990)} {random.choice(['тыс','млн'])} ₽"
def _pct(a,b): return f"{random.randint(a,b)}%"
def _r(a,b): return str(random.randint(a,b))
def _p(o): return random.choice(o)
# (title, headers, [per-column generator fns])
DOMAINS=[
 ("продажи по регионам",["Регион","Выручка","Рост","Доля"],
   [lambda:_p(["Москва","СПб","Урал","Сибирь","Юг","Поволжье","Дальний Восток","Казань","Новосибирск"]),lambda:_money(),lambda:_pct(-20,45),lambda:_pct(1,40)]),
 ("прайс-лист услуг",["Услуга","Тариф","Срок","Гарантия"],
   [lambda:_p(["Базовый","Стандарт","Премиум","Корпоративный","Старт","Бизнес"]),lambda:_money(),lambda:f"{_r(1,30)} дн",lambda:f"{_r(6,36)} мес"]),
 ("спецификация оборудования",["Параметр","Значение","Ед.","Норма"],
   [lambda:_p(["Мощность","Напряжение","Частота","Масса","Ток","Темп. режим","Давление"]),lambda:_r(1,500),lambda:_p(["Вт","В","Гц","кг","А","°C","бар"]),lambda:_r(1,500)]),
 ("метрики мониторинга",["Сервис","CPU","RAM","Запросы/с","Аптайм"],
   [lambda:_p(["api","db","cache","worker","gateway","auth","queue"]),lambda:_pct(5,95),lambda:f"{_r(1,64)} ГБ",lambda:_r(10,5000),lambda:f"{random.randint(95,99)}.{_r(0,99)}%"]),
 ("расписание занятий",["День","Время","Предмет","Аудитория"],
   [lambda:_p(["Пн","Вт","Ср","Чт","Пт"]),lambda:f"{_r(8,18)}:00",lambda:_p(["Математика","Физика","История","Химия","Биология","Информатика","Право"]),lambda:f"№{_r(100,400)}"]),
 ("бюджет проекта",["Статья","План","Факт","Откл."],
   [lambda:_p(["Разработка","Маркетинг","ФОТ","Аренда","Оборудование","Логистика","Лицензии"]),lambda:_money(),lambda:_money(),lambda:_pct(-30,30)]),
 ("складские остатки",["Артикул","Наименование","Кол-во","Цена"],
   [lambda:f"A-{_r(1000,9999)}",lambda:_p(["Болт","Гайка","Шайба","Кабель","Плата","Корпус","Винт","Реле"]),lambda:_r(0,500),lambda:_money()]),
 ("результаты тестов",["Набор","Прошло","Упало","Время"],
   [lambda:_p(["unit","integration","e2e","smoke","нагрузочный","регресс"]),lambda:_r(50,500),lambda:_r(0,20),lambda:f"{_r(1,300)} с"]),
 ("сравнение моделей",["Модель","Точность","Скорость","Память"],
   [lambda:_p(["Base","Large","Mini","Pro","v1","v2","Turbo"]),lambda:_pct(60,99),lambda:f"{_r(5,500)} мс",lambda:f"{_r(1,32)} ГБ"]),
 ("финансовые показатели",["Показатель","2023","2024","2025"],
   [lambda:_p(["Выручка","Прибыль","EBITDA","Расходы","Активы","Долг","Маржа"]),lambda:_money(),lambda:_money(),lambda:_money()]),
 ("курсы валют",["Валюта","Покупка","Продажа","Δ день"],
   [lambda:_p(["USD","EUR","CNY","GBP","JPY","TRY","AED"]),lambda:_r(60,120),lambda:_r(60,120),lambda:_pct(-5,5)]),
 ("тарифы доставки",["Направление","Срок","Вес до","Стоимость"],
   [lambda:_p(["Москва–СПб","Москва–Екб","СПб–Казань","по региону","межгород"]),lambda:f"{_r(1,10)} дн",lambda:f"{_r(1,30)} кг",lambda:_money()]),
]
INTROS=["Ниже приведены ключевые данные ({t}).","В таблице представлена сводка: {t}.","Рассмотрим основные показатели ({t}).","Для наглядности данные по теме «{t}» сведены в таблицу.","{t}: итоговые цифры за отчётный период приведены ниже.","Документ содержит {t}; детали в таблице."]
MIDS=["Как видно из таблицы, значения распределены неравномерно.","Обратите внимание на крайние позиции — они требуют разбора.","Большинство строк укладывается в норму, но есть исключения.","Динамика по строкам различается, что важно при планировании.","Эти данные используются для дальнейшего анализа."]
ENDS=["В следующем периоде планируется корректировка.","По итогам будет подготовлен расширенный отчёт.","Решение принимается на основании этих цифр.","Таблица обновляется ежемесячно ответственным отделом.","Пояснения доступны в приложении к документу."]
def _mktable(dom):
    title,headers,gens=dom
    if len(headers)>3 and random.random()<0.3: headers,gens=headers[:-1],gens[:-1]
    nc=len(headers); nr=random.randint(2,7)
    head="| "+" | ".join(headers)+" |"; sep="|"+"|".join(["---"]*nc)+"|"
    body="\n".join("| "+" | ".join(str(g()) for g in gens)+" |" for _ in range(nr))
    return title, head+"\n"+sep+"\n"+body
CODE_BLOCKS=[
"```python\ndef process(items):\n    return [x*2 for x in items if x > 0]\n```",
"```bash\nfor f in *.log; do\n  grep ERROR \"$f\" | wc -l\ndone\n```",
"```sql\nSELECT region, SUM(revenue) AS total\nFROM sales\nGROUP BY region\nORDER BY total DESC;\n```",
"```json\n{\n  \"timeout\": 30,\n  \"retries\": 3,\n  \"endpoints\": [\"api\", \"db\"]\n}\n```",
"```yaml\nservice:\n  port: 8080\n  replicas: 3\n  env:\n    - LOG_LEVEL=info\n```",
"```python\nclass Cache:\n    def __init__(self, ttl):\n        self.ttl = ttl\n        self.store = {}\n```",
]
CODE_INTRO=["Рассмотрим пример реализации.","Ниже приведён фрагмент кода.","Базовая конфигурация выглядит так.","Пример использования приведён ниже."]
CODE_AFTER=["Этот фрагмент решает задачу в общем виде.","После изменений сервис нужно перезапустить.","Код адаптируется под конкретные требования.","Обратите внимание на обработку граничных случаев."]
EXTRA=["Данные собраны за отчётный период.","Источник — внутренняя система учёта.","Значения приведены без округления.","Методика расчёта не менялась с прошлого квартала.","Ответственное подразделение подтвердило корректность.","Сведения носят справочный характер."]
EXTRA2=["Это влияет на планирование следующего этапа.","Рекомендуется перепроверить крайние значения.","Подробности обсуждаются с ответственными лицами.","Итоговое решение фиксируется в протоколе.","Дополнительный анализ запланирован отдельно.","При необходимости данные пересматриваются."]
def synth_doc(idx):
    if random.random()<0.22:  # code-doc (длиннее → гейт проходит)
        parts=[]
        if random.random()<0.6: parts.append(f"# {_p(['Реализация','Конфигурация','Пример','Решение'])}: {_p(['обработка данных','сервис','интеграция','пайплайн'])}")
        parts.append("Рассмотрим типовую задачу и её решение. "+_p(EXTRA)+" "+random.choice(CODE_INTRO))
        parts.append(random.choice(CODE_BLOCKS))
        parts.append(random.choice(CODE_AFTER)+" "+_p(EXTRA2)+" "+_p(["Тесты подтверждают корректность.","Производительность в пределах нормы.","Решение готово к развёртыванию."]))
        if random.random()<0.5:
            parts.append("Дополнительно стоит учесть граничные случаи и обработку ошибок. "+_p(EXTRA2))
            parts.append(random.choice(CODE_BLOCKS))
            parts.append(random.choice(CODE_AFTER)+" "+_p(EXTRA))
        return "\n\n".join(parts)
    multi=random.random()<0.45
    doms=random.sample(DOMAINS, 2 if multi else 1)
    parts=[]
    if random.random()<0.7: parts.append(f"# {_p(['Отчёт','Сводка','Справка','Обзор'])}: {doms[0][0].capitalize()}")
    parts.append(random.choice(INTROS).format(t=doms[0][0])+" "+_p(EXTRA)+" "+_p(EXTRA2))
    for d in doms:
        title,tbl=_mktable(d)
        parts.append(random.choice(INTROS).format(t=title)+" "+_p(EXTRA))
        parts.append(tbl)
        parts.append(random.choice(MIDS)+" "+random.choice(ENDS)+" "+_p(EXTRA2))
    return "\n\n".join(parts)

def cyr_ratio(s):
    a=sum(c.isalpha() for c in s); return (sum('а'<=c.lower()<='я' or c.lower()=='ё' for c in s)/a) if a else 0

def iter_docs(target):
    """Yield (source, markdown) interleaved by proportion (synthetic in round-robin)."""
    from datasets import load_dataset
    random.seed(13)
    def synth_gen():
        i=0
        while True:
            yield ("synthetic",synth_doc(i)); i+=1
    SYNTH_ONLY=bool(os.environ.get("SYNTH_ONLY"))
    gens=[("synthetic",synth_gen(),1.0 if SYNTH_ONLY else 0.25,None)]
    if not SYNTH_ONLY:
        try: gens.append(("cultura",iter(load_dataset("deepvk/cultura_ru_edu",split="train",streaming=True)),0.45,["text"]))
        except Exception as e: print("cultura load fail",str(e)[:80])
        try: gens.append(("habr",iter(load_dataset("IlyaGusev/habr",split="train",streaming=True)),0.30,["text_markdown"]))
        except Exception as e: print("habr load fail",str(e)[:80])
    quotas={name:max(1,int(target*w)) for name,_,w,_ in gens}
    counts={name:0 for name,_,_,_ in gens}
    active=True
    while active:
        active=False
        for name,it,w,fields in gens:
            if counts[name]>=quotas[name]: continue
            active=True
            tot=sum(counts.values())
            if tot>40 and counts[name] > w*tot: continue   # proportion guard: skip over-represented source this round
            try: ex=next(it)
            except StopIteration: counts[name]=quotas[name]; continue
            if name=="synthetic": counts[name]+=1; yield ex; continue
            txt=""
            for f in fields:
                v=ex.get(f)
                if isinstance(v,str) and len(v)>len(txt): txt=v
            if not txt: continue
            txt=habr_to_md(txt) if name=="habr" else txt
            if not (800<len(txt)<7000): continue
            if cyr_ratio(txt)<0.6: continue
            counts[name]+=1
            yield (name,txt)

# ---------- main ----------
def process(doc):
    src,md=doc
    units=segment_units(md)
    if not (5<=len(units)<=120): return None
    wc=random.choice(WORD_COUNTS)
    try: res=label(units,wc)
    except Exception: return None
    g=gate(units,res)
    if not g: return None
    row=to_alpaca(units,g); row["_src"]=src; row["_nunits"]=len(units); row["_nsplits"]=len(g["splits"])
    return row

def main():
    target=int(sys.argv[1]) if len(sys.argv)>1 else 200
    out=sys.argv[2] if len(sys.argv)>2 else "train.jsonl"
    workers=int(sys.argv[3]) if len(sys.argv)>3 else 24
    random.seed(7)
    docs=iter_docs(int(target*1.4))  # overshoot for gate rejection; interleaved => proportions hold
    seen=set(); rows=[]; done=0; fail=0
    rawf=open(out.replace(".jsonl","_raw.jsonl"),"w")  # durable incremental output
    print(f"target={target} workers={workers}",flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs={}
        def submit_more(it,k):
            for _ in range(k):
                try: d=next(it)
                except StopIteration: return False
                h=hashlib.md5(d[1][:500].encode()).hexdigest()
                if h in seen: continue
                seen.add(h); futs[ex.submit(process,d)]=1
            return True
        it=iter(docs); submit_more(it,workers*3)
        while futs and len(rows)<target:
            for fut in as_completed(list(futs)):
                del futs[fut]
                r=fut.result()
                if r: rows.append(r); rawf.write(json.dumps(r,ensure_ascii=False)+"\n"); rawf.flush()
                else: fail+=1
                done+=1
                if done%50==0: print(f"  done={done} ok={len(rows)} fail={fail}",flush=True)
                if len(rows)>=target: break
                submit_more(it,1)
            if not futs: break
    random.shuffle(rows)
    hn=max(1,min(len(rows)//10,1500)); hold=rows[:hn]; train=rows[hn:]
    with open(out,"w") as f:
        for r in train: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    with open(out.replace(".jsonl","_holdout.jsonl"),"w") as f:
        for r in hold: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    from collections import Counter
    print(f"\nDONE: train={len(train)} holdout={len(hold)} fail={fail}")
    print("by source:",dict(Counter(r["_src"] for r in rows)))
    print("avg units:",round(sum(r["_nunits"] for r in rows)/max(1,len(rows)),1),
          "avg splits:",round(sum(r["_nsplits"] for r in rows)/max(1,len(rows)),1))

if __name__=="__main__": main()
