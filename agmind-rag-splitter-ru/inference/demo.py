import sys, json, re, urllib.request
sys.path.insert(0,'/home/beelinknode/ru-splitter-data')
from gen import segment_units
URL="http://192.168.1.73:8085/completion"
DOC = """# Обзор новых тарифов мобильной связи

Оператор представил три новых тарифных плана для частных клиентов. Все они включают безлимитные звонки внутри сети. Подключение бесплатное при переходе с другого оператора.

| Тариф | Цена в месяц | Гигабайты | Минуты |
|-------|-------------|-----------|--------|
| Лайт | 350 ₽ | 15 | 300 |
| Стандарт | 550 ₽ | 40 | безлимит |
| Премиум | 900 ₽ | 100 | безлимит |

Как видно, тариф «Стандарт» предлагает лучшее соотношение цены и объёма. Премиум подойдёт активным пользователям мобильного интернета. Переход на новые тарифы доступен в личном кабинете и приложении.

Дополнительно действует акция для новых абонентов. Первый месяц предоставляется со скидкой пятьдесят процентов. Акция продлится до конца квартала."""
units=segment_units(DOC)
numbered="\n".join(f"[{k+1}] {u[1]}" for k,u in enumerate(units))
instr="Раздели документ на смысловые части для системы поиска (RAG). Каждая часть читается независимо, не разрывая предложений, таблиц и кода. Верни ТОЛЬКО номера единиц, после которых проходит граница, в формате JSON."
prompt=f"### Instruction:\n{instr}\n\n### Input:\n{numbered}\n\n### Response:\n"
data=json.dumps({"prompt":prompt,"n_predict":200,"temperature":0}).encode()
out=json.loads(urllib.request.urlopen(urllib.request.Request(URL,data=data,headers={"Content-Type":"application/json"}),timeout=120).read())["content"]
print("ВЫВОД МОДЕЛИ:",out.strip(),"\n")
splits=sorted(set(int(x) for x in json.loads(re.search(r'\{.*\}',out,re.S).group(0))["splits"] if 0<int(x)<len(units)))
bounds=[0]+splits+[len(units)]
print(f"единиц: {len(units)} | границы: {splits}\n"+"="*60)
for ci,(a,b) in enumerate(zip(bounds,bounds[1:])):
    seg=units[a:b]
    txt="\n".join(u[1] for u in seg).strip()
    tab=any(u[0]=="table" for u in seg)
    print(f"\n━━━ ЧАНК {ci+1}{'  ◀ ТАБЛИЦА ЦЕЛИКОМ' if tab else ''} ━━━")
    print(txt)
