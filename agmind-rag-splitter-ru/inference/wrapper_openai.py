"""OpenAI-compatible wrapper: Dify шлёт документ как chat-сообщение -> razdel+нумерация ->
splitter-модель (llama.cpp :8085) -> реконструкция -> возвращает смысловые чанки.
Lossless (режем оригинал), таблицы/код атомарны."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import re, json, time, requests
from razdel import sentenize

SPLITTER = "http://192.168.1.73:8085/completion"  # splitter GGUF на beelink2 (AMD Vulkan)
MODELID = "ru-splitter-chunker"
INSTR = ("Раздели документ на смысловые части для системы поиска (RAG). Каждая часть читается "
         "независимо, не разрывая предложений, таблиц и кода. Верни ТОЛЬКО номера единиц, после "
         "которых проходит граница, в формате JSON.")
app = FastAPI()

def segment_units(md):
    units=[]; lines=md.split("\n"); i=0; n=len(lines); buf=[]
    def flush(b):
        t=" ".join(x.strip() for x in b if x.strip())
        for s in sentenize(t):
            st=s.text.strip()
            if st: units.append(("sent",st))
    while i<n:
        ln=lines[i]
        if ln.strip().startswith("```"):
            blk=[ln]; i+=1
            while i<n and not lines[i].strip().startswith("```"): blk.append(lines[i]); i+=1
            if i<n: blk.append(lines[i]); i+=1
            flush(buf); buf=[]; units.append(("code","\n".join(blk))); continue
        if "|" in ln and i+1<n and re.match(r"^\s*\|?[\s:|-]+\|?\s*$",lines[i+1]) and "-" in lines[i+1]:
            flush(buf); buf=[]; blk=[ln]; i+=1
            while i<n and "|" in lines[i]: blk.append(lines[i]); i+=1
            units.append(("table","\n".join(blk))); continue
        if ln.strip()=="": flush(buf); buf=[]; i+=1; continue
        if re.match(r"^#{1,6}\s",ln.strip()): flush(buf); buf=[]; units.append(("head",ln.strip())); i+=1; continue
        buf.append(ln); i+=1
    flush(buf); return units

def chunk_document(text):
    units=segment_units(text)
    if len(units)<2: return [text.strip()], ""
    numbered="\n".join(f"[{k+1}] {u[1]}" for k,u in enumerate(units))
    prompt=f"### Instruction:\n{INSTR}\n\n### Input:\n{numbered}\n\n### Response:\n"
    out=requests.post(SPLITTER,json={"prompt":prompt,"n_predict":256,"temperature":0,"cache_prompt":False},timeout=180).json().get("content","")
    m=re.search(r'\{.*?\}',out,re.S); splits=[]; topic=""
    if m:
        try:
            o=json.loads(m.group(0)); splits=sorted(set(int(x) for x in o.get("splits",[]) if 0<int(x)<len(units))); topic=o.get("topic","")
        except Exception: pass
    bounds=[0]+splits+[len(units)]
    chunks=["\n".join(u[1] for u in units[a:b]).strip() for a,b in zip(bounds,bounds[1:])]
    return chunks, topic

def render(text):
    if not text.strip(): return "Пришли текст документа — верну смысловые чанки (таблицы целиком)."
    chunks,topic=chunk_document(text)
    out=[f"📄 Тема: {topic}".rstrip(), f"Чанков: {len(chunks)}",""]
    for i,c in enumerate(chunks):
        tag=" [таблица/код]" if ("|---" in c or "|--" in c or "```" in c) else ""
        out.append(f"━━━ ЧАНК {i+1}{tag} ━━━\n{c}\n")
    return "\n".join(out)

@app.get("/v1/models")
def models():
    return {"object":"list","data":[{"id":MODELID,"object":"model","owned_by":"agmind"}]}

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body=await req.json(); msgs=body.get("messages",[]); text=""
    for m in reversed(msgs):
        if m.get("role")=="user":
            c=m.get("content"); text=c if isinstance(c,str) else json.dumps(c,ensure_ascii=False); break
    try: content=render(text)
    except Exception as e: content=f"Ошибка чанкинга: {e}"
    cr=int(time.time())
    base={"id":"chatcmpl-spl","created":cr,"model":MODELID}
    if body.get("stream"):
        def g():
            d={**base,"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":content},"finish_reason":None}]}
            yield f"data: {json.dumps(d,ensure_ascii=False)}\n\n"
            d2={**base,"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
            yield f"data: {json.dumps(d2,ensure_ascii=False)}\n\n"; yield "data: [DONE]\n\n"
        return StreamingResponse(g(),media_type="text/event-stream")
    resp={**base,"object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
    return JSONResponse(resp)
