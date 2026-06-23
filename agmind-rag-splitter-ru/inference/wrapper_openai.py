"""OpenAI-compatible chunking service: принимает документ как chat-сообщение,
делает razdel-нумерацию -> splitter-модель (llama.cpp) -> реконструкцию,
возвращает смысловые чанки. Lossless (режет оригинал), таблицы/код атомарны.

Любой OpenAI-совместимый клиент шлёт {"messages":[{"role":"user","content": "<документ>"}]}
на /v1/chat/completions и получает чанки в ответном сообщении.

Запуск:  SPLITTER_URL=http://<host>:8085/completion uvicorn wrapper_openai:app --host 0.0.0.0 --port 8086
"""
import os, sys, re, json, time, requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segmenter import segment_units  # единый сегментатор — см. segmenter.py (no train/serve skew)

SPLITTER = os.environ.get("SPLITTER_URL", "http://127.0.0.1:8085/completion")  # llama-server /completion
MODELID = "ru-splitter-chunker"
INSTR = ("Раздели документ на смысловые части для системы поиска (RAG). Каждая часть читается "
         "независимо, не разрывая предложений, таблиц и кода. Верни ТОЛЬКО номера единиц, после "
         "которых проходит граница, в формате JSON.")
app = FastAPI()

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
    return ["\n".join(u[1] for u in units[a:b]).strip() for a,b in zip(bounds,bounds[1:])], topic

def render(text):
    if not text.strip(): return "Пришли текст документа — верну смысловые чанки (таблицы целиком)."
    chunks,topic=chunk_document(text)
    out=[f"📄 Тема: {topic}".rstrip(), f"Чанков: {len(chunks)}",""]
    for i,c in enumerate(chunks):
        tag=" [таблица/код]" if ("|---" in c or "|--" in c or "```" in c) else ""
        out.append(f"━━━ ЧАНК {i+1}{tag} ━━━\n{c}\n")
    return "\n".join(out)

@app.get("/v1/models")
def models(): return {"object":"list","data":[{"id":MODELID,"object":"model","owned_by":"agmind"}]}

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
    cr=int(time.time()); base={"id":"chatcmpl-spl","created":cr,"model":MODELID}
    if body.get("stream"):
        def g():
            d={**base,"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":content},"finish_reason":None}]}
            yield f"data: {json.dumps(d,ensure_ascii=False)}\n\n"
            d2={**base,"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
            yield f"data: {json.dumps(d2,ensure_ascii=False)}\n\n"; yield "data: [DONE]\n\n"
        return StreamingResponse(g(),media_type="text/event-stream")
    resp={**base,"object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
    return JSONResponse(resp)
