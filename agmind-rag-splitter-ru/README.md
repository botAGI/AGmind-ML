# agmind-rag-splitter-ru

Русскоязычный **context-aware сплиттер документов** для RAG: режет документ на самодостаточные смысловые чанки, **держит таблицы и блоки кода целиком** и выдаёт структурные границы, по которым чанки восстанавливаются **байт-в-байт** из исходника.

Дообучен из **`t-tech/T-lite-it-2.1`** (Qwen3-8B, Apache-2.0) дистилляцией от фронтир-модели-учителя, обучен bf16 LoRA на одной RTX 5090, квантизован в GGUF и развёрнут на **AMD (Vulkan)** через `llama.cpp`.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Base](https://img.shields.io/badge/base-T--lite--it--2.1-orange)
![GGUF](https://img.shields.io/badge/GGUF-Q5__K__M-green)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-AGmind%2Fagmind--rag--splitter--ru-yellow)](https://huggingface.co/AGmind/agmind-rag-splitter-ru)

**🤗 Модель:** [`AGmind/agmind-rag-splitter-ru`](https://huggingface.co/AGmind/agmind-rag-splitter-ru) — safetensors (transformers) + GGUF Q5_K_M (llama.cpp).

> Идея вдохновлена [`mhenrichsen/context-aware-splitter`](https://huggingface.co/mhenrichsen/context-aware-splitter-1b) (датский) — пересобрано под русский с кириллическим токенайзером и lossless-выводом по индексам.

---

## Результаты

Оценка на отложенной выборке 1500 примеров (согласие границ с метками учителя), greedy-декодинг:

| Метрика | HF-модель (RTX 5090) | GGUF Q5_K_M (AMD Vulkan) |
|---|---|---|
| Валидный JSON | **100%** | **100%** |
| boundary-F1 @0 (точная позиция) | **0.656** | 0.639 |
| boundary-F1 @±1 (±1 предложение) | **0.821** | 0.817 |
| exact-set-match | 29% | 25% |

- **100% парсимого структурного вывода** — модель всегда возвращает `{"splits":[...], "topic":"..."}`.
- GGUF/Vulkan совпадает с FP16 в пределах шума квантизации → **токенайзер и Q5 сохраняют качество**.
- Таблицы остаются **атомарными** (проверено на живых документах — `inference/demo.py`).

*(Это согласие с моделью-учителем дистилляции, не human-ground-truth — см. [Ограничения](#ограничения).)*

---

## Подход

```
                        ┌──────────────── ОБУЧЕНИЕ (однократно) ────────────────┐
 RU-корпуса  ──►  razdel-деление на предложения + таблицы/код как атомарные юниты  ──►  нумерованные юниты
 (wiki/habr/синт)                                                                          │
                                                                                           ▼
                              Учитель (DeepSeek-V4-Flash)  размечает индексы границ + topic
                                                                                           │
                                                                                           ▼
                        bf16 LoRA (r32, rsLoRA, all-linear, response-only)  на  T-lite-it-2.1
                                                                                           │
                                                              merge ──► GGUF (Q5_K_M) ──► AMD Vulkan
 ────────────────────────────────────────────────────────────────────────────────────────────────────
                        ┌──────────────── ИНФЕРЕНС ───────────────────────────────┐
 документ ─► [хост] деление на нумерованные юниты ─► [модель] {"splits":[i,...],"topic":...}
          ─► [хост] нарезка ОРИГИНАЛА по индексам ─► чанки (байт-в-байт, таблицы целиком)
```

**Ключевые решения**

- **Кириллическая база.** `T-lite-it-2.1` (continued-pretrain Qwen3-8B от T-Bank) несёт переработанный токенайзер (~2.4 токена/слово на русском против ~3.9 у ванильного). Токенайзер Llama-2 (как в датском оригинале) кириллично-неэффективен.
- **Вывод индексов, а не переписывание текста.** Модель возвращает **индексы** границ, а хост режет оригинал. Это **lossless** (никаких перефразов ё/кавычек), **~в 10× дешевле** (≈40 токенов вывода вместо переписывания всего документа) и **гарантирует целостность таблиц**.
- **Дистилляция от учителя.** Сильная модель размечает, где резать; без ручной разметки. Self-consistency + жёсткие гейты фильтруют метки.
- **Response-only loss.** В лоссе только короткий JSON-вывод — модель не тратится на воспроизведение входа.

---

## Структура

```
data/
  generate.py          # генератор данных дистилляцией (корпуса → нумерованные юниты → метки учителя → Alpaca JSONL)
  sample_train.jsonl   # сэмпл 600 примеров (полный ~17k — см. data/README.md)
  sample_holdout.jsonl
training/
  train.py             # bf16 LoRA (Unsloth + TRL), response-only, rsLoRA
  patch_tokenizer_hash.py  # регистрирует BPE-pre-tokenizer T-lite для конверсии в GGUF
  config.md            # гиперпараметры + точные пины окружения (Blackwell)
eval/
  eval_hf.py           # boundary-F1 / валидный JSON / exact-match на HF-модели
  eval_gguf.py         # те же метрики против развёрнутого GGUF-эндпоинта
  results.md           # цифры + методология
inference/
  demo.py              # живая нарезка документа (показывает, что таблица цела)
  wrapper_openai.py    # OpenAI-совместимый сервис (текст → чанки)
docs/
  methodology.md       # решения база/PEFT/данные/деплой с обоснованием
  model_card.md        # карточка модели HuggingFace
```

---

## Быстрый старт

**1. Сгенерировать датасет** (нужен OpenAI-совместимый эндпоинт учителя):
```bash
pip install -r requirements.txt
# отредактируй URL/имя учителя в data/generate.py
python data/generate.py 17000 train.jsonl 48      # цель, выходной файл, воркеры
```

**2. Обучить** (RTX 5090 / Blackwell, WSL2; точные пины в `training/config.md`):
```bash
DATA=train.jsonl HOLD=train_holdout.jsonl OUTDIR=out EPOCHS=2 MAXLEN=4096 BS=2 GA=8 \
  python training/train.py
```

**3. Конвертировать в GGUF** (сначала пропатчить хэш токенайзера):
```bash
python training/patch_tokenizer_hash.py          # регистрирует pre-tokenizer T-lite
python llama.cpp/convert_hf_to_gguf.py out_merged --outfile model-f16.gguf --outtype f16
./llama.cpp/build/bin/llama-quantize model-f16.gguf model-Q5_K_M.gguf Q5_K_M
```

**4. Запустить** (AMD Vulkan):
```bash
llama-server -m model-Q5_K_M.gguf -ngl 99 -c 8192 --host 0.0.0.0 --port 8085
```

**5. Оценка / демо:**
```bash
MODEL=out_merged N=300 python eval/eval_hf.py
python inference/demo.py
```

---

## Железо

- **Обучение:** RTX 5090 (Blackwell, 32 ГБ) под WSL2 — bf16 LoRA, ~3.5 ч, пик 25.4 ГБ VRAM, 2122 шага (2 эпохи на ~17k примеров).
- **Инференс:** AMD Strix Halo (gfx1151) через `llama.cpp` **Vulkan** (без CUDA) — Q5_K_M, ~5.9 ГБ.
- **Учитель / генерация данных:** любой OpenAI-совместимый эндпоинт (использовался self-hosted DeepSeek-V4-Flash).

---

## Ограничения

- Метрики — это **согласие границ с метками учителя**, не human-ground-truth. Следующий шаг — downstream RAG-оценка (hit-rate / faithfulness).
- Модель **слегка пере-сегментирует** (изредка лишняя граница) — наследие гранулярности учителя; лечится пост-мёржем или подкруткой промпта учителя.
- Рассчитан на **прозу + мелкие/средние таблицы**. **Очень большие таблицы** (превышающие бюджет эмбеддера) требуют отдельной стратегии (построчная нарезка с повтором шапки + whole-table retrieval), а не боундари-сплиттера.

## Лицензия

Apache-2.0 (соответствует базе `T-lite-it-2.1` и пермиссивным корпусам). Обучающие данные — из публичных русских корпусов; по-источниковые лицензии в `data/README.md`.

## Благодарности

- Базовая модель: [`t-tech/T-lite-it-2.1`](https://huggingface.co/t-tech/T-lite-it-2.1) (T-Bank)
- Идея-вдохновение: [`mhenrichsen/context-aware-splitter`](https://huggingface.co/mhenrichsen/context-aware-splitter-1b)
- Обучение: [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl); инференс: [llama.cpp](https://github.com/ggml-org/llama.cpp)
