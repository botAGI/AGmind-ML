---
language:
- ru
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
pipeline_tag: text-generation
tags:
- search
- query-expansion
- rag
- russian
- qmd
- gguf
---

# qmd-query-expansion-ru

**Русская модель расширения поисковых запросов для [qmd](https://github.com/tobi/qmd)** — локального поискового движка Тоби Лютке. Drop-in замена стоковой `tobil/qmd-query-expansion-1.7B`, которая на русских запросах не работает.

*EN TL;DR: Russian query-expansion model for qmd. The stock model is English-only and fails on Russian queries ([#774](https://github.com/tobi/qmd/issues/774), [#454](https://github.com/tobi/qmd/issues/454)); this is a drop-in fix trained with the upstream `finetune/` recipe. Benchmark below.*

## Зачем

Стоковая модель qmd обучена на 100% английских данных. Русский запрос она «переводит» в английский шаблон-галлюцинацию (*«…is an important concept… in software development»* — про борщ), а формат вывода разваливается ([issue #774](https://github.com/tobi/qmd/issues/774)). Вдобавок BM25-поиск qmd использует Porter-стеммер, который **не понимает русскую морфологию** — поэтому лексические расширения обязаны сами покрывать словоформы и синонимы («получить получение вернуть возврат»). Эта модель обучена делать и то, и другое.

## Подключение (одна строка)

В `~/.config/qmd/index.yml` (или проектном `.qmd/index.yml`):

```yaml
models:
  generate: "hf:AGmind/qmd-query-expansion-ru/qmd-query-expansion-ru-q4_k_m.gguf"
```

Переиндексация не нужна. Контракт тот же, что у стока: `/no_think Expand this search query: {запрос}` → построчный вывод `hyde:` / `lex:` / `vec:`.

## Бенчмарк: 450 отложенных русских запросов, наша vs сток

Механические метрики (без LLM-судей), сэмплинг как в рантайме qmd (temp 0.7, top-k 20, top-p 0.8). Скрипт: `bench_qmd.py` в [репозитории](https://github.com/botAGI/AGmind-ML/tree/main/qmd-query-expansion-ru).

| метрика | **наша** | сток `tobil/…-1.7B` |
|---|---|---|
| валидный контракт qmd (ровно 1 `hyde:` + 3 `lex:` + 2 `vec:`, без мусора) | **99.8%** | 0.0% |
| вывод на русском (≥70% кириллицы) | **99.3%** | 26.2% |
| английский шаблон-боилерплейт («…is an important concept…») | **0.0%** | 22.9% |
| `hyde` в размерном контракте (50–250 симв.) | **96.0%** | 19.8% |
| дословное эхо запроса вместо расширения | 0.4% | 1.3% |

Сток не выдал валидный контракт **ни на одном из 450** русских запросов и в ~23% ответов галлюцинирует английский шаблон. Наша держит контракт на 99.8%, оставаясь русской.

## Пример

Запрос: **«как оформить налоговый вычет за лечение»**

```
hyde: Для оформления налогового вычета за лечение необходимо подтвердить расходы на медицинские услуги...
lex: налоговый вычет лечение оформление декларация
lex: расходы медицинские услуги справки врачей
lex: вычет налога подача декларация
vec: Как получить налоговый вычет за медицинские расходы
vec: Подача заявки на налоговый вычет за лечение и справки
```

Сток на тот же запрос: `hyde: Как оформить налоговый вычет за лечение is an important concept that relates to... It provides functionality for various use cases in software development.`

## Обучение

- База: Qwen/Qwen3-1.7B (та же, что у стока), LoRA r16/α32 all-proj, 5 эпох — точный SFT-рецепт апстрима (`tobi/qmd/finetune`).
- Данные: 5 075 русских запросов (MIRACL-ru apache-2.0, Mr.TyDi-ru apache-2.0, Яндекс.Кью CC0) → дистилляция учителем (DeepSeek, n=2 сэмпла + rule-based reward-фильтр ≥70, средний скор 94.8).
- Формат идентичен трейн-схеме апстрима (`{"query", "output": [["hyde",…],["lex",…],["vec",…]]}`, hyde первой).

## Ограничения

- Модель 1.7B: `hyde:`-пассажи могут фантазировать детали (номера деклараций, ингредиенты). Для расширения поиска это терпимо — термины остаются в теме, — но не считайте hyde фактами.
- Обучена для русского; для английских запросов используйте сток.

## Файлы

- `qmd-query-expansion-ru-q4_k_m.gguf` — для qmd / llama.cpp (рекомендуется)
- safetensors — merged fp16 для transformers

Код, генератор датасета, трейн- и бенч-скрипты: [github.com/botAGI/AGmind-ML → qmd-query-expansion-ru](https://github.com/botAGI/AGmind-ML/tree/main/qmd-query-expansion-ru). Родственная модель: [AGmind/agmind-rag-splitter-ru](https://huggingface.co/AGmind/agmind-rag-splitter-ru).
