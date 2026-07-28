#!/bin/bash
# ПРИМЕЧАНИЕ (добавлено при публикации): вызовы pipeline_bench.py ниже написаны под ПЕРВУЮ
# версию харнесса (eval/pipeline_bench_v1.py), которая не принимала аргумент корпуса.
# Скрипт оставлен как исторический протокол прогона и НЕ приведён к текущей сигнатуре:
# для новых замеров используйте eval/runners/run_final.sh и eval/pipeline_bench.py.
set -u
cd ~/coresident_bench
{
echo "### ПОВТОРНЫЙ ПРОГОН С ИСПРАВЛЕННЫМИ ХАРНЕССАМИ ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
echo "# fix 1: achieved_rps считается по измеряемому окну (было: числитель включал ramp-up и warmup)"
echo "# fix 2: TTFT — таймер до отправки запроса (было: после получения заголовков, очередь и prefill выпадали)"
echo "# fix 3: top-4 документа после rerank реально уходят в prompt LLM (было: ответ reranker отбрасывался)"
echo ""
echo "## co-resident matrix"
for cond in "- baseline" "8084 online 200" "8085 online 200" "8084 indexing" "8085 indexing" "- baseline"; do
  echo "== $cond =="; python3 coresident_bench.py 8086 $cond; echo ""
done
echo "## prod-pattern: юзер-свип + фоновая индексация (реальный контекст в prompt)"
for U in 1 2 4 8; do echo "== A users=$U strizh =="; python3 pipeline_bench.py $U 8084 8087 8086 75; done
for U in 1 2 4 8; do echo "== B users=$U bge =="; python3 pipeline_bench.py $U 8085 8087 8086 75; done
echo "== C users=4 strizh + indexing =="; python3 pipeline_bench.py 4 8084 8087 8086 75 --index-port 8084
echo "== C users=4 bge + indexing ==";    python3 pipeline_bench.py 4 8085 8087 8086 75 --index-port 8085
} 2>&1 | tee rerun_fixed.log
echo DONE
