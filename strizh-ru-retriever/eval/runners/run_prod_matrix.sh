#!/bin/bash
# Прод-сценарии: полная RAG-цепочка (embed→rerank→LLM) под мультиюзером на одном iGPU + кривые.
set -u
cd ~/coresident_bench
OUT=~/coresident_bench/prod_matrix.log
{
echo "# Production-pattern RAG load — full chain embed→rerank→LLM on one AMD Strix Halo iGPU"
echo "# date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "# host: $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //'); kernel $(uname -r); RAM 124G unified"
echo "# servers (все co-resident на одном iGPU):"
echo "#   LLM    8086: Qwen3.6-35B-A3B Q4_K_M, server-vulkan(latest), -ngl 999 -c 16384 --parallel 4"
echo "#   embed  8084: strizh Q8_0, b9049, --pooling mean -c 65536 -b 8192 -ub 8192 -np 8"
echo "#   embed  8085: bge-m3 Q8_0, b9049, --pooling cls -c 8192 -b 4096 -ub 4096 -np 8"
echo "#   rerank 8087: bge-reranker-v2-m3 Q8_0, b9049, --rerank -c 8192 -b 4096 -ub 4096 --parallel 4"
echo "# sha256(16): qwen=$(sha256sum /var/lib/agmind/models/Qwen3.6-35B-A3B-Q4_K_M.gguf | cut -c1-16) strizh=$(sha256sum /var/lib/agmind/models/strizh-ru-retriever.Q8_0.gguf | cut -c1-16) bge=$(sha256sum /var/lib/agmind/models/bge-m3-Q8_0.gguf | cut -c1-16) rerank=$(sha256sum /var/lib/agmind/models/bge-reranker-v2-m3-Q8_0.gguf | cut -c1-16)"
echo "# юзер-цикл: embed(короткий RU-запрос) → rerank(запрос + 8 доков ~150 ток., top_n 4) → LLM completion 128 ток. (stream, TTFT); 75s на точку"
echo "# фоновая переиндексация: 3 воркера closed-loop, батчи 4×~350 ток. на тот же embedding-порт"
echo "# скрипты: eval/pipeline_bench.py, eval/loadtest_v2.py, eval/batch_thr.py (этот репозиторий)"
echo ""
echo "### СЦЕНАРИЙ A: юзер-свип, эмбеддер = STRIZH"
for U in 1 2 4 8; do echo "== A users=$U strizh =="; python3 pipeline_bench.py $U 8084 8087 8086 75; done
echo ""
echo "### СЦЕНАРИЙ B: юзер-свип, эмбеддер = bge-m3"
for U in 1 2 4 8; do echo "== B users=$U bge =="; python3 pipeline_bench.py $U 8085 8087 8086 75; done
echo ""
echo "### СЦЕНАРИЙ C: 4 юзера + фоновая переиндексация тем же эмбеддером"
echo "== C users=4 strizh + indexing =="; python3 pipeline_bench.py 4 8084 8087 8086 75 --index-port 8084
echo "== C users=4 bge + indexing ==";    python3 pipeline_bench.py 4 8085 8087 8086 75 --index-port 8085
echo ""
echo "### КРИВАЯ ПО КОНКУРЕНЦИИ (изолированный эмбеддер, sustained 15s/точка, 1 текст/запрос)"
for C in 1 2 4 8 16 32 64; do python3 loadtest_v2.py 8084 $C t:15 texts_short_ru.json strizh-conc$C; done
for C in 1 2 4 8 16 32 64; do python3 loadtest_v2.py 8085 $C t:15 texts_short_ru.json bge-conc$C; done
echo ""
echo "### КРИВАЯ ПО БАТЧАМ (изолированный эмбеддер, длинные ~350 ток., texts/s)"
for B in 1 2 4 8 16 32 64 128; do NB=$(( 512 / B > 6 ? 512 / B : 6 )); python3 batch_thr.py 8084 $B $NB texts_long_ru.json strizh-b$B; done
for B in 1 2 4 8 16 32 64 128; do NB=$(( 512 / B > 6 ? 512 / B : 6 )); python3 batch_thr.py 8085 $B $NB texts_long_ru.json bge-b$B; done
} 2>&1 | tee $OUT
echo "DONE -> $OUT"
