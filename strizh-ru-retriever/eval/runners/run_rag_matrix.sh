#!/bin/bash
set -u
cd ~/coresident_bench
{
echo "# Production RAG pipeline under multi-user load — one AMD Strix Halo iGPU"
echo "# date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "# host: $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //'); kernel $(uname -r); RAM 124G unified"
echo "# ЦЕПОЧКА НА ТРАНЗАКЦИЮ: embed(запрос) → косинусный поиск по индексу (top-8) → rerank → top-4 в prompt → generation 128 ток."
echo "#   индекс: 99 чанков корпуса AGmind (медиана ~1270 символов), строится тем же эмбеддером на старте каждой точки"
echo "# ОКНО: жёсткий дедлайн 90с; после дедлайна новые транзакции не стартуют, in-flight дожидаются (join)"
echo "#   в throughput идут ТОЛЬКО завершённые внутри окна (поля started / completed_in_window / completed_after_window)"
echo "# серверы (co-resident на одном iGPU):"
echo "#   LLM    8086: Qwen3.6-35B-A3B Q4_K_M, image $(docker inspect qwen-llm-bench --format '{{.Image}}' | cut -c8-19), -ngl 999 -c 16384 --parallel 4"
echo "#   embed  8084: strizh Q8_0, b9049, --pooling mean -c 65536 -b 8192 -ub 8192 -np 8"
echo "#   embed  8085: bge-m3 Q8_0, b9049, --pooling cls  -c 8192  -b 8192 -ub 8192 -np 8"
echo "#   rerank 8087: bge-reranker-v2-m3 Q8_0, b9049, --rerank -c 8192 -b 8192 -ub 8192 --parallel 4"
echo "# sha256(16): qwen=$(sha256sum /var/lib/agmind/models/Qwen3.6-35B-A3B-Q4_K_M.gguf|cut -c1-16) strizh=$(sha256sum /var/lib/agmind/models/strizh-ru-retriever.Q8_0.gguf|cut -c1-16) bge=$(sha256sum /var/lib/agmind/models/bge-m3-Q8_0.gguf|cut -c1-16) rerank=$(sha256sum /var/lib/agmind/models/bge-reranker-v2-m3-Q8_0.gguf|cut -c1-16)"
echo "# фоновая индексация: 3 воркера closed-loop, батчи по 4 чанка на тот же embedding-порт"
echo "# скрипт: eval/pipeline_bench.py"
echo ""
for U in 1 2 4 8; do echo "== A users=$U strizh =="; python3 pipeline_bench.py $U 8084 8087 8086 90 corpus_ru.json; done
for U in 1 2 4 8; do echo "== B users=$U bge =="; python3 pipeline_bench.py $U 8085 8087 8086 90 corpus_ru.json; done
echo "== C users=4 strizh + indexing =="; python3 pipeline_bench.py 4 8084 8087 8086 90 corpus_ru.json --index-port 8084
echo "== C users=4 bge + indexing ==";    python3 pipeline_bench.py 4 8085 8087 8086 90 corpus_ru.json --index-port 8085
} 2>&1 | tee rag_matrix.log
echo DONE
