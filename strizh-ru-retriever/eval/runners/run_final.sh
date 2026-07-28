#!/bin/bash
set -u
cd ~/coresident_bench
D=420
{
echo "# Full RAG pipeline under multi-user load — one AMD Strix Halo iGPU (long run, repeated, order-alternated)"
echo "# date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "# host: $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //'); kernel $(uname -r); RAM 124G unified"
echo "# ТРАНЗАКЦИЯ: embed(запрос) → косинусный поиск по индексу (top-8) → rerank → top-4 в prompt → generation 128 ток."
echo "#   индекс: 99 чанков корпуса AGmind, строится тем же эмбеддером на старте каждой точки"
echo "#   поиск: линейный cosine scan в Python по 99 векторам (НЕ тест масштабирования векторной БД)"
echo "# ОКНО: жёсткий дедлайн ${D}с. throughput = завершённые ВНУТРИ окна; латентности = все стартовавшие до дедлайна и завершившиеся (без right-censoring)"
echo "# ФОНОВАЯ ИНДЕКСАЦИЯ: 3 closed-loop воркера, циклический проход ВСЕГО корпуса батчами по 4 чанка (index_corpus_passes = сколько раз корпус пройден)"
echo "# ПОВТОРЫ: каждая точка снята дважды, во втором раунде порядок обратный (контроль порядка и прогрева)"
echo "# серверы: LLM 8086 Qwen3.6-35B-A3B Q4_K_M (35B всего, ~3B активных на токен) -c 16384 --parallel 4 | embed 8084 strizh | embed 8085 bge-m3 | rerank 8087 bge-reranker-v2-m3"
echo "#   на iGPU одновременно резидентны все четыре сервера; нагрузка в каждой точке идёт только в один embedding-порт"
echo "# sha256(16): qwen=$(sha256sum /var/lib/agmind/models/Qwen3.6-35B-A3B-Q4_K_M.gguf|cut -c1-16) strizh=$(sha256sum /var/lib/agmind/models/strizh-ru-retriever.Q8_0.gguf|cut -c1-16) bge=$(sha256sum /var/lib/agmind/models/bge-m3-Q8_0.gguf|cut -c1-16)"
echo ""
echo "### РАУНД 1 (прямой порядок)"
echo "== R1 4 юзера, strizh, без индексации =="; python3 pipeline_bench.py 4 8084 8087 8086 $D corpus_ru.json
echo "== R1 4 юзера, bge, без индексации ==";    python3 pipeline_bench.py 4 8085 8087 8086 $D corpus_ru.json
echo "== R1 4 юзера, strizh + индексация ==";    python3 pipeline_bench.py 4 8084 8087 8086 $D corpus_ru.json --index-port 8084
echo "== R1 4 юзера, bge + индексация ==";       python3 pipeline_bench.py 4 8085 8087 8086 $D corpus_ru.json --index-port 8085
echo ""
echo "### РАУНД 2 (обратный порядок)"
echo "== R2 4 юзера, bge + индексация ==";       python3 pipeline_bench.py 4 8085 8087 8086 $D corpus_ru.json --index-port 8085
echo "== R2 4 юзера, strizh + индексация ==";    python3 pipeline_bench.py 4 8084 8087 8086 $D corpus_ru.json --index-port 8084
echo "== R2 4 юзера, bge, без индексации ==";    python3 pipeline_bench.py 4 8085 8087 8086 $D corpus_ru.json
echo "== R2 4 юзера, strizh, без индексации =="; python3 pipeline_bench.py 4 8084 8087 8086 $D corpus_ru.json
} 2>&1 | tee rag_final.log
echo DONE
