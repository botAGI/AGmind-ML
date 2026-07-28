#!/bin/bash
# Co-resident matrix: LLM на том же iGPU + embedding-нагрузка. Лог — публикуемый.
set -u
OUT=~/coresident_bench/coresident_run.log
LLM_PORT=8086; STRIZH_PORT=8084; BGE_PORT=8085
{
echo "# Co-resident benchmark — Qwen3.6-35B-A3B (Q4_K_M) + embedders on one AMD Strix Halo iGPU"
echo "# date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "# host: $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //'); kernel $(uname -r); RAM $(free -g | awk '/Mem/{print $2}')G unified"
echo "# images: LLM = ghcr.io/ggml-org/llama.cpp:server-vulkan (latest, imageId 2327a745588d; b9049 не грузит SSM-тензоры Qwen3.6); embedders = server-vulkan-b9049"
echo "# sha256(16): qwen=$(sha256sum /var/lib/agmind/models/Qwen3.6-35B-A3B-Q4_K_M.gguf | cut -c1-16) strizh=$(sha256sum /var/lib/agmind/models/strizh-ru-retriever.Q8_0.gguf | cut -c1-16) bge=$(sha256sum /var/lib/agmind/models/bge-m3-Q8_0.gguf | cut -c1-16)"
echo "# LLM server: --model Qwen3.6...Q4_K_M.gguf -ngl 999 --ctx-size 8192 --parallel 1 (порт $LLM_PORT)"
echo "# strizh server (порт $STRIZH_PORT): --embeddings --pooling mean -ngl 999 -c 65536 -b 8192 -ub 8192 -np 8"
echo "# bge server (порт $BGE_PORT):     --embeddings --pooling cls  -ngl 999 -c 8192  -b 4096 -ub 4096 -np 8"
echo "# LLM-замер: 1 warmup + 5 промптов (RU, n_predict=200, temp=0, stream, cache_prompt=false); медианы TTFT/tok-s"
echo "# online-нагрузка: OPEN-LOOP фиксированный QPS (короткие RU-запросы, 1 текст/запрос) — одинаковая работа для обеих моделей"
echo "# indexing-нагрузка: closed-loop 6 воркеров, батчи 4×(~350 ток.) — полная тяга «переиндексации»"
echo "# порядок: baseline → strizh@200 → bge@200 → strizh-indexing → bge-indexing → closing-baseline (контроль дрейфа)"
echo ""
run() { local label=$1; shift; echo "== $label =="; python3 ~/coresident_bench/coresident_bench.py "$@" ; echo ""; sleep 5; }
run baseline           $LLM_PORT - baseline
run strizh@200qps      $LLM_PORT $STRIZH_PORT online 200
run bge-m3@200qps      $LLM_PORT $BGE_PORT online 200
run strizh-indexing    $LLM_PORT $STRIZH_PORT indexing
run bge-m3-indexing    $LLM_PORT $BGE_PORT indexing
run closing-baseline   $LLM_PORT - baseline
} 2>&1 | tee $OUT
echo "DONE -> $OUT"
