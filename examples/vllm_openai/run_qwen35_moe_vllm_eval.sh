#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NPU_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export TP_SIZE="${TP_SIZE:-2}"
export DP_SIZE="${DP_SIZE:-8}"
export API_SERVER_COUNT="${API_SERVER_COUNT:-16}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
export NUM_CONCURRENT="${NUM_CONCURRENT:-500}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
export RENDERER_NUM_WORKERS="${RENDERER_NUM_WORKERS:-32}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export PREFIX_AWARE_QUEUE="${PREFIX_AWARE_QUEUE:-0}"
export DATA_PARALLEL_STICKY_ROUTING="${DATA_PARALLEL_STICKY_ROUTING:-0}"
export VLLM_SCHEDULER_OVERRIDES='{"max_num_encoder_input_tokens":32768}'
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  export COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2,4,8,16,24,32,40,48,56,64,80,96,112,128]}'
fi

export TASKS="${TASKS:-videomme_v2}"

exec bash "/opt/tiger/lmms-eval-extra/examples/vllm_openai/run_qwen3vl_vllm_eval.sh" \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  "$@"
