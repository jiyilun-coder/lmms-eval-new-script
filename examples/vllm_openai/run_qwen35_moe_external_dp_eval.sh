#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TP_SIZE=2
export DP_SIZE=2
export DP_SIZE_LOCAL=1
# 外部DP实例数目：一共8组，每组TP2；总卡16
BACKEND_COUNT=8
PROXY_PORT=9000
PROXY_HOST="127.0.0.1"
# backend端口从8100开始
START_BACKEND_PORT=8100

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

# 后端NPU分配数组；每组TP占用连续2张卡
BACKEND_NPU_LIST=(
  "0,1"
  "2,3"
  "4,5"
  "6,7"
  "8,9"
  "10,11"
  "12,13"
  "14,15"
)

# backend端口数组
BACKEND_PORTS=()
for ((i=0;i<BACKEND_COUNT;i++)); do
  BACKEND_PORTS+=($((START_BACKEND_PORT + i)))
done

# proxy后端地址列表
DP_HOSTS=()
for ((i=0;i<BACKEND_COUNT;i++)); do
  DP_HOSTS+=("127.0.0.1")
done

# -------- 子脚本路径：复用原有run_qwen3vl_vllm_eval.sh，但是做改造调用方式 --------
EVAL_SCRIPT="/opt/tiger/lmms-eval-extra/examples/vllm_openai/run_qwen3vl_vllm_eval.sh"
PROXY_PY="${SCRIPT_DIR}/dp_load_balance_proxy_server.py"

# 存放后台pid
declare -a BACKEND_PIDS=()
PROXY_PID=""
RUN_ID="${RUN_ID:-qwen35_moe_extdp_$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/logs/${RUN_ID}}"
mkdir -p "${RUN_DIR}"

cleanup() {
  local status=$?
  echo "==== Cleaning up backend vllm instances and proxy ===="
  if [[ -n "${PROXY_PID}" ]]; then
    if kill -0 "${PROXY_PID}" >/dev/null 2>&1; then
      kill "${PROXY_PID}" || true
      wait "${PROXY_PID}" || true
    fi
  fi
  for pid in "${BACKEND_PIDS[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" || true
      wait "${pid}" || true
    fi
  done
  exit "${status}"
}
trap cleanup EXIT INT TERM

# 启动每一个后端vllm实例；每个后端内部 DP_SIZE=1；TP=2
echo "============ Starting ${BACKEND_COUNT} backend vLLM instances (each TP=${TP_SIZE}, no internal DP) ============"
for ((idx=0; idx<BACKEND_COUNT; idx++)); do
  npu_devices="${BACKEND_NPU_LIST[$idx]}"
  port="${BACKEND_PORTS[$idx]}"
  LOG_FILE="${RUN_DIR}/vllm_backend_${idx}_port${port}.log"
  echo "Start backend[${idx}] NPU=${npu_devices}, port=${port}, log=${LOG_FILE}"

  # 注意关键点：DP_SIZE=1；API_SERVER_COUNT=1；不再使用内部data‑parallel；
  # USE_EXISTING_SERVER=0；STOP_SERVER_ON_EXIT=0（由顶层脚本统一trap清理）
  (
    export NPU_DEVICES="${npu_devices}"
    export TP_SIZE=2
    export DP_SIZE=1
    export API_SERVER_COUNT=1
    export STOP_SERVER_ON_EXIT=0
    export USE_EXISTING_SERVER=0
    export PORT="${port}"
    exec bash "${EVAL_SCRIPT}" \
      --enable-prefix-caching \
      --mamba-cache-mode align \
      "$@"
  ) > "${LOG_FILE}" 2>&1 &
  backend_pid=$!
  BACKEND_PIDS+=("${backend_pid}")
done

# wait for all backends healthy
echo "==== Wait for all 8 vLLM backends to become ready ===="
READY_DEADLINE=$((SECONDS + 1800))
all_ok=0
while (( SECONDS < READY_DEADLINE )); do
  all_ok=1
  for p in "${BACKEND_PORTS[@]}"; do
    if ! curl --max-time 8 -s "http://127.0.0.1:${p}/v1/models" >/dev/null ; then
      all_ok=0
      break
    fi
  done
  if (( all_ok == 1 )); then
    echo "✅ All backend vllm services are healthy now."
    break
  fi
  echo "⏳ Still waiting for backends ..."
  sleep 8
done
if (( all_ok !=1 )); then
  echo "❌ Timeout: some backend vllm instances failed to start within timeout"
  exit 1
fi

# launch external dp proxy
echo "==== Starting external DP load‑balance proxy server at port ${PROXY_PORT} ===="
PROXY_LOG="${RUN_DIR}/proxy_server.log"

python3 "${PROXY_PY}" \
  --host "${PROXY_HOST}" \
  --port "${PROXY_PORT}" \
  --dp-hosts "${DP_HOSTS[@]}" \
  --dp-ports "${BACKEND_PORTS[@]}" \
  >"${PROXY_LOG}" 2>&1 &
PROXY_PID=$!

# wait proxy ready
PROXY_READY_DEADLINE=$((SECONDS + 300))
proxy_up=0
while (( SECONDS < PROXY_READY_DEADLINE )); do
  if curl --max-time 5 -s "http://${PROXY_HOST}:${PROXY_PORT}/healthcheck" >/dev/null; then
    proxy_up=1
    echo "✅ Proxy server ready at ${PROXY_HOST}:${PROXY_PORT}"
    break
  fi
  sleep 3
done
if (( proxy_up !=1 )); then
  echo "❌ Proxy server failed to start"
  exit 1
fi

# ---------- 调用lmms‑eval，直接访问proxy；不再拉起vllm服务 ----------
echo "==== Launch lmms‑eval evaluation against proxy endpoint ===="
EVAL_LOG="${RUN_DIR}/lmms_eval_run.log"

# 复用原有环境变量，BASE_URL指向proxy；USE_EXISTING_SERVER=1，STOP_SERVER_ON_EXIT=0
export BASE_URL="http://${PROXY_HOST}:${PROXY_PORT}/v1"
export USE_EXISTING_SERVER=1
export STOP_SERVER_ON_EXIT=0
# 注意：MODEL_PATH仍然需要传入，只是后端已经启动；
exec bash "${EVAL_SCRIPT}" \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  "$@"
