
#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=2
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PATCH_ROOT="/opt/tiger/lmms-eval-extra/vllm"
if [[ -d "${PATCH_ROOT}" ]]; then
export PYTHONPATH="${PYTHONPATH:-}${PATCH_ROOT:+:${PATCH_ROOT}}"
else
echo "WARNING: patch dir ${PATCH_ROOT} not found, skip patch load!" >&2
fi

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_VERSION:-MyModel}}"
MODEL_VERSION="${MODEL_VERSION:-${SERVED_MODEL_NAME}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-${PORT1:-8000}}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}/v1}"
HEALTHCHECK_URL="${HEALTHCHECK_URL:-http://127.0.0.1:${PORT}/v1/models}"
API_KEY="${API_KEY:-EMPTY}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-}"
export HF_HOME="${HF_HOME:-/mnt/bn/commercial-ai-mllm-shared-hl/home/panjunwen/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
NPU_DEVICES=${NPU_DEVICES:-${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}}}
NPU_DEVICES="${NPU_DEVICES// /}"
IFS=',' read -r -a NPU_DEVICE_ARRAY <<< "${NPU_DEVICES}"
NPU_COUNT=${#NPU_DEVICE_ARRAY[@]}
if [[ "${NPU_COUNT}" -lt 1 || -z "${NPU_DEVICE_ARRAY[0]}" ]]; then
echo "NPU_DEVICES is empty. Set it like: NPU_DEVICES=0,1,2,3 bash ${0}" >&2
exit 1
fi
TP_SIZE="${TP_SIZE:-1}"
if ! [[ "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
echo "TP_SIZE must be a positive integer, got: ${TP_SIZE}" >&2
exit 1
fi
if (( TP_SIZE > NPU_COUNT )); then
echo "TP_SIZE=${TP_SIZE} exceeds visible NPU count=${NPU_COUNT}" >&2
exit 1
fi
if [[ -n "${DP_SIZE:-}" ]]; then
if ! [[ "${DP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
echo "DP_SIZE must be a positive integer, got: ${DP_SIZE}" >&2
exit 1
fi
else
if (( NPU_COUNT % TP_SIZE != 0 )); then
echo "NPU_COUNT=${NPU_COUNT} must be divisible by TP_SIZE=${TP_SIZE} when DP_SIZE is not set" >&2
exit 1
fi
DP_SIZE=$((NPU_COUNT / TP_SIZE))
fi
DP_SIZE_LOCAL="${DP_SIZE_LOCAL:-${DP_SIZE}}"
if ! [[ "${DP_SIZE_LOCAL}" =~ ^[1-9][0-9]*$ ]]; then
echo "DP_SIZE_LOCAL must be a positive integer, got: ${DP_SIZE_LOCAL}" >&2
exit 1
fi
if (( DP_SIZE_LOCAL > DP_SIZE )); then
echo "DP_SIZE_LOCAL=${DP_SIZE_LOCAL} exceeds DP_SIZE=${DP_SIZE}" >&2
exit 1
fi
if (( DP_SIZE_LOCAL * TP_SIZE > NPU_COUNT )); then
echo "DP_SIZE_LOCAL=${DP_SIZE_LOCAL} with TP_SIZE=${TP_SIZE} requires $((DP_SIZE_LOCAL * TP_SIZE)) NPUs, but only ${NPU_COUNT} are visible" >&2
exit 1
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29580}"
DATA_PARALLEL_RPC_PORT="${DATA_PARALLEL_RPC_PORT:-29680}"
API_SERVER_COUNT="${API_SERVER_COUNT:-$((DP_SIZE * 2))}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_PIXELS_EXPLICIT=0
MIN_PIXELS_EXPLICIT=0
if [[ -n "${MAX_PIXELS+x}" ]]; then
MAX_PIXELS_EXPLICIT=1
fi
if [[ -n "${MIN_PIXELS+x}" ]]; then
MIN_PIXELS_EXPLICIT=1
fi
MAX_PIXELS="${MAX_PIXELS:-262144}"
MIN_PIXELS="${MIN_PIXELS:-16384}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-64}"
LIMIT_MM_FRAMES="${LIMIT_MM_FRAMES:-256}"

LIMIT_MM_WIDTH="${LIMIT_MM_WIDTH:-1024}"
LIMIT_MM_HEIGHT="${LIMIT_MM_HEIGHT:-1024}"
VIDEO_FPS="${VIDEO_FPS:-}"
PASS_VIDEO_URL="${PASS_VIDEO_URL:-0}"

ADDITIONAL_CONFIG=$(jq -n '{
  "enable_flashcomm":1,
  "enable_fused_mc2":1,
  "scheduler_config":{
    "enable_balance_scheduling":true
  }
}')

MM_PROCESSOR_KWARGS=$(jq -n \
  --argjson max_pixels "${MAX_PIXELS}" \
  --argjson min_pixels "${MIN_PIXELS}" \
  '{max_pixels: $max_pixels, min_pixels: $min_pixels}')

if [[ "${PASS_VIDEO_URL}" == "0" || "${PASS_VIDEO_URL,,}" == "false" ]]; then
LIMIT_MM_PER_PROMPT=$(jq -n \
  --argjson img_cnt "${MAX_FRAMES_NUM}" \
  --argjson vid_frames "${LIMIT_MM_FRAMES}" \
  --argjson w "${LIMIT_MM_WIDTH}" \
  --argjson h "${LIMIT_MM_HEIGHT}" \
  '{
    image: {count: $img_cnt, width: $w, height: $h},
    video: {count: 1, num_frames: $vid_frames, width: $w, height: $h}
  }')
else
LIMIT_MM_PER_PROMPT=$(jq -n \
  --argjson vid_frames "${LIMIT_MM_FRAMES}" \
  --argjson w "${LIMIT_MM_WIDTH}" \
  --argjson h "${LIMIT_MM_HEIGHT}" \
  '{
    video: {count: 1, num_frames: $vid_frames, width: $w, height: $h}
  }')
fi

if [[ -z "${ALLOWED_LOCAL_MEDIA_PATH}" ]]; then
if [[ "${PASS_VIDEO_URL}" == "0" || "${PASS_VIDEO_URL,,}" == "false" ]]; then
ALLOWED_LOCAL_MEDIA_PATH="/dev/shm"
else
ALLOWED_LOCAL_MEDIA_PATH="/mnt/bn/"
fi
fi
PREDECODE_VIDEO_TO_IMAGE_FILES="${PREDECODE_VIDEO_TO_IMAGE_FILES:-1}"
VIDEO_DECODE_BACKEND="${VIDEO_DECODE_BACKEND:-decord}"
MEDIA_IO_VIDEO_BACKEND="${MEDIA_IO_VIDEO_BACKEND:-}"
VIDEO_DO_SAMPLE_FRAMES="${VIDEO_DO_SAMPLE_FRAMES:-}"
MM_PROCESSOR_DO_RESIZE="${MM_PROCESSOR_DO_RESIZE:-}"
USE_QWEN3_VL_OPENAI_MESSAGES="${USE_QWEN3_VL_OPENAI_MESSAGES:-0}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
RENDERER_NUM_WORKERS="${RENDERER_NUM_WORKERS:-16}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-0}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-mp}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-0}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
COMPILATION_CONFIG='{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
fi
export VLLM_MQ_MAX_CHUNK_BYTES_MB="${VLLM_MQ_MAX_CHUNK_BYTES_MB:-1024}"
if [[ "${PREDECODE_VIDEO_TO_IMAGE_FILES}" == "1" || "${PREDECODE_VIDEO_TO_IMAGE_FILES,,}" == "true" ]]; then
if [[ -z "${LMMS_VIDEO_PREDECODE_CACHE_DIR:-}" && -d /dev/shm ]]; then
export LMMS_VIDEO_PREDECODE_CACHE_DIR="/dev/shm/lmms-eval-video-predecode-${MAX_FRAMES_NUM}-${VIDEO_DECODE_BACKEND:-pyav}"
fi
fi
TASKS="${TASKS:-videomme_v2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-}"
NUM_CONCURRENT="${NUM_CONCURRENT:-128}"
PREFIX_AWARE_QUEUE="${PREFIX_AWARE_QUEUE:-1}"
DATA_PARALLEL_STICKY_ROUTING="${DATA_PARALLEL_STICKY_ROUTING:-0}"
TIMEOUT="${TIMEOUT:-6000}"
MAX_RETRIES="${MAX_RETRIES:-3}"
LIMIT="${LIMIT:-}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1800}"
READY_INTERVAL_S="${READY_INTERVAL_S:-5}"
RUN_ID="${RUN_ID:-qwen3vl_vllm_eval_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/logs/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/${TASKS//,/_}/vllm}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/vllm_openai_${PORT}.log}"
MODELS_JSON="${MODELS_JSON:-${RUN_DIR}/${TASKS//,/_}/models.json}"
EVAL_STDOUT_LOG="${EVAL_STDOUT_LOG:-${RUN_DIR}/${TASKS//,/_}/eval.stdout.log}"
OUTPUT_PATH="${OUTPUT_PATH:-${RUN_DIR}/${TASKS//,/_}/results}"
STOP_SERVER_ON_EXIT="${STOP_SERVER_ON_EXIT:-1}"
USE_EXISTING_SERVER="${USE_EXISTING_SERVER:-0}"
STREAM_LOGS="${STREAM_LOGS:-1}"
LOG_SAMPLES="${LOG_SAMPLES:-1}"
RESPONSE_CACHE="${RESPONSE_CACHE:-}"
ENABLE_THINKING="${ENABLE_THINKING:-}"
if [[ -n "${ENABLE_THINKING}" && "${ENABLE_THINKING,,}" != "true" && "${ENABLE_THINKING,,}" != "false" ]]; then
echo "ENABLE_THINKING must be true, false, or empty, got: ${ENABLE_THINKING}" >&2
exit 1
fi
if [[ -n "${VIDEO_DO_SAMPLE_FRAMES}" && "${VIDEO_DO_SAMPLE_FRAMES,,}" != "true" && "${VIDEO_DO_SAMPLE_FRAMES,,}" != "false" ]]; then
echo "VIDEO_DO_SAMPLE_FRAMES must be true, false, or empty, got: ${VIDEO_DO_SAMPLE_FRAMES}" >&2
exit 1
fi
if [[ -n "${MM_PROCESSOR_DO_RESIZE}" && "${MM_PROCESSOR_DO_RESIZE,,}" != "true" && "${MM_PROCESSOR_DO_RESIZE,,}" != "false" ]]; then
echo "MM_PROCESSOR_DO_RESIZE must be true, false, or empty, got: ${MM_PROCESSOR_DO_RESIZE}" >&2
exit 1
fi
if [[ -n "${VIDEO_FPS}" ]] && ! [[ "${VIDEO_FPS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
echo "VIDEO_FPS must be a non-negative number or empty, got: ${VIDEO_FPS}" >&2
exit 1
fi
if ! [[ "${MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
echo "MAX_NEW_TOKENS must be a positive integer, got: ${MAX_NEW_TOKENS}" >&2
exit 1
fi
mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${OUTPUT_PATH}"
server_pid=""
log_stream_pids=()
start_log_stream() {
local label=$1
local file=$2
local stream=${3:-stderr}
if [[ "${STREAM_LOGS}" != "1" ]]; then
return 0
fi
touch "${file}"
if [[ "${stream}" == "stdout" ]]; then
tail -n +1 -F "${file}" 2>/dev/null | sed -u "s/^/[${label}] /" &
else
tail -n +1 -F "${file}" 2>/dev/null | sed -u "s/^/[${label}] /" >&2 &
fi
log_stream_pids+=("$!")
}
stop_log_streams() {
local pid
for pid in "${log_stream_pids[@]}"; do
if kill -0 "${pid}" >/dev/null 2>&1; then
kill "${pid}" >/dev/null 2>&1 || true
wait "${pid}" >/dev/null 2>&1 || true
fi
done
}
cleanup() {
local status=$?
if [[ "${STOP_SERVER_ON_EXIT}" == "1" && -n "${server_pid}" ]]; then
if kill -0 "${server_pid}" >/dev/null 2>&1; then
echo "Stopping vLLM server pid=${server_pid}" >&2
kill "${server_pid}" >/dev/null 2>&1 || true
wait "${server_pid}" >/dev/null 2>&1 || true
fi
fi
stop_log_streams
exit "${status}"
}
trap cleanup EXIT INT TERM
health_check() {
if command -v curl >/dev/null 2>&1; then
curl -g -fsS --max-time 10 "${HEALTHCHECK_URL}" -o "${MODELS_JSON}" 2>/dev/null
return $?
fi
python - "${HEALTHCHECK_URL}" "${MODELS_JSON}" <<'PY'
import sys
import urllib.request
url, output = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        body = response.read()
        if not 200 <= response.status < 300:
            raise SystemExit(1)
    with open(output, "wb") as f:
        f.write(body)
except Exception:
    raise SystemExit(1)
PY
}
port_in_use() {
local port=$1
if command -v ss >/dev/null 2>&1; then
ss -H -ltn "sport = :${port}" 2>/dev/null | grep -q .
return $?
fi
if command -v netstat >/dev/null 2>&1; then
netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"
return $?
fi
python - "${port}" >/dev/null 2>&1 <<'PY'
import socket
import sys
port = int(sys.argv[1])
for family, host in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError:
        raise SystemExit(0)
    finally:
        sock.close()
raise SystemExit(1)
PY
}
wait_for_ready() {
local deadline=$((SECONDS + READY_TIMEOUT_S))
local attempt=0
while (( SECONDS < deadline )); do
attempt=$((attempt + 1))
if health_check; then
echo "vLLM server is ready: ${HEALTHCHECK_URL}"
return 0
fi
if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
echo "vLLM server exited before readiness. Log: ${LOG_FILE}" >&2
tail -200 "${LOG_FILE}" >&2 || true
return 1
fi
echo "Waiting for vLLM server readiness attempt=${attempt} url=${HEALTHCHECK_URL}" >&2
sleep "${READY_INTERVAL_S}"
done
echo "Timed out after ${READY_TIMEOUT_S}s waiting for vLLM server. Log: ${LOG_FILE}" >&2
tail -200 "${LOG_FILE}" >&2 || true
return 1
}
echo "run_dir=${RUN_DIR}"
echo "model_path=${MODEL_PATH}"
echo "served_model_name=${SERVED_MODEL_NAME}"
echo "base_url=${BASE_URL}"
echo "healthcheck_url=${HEALTHCHECK_URL}"
echo "output_path=${OUTPUT_PATH}"
echo "tasks=${TASKS}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "temperature=${TEMPERATURE}"
echo "top_p=${TOP_P}"
echo "top_k=${TOP_K}"
echo "presence_penalty=${PRESENCE_PENALTY:-model_default}"
echo "npu_devices=${NPU_DEVICES}"
echo "tp_size=${TP_SIZE}"
echo "dp_size=${DP_SIZE}"
echo "dp_size_local=${DP_SIZE_LOCAL}"
echo "api_server_count=${API_SERVER_COUNT}"
echo "max_frames_num=${MAX_FRAMES_NUM}"
echo "video_fps=${VIDEO_FPS:-frame_count_only}"
echo "pass_video_url=${PASS_VIDEO_URL}"
echo "predecode_video_to_image_files=${PREDECODE_VIDEO_TO_IMAGE_FILES}"
echo "video_decode_backend=${VIDEO_DECODE_BACKEND}"
echo "media_io_video_backend=${MEDIA_IO_VIDEO_BACKEND:-server_default}"
echo "video_do_sample_frames=${VIDEO_DO_SAMPLE_FRAMES:-processor_default}"
echo "mm_processor_do_resize=${MM_PROCESSOR_DO_RESIZE:-processor_default}"
echo "max_pixels=${MAX_PIXELS}"
echo "min_pixels=${MIN_PIXELS}"
echo "use_qwen3_vl_openai_messages=${USE_QWEN3_VL_OPENAI_MESSAGES}"
echo "async_scheduling=${ASYNC_SCHEDULING}"
echo "renderer_num_workers=${RENDERER_NUM_WORKERS}"
echo "mm_processor_cache_gb=${MM_PROCESSOR_CACHE_GB}"
echo "distributed_executor_backend=${DISTRIBUTED_EXECUTOR_BACKEND}"
echo "enable_expert_parallel=${ENABLE_EXPERT_PARALLEL}"
echo "compilation_config=${COMPILATION_CONFIG}"
echo "prefix_aware_queue=${PREFIX_AWARE_QUEUE}"
echo "data_parallel_sticky_routing=${DATA_PARALLEL_STICKY_ROUTING}"
echo "stream_logs=${STREAM_LOGS}"
echo "log_samples=${LOG_SAMPLES}"
echo "response_cache=${RESPONSE_CACHE:-disabled}"
echo "enable_thinking=${ENABLE_THINKING:-model_default}"
echo "vllm_mq_max_chunk_bytes_mb=${VLLM_MQ_MAX_CHUNK_BYTES_MB}"
echo "lmms_video_predecode_cache_dir=${LMMS_VIDEO_PREDECODE_CACHE_DIR:-}"
echo "PYTHONPATH=${PYTHONPATH}"
if health_check; then
if [[ "${USE_EXISTING_SERVER}" != "1" ]]; then
echo "Endpoint is already healthy before launch: ${HEALTHCHECK_URL}" >&2
echo "Set USE_EXISTING_SERVER=1 to evaluate against the existing service, or choose another PORT." >&2
exit 1
fi
echo "Using existing vLLM server: ${HEALTHCHECK_URL}"
elif port_in_use "${PORT}"; then
echo "Port ${PORT} is already in use, but ${HEALTHCHECK_URL} is not healthy." >&2
echo "Stop the process on that port or choose another PORT." >&2
exit 1
else
: >"${LOG_FILE}"
start_log_stream "vllm" "${LOG_FILE}" "stderr"
  (
cd /tmp

CANN_ENV="/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
source "${CANN_ENV}"

export ASCEND_VISIBLE_DEVICES="${NPU_DEVICES}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}"

EXTRA_VLLM_ARGS=()
if [[ "${ASYNC_SCHEDULING}" == "1" || "${ASYNC_SCHEDULING,,}" == "true" ]]; then
EXTRA_VLLM_ARGS+=(--async-scheduling)
fi
if [[ -n "${RENDERER_NUM_WORKERS}" ]]; then
EXTRA_VLLM_ARGS+=(--renderer-num-workers "${RENDERER_NUM_WORKERS}")
fi
if [[ -n "${MM_PROCESSOR_CACHE_GB}" ]]; then
EXTRA_VLLM_ARGS+=(--mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB}")
fi
if [[ -n "${DISTRIBUTED_EXECUTOR_BACKEND}" ]]; then
EXTRA_VLLM_ARGS+=(--distributed-executor-backend "${DISTRIBUTED_EXECUTOR_BACKEND}")
fi
if [[ "${ENABLE_EXPERT_PARALLEL}" == "1" || "${ENABLE_EXPERT_PARALLEL,,}" == "true" ]]; then
EXTRA_VLLM_ARGS+=(--enable-expert-parallel)
fi

VLLM_ARGS=(
--host "${HOST}"
--port "${PORT}"
--model "${MODEL_PATH}"
--served-model-name "${SERVED_MODEL_NAME}"
--trust-remote-code
--allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}"
--enable-chunked-prefill 
--tensor-parallel-size "${TP_SIZE}"
--master-addr "${MASTER_ADDR}"
--master-port "${MASTER_PORT}"
--data-parallel-size "${DP_SIZE}"
--api-server-count "${API_SERVER_COUNT}"
--data-parallel-size-local "${DP_SIZE_LOCAL}"
--data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT}"
--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
--max-model-len "${MAX_MODEL_LEN}"
--max-num-seqs "${MAX_NUM_SEQS}"
--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
--limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}"
--mm-processor-kwargs "${MM_PROCESSOR_KWARGS}"
--mm-encoder-tp-mode 'data'
--media-io-kwargs '{"video": {"backend": "pyav"}}'
--compilation-config "${COMPILATION_CONFIG}"
--additional-config "${ADDITIONAL_CONFIG}"
)
VLLM_ARGS+=("${EXTRA_VLLM_ARGS[@]}")
VLLM_ARGS+=("$@")

exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
  ) >"${LOG_FILE}" 2>&1 &
server_pid="$!"
echo "Started vLLM server pid=${server_pid} log=${LOG_FILE}"
wait_for_ready
fi
echo "Starting evaluation request"
EXTRA_ARGS=()
GEN_KWARGS="max_new_tokens=${MAX_NEW_TOKENS},temperature=${TEMPERATURE},top_p=${TOP_P},top_k=${TOP_K}"
if [[ -n "${PRESENCE_PENALTY}" ]]; then
GEN_KWARGS+=",presence_penalty=${PRESENCE_PENALTY}"
fi
EXTRA_ARGS+=(--gen_kwargs "${GEN_KWARGS}")
if [[ -n "${LIMIT}" ]]; then
EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [[ "${LOG_SAMPLES}" == "1" || "${LOG_SAMPLES,,}" == "true" ]]; then
EXTRA_ARGS+=(--log_samples)
fi
if [[ -n "${RESPONSE_CACHE}" ]]; then
mkdir -p "${RESPONSE_CACHE}"
EXTRA_ARGS+=(--use_cache "${RESPONSE_CACHE}")
fi
MODEL_ARGS="base_url=${BASE_URL},model_version=${MODEL_VERSION},api_key=${API_KEY},pass_video_url=${PASS_VIDEO_URL},predecode_video_to_image_files=${PREDECODE_VIDEO_TO_IMAGE_FILES},video_decode_backend=${VIDEO_DECODE_BACKEND},use_qwen3_vl_openai_messages=${USE_QWEN3_VL_OPENAI_MESSAGES},max_frames_num=${MAX_FRAMES_NUM},timeout=${TIMEOUT},max_retries=${MAX_RETRIES},num_concurrent=${NUM_CONCURRENT},prefix_aware_queue=${PREFIX_AWARE_QUEUE},data_parallel_size=${DP_SIZE},data_parallel_sticky_routing=${DATA_PARALLEL_STICKY_ROUTING},httpx_trust_env=False"
if [[ "${MAX_PIXELS_EXPLICIT}" == "1" ]]; then
MODEL_ARGS+=",max_pixels=${MAX_PIXELS}"
fi
if [[ "${MIN_PIXELS_EXPLICIT}" == "1" ]]; then
MODEL_ARGS+=",min_pixels=${MIN_PIXELS}"
fi
if [[ -n "${ENABLE_THINKING}" ]]; then
MODEL_ARGS+=",enable_thinking_kwarg=${ENABLE_THINKING,,}"
fi
if [[ -n "${VIDEO_DO_SAMPLE_FRAMES}" ]]; then
MODEL_ARGS+=",video_do_sample_frames=${VIDEO_DO_SAMPLE_FRAMES,,}"
fi
if [[ -n "${VIDEO_FPS}" ]]; then
MODEL_ARGS+=",video_fps=${VIDEO_FPS}"
fi
if [[ -n "${MEDIA_IO_VIDEO_BACKEND}" ]]; then
MODEL_ARGS+=",media_io_video_backend=${MEDIA_IO_VIDEO_BACKEND}"
fi
if [[ -n "${MM_PROCESSOR_DO_RESIZE}" ]]; then
MODEL_ARGS+=",mm_processor_do_resize=${MM_PROCESSOR_DO_RESIZE,,}"
fi
: >"${EVAL_STDOUT_LOG}"
start_log_stream "eval stdout" "${EVAL_STDOUT_LOG}" "stdout"
if ! (

VENV_PATH="/opt/tiger/lmms-eval/lmms_env"
source "${VENV_PATH}/bin/activate"

echo "Eval shell VIRTUAL_ENV=${VIRTUAL_ENV}"
echo "Eval shell PYTHONPATH=${PYTHONPATH}"

cd "${REPO_ROOT}"
python -m lmms_eval \
--model openai \
--model_args "${MODEL_ARGS}" \
--tasks "${TASKS}" \
--batch_size "${BATCH_SIZE}" \
--output_path "${OUTPUT_PATH}" \
"${EXTRA_ARGS[@]}"
  ) >"${EVAL_STDOUT_LOG}" 2>&1; then
echo "Evaluation failed. Log tail: ${EVAL_STDOUT_LOG}" >&2
tail -120 "${EVAL_STDOUT_LOG}" >&2 || true
exit 1
fi
if ! find "${OUTPUT_PATH}" -type f -name '*_results.json' -print -quit | grep -q .; then
echo "Evaluation completed without result JSON files under ${OUTPUT_PATH}" >&2
echo "stdout tail: ${EVAL_STDOUT_LOG}" >&2
tail -120 "${EVAL_STDOUT_LOG}" >&2 || true
exit 1
fi
echo "Evaluation finished. Results: ${OUTPUT_PATH}"
echo "Logs: ${RUN_DIR}"
