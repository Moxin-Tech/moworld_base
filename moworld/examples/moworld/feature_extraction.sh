#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MINDSPEED_PATH=${MINDSPEED_PATH:-${PROJECT_DIR}/../MindSpeed}
MM_DATA=${MM_DATA:-${PROJECT_DIR}/examples/moworld/data_process_data.json}
MM_MODEL=${MM_MODEL:-${PROJECT_DIR}/examples/moworld/data_process_model.json}
MM_TOOL=${MM_TOOL:-${PROJECT_DIR}/examples/moworld/data_process_tools.json}
NPUS_PER_NODE=${NPUS_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6010}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$((NPUS_PER_NODE * NNODES))

ASCEND_ENV=${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
if [[ ! -f "${ASCEND_ENV}" ]]; then
    ASCEND_ENV=/usr/local/Ascend/cann/set_env.sh
fi
[[ -f "${ASCEND_ENV}" ]] || { echo "Ascend environment script not found" >&2; exit 1; }
source "${ASCEND_ENV}"

for path in "${MINDSPEED_PATH}" "${MM_DATA}" "${MM_MODEL}" "${MM_TOOL}"; do
    [[ -e "${path}" ]] || { echo "required path does not exist: ${path}" >&2; exit 1; }
done

export PYTHONPATH="${MINDSPEED_PATH}:${PROJECT_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT:-0}
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export COMBINED_ENABLE=${COMBINED_ENABLE:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}

cd "${PROJECT_DIR}"
mkdir -p logs
torchrun \
    --nproc_per_node "${NPUS_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    mindspeed_mm/tools/feature_extraction/get_moworld_feature.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --micro-batch-size 1 \
    --global-batch-size "${WORLD_SIZE}" \
    --num-workers "${NUM_WORKERS:-1}" \
    --mm-data "${MM_DATA}" \
    --mm-model "${MM_MODEL}" \
    --mm-tool "${MM_TOOL}" \
    --distributed-backend nccl \
    2>&1 | tee "logs/moworld_feature_$(date +%Y%m%d_%H%M%S).log"
