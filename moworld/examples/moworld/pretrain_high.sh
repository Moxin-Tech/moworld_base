#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MINDSPEED_PATH=${MINDSPEED_PATH:-${PROJECT_DIR}/../MindSpeed}
MM_DATA=${MM_DATA:-${PROJECT_DIR}/examples/moworld/data_feature.json}
MM_MODEL=${MM_MODEL:-${PROJECT_DIR}/examples/moworld/pretrain_model_feature_high.json}
MM_TOOL=${MM_TOOL:-${PROJECT_DIR}/mindspeed_mm/tools/tools.json}
LOAD_PATH=${LOAD_PATH:-${PROJECT_DIR}/../checkpoints/base/dcp/high_noise_model}
SAVE_PATH=${SAVE_PATH:-${PROJECT_DIR}/../checkpoints/train/high_noise_model}
FSDP_CONFIG=${FSDP_CONFIG:-${PROJECT_DIR}/examples/moworld/fsdp2_config.yaml}

ASCEND_ENV=${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
[[ -f "${ASCEND_ENV}" ]] || ASCEND_ENV=/usr/local/Ascend/cann/set_env.sh
[[ -f "${ASCEND_ENV}" ]] || { echo "Ascend environment script not found" >&2; exit 1; }
source "${ASCEND_ENV}"

NPUS_PER_NODE=${NPUS_PER_NODE:-16}
if [[ -n "${VC_WORKER_HOSTS:-}" ]]; then
    IFS=',' read -r -a NODE_IPS <<< "${VC_WORKER_HOSTS}"
    MASTER_ADDR=${MASTER_ADDR:-${NODE_IPS[0]}}
    NNODES=${NNODES:-${#NODE_IPS[@]}}
    NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX:-${RANK:-0}}}
else
    MASTER_ADDR=${MASTER_ADDR:-localhost}
    NNODES=${NNODES:-1}
    NODE_RANK=${NODE_RANK:-0}
fi
MASTER_PORT=${MASTER_PORT:-6006}
WORLD_SIZE=$((NPUS_PER_NODE * NNODES))

TP=${TP:-1}; PP=${PP:-1}; VP=${VP:-1}; CP=${CP:-2}; MBS=${MBS:-1}
GRAD_ACC_STEP=${GRAD_ACC_STEP:-1}
(( WORLD_SIZE % (TP * PP * CP) == 0 )) || { echo "WORLD_SIZE must be divisible by TP*PP*CP" >&2; exit 2; }
DP=$((WORLD_SIZE / TP / PP / CP))
GBS=${GBS:-$((MBS * GRAD_ACC_STEP * DP))}

for path in "${MINDSPEED_PATH}" "${MM_DATA}" "${MM_MODEL}" "${MM_TOOL}" "${LOAD_PATH}" "${FSDP_CONFIG}"; do
    [[ -e "${path}" ]] || { echo "required path does not exist: ${path}" >&2; exit 1; }
done

export PYTHONPATH="${MINDSPEED_PATH}:${PROJECT_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-16}
export ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT:-0}
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export COMBINED_ENABLE=${COMBINED_ENABLE:-1}
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1200}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}

cd "${PROJECT_DIR}"
mkdir -p logs "${SAVE_PATH}"
torchrun \
    --nproc_per_node "${NPUS_PER_NODE}" --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" \
    pretrain_sora.py \
    --tensor-model-parallel-size "${TP}" --pipeline-model-parallel-size "${PP}" \
    --virtual-pipeline-model-parallel-size "${VP}" --context-parallel-size "${CP}" \
    --context-parallel-algo ulysses_cp_algo --micro-batch-size "${MBS}" --global-batch-size "${GBS}" \
    --num-workers "${NUM_WORKERS:-16}" --lr "${LR:-1e-5}" --min-lr "${MIN_LR:-1e-5}" \
    --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --lr-decay-style constant \
    --weight-decay 1e-2 --lr-warmup-iters 0 --train-iters "${TRAIN_ITERS:-10}" \
    --save-interval "${SAVE_INTERVAL:-600}" --log-interval 1 --eval-interval 10000 --eval-iters 10 \
    --no-gradient-accumulation-fusion --no-load-optim --no-load-rng --no-save-optim --no-save-rng \
    --downcast-to-bf16 --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES:-120}" \
    --use-torch-fsdp2 --use-fused-rmsnorm --untie-embeddings-and-output-weights \
    --fsdp2-config-path "${FSDP_CONFIG}" --optimizer-selection fused_torch_adamw \
    --use-cpu-initialization --attention-mask-type general \
    --mm-data "${MM_DATA}" --mm-model "${MM_MODEL}" --mm-tool "${MM_TOOL}" \
    --load "${LOAD_PATH}" --save "${SAVE_PATH}" --ckpt-format torch_dcp \
    --distributed-backend nccl \
    2>&1 | tee "logs/moworld_train_high_$(date +%Y%m%d_%H%M%S).log"
