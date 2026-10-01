#!/bin/bash
# MoWorld training script (high noise expert)
# Optimized for 3-Node 16-Card Distributed Training

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${PROJECT_DIR}" || exit 1
ASCEND_ENV=${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
source "${ASCEND_ENV}"
MINDSPEED_PATH=${MINDSPEED_PATH:-${PROJECT_DIR}/../MindSpeed}
export PYTHONPATH="${MINDSPEED_PATH}:${PROJECT_DIR}:${PYTHONPATH:-}"

# ======== 环境变量优化 ========



export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=300
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore"

# =======================================Distributed Training=======================================

if [ -n "$VC_WORKER_HOSTS" ]; then
    IFS=',' read -r -a NODE_IPS <<< "$VC_WORKER_HOSTS"
    MASTER_ADDR=${NODE_IPS[0]}
    echo "MASTER_ADDR: $MASTER_ADDR"
    NNODES=${#NODE_IPS[@]}
    NODE_RANK=${VC_TASK_INDEX:-0} 
else
    # 本地调试环境
    MASTER_ADDR=localhost
    NNODES=1
    NODE_RANK=0
fi

MASTER_PORT=${MASTER_PORT:-19980}

NPUS_PER_NODE=16

WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# 打印关键信息
echo "=========================================="
echo "Distributed Training Config:"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_RANK:   $NODE_RANK"
echo "NNODES:      $NNODES"
echo "WORLD_SIZE:  $WORLD_SIZE"
echo "=========================================="
# ==================================================================================================

# ======== 并行度优化配置 (TP=4, MBS=4) ========
TP=1
PP=1
VP=1
CP=2
MBS=4
GRAD_ACC_STEP=1 

# 自动计算 DP 和 GBS
# DP = 48 / (4*1*2) = 6
# GBS = 4 * 6 = 24
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

MM_DATA=${MM_DATA:-${PROJECT_DIR}/examples/moworld/data_feature.json}
MM_MODEL=${MM_MODEL:-${PROJECT_DIR}/examples/moworld/pretrain_model_feature_high.json}
MM_TOOL=${MM_TOOL:-${PROJECT_DIR}/mindspeed_mm/tools/tools.json}
LOAD_PATH=${LOAD_PATH:-${PROJECT_DIR}/../checkpoints/base/dcp/high_noise_model}
SAVE_PATH=${SAVE_PATH:-${PROJECT_DIR}/../checkpoints/train/high_noise_model}
FSDP_CONFIG=${FSDP_CONFIG:-${PROJECT_DIR}/examples/moworld/fsdp2_config.yaml}

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --virtual-pipeline-model-parallel-size ${VP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --num-workers 16 \
    --lr 1e-5 \
    --min-lr 1e-5 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr-decay-style constant \
    --weight-decay 1e-2 \
    --lr-warmup-init 0 \
    --lr-warmup-iters 0 \
    --train-iters 10 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --downcast-to-bf16 \
    --distributed-timeout-minutes 600 \
    --use-torch-fsdp2 \
    --use-fused-rmsnorm \
    --untie-embeddings-and-output-weights \
    --fsdp2-config-path ${FSDP_CONFIG} \
    --optimizer-selection fused_torch_adamw \
    --use-cpu-initialization \
    --attention-mask-type general \
"

MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 10000 \
    --eval-iters 200 \
    --load $LOAD_PATH \
    --save $SAVE_PATH \
    --ckpt-format torch_dcp \
"

# 关闭 Profiler 以避免性能影响和潜在的卡顿
# PROFILER_ARGS="..."

logfile=opengenie3_high_$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

echo "[INFO] Starting training with torchrun..."
echo "[INFO] Command: torchrun $DISTRIBUTED_ARGS pretrain_sora.py ..."

accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_machines $NNODES \
    --machine_rank $NODE_RANK \
    --num_processes $WORLD_SIZE \
    --multi_npu \
    --main_process_port $MASTER_PORT \
    pretrain_sora.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    2>&1 | tee logs/train_${logfile}.log
