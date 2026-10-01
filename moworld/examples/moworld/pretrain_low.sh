#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MM_MODEL=${MM_MODEL:-${SCRIPT_DIR}/pretrain_model_feature_low.json}
export LOAD_PATH=${LOAD_PATH:-${SCRIPT_DIR}/../../../checkpoints/base/dcp/low_noise_model}
export SAVE_PATH=${SAVE_PATH:-${SCRIPT_DIR}/../../../checkpoints/train/low_noise_model}
exec bash "${SCRIPT_DIR}/pretrain_high.sh"
