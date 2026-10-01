#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MINDSPEED_PATH=${MINDSPEED_PATH:-${PROJECT_DIR}/../MindSpeed}
SOURCE_ROOT=${SOURCE_ROOT:-${PROJECT_DIR}/../models/base}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_DIR}/../checkpoints/base}
EXPERTS=${EXPERTS:-high_noise_model,low_noise_model}
TQDM_MININTERVAL=${TQDM_MININTERVAL:-1}

export PYTHONPATH="${MINDSPEED_PATH}:${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL
cd "${PROJECT_DIR}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_conversion() {
    local stage=$1
    shift
    local started_at=$SECONDS

    log "START ${stage}"
    "$@"
    log "DONE  ${stage} (elapsed $((SECONDS - started_at))s)"
}

if command -v mm-convert >/dev/null 2>&1; then
    CONVERTER=(mm-convert)
else
    CONVERTER=(python -m checkpoint.convert_cli)
fi

IFS=',' read -r -a expert_list <<< "${EXPERTS}"
total_experts=${#expert_list[@]}
log "MoWorld weight conversion"
log "source=${SOURCE_ROOT} output=${OUTPUT_ROOT} experts=${EXPERTS}"
log "converter=${CONVERTER[*]}"

for expert_index in "${!expert_list[@]}"; do
    expert=${expert_list[$expert_index]}
    # Permit a space after the comma in EXPERTS=high_noise_model, low_noise_model.
    expert=${expert##+([[:space:]])}
    expert=${expert%%+([[:space:]])}
    case "${expert}" in
        high_noise_model|low_noise_model) ;;
        *) log "ERROR unsupported expert: ${expert}" >&2; exit 2 ;;
    esac

    source_path="${SOURCE_ROOT}/${expert}"
    mm_path="${OUTPUT_ROOT}/mm/${expert}"
    dcp_path="${OUTPUT_ROOT}/dcp/${expert}"
    log "[expert $((expert_index + 1))/${total_experts}] checking ${expert}"
    [[ -f "${source_path}/diffusion_pytorch_model.safetensors" ]] || {
        log "ERROR missing safetensors: ${source_path}/diffusion_pytorch_model.safetensors" >&2
        exit 1
    }
    [[ ! -e "${mm_path}" && ! -e "${dcp_path}" ]] || {
        log "ERROR target already exists; move it away before retrying: ${mm_path} or ${dcp_path}" >&2
        exit 1
    }

    mkdir -p "$(dirname "${mm_path}")" "$(dirname "${dcp_path}")"
    run_conversion "${expert} [1/2] DiffSynth safetensors -> MindSpeed-MM" \
        "${CONVERTER[@]}" WanConverter lora_hf_to_mm \
        --cfg.source_path "${source_path}" \
        --cfg.target_path "${mm_path}"

    run_conversion "${expert} [2/2] MindSpeed-MM -> torch DCP" \
        "${CONVERTER[@]}" WanConverter mm_to_dcp \
        --cfg.source_path "${mm_path}" \
        --cfg.target_path "${dcp_path}"
    log "[expert $((expert_index + 1))/${total_experts}] ${expert} complete"
done

log "DCP checkpoints written under ${OUTPUT_ROOT}/dcp"
