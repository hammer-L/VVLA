#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: run_classifier_policy_server.sh off|rerank|gradient|gradient_rerank}"
case "${MODE}" in
  off|rerank|gradient|gradient_rerank) ;;
  *) echo "Unknown classifier mode: ${MODE}" >&2; exit 2 ;;
esac

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../../" && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
BASE_CKPT="${BASE_CKPT:?Set BASE_CKPT to the complete QwenGR00T checkpoint}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-}"
PORT="${PORT:-10093}"
GPU_ID="${GPU_ID:-0}"
NUM_CANDIDATES="${NUM_CANDIDATES:-1}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.0}"

CMD=(
  "${STARVLA_PYTHON}" deployment/model_server/server_policy.py
  --base_ckpt_path "${BASE_CKPT}"
  --classifier_mode "${MODE}"
  --num_candidates "${NUM_CANDIDATES}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --port "${PORT}"
)
if [[ "${MODE}" != "off" ]]; then
  : "${CLASSIFIER_CKPT:?Set CLASSIFIER_CKPT for classifier-assisted modes}"
fi
if [[ -n "${CLASSIFIER_CKPT}" ]]; then
  CMD+=(--classifier_ckpt_path "${CLASSIFIER_CKPT}")
fi
if [[ "${USE_BF16:-1}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
exec env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
