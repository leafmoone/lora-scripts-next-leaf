#!/usr/bin/env bash
# Start Gemma-4-E4B vLLM server for Anima Train mode.
# Run from project root (lora-scripts-next).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${ROOT}/models/gemma-4-E3B-it"
PYTHON="${ROOT}/.venv/bin/python"
VLLM="${ROOT}/.venv/bin/vllm"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3 || command -v python)"
  VLLM="$(command -v vllm)"
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Gemma weights not found at ${MODEL_DIR}" >&2
  echo "Download first:" >&2
  echo "  modelscope download spawner/spawner-gemma-4-E4B-it --local_dir ${MODEL_DIR}" >&2
  exit 1
fi

PY_SITE="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="${PY_SITE}/nvidia/cu13/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export TORCHDYNAMO_DISABLE=1

CUSTOM_OPS_ARGS=()
if [[ "${VLLM_ENABLE_CUSTOM_OPS:-0}" != "1" ]]; then
  CUSTOM_OPS_ARGS=(-cc.custom_ops '["none"]')
fi

exec "${VLLM}" serve "${MODEL_DIR}" \
  --served-model-name spawner-gemma-4-e4b-it \
  --port 9002 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.42 \
  --max-num-seqs 8 \
  --trust-remote-code \
  --dtype bfloat16 \
  --enforce-eager \
  --generation-config vllm \
  "${CUSTOM_OPS_ARGS[@]}" \
  --limit-mm-per-prompt '{"image": 1}'
