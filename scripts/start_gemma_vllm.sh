#!/usr/bin/env bash
# Start Gemma-4-E4B vLLM server for Anima Train mode.
# Run from project root (lora-scripts-next).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${ROOT}/models/gemma-4-E3B-it"
SIDECAR_ENV="${ANIMA_GEMMA_VLLM_ENV:-${ROOT}/.venv}"
CUDA_HOME="${ANIMA_VLLM_CUDA_HOME:-}"
PYTHON="${SIDECAR_ENV}/bin/python"
VLLM="${ANIMA_GEMMA_VLLM_BIN:-${SIDECAR_ENV}/bin/vllm}"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
fi
if [[ ! -x "${VLLM}" ]]; then
  VLLM="${ROOT}/.venv/bin/vllm"
fi
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3 || command -v python)"
fi
if [[ ! -x "${VLLM}" ]]; then
  VLLM="$(command -v vllm)"
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Gemma weights not found at ${MODEL_DIR}" >&2
  echo "Download first:" >&2
  echo "  modelscope download spawner/spawner-gemma-4-E4B-it --local_dir ${MODEL_DIR}" >&2
  exit 1
fi

PY_SITE="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PATH_ENTRIES=("$(dirname "${VLLM}")")
LIB_ENTRIES=("${PY_SITE}/nvidia/cuda_runtime/lib")
if [[ -n "${CUDA_HOME}" && -d "${CUDA_HOME}" ]]; then
  export CUDA_HOME
  export CUDA_PATH="${CUDA_HOME}"
  PATH_ENTRIES=("${CUDA_HOME}/bin" "${PATH_ENTRIES[@]}")
  LIB_ENTRIES=("${CUDA_HOME}/lib64" "${CUDA_HOME}/lib" "${CUDA_HOME}/targets/x86_64-linux/lib" "${LIB_ENTRIES[@]}")
fi
export PATH="$(IFS=:; echo "${PATH_ENTRIES[*]}"):${PATH}"
export LD_LIBRARY_PATH="$(IFS=:; echo "${LIB_ENTRIES[*]}"):${LD_LIBRARY_PATH:-}"
export TORCHDYNAMO_DISABLE=1

CUSTOM_OPS_ARGS=()
if [[ "${VLLM_DISABLE_CUSTOM_OPS:-0}" == "1" ]]; then
  CUSTOM_OPS_ARGS=(-cc.custom_ops '["none"]')
fi

exec "${VLLM}" serve "${MODEL_DIR}" \
  --served-model-name spawner-gemma-4-e4b-it \
  --host 127.0.0.1 \
  --port 9003 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --enforce-eager \
  --generation-config vllm \
  "${CUSTOM_OPS_ARGS[@]}" \
  --limit-mm-per-prompt '{"image": 1}'
