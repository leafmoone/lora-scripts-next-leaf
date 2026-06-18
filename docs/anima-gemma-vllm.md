# Anima Train Gemma vLLM Environment

This document records the verified single-environment Gemma-4-E4B vLLM setup
for Tag-Edit-Leaf Anima Train captioning.

## Current Layout

The project now uses the project `.venv` for UI, training, WD14, and Gemma vLLM.
New installs should be created with `uv`:

```bash
cd /root/autodl-tmp/lora-scripts-next-leaf
uv sync
```

Verified core stack:

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| CUDA runtime | cu128 |
| torch | `2.10.0+cu128` |
| torchvision | `0.25.0+cu128` |
| torchaudio | `2.10.0+cu128` |
| vLLM | `0.19.1` |
| transformers | `>=4.56,<5` |
| numpy | `>=2,<3` |
| gradio | `>=5,<6` |

`pyproject.toml` pins this stack and maps torch packages to the official
PyTorch cu128 index. `requirements-vllm-cu128.txt` is kept only as a legacy pip
constraint file; prefer `uv sync`.

## Current Config

`config/anima_caption_models.json` points Gemma to the local OpenAI-compatible
server:

```json
{
  "default_api_url": "http://127.0.0.1:9003/v1/chat/completions",
  "default_served_name": "spawner-gemma-4-e4b-it",
  "port": 9003,
  "gemma_vlm_backend": "vllm",
  "vllm_serve": {
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9,
    "max_num_seqs": 4,
    "enable_custom_ops": true
  }
}
```

`mikazuki/utils/vllm_manager.py` resolves `vllm` from the current environment.
`scripts/start_gemma_vllm.sh` uses `${PROJECT_ROOT}/.venv/bin/vllm` by default.

## Start vLLM

Download Gemma weights if missing:

```bash
uv run modelscope download spawner/spawner-gemma-4-E4B-it \
  --local_dir ./models/gemma-4-E3B-it
```

Manual start:

```bash
bash scripts/start_gemma_vllm.sh
```

Health check:

```bash
curl -fsS http://127.0.0.1:9003/v1/models
```

Text generation probe:

```bash
curl -sS http://127.0.0.1:9003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "spawner-gemma-4-e4b-it",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "max_tokens": 16,
    "temperature": 0
  }'
```

The response must contain readable text such as `OK`, not empty content or pad
tokens.

## Caption Pipeline

For each image, Anima Train runs:

```text
WD14 tags -> Gemma refine_wd14_tags -> Gemma generate_natural_caption -> .txt
```

The output format is:

```text
tag one, tag two, tag three,

Natural-language training caption.
```

The first line keeps the original WD14 tag set for training stability. The VLM
refine step is used as context for the natural-language caption, not as a
replacement for the WD14 training tag line.

Current measured single-image test with an already-running vLLM server:

| Image | Mode | Time | Output |
|-------|------|------|--------|
| `/root/autodl-tmp/DiffSynth-Studio/data/qwen_image/test/comfy_0001.png` | GPU WD14 + Gemma vLLM | ~46 s | `/tmp/tagger-leaf-comfy0001-afterfix` |

## Environment Policy

Recommended policy:

| Component | Policy |
|-----------|--------|
| Project `.venv` | Build with `uv sync`; contains UI/training/WD14/vLLM. |
| Gemma vLLM | Run from the project `.venv`, port 9003. |
| Dependency constraints | Keep torch `2.10.0+cu128` and vLLM `0.19.1`. |
| Config | Use `gemma_vlm_backend: "vllm"` and served name `spawner-gemma-4-e4b-it`. |

## Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `Gemma vLLM sidecar executable not found` | Old config or old UI process still running | Restart the UI after updating; current config does not use the sidecar path. |
| Empty Gemma output or pad-only output | Broken vLLM backend or custom ops mismatch | Run the text probe above; keep vLLM `0.19.1` with torch `2.10.0+cu128`. |
| WD14 OOM | vLLM already reserved most GPU memory | Stop vLLM before WD14, or run WD14 on CPU. |
| Port busy | Previous vLLM still running on 9003 | Stop it or reuse the existing healthy server. |

Environment variables accepted by the manual start script:

| Variable | Meaning |
|----------|---------|
| `ANIMA_GEMMA_VLLM_ENV` | Override env path; defaults to project `.venv`. |
| `ANIMA_GEMMA_VLLM_BIN` | Override `vllm` executable. |
| `ANIMA_VLLM_CUDA_HOME` | Optional CUDA toolkit path. |
| `VLLM_DISABLE_CUSTOM_OPS=1` | Disable vLLM custom ops for debugging only. |
