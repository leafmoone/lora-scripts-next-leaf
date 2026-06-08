#!/bin/bash
# =============================================================================
# Smart Tag — 高精度多标签器共识 + ToriiGate VLM 自然语言描述
#
# 用法：
#   ./run_smart_tag.sh [输入目录]
#
# 默认输入：/root/lanyun-tmp/chatgpt2api/data/images/2026/05/28
# =============================================================================

set -euo pipefail

# ─── 配置 ──────────────────────────────────────────────────────────────
INPUT="${1:-/root/lanyun-tmp/chatgpt2api/data/images/2026/05/28}"
OUTPUT="./output/smart-tag-$(date +%Y%m%d-%H%M%S)"
TRIGGER="kisaragi_yuuna"
PURPOSE="character"

# 多标签器共识配置
TAGGER1="wd-swinv2-tagger-v3"
TAGGER2="wd-eva02-large-tagger-v3"
CONSENSUS_MIN=2

# ─── 环境准备 ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# HuggingFace 缓存目录
export HF_HOME="/root/lanyun-tmp/cache"
mkdir -p "$HF_HOME"

# 模型存储目录（避免撑爆 overlay）
export STANDALONE_TAGGER_DATA_DIR="/root/lanyun-tmp/cache/standalone-tagger-data"
mkdir -p "$STANDALONE_TAGGER_DATA_DIR"

# 激活网络加速（如果可用）
if [ -f /etc/network_turbo ]; then
    echo "[*] 启用学术加速..."
    source /etc/network_turbo
fi

# 创建虚拟环境（如果不存在）
if [ ! -d ".venv" ]; then
    echo "[*] 创建 Python 虚拟环境..."
    uv venv
fi

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖（如果缺失）
python -c "import torch; import transformers" 2>/dev/null || {
    echo "[*] 安装 PyTorch + Transformers（ToriiGate VLM 需要）..."
    source /etc/network_turbo 2>/dev/null || true
    uv pip install torch transformers accelerate
}

# ─── 运行 ──────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Smart Tag — 高精度多标签器共识 + VLM"
echo "============================================"
echo " 输入目录:    $INPUT"
echo " 输出目录:    $OUTPUT"
echo " 标签器:      $TAGGER1 + $TAGGER2"
echo " 共识权重:    >= $CONSENSUS_MIN"
echo " 触发词:      $TRIGGER"
echo " VLM 方向:    $PURPOSE"
echo " VLM 模型:    ToriiGate 0.5"
echo "============================================"
echo ""

python main.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --smart \
    --vlm \
    --purpose "$PURPOSE" \
    --trigger "$TRIGGER" \
    --taggers "$TAGGER1" "$TAGGER2" \
    --consensus "$CONSENSUS_MIN" \
    --save-captions

echo ""
echo "============================================"
echo " 完成！结果保存在: $OUTPUT/"
echo "  - results.json   完整 JSON 结果"
echo "  - captions/      每张图片的 .txt 标题文件"
echo "============================================"
