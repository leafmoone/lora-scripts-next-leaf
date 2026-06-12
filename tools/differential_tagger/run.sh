#!/bin/bash
# =============================================================================
# Standalone AI Image Tagger — 一键打标脚本
#
# 用法:
#   ./run.sh <输入目录> [输出目录] [选项]
#
# 示例:
#   # 简单打标（仅 WD14 标签）
#   ./run.sh ./images --simple
#
#   # 高精度 Smart Tag（多标签器共识 + ToriiGate VLM）
#   ./run.sh ./images --smart --vlm --trigger "my_character"
#
#   # 指定触发词 + 训练目的
#   ./run.sh ./images --smart --vlm --purpose character --trigger "kisaragi_yuuna"
#
#   # 仅 CPU
#   ./run.sh ./images --simple --cpu
#
#   # 自定义标签器
#   ./run.sh ./images --smart --vlm --taggers wd-swinv2-tagger-v3 wd-eva02-large-tagger-v3
#
#   # 递归扫描子目录
#   ./run.sh ./images --simple -r
# =============================================================================

set -euo pipefail

# ─── 默认值 ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="${TAGGER_INPUT:-}"
OUTPUT="${TAGGER_OUTPUT:-}"
MODE="smart"
USE_VLM=true
USE_GPU=true
PURPOSE="character"
TRIGGER=""
TAGGER_MODEL="wd-eva02-large-tagger-v3"
TAGGERS=()
CONSENSUS=2
RECURSIVE=false
SAVE_CAPTIONS=true
MAX_TAGS=0
BLACKLIST=()
VERBOSE=false
THRESHOLD=""
CHAR_THRESHOLD=""

# ─── 帮助 ────────────────────────────────────────────────────────────────
usage() {
    cat << 'EOF'
AI 图片打标工具 — 支持 WD14 booru 标签 + ToriiGate VLM 自然语言

用法:
  ./run.sh <输入目录> [输出目录] [选项]

选项:
  --simple              简单模式：仅 WD14 标签（默认：smart）
  --smart               Smart Tag 多阶段流水线（默认）

  --vlm                 启用 ToriiGate VLM 自然语言（smart 模式默认开启）
  --no-vlm              禁用 VLM（仅 booru 标签）

  --purpose <目的>      VLM 描述方向: style / character / general / concept
                        （默认: character）
  --trigger <词>        触发词，注入标题最前面（须为单个词）

  --model <模型名>      单个 WD14 模型名（默认: wd-eva02-large-tagger-v3）
  --taggers <模型1> <模型2> ...
                        多标签器共识模式（指定 2+ 个模型）
  --consensus <N>       最小共识权重（默认: 2）

  --threshold <值>      通用标签置信度阈值（默认: 0.35）
  --char-threshold <值> 角色标签阈值（默认: 0.85）

  --cpu                 仅使用 CPU（默认：使用 GPU）
  -r, --recursive       递归扫描输入子目录

  --max-tags <N>        每张图最大标签数（0=不限）
  --blacklist <tag> ... 过滤标签列表
  --save-captions       额外保存 .txt 标题文件

  -v, --verbose         详细日志
  -h, --help            显示帮助

示例:
  ./run.sh ./my_images                        # 默认 Smart Tag + VLM
  ./run.sh ./my_images --simple               # 仅 WD14 标签
  ./run.sh ./my_images --smart --vlm --trigger "my_char"  # 含触发词
  ./run.sh ./my_images --smart --vlm --purpose style      # 风格 LoRA
  ./run.sh ./my_images --simple --cpu         # CPU 模式
  ./run.sh ./my_images --smart --no-vlm --taggers A B     # 多标签器无 VLM
EOF
    exit 0
}

# ─── 参数解析 ────────────────────────────────────────────────────────────
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --simple)        MODE="simple" ;;
            --smart)         MODE="smart" ;;
            --vlm)           USE_VLM=true ;;
            --no-vlm)        USE_VLM=false ;;
            --cpu)           USE_GPU=false ;;
            -r|--recursive)  RECURSIVE=true ;;
            --save-captions) SAVE_CAPTIONS=true ;;
            -v|--verbose)    VERBOSE=true ;;
            -h|--help)       usage ;;

            --purpose)
                shift; PURPOSE="$1" ;;
            --trigger)
                shift; TRIGGER="$1" ;;
            --model)
                shift; TAGGER_MODEL="$1" ;;
            --taggers)
                shift
                TAGGERS=()
                while [[ $# -gt 0 && "$1" != -* ]]; do
                    TAGGERS+=("$1"); shift
                done
                continue ;;
            --consensus)
                shift; CONSENSUS="$1" ;;
            --threshold)
                shift; THRESHOLD="$1" ;;
            --char-threshold)
                shift; CHAR_THRESHOLD="$1" ;;
            --max-tags)
                shift; MAX_TAGS="$1" ;;
            --blacklist)
                shift
                while [[ $# -gt 0 && "$1" != -* ]]; do
                    BLACKLIST+=("$1"); shift
                done
                continue ;;
            -*)
                echo "ERROR: 未知选项: $1"
                usage ;;
            *)
                if [[ -z "$INPUT" ]]; then
                    INPUT="$1"
                elif [[ -z "$OUTPUT" ]]; then
                    OUTPUT="$1"
                else
                    echo "ERROR: 多余参数: $1"
                    usage
                fi ;;
        esac
        shift
    done

    # 验证
    if [[ -z "$INPUT" ]]; then
        echo "ERROR: 请指定输入目录"
        usage
    fi
    if [[ ! -d "$INPUT" ]]; then
        echo "ERROR: 输入目录不存在: $INPUT"
        exit 1
    fi
    if [[ -z "$OUTPUT" ]]; then
        OUTPUT="$SCRIPT_DIR/output/tag-$(date +%Y%m%d-%H%M%S)"
    fi
}

# ─── 主流程 ──────────────────────────────────────────────────────────────
main() {
    parse_args "$@"

    cd "$SCRIPT_DIR"
    OUTPUT="$(realpath -m "$OUTPUT")"

    # ── 网络加速 ──────────────────────────────────────────────────────
    if [[ -f /etc/network_turbo ]]; then
        source /etc/network_turbo
        echo "[*] 已启用学术加速"
    fi

    # ── cuDNN 路径（ONNX Runtime GPU 需要，优先用系统 Python 检测）───
    if [[ -z "${LD_LIBRARY_PATH:-}" ]] || ! echo "$LD_LIBRARY_PATH" | grep -q cudnn; then
        CUDNN_LIB_DIR="$(python3 -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])' 2>/dev/null)/lib"
        if [[ -d "$CUDNN_LIB_DIR" ]]; then
            export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:${LD_LIBRARY_PATH:-}"
        fi
    fi

    # ── 环境 ──────────────────────────────────────────────────────────
    if [[ ! -d ".venv" ]]; then
        echo "[*] 创建虚拟环境..."
        uv venv
    fi
    source .venv/bin/activate

    # 检测并安装缺失依赖
    python -c "import torch; import transformers; import onnxruntime" 2>/dev/null || {
        echo "[*] 安装依赖..."
        source /etc/network_turbo 2>/dev/null || true
        uv pip install Pillow numpy onnxruntime-gpu "huggingface-hub>=0.20" transformers accelerate
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    }

    # 确保 PyTorch 是 cu128 版本（避免自动装 cu130 导致驱动不兼容）
    if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
        echo "[*] PyTorch CUDA 不可用，重装 cu128 版本..."
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
    fi

    # ── 环境变量 ──────────────────────────────────────────────────────

    export HF_HOME="${HF_HOME:-/root/lanyun-tmp/cache}"
    export STANDALONE_TAGGER_DATA_DIR="${STANDALONE_TAGGER_DATA_DIR:-/root/lanyun-tmp/cache/standalone-tagger-data}"
    mkdir -p "$HF_HOME" "$STANDALONE_TAGGER_DATA_DIR"

    # ── 构建命令行参数 ────────────────────────────────────────────────
    ARGS=(
        --input "$INPUT"
        --output "$OUTPUT"
        "--$MODE"
    )

    [[ "$RECURSIVE" == true ]] && ARGS+=(-r)
    [[ "$SAVE_CAPTIONS" == true ]] && ARGS+=(--save-captions)
    [[ "$USE_GPU" == false ]] && ARGS+=(--cpu)
    [[ "$VERBOSE" == true ]] && ARGS+=(-v)
    [[ -n "$TRIGGER" ]] && ARGS+=(--trigger "$TRIGGER")

    # 打标器
    if [[ ${#TAGGERS[@]} -ge 2 ]]; then
        ARGS+=(--taggers "${TAGGERS[@]}" --consensus "$CONSENSUS")
    else
        ARGS+=(--model "$TAGGER_MODEL")
    fi

    # 阈值
    [[ -n "$THRESHOLD" ]] && ARGS+=(--threshold "$THRESHOLD")
    [[ -n "$CHAR_THRESHOLD" ]] && ARGS+=(--character-threshold "$CHAR_THRESHOLD")
    [[ "$MAX_TAGS" -gt 0 ]] && ARGS+=(--max-tags "$MAX_TAGS")

    # 黑名单
    if [[ ${#BLACKLIST[@]} -gt 0 ]]; then
        ARGS+=(--blacklist "${BLACKLIST[@]}")
    fi

    # Smart Tag 选项
    if [[ "$MODE" == "smart" ]]; then
        ARGS+=(--purpose "$PURPOSE")
        if [[ "$USE_VLM" == true ]]; then
            ARGS+=(--vlm)
        else
            ARGS+=(--no-vlm)
        fi
    fi

    # ── 打印配置 ──────────────────────────────────────────────────────
    echo ""
    echo "============================================"
    echo " Standalone AI Image Tagger"
    echo "============================================"
    echo " 输入:        $INPUT"
    echo " 输出:        $OUTPUT"
    echo " 模式:        $MODE"
    echo " GPU:         $([[ "$USE_GPU" == true ]] && echo '是' || echo '否')"
    echo " 递归:        $([[ "$RECURSIVE" == true ]] && echo '是' || echo '否')"
    if [[ ${#TAGGERS[@]} -ge 2 ]]; then
        echo " 标签器:      共识模式 (${TAGGERS[*]}, 最小投票: $CONSENSUS)"
    else
        echo " 标签器:      $TAGGER_MODEL"
    fi
    if [[ "$MODE" == "smart" ]]; then
        echo " VLM:         $([[ "$USE_VLM" == true ]] && echo 'ToriiGate' || echo '禁用')"
        echo " 训练目的:    $PURPOSE"
        [[ -n "$TRIGGER" ]] && echo " 触发词:      $TRIGGER"
    fi
    echo "============================================"
    echo ""

    # ── 运行 ──────────────────────────────────────────────────────────
    START=$(date +%s)
    python main.py "${ARGS[@]}"
    ELAPSED=$(( $(date +%s) - START ))

    # ── 结果 ──────────────────────────────────────────────────────────
    echo ""
    echo "============================================"
    printf " 完成! 耗时: %d 分 %d 秒\n" $((ELAPSED / 60)) $((ELAPSED % 60))
    echo "============================================"
    echo " 结果:  $OUTPUT/results.json"
    [[ "$SAVE_CAPTIONS" == true ]] && echo " 标题:  $OUTPUT/captions/"
    echo "============================================"
}

main "$@"
