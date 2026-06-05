#!/usr/bin/env python3
"""
Kohya LoRA → Base Model 融合工具

将 Kohya 格式的 LoRA safetensors 权重融入底模 safetensors。
用于 Differential LoRA 训练中 Step 1 结束后合并 LoRA1 到底模，
使 Step 2 只学习差异部分。

Kohya LoRA 格式: {prefix}.lora_A.weight (rank × in_dim) + {prefix}.lora_B.weight (out_dim × rank)
融合公式: W_new = W + scale * (lora_B @ lora_A)

用法:
  python merge_lora_to_base.py --base base.safetensors --lora lora.safetensors --output merged.safetensors
  python merge_lora_to_base.py --base base.safetensors --lora lora.safetensors --output merged.safetensors --scale 0.8
  python merge_lora_to_base.py --base base.safetensors --lora lora.safetensors --output merged.safetensors --dtype fp16
"""

import argparse
import os
import sys
from typing import Optional

import torch
from safetensors.torch import load_file, save_file


def find_lora_pairs(state_dict: dict) -> list[tuple[str, str, str]]:
    """
    在 state_dict 中匹配 lora_A / lora_B 对。

    返回: [(base_key_prefix, lora_A_key, lora_B_key), ...]
    其中 base_key_prefix 是去掉 .lora_A.weight 后的前缀，
    对应底模中的 {base_key_prefix}.weight。
    """
    pairs = []
    seen = set()
    for key in sorted(state_dict):
        if not key.endswith(".lora_A.weight"):
            continue
        base_prefix = key[: -len(".lora_A.weight")]
        if base_prefix in seen:
            continue
        b_key = key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key in state_dict:
            seen.add(base_prefix)
            pairs.append((base_prefix, key, b_key))
    return pairs


def find_lora_alpha(state_dict: dict, lora_pairs: list) -> Optional[float]:
    """尝试从 state_dict 中找到 lora alpha 值。"""
    for key in state_dict:
        if "alpha" in key.lower():
            val = state_dict[key]
            if isinstance(val, (int, float)) or (hasattr(val, "numel") and val.numel() == 1):
                alpha = float(val.item() if hasattr(val, "item") else val)
                return alpha
    # 常见位置: 根级别的 alpha tensor
    for key in ["alpha", "lora_alpha", "network_alpha"]:
        if key in state_dict:
            val = state_dict[key]
            if hasattr(val, "item"):
                return float(val.item())
    return None


def merge_lora_to_base(
    base_state: dict,
    lora_state: dict,
    scale: float = 1.0,
    dtype: Optional[torch.dtype] = None,
    device: str = "cuda",
    verbose: bool = False,
) -> dict:
    """
    将 LoRA 权重融入底模。

    Args:
        base_state: 底模的 state_dict
        lora_state: LoRA 的 state_dict
        scale: 融合权重倍数 (默认 1.0)
        dtype: 输出精度 (默认保持原精度)
        device: 计算设备
        verbose: 是否输出详细日志

    Returns:
        融合后的 state_dict
    """
    lora_pairs = find_lora_pairs(lora_state)

    if not lora_pairs:
        print("警告: 未找到 lora_A / lora_B 配对，可能不是标准 Kohya LoRA 格式", file=sys.stderr)

    # 检测 LoRA alpha，自动计算 scale
    lora_alpha = find_lora_alpha(lora_state, lora_pairs)
    if lora_alpha is not None and scale == 1.0:
        # 如果有 dim/key 信息，用 alpha/dim 标准公式
        import re
        dim = None
        for key in lora_state:
            m = re.search(r"lora_A\.weight$", key)
            if m:
                a_key = key
                dim = lora_state[a_key].shape[0]  # rank
                break
        if dim and dim > 0:
            auto_scale = lora_alpha / dim
            if verbose:
                print(f"检测到 alpha={lora_alpha}, dim={dim}, 自动 scale={auto_scale:.6f}")
            scale = auto_scale

    if verbose:
        print(f"融合 scale = {scale}")
        print(f"找到 {len(lora_pairs)} 对 LoRA 矩阵")
        print(f"底模 key 总数: {len(base_state)}")
        print(f"LoRA key 总数: {len(lora_state)}")

    # 检查哪些 key 不是标准 LoRA 参数（如 alpha, dora_scale 等）
    lora_keys = set()
    for _, a_key, b_key in lora_pairs:
        lora_keys.add(a_key)
        lora_keys.add(b_key)
    non_lora_keys = set(lora_state.keys()) - lora_keys
    if non_lora_keys and verbose:
        print(f"非 LoRA key ({len(non_lora_keys)} 个): {sorted(non_lora_keys)[:10]}")

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        if verbose:
            print("CUDA 不可用，回退到 CPU")

    target_dtype = dtype
    merged = {}

    for base_key, base_tensor in base_state.items():
        merged[base_key] = base_tensor.clone()

    merged_count = 0
    skipped_count = 0

    for base_prefix, a_key, b_key in lora_pairs:
        target_key = f"{base_prefix}.weight"

        if target_key not in base_state:
            if verbose:
                print(f"  跳过 {base_prefix}: 底模中无 {target_key}")
            skipped_count += 1
            continue

        lora_a = lora_state[a_key].to(device=device, dtype=torch.float32)
        lora_b = lora_state[b_key].to(device=device, dtype=torch.float32)
        orig_tensor = base_state[target_key].to(device=device, dtype=torch.float32)

        # ΔW = lora_B @ lora_A
        # lora_A: (rank, in_dim), lora_B: (out_dim, rank)
        # ΔW: (out_dim, in_dim)
        delta = lora_b @ lora_a  # (out_dim, rank) @ (rank, in_dim) = (out_dim, in_dim)

        # 检查维度匹配
        if delta.shape != orig_tensor.shape:
            if verbose:
                print(f"  跳过 {base_prefix}: 维度不匹配 (ΔW {list(delta.shape)} vs 原权重 {list(orig_tensor.shape)})")
            skipped_count += 1
            continue

        # 融合
        merged_tensor = orig_tensor + scale * delta

        if target_dtype is not None:
            merged_tensor = merged_tensor.to(target_dtype)
        else:
            merged_tensor = merged_tensor.to(base_state[target_key].dtype)

        merged[target_key] = merged_tensor.cpu()
        merged_count += 1

        if verbose and merged_count <= 5:
            short_name = base_prefix.split(".")[-4:] if len(base_prefix.split(".")) > 4 else base_prefix
            delta_norm = delta.norm().item()
            orig_norm = orig_tensor.norm().item()
            ratio = delta_norm / (orig_norm + 1e-8)
            print(f"  融合 {'.'.join(short_name)}: ΔW范数={delta_norm:.4f}, 原范数={orig_norm:.4f}, 比率={ratio:.4f}")

    if verbose:
        print(f"\n融合完成: {merged_count} 层已合并, {skipped_count} 层跳过")
        print(f"输出 key 总数: {len(merged)}")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Kohya LoRA → Base Model 融合工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --base anima-base.safetensors --lora step1_lora.safetensors -o merged.safetensors
  %(prog)s --base base.safetensors --lora lora.safetensors -o merged.safetensors --scale 0.8
  %(prog)s --base base.safetensors --lora lora.safetensors -o merged.safetensors --dtype fp16
  %(prog)s --base base.safetensors --lora lora.safetensors -o merged.safetensors -v
        """,
    )
    parser.add_argument("--base", "-b", required=True, help="底模 .safetensors 路径")
    parser.add_argument("--lora", "-l", required=True, help="LoRA .safetensors 路径")
    parser.add_argument("--output", "-o", required=True, help="输出 .safetensors 路径")
    parser.add_argument("--scale", "-s", type=float, default=1.0, help="融合权重倍数 (默认: 1.0, 自动检测 alpha/dim)")
    parser.add_argument(
        "--dtype", "-d",
        choices=["fp32", "fp16", "bf16"],
        default=None,
        help="输出精度 (默认: 保持底模原始精度)",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="计算设备 (默认: cuda)")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    args = parser.parse_args()

    for path, label in [(args.base, "底模"), (args.lora, "LoRA")]:
        if not os.path.isfile(path):
            print(f"错误: {label}文件不存在: {path}", file=sys.stderr)
            sys.exit(1)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    output_dtype = dtype_map.get(args.dtype) if args.dtype else None

    print(f"加载底模: {args.base}")
    base_state = load_file(args.base)

    print(f"加载 LoRA: {args.lora}")
    lora_state = load_file(args.lora)

    merged = merge_lora_to_base(
        base_state,
        lora_state,
        scale=args.scale,
        dtype=output_dtype,
        device=args.device,
        verbose=args.verbose,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_file(merged, args.output)
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n已保存: {args.output} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
