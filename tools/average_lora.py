#!/usr/bin/env python3
"""
Differential LoRA 权重合并脚本 — 支持朴素平均 / SVD 合并

移植自: DiffSynth-Studio/examples/differential_lora/average_lora.py
依赖: safetensors, torch (已在 lora-scripts-next requirements.txt)

SVD 合并原理:
  1. 展开每个 LoRA 的 ΔW = B @ A
  2. 求平均 ΔW_avg = mean(ΔW_i)
  3. SVD 分解: ΔW_avg = U @ S @ Vt
  4. 取 top-rank 主成分
  5. 重构: lora_A = sqrt(S) @ Vt, lora_B = U @ sqrt(S)

用法:
  python average_lora.py <文件夹路径> [--method svd|naive] [--rank N]
  python average_lora.py ./differential --method svd --rank 32
  python average_lora.py ./differential --method naive
  python average_lora.py <单个文件1> <单个文件2> ... --files
"""

import sys
import os
import glob
import argparse
import warnings

import torch
from safetensors.torch import load_file, save_file

RANK_PAD_WARNED = False
SVD_FALLBACK_WARNED = False
NON_LORA_WARNED = False


def find_lora_pairs(state_dict: dict) -> list:
    """在 state_dict 中匹配 lora_A / lora_B 对，返回 [(base_prefix, a_key, b_key), ...]"""
    pairs = []
    seen = set()
    for key in state_dict:
        if ".lora_A." not in key:
            continue
        base = key[: key.index(".lora_A.")]
        if base in seen:
            continue
        b_key = key.replace(".lora_A.", ".lora_B.")
        if b_key in state_dict:
            seen.add(base)
            pairs.append((base, key, b_key))
    return pairs


def svd_merge(file_paths: list, output_path: str, rank: int = None, dtype=None) -> dict:
    """SVD 合并: 展开 ΔW=B@A → 平均 → SVD → 提取主成分"""
    global RANK_PAD_WARNED, SVD_FALLBACK_WARNED
    RANK_PAD_WARNED = SVD_FALLBACK_WARNED = False

    if len(file_paths) < 2:
        print(f"错误: 至少需要 2 个文件，当前 {len(file_paths)} 个", file=sys.stderr)
        return None

    states = [load_file(f) for f in file_paths]

    pairs = find_lora_pairs(states[0])
    if not pairs:
        print(f"错误: 未找到 lora_A / lora_B 配对", file=sys.stderr)
        return None

    if rank is None:
        rank = states[0][pairs[0][1]].shape[0]
        print(f"自动推断 rank = {rank}")

    print(f"找到 {len(pairs)} 对 LoRA 矩阵，目标 rank = {rank}")

    all_keys = set()
    for s in states:
        all_keys.update(s.keys())
    lora_keys = {p[1] for p in pairs} | {p[2] for p in pairs}
    non_lora_keys = all_keys - lora_keys
    _warn_non_lora(non_lora_keys)

    result = {}
    rank_pad_count = 0
    svd_fail_count = 0

    for base_prefix, a_key, b_key in pairs:
        d_out = states[0][b_key].shape[0]
        d_in = states[0][a_key].shape[1]

        delta_sum = None
        count = 0
        for s in states:
            if a_key not in s or b_key not in s:
                continue
            A = s[a_key].float()
            B = s[b_key].float()
            delta = B @ A
            delta_sum = delta if delta_sum is None else delta_sum + delta
            count += 1

        if delta_sum is None or count == 0:
            continue

        delta_avg = delta_sum / count

        try:
            U, S_vec, Vt = torch.linalg.svd(delta_avg, full_matrices=False)
        except Exception as e:
            _warn_svd_fallback(base_prefix, e)
            svd_fail_count += 1
            result[a_key] = torch.eye(rank, d_in)
            result[b_key] = delta_avg[:, :rank]
            continue

        r_actual = min(rank, len(S_vec))
        if r_actual < rank:
            _warn_rank_pad(base_prefix, r_actual, rank)
            rank_pad_count += 1

        U_r = U[:, :r_actual]
        S_r = S_vec[:r_actual]
        Vt_r = Vt[:r_actual, :]
        S_sqrt = torch.sqrt(S_r)

        new_A = torch.diag(S_sqrt) @ Vt_r
        new_B = U_r @ torch.diag(S_sqrt)

        if r_actual < rank:
            new_A = torch.cat([new_A, torch.zeros(rank - r_actual, d_in)], dim=0)
            new_B = torch.cat([new_B, torch.zeros(d_out, rank - r_actual)], dim=1)

        orig_dtype = dtype or states[0][a_key].dtype
        result[a_key] = new_A.to(orig_dtype)
        result[b_key] = new_B.to(orig_dtype)

        if len(result) <= 8 * len(pairs):
            explained = (S_r**2).sum() / (S_vec**2).sum()
            recon = (new_B.float() @ new_A.float() - delta_avg).norm() / delta_avg.norm()
            short = base_prefix.rsplit("diffusion_model.", 1)[-1] if "diffusion_model." in base_prefix else base_prefix
            print(f"  {short[:60]}: top-{r_actual} 解释方差 {explained:.1%}, 重构误差 {recon:.4f}")

    for key in non_lora_keys:
        tensors = [s[key].float() for s in states if key in s]
        if tensors:
            result[key] = torch.stack(tensors).mean(dim=0)
            if dtype is not None:
                result[key] = result[key].to(dtype)

    save_file(result, output_path)

    print(f"\nSVD 合并完成: {len(file_paths)} 文件 → {os.path.basename(output_path)}")
    print(f"  {len(pairs)} 对 LoRA + {len(non_lora_keys)} 独立 key")
    if rank_pad_count > 0:
        print(f"  !!! {rank_pad_count} 层 rank 不足，已补零（合并结果可能含死神经元）")
    if svd_fail_count > 0:
        print(f"  !!! {svd_fail_count} 层 SVD 失败，已回退为 QR 近似")

    return result


def naive_average(file_paths: list, output_path: str) -> dict:
    """朴素平均：分别平均 lora_A 和 lora_B"""
    global NON_LORA_WARNED
    NON_LORA_WARNED = False

    if len(file_paths) < 2:
        print(f"错误: 至少需要 2 个文件，当前 {len(file_paths)} 个", file=sys.stderr)
        return None

    states = [load_file(f) for f in file_paths]

    base_keys = set(states[0].keys())
    for i, s in enumerate(states[1:], 2):
        missing = base_keys - set(s.keys())
        extra = set(s.keys()) - base_keys
        if missing or extra:
            print(f"  !!! {os.path.basename(file_paths[i-1])} key 不一致 (缺 {len(missing)}, 多 {len(extra)})", file=sys.stderr)

    pairs = find_lora_pairs(states[0])
    lora_keys = {p[1] for p in pairs} | {p[2] for p in pairs}
    non_lora_keys = base_keys - lora_keys
    _warn_non_lora(non_lora_keys)

    avg = {}
    for key in base_keys:
        tensors = [s[key].float() for s in states if key in s]
        if tensors:
            avg[key] = torch.stack(tensors).mean(dim=0)

    save_file(avg, output_path)
    print(f"\n朴素平均完成: {len(file_paths)} 文件 → {os.path.basename(output_path)} ({len(avg)} keys)")
    return avg


def _warn_rank_pad(base, actual, target):
    global RANK_PAD_WARNED
    name = base.rsplit("diffusion_model.", 1)[-1] if "diffusion_model." in base else base
    if not RANK_PAD_WARNED:
        print(f"\n  [W] RANK 不足开始补零:", file=sys.stderr)
        RANK_PAD_WARNED = True
    print(f"     {name[:60]}: actual={actual} < target={target}", file=sys.stderr)
    print(f"     → 缺失的 {target - actual} 维以零填充，推理时这部分维度不产生任何效果。" f"建议降低 --rank 或增加训练图片组数。", file=sys.stderr)


def _warn_svd_fallback(base, error):
    global SVD_FALLBACK_WARNED
    name = base.rsplit("diffusion_model.", 1)[-1] if "diffusion_model." in base else base
    if not SVD_FALLBACK_WARNED:
        print(f"\n  [W] SVD 分解失败，回退为 QR 近似:", file=sys.stderr)
        SVD_FALLBACK_WARNED = True
    print(f"     {name[:60]}: {error}", file=sys.stderr)
    print(f"     → 该层改用 QR 分解，合并质量可能下降。" f"通常是 ΔW 矩阵数值不稳定（含 NaN/Inf 或极端值）。", file=sys.stderr)


def _warn_non_lora(keys):
    global NON_LORA_WARNED
    if keys and not NON_LORA_WARNED:
        print(f"\n  [W] 发现 {len(keys)} 个非 lora_A/B 的 key，将使用简单平均（非 SVD 合并）:", file=sys.stderr)
        for k in sorted(keys)[:20]:
            print(f"     {k}", file=sys.stderr)
        if len(keys) > 20:
            print(f"     ... 还有 {len(keys) - 20} 个", file=sys.stderr)
        print(f"     → 这些 key 不是标准 LoRA 矩阵，无法做 ΔW 展开。" f"检查是否有多余的 alpha/bias 参数或数据损坏。", file=sys.stderr)
        NON_LORA_WARNED = True


def main():
    parser = argparse.ArgumentParser(description="Differential LoRA 权重合并")
    parser.add_argument("path", nargs="?", help="文件夹路径或文件")
    parser.add_argument("--glob", "-g", default="*_comfyui.safetensors", help="文件匹配模式 (默认: *_comfyui.safetensors)")
    parser.add_argument("--output", "-o", default=None, help="输出路径 (默认: <path>/merged_lora.safetensors)")
    parser.add_argument("--files", "-f", nargs="+", default=None, help="手动指定文件列表")
    parser.add_argument("--method", "-m", choices=["svd", "naive"], default="svd", help="合并模式: svd (推荐) | naive (分别平均A/B)")
    parser.add_argument("--rank", "-r", type=int, default=None, help="SVD 截断 rank (默认: 自动检测)")
    parser.add_argument("--dtype", "-d", choices=["float32", "float16", "bfloat16"], default=None, help="输出精度 (默认: 与输入一致)")
    args = parser.parse_args()

    if args.files:
        files = args.files
    elif args.path and os.path.isfile(args.path):
        files = [args.path]
    elif args.path and os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "**", args.glob), recursive=True))
    else:
        print("错误: 请指定路径或文件列表", file=sys.stderr)
        sys.exit(1)

    if not files:
        print(f"未找到匹配 '{args.glob}' 的文件", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output = args.output
    else:
        base_dir = args.path if (args.path and os.path.isdir(args.path)) else os.path.dirname(args.path or ".")
        output = os.path.join(base_dir, "merged_lora.safetensors")

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(args.dtype) if args.dtype else None

    print(f"参与合并的文件 ({len(files)} 个):")
    for f in files:
        print(f"  {os.path.basename(f)}")
    print(f"方法: {args.method}")

    if args.method == "svd":
        svd_merge(files, output, rank=args.rank, dtype=dtype)
    else:
        naive_average(files, output)


if __name__ == "__main__":
    main()
