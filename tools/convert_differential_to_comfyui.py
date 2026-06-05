#!/usr/bin/env python3
"""
Differential LoRA → ComfyUI 格式转换工具

为所有 safetensors key 添加 "diffusion_model." 前缀，
使其兼容 ComfyUI 的 Anima LoRA 加载器。

用法:
  python convert_differential_to_comfyui.py <文件夹路径>
  python convert_differential_to_comfyui.py <单个文件>
  python convert_differential_to_comfyui.py .    # 递归处理所有子目录
"""

import sys
import os
import glob
from safetensors.torch import load_file, save_file


def convert_file(src: str, dst: str) -> int:
    """转换单个文件，返回 key 数量"""
    state = load_file(src)
    new_state = {}
    for k, v in state.items():
        new_state[f"diffusion_model.{k}"] = v
    save_file(new_state, dst)
    return len(new_state)


def main():
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <文件夹路径或文件>")
        print(f"示例: python {sys.argv[0]} ./differential_lora_output/")
        print(f"      python {sys.argv[0]} my_lora.safetensors")
        sys.exit(1)

    base = sys.argv[1]
    if not os.path.exists(base):
        print(f"错误: 路径不存在: {base}", file=sys.stderr)
        sys.exit(1)

    if os.path.isfile(base):
        files = [base]
    else:
        files = sorted(glob.glob(os.path.join(base, "**", "*.safetensors"), recursive=True))
        files = [f for f in files if "_comfyui" not in f]

    if not files:
        print("未找到需要转换的 .safetensors 文件")
        return

    for src in files:
        dst = src.replace(".safetensors", "_comfyui.safetensors")
        if dst == src:
            continue
        n = convert_file(src, dst)
        os.remove(src)
        print(f"  {os.path.basename(src)} → {os.path.basename(dst)} ({n} keys)")

    print(f"\n转换完成: {len(files)} 个文件")


if __name__ == "__main__":
    main()
