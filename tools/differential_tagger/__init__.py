"""图像标注工具 — 为差分 LoRA 训练自动生成标签文件。

用法:
    from diffsynth.tools.tagger import auto_tag

    # 为文件夹内所有图片生成 .txt 标签文件
    auto_tag("/path/to/images/")

    # 自定义阈值和触发词
    auto_tag("/path/to/images/", general_threshold=0.35, trigger="")

    # 开启 VLM 自然语言描述（需要 ~10GB VRAM）
    auto_tag("/path/to/images/", use_vlm=True, purpose="general")
"""

import os


def auto_tag(input_dir, general_threshold=0.35, character_threshold=0.85,
             trigger="", use_vlm=False, purpose="general", max_tags=0,
             recursive=False, verbose=False):
    """
    为目录中所有图片生成 .txt 标签文件（与图片同名）。

    参数:
        input_dir: 图片文件夹路径
        general_threshold: 通用标签置信度阈值
        character_threshold: 角色标签置信度阈值
        trigger: 注入到标签开头的触发词
        use_vlm: 是否启用 ToriiGate VLM 自然语言描述
        purpose: VLM 描述方向 (general/character/style/concept)
        max_tags: 每张图最大标签数，0=不限
        recursive: 是否递归扫描子目录
        verbose: 详细日志
    """
    from .main import auto_tag_images
    auto_tag_images(
        input_path=input_dir,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        trigger=trigger,
        use_vlm=use_vlm,
        purpose=purpose,
        max_tags=max_tags,
        recursive=recursive,
        verbose=verbose,
    )
