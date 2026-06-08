Schema.intersect([
    Schema.object({
        model_train_type: Schema.const("differential-lora").default("differential-lora").disabled().description("训练种类"),
        training_method: Schema.const("differential-lora").default("differential-lora").hidden(),
        folder_a: Schema.string().role('filepicker', { type: "folder" }).description("原风格图片目录 (folder A)。图片需与 folder B 同名配对"),
        folder_b: Schema.string().role('filepicker', { type: "folder" }).description("目标风格图片目录 (folder B)。与 folder A 同名的图片自动配对"),
        output_dir: Schema.string().role('filepicker', { type: "folder" }).default("./models/differential_lora").description("差分 LoRA 输出目录"),
        tag_dir: Schema.string().role('filepicker', { type: "folder" }).description("标签 .txt 目录。默认与 folder A 相同，每个图片 cat.jpg 对应 cat.txt"),
    }).description("差分训练配置"),

    Schema.object({
        trigger_word: Schema.string().default("Character_Splitting").description("差分训练触发词。推理时加上此词即可激活差分效果"),
        remove_tokens: Schema.string().description("从基础标签中删除的词（逗号分隔，如 boots,gloves）。Step2 会使用清理后的标签"),
    }).description("差分训练参数"),

    Schema.object({
        pretrained_model_name_or_path: Schema.string().role('filepicker', { type: "model-file" }).default("./sd-models/anima/anima-base-v1.0.safetensors").description("Anima 基础 DiT 模型路径"),
        vae: Schema.string().role('filepicker', { type: "model-file" }).default("./sd-models/anima/qwen_image_vae.safetensors").description("Qwen Image VAE 模型路径"),
        qwen3: Schema.string().role('filepicker', { type: "model-file" }).default("./sd-models/anima/qwen_3_06b_base.safetensors").description("Qwen3 文本模型路径"),
        attn_mode: Schema.union(["", "torch", "xformers", "sageattn", "flash"]).default("").description("Attention 加速实现。留空使用 torch 保底"),
        discrete_flow_shift: Schema.number().step(0.001).default(3.0).description("Rectified Flow 位移"),
    }).description("模型设置"),

    Schema.object({
        lora_rank: Schema.number().min(1).max(256).default(32).description("LoRA 秩 (rank)。越大学习能力越强，但文件也越大"),
        conv_dim: Schema.number().min(0).default(0).description("LoRA Conv 维度。0=不训练 Conv 层"),
        conv_alpha: Schema.number().min(1).default(1).description("LoRA Conv alpha"),
        lora_exclude_modules: Schema.string().default("").description("排除的模块字符串（如 llm_adapter.）"),
    }).description("LoRA 网络设置"),

    Schema.object({
        learning_rate: Schema.string().default("1e-4").description("学习率。差分训练需要较大 lr 以确保过拟合"),
        num_epochs: Schema.number().min(1).default(5).description("每组图片的训练轮数"),
        dataset_repeat: Schema.number().min(1).default(1000).description("单图高重复次数，确保过拟合"),
        resolution: Schema.string().default("1024,1024").description("训练图片分辨率，宽x高"),
        enable_bucket: Schema.boolean().default(true).description("启用 arb 桶以允许非固定宽高比"),
        gradient_accumulation_steps: Schema.number().min(1).default(1).description("梯度累加步数"),
        gradient_checkpointing: Schema.boolean().default(false).description("梯度检查点（省显存）"),
        mixed_precision: Schema.union(["no", "fp16", "bf16"]).default("bf16").description("混合精度"),
        optimizer_type: Schema.union([
            "AdamW8bit", "AdamW", "AdamW32bit", "SGDNesterov8bit",
            "SGDNesterov", "Lion8bit", "Lion", "DAdaptation",
            "Prodigy", "AdaFactor", "DAdaptAdaGrad", "DAdaptAdam",
            "DAdaptLion", "DAdaptSGD",
        ]).default("AdamW8bit").description("优化器类型"),
        lr_scheduler: Schema.union([
            "constant", "linear", "cosine", "cosine_with_restarts",
            "polynomial", "constant_with_warmup",
        ]).default("constant").description("学习率调度器"),
        max_grad_norm: Schema.number().min(0).default(1.0).description("梯度裁剪阈值"),
        seed: Schema.number().default(42).description("随机种子"),
        save_precision: Schema.union(["float", "fp16", "bf16"]).default("fp16").description("保存精度"),
    }).description("训练超参数"),

    Schema.object({
        auto_tag: Schema.boolean().default(false).description("训练前自动为 folder A 打标（使用 DiffSynth 标注器 run.sh）"),
        tagger_mode: Schema.union(["smart", "simple"]).default("smart").description("标注模式: smart=多阶段流水线+VLM, simple=仅 WD14"),
        tagger_use_vlm: Schema.boolean().default(true).description("启用 ToriiGate VLM 自然语言描述（smart 模式）"),
        tagger_use_cpu: Schema.boolean().default(false).description("仅用 CPU（默认 GPU）"),
        tagger_recursive: Schema.boolean().default(false).description("递归扫描子目录"),
        tagger_model: Schema.string().default("wd-eva02-large-tagger-v3").description("标注模型（如 wd-eva02-large-tagger-v3）"),
        tagger_threshold: Schema.number().min(0).max(1).default(0.35).description("标签置信度阈值"),
        tagger_char_threshold: Schema.number().min(0).max(1).default(0.85).description("角色标签阈值"),
        tagger_max_tags: Schema.number().min(0).default(0).description("每张图最大标签数（0=不限）"),
        tagger_blacklist: Schema.string().description("过滤标签（逗号分隔，如 watermark,signature）"),
        tagger_purpose: Schema.union(["character", "style", "general", "concept"]).default("character").description("VLM 描述方向"),
    }).description("自动标注 (DiffSynth Tagger)"),

    Schema.object({
        logging_dir: Schema.string().role('filepicker', { type: "folder" }).default("./logs/differential_lora").description("日志目录"),
        sample_every: Schema.number().min(0).default(10000).description("采样步数间隔（0 禁用）"),
        sample_prompts: Schema.string().role('textarea').description("采样提示词（留空使用训练提示词）"),
        sample_sampler: Schema.union(["euler", "k_euler"]).default("euler").description("采样器"),
        noise_offset: Schema.number().min(0).default(0).description("噪声偏移"),
        data_enhancement: Schema.string().description("数据增强（逗号分隔 flip_aug,color_aug,random_crop 等）"),
    }).description("日志与采样"),

    Schema.object({
        postprocess_comfyui: Schema.boolean().default(true).description("转换为 ComfyUI 格式（添加 diffusion_model. 前缀，删除原始文件）"),
        postprocess_svd: Schema.boolean().default(true).description("SVD 合并所有差分 LoRA 为一个文件"),
        keep_temp: Schema.boolean().default(false).description("保留临时文件（调试用）"),
    }).description("后处理"),
])
