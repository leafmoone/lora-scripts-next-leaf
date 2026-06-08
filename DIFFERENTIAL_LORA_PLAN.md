# Differential LoRA 移植技术方案与实施规划

## 1. Differential LoRA 核心概念

### 1.1 原理

Differential LoRA（差分 LoRA）是一种**双步骤 LoRA 训练策略**，用于捕捉两张配对图片之间的风格/特征差异：

```
Step 1 (图A): prompt = 基础标签 → LoRA_A 过拟合到图A的特征
Step 2 (图B): prompt = 基础标签 + 触发词 → LoRA_B 学习差异（LoRA_A已融入底模）
```

**推理时**: 基础标签生成 A 风格，加上触发词激活 B 风格差异。例如：
- `"1girl, blue sky"` → 生成 A 风格（原图风格）
- `"Character_Splitting, 1girl, blue sky"` → 生成 B 风格（目标风格）

### 1.2 与标准 LoRA 的区别

| 项目 | 标准 LoRA | Differential LoRA |
|------|----------|-------------------|
| 数据量 | 多张图 | 每组仅 **1 对图片** |
| 训练策略 | 一次训练 | **两次训练**，第二次以第一次的 LoRA 为底模 |
| 数据集 | 多样化图像 | 单一图像 × 高重复（过拟合） |
| 目标 | 学习概念 | **学习两个风格之间的精确差异** |
| 输入格式 | 单文件夹 | 双文件夹（A=原风格, B=目标风格），同名图片自动配对 |

---

## 2. DiffSynth-Studio 现有实现分析

### 2.1 入口脚本

`/root/lanyun-tmp/DiffSynth-Studio-leaf/examples/differential_lora/train_differential_lora.sh`

**参数:**
```bash
bash train_differential_lora.sh <folder_A> <folder_B> [output_dir] [tag_dir]
```

**目录结构示例:**
```
before/                    # folder_A
├── img_001.jpg
├── img_001.txt            # "1girl, blue sky, standing"
└── img_002.jpg
    img_002.txt            # "1boy, red shirt, sitting"

after/                     # folder_B
├── img_001.jpg            # 同名配对
└── img_002.jpg
```

### 2.2 完整训练流程

```
1. 自动打标 (AUTO_TAG)
   └── 为 folder_A 中的每张图片调用 tagger 生成 .txt 标签

2. 图片配对
   └── 遍历 folder_A 的图片，在 folder_B 中找同名文件

3. 对每组配对，执行：
   ├── Step 1: 训练 LoRA1
   │   ├── 数据: 图A + BASE_TAGS (从同名 .txt 读取)
   │   ├── 命令: python train.py --dataset_base_path ... 
   │   ├── 模型: Anima Pipeline (DiT + VAE + Qwen3)
   │   └── 输出: LoRA1 .safetensors
   │
   └── Step 2: 训练 LoRA2 (differential)
       ├── 数据: 图B + cleaned_tags + TRIGGER_WORD
       ├── 命令: python train.py --preset_lora_path <LoRA1> --preset_lora_model dit
       ├── 关键: LoRA1 融入底模 → 新 LoRA2 只学差异
       └── 输出: 差分 LoRA .safetensors

4. 后处理:
   ├── 转换 ComfyUI 格式 (添加 diffusion_model. 前缀)
   ├── SVD 合并所有差分 LoRA
   └── 采样验证
```

### 2.3 DiffSynth 核心训练脚本

文件: `examples/anima/model_training/train.py`

**AnimaTrainingModule** (继承 DiffusionTrainingModule):
- `__init__()`: 加载 AnimaImagePipeline（DiT + Qwen3 + VAE）
- `switch_pipe_to_training_mode()`: 添加 LoRA 层 + 可选 preset LoRA
- `forward()`: 前向传播 → FlowMatchSFTLoss

**关键参数:**
| 参数 | 说明 |
|------|------|
| `model_id_with_origin_paths` | 模型路径（DiT, Qwen3, VAE） |
| `lora_base_model` | LoRA 应用目标 ("dit") |
| `lora_target_modules` | LoRA 目标模块 |
| `lora_rank` | LoRA 秩 (默认 32) |
| `preset_lora_path` | **Step 2 使用**: 预加载 LoRA1 融入底模 |
| `preset_lora_model` | preset LoRA 应用目标 ("dit") |
| `lora_exclude_modules` | 排除的模块（如 "llm_adapter."） |

### 2.4 无法复用的 DiffSynth 组件

| 组件 | 原因 |
|------|------|
| `diffsynth.pipelines.anima_image.AnimaImagePipeline` | DiffSynth 专有，不可移植 |
| `diffsynth.core.UnifiedDataset` | DiffSynth 数据加载器 |
| `diffsynth.diffusion.DiffusionTrainingModule` | DiffSynth 训练框架基类 |
| `diffsynth.diffusion.FlowMatchSFTLoss` | DiffSynth loss 函数 |
| `diffsynth.diffusion.launch_training_task` | DiffSynth 训练启动器 |

### 2.5 可移植的工具脚本

| 脚本 | 功能 | 移植难度 |
|------|------|---------|
| `average_lora.py` | SVD/朴素 LoRA 权重合并 | 低（仅依赖 safetensors + torch） |
| `sample_lora.py` | 加载 LoRA 采样验证 | 中（依赖 AnimaImagePipeline） |
| `tagger/run.sh` | 自动打标 | 已在 lora-scripts-next 中有等价功能 |

---

## 3. lora-scripts-next 可复用接口分析

### 3.1 训练启动 (mikazuki/process.py)

```python
# 核心函数 - 用于启动标准 Kohya 训练
def build_accelerate_train_command(trainer_file, toml_path, cpu_threads, gpu_ids) 
    -> tuple[list[str], dict[str, str], Optional[str]]
    # 返回 (命令行参数, 环境变量, 混合精度设置)

def run_train(toml_path, trainer_file, gpu_ids, cpu_threads):
    # 将训练提交到 TaskManager，返回 APIResponse
    # 自动处理 stdout 流式日志、GPU 选择等
```

### 3.2 任务管理 (mikazuki/tasks.py)

```python
class Task:
    # 属性: task_id, command, status (CREATED/RUNNING/FINISHED/TERMINATED/FAILED)
    # 方法: start(), terminate(), _pump_stdout() → TrainLogHub

class TaskManager:
    # 单例: tm
    # 方法: create_task(), find_task(), terminate_task()
    # 限制: 最多 1 个并发任务
```

### 3.3 日志流 (mikazuki/train_log_hub.py)

```python
class TrainLogHub:
    # 单例: hub
    # 方法: feed(task_id, text), get(task_id), get_stream(task_id)
    # SSE 端点: GET /api/train/log/stream/{task_id}
```

### 3.4 训练器映射 (mikazuki/app/api.py)

```python
trainer_mapping = {
    "anima-lora": "./scripts/dev/anima_train_network.py",
    # ...
}

# 新增 Differential LoRA 类型:
# "differential-lora" → 自定义处理流程
```

### 3.5 标注器系统 (mikazuki/tagger/)

```python
# 已有功能:
# - WD14/CL 标注 (10 种模型)
# - 批量标注 via POST /api/interrogate
# - 进度追踪 (TaggerProgress)
# - 直接可复用于 Differential LoRA 的 auto_tag
```

### 3.6 TOML 配置系统

```python
# 使用 toml 库读写
import toml

# 从 config["key"] 读取
# sanitize_config(config) - 清理无效值
# 训练使用: trainer_file --config_file toml_path
```

### 3.7 工具脚本

```python
avaliable_scripts = [
    "networks/extract_lora_from_models.py",
    "networks/extract_lora_from_dylora.py",
    "networks/merge_lora.py",
    "tools/merge_models.py",
]
```

### 3.8 训练数据验证

```python
# mikazuki/utils/train_utils.py
def validate_model(pretrained_model_name_or_path, model_train_type) -> (bool, str)
def validate_data_dir(train_data_dir) -> bool
def get_total_images(train_data_dir) -> int
def build_sample_prompt_line(...)
```

---

## 4. 技术方案设计

### 4.1 核心决策: 双步 Kohya 训练 + 中间合并

**不使用 DiffSynth 库的前提下，最可行的方案是:**

```
对每组配对图片:
  Step 1: anima_train_network.py → LoRA1.safetensors (标准 Kohya 单图训练)
  ↓
  合并: merge_lora_to_base.py → merged_model.safetensors (Python 脚本)
  ↓
  Step 2: anima_train_network.py → LoRA2_differential.safetensors 
          (以 merged_model 为底模)
```

**为什么选这个方案:**
- ✅ 完全复用现有 Kohya 训练框架
- ✅ 不需要修改任何 Kohya 代码
- ✅ 合并脚本是独立的 Python 工具，不依赖 DiffSynth
- ✅ 训练日志、任务管理、SSE 流全部复用
- ✅ 现有 TOML 配置系统可直接使用

### 4.2 合并脚本设计 (核心)

`tools/merge_lora_to_base.py`:

```python
def merge_lora_to_base(base_safetensors, lora_safetensors, scale=1.0, device="cuda"):
    """
    将 Kohya 格式 LoRA 权重融入底模 safetensors
    
    Kohya LoRA 格式: {prefix}.lora_A.weight, {prefix}.lora_B.weight
    融合公式: W_new = W + scale * (lora_B @ lora_A)
    
    Args:
        base_safetensors: 底模 .safetensors 路径 (如 anima-base-v1.0.safetensors)
        lora_safetensors: LoRA .safetensors 路径
        scale: 融合权重 (默认 1.0)
    
    Returns:
        dict: 合并后的 state_dict (可直接 save_file)
    """
    base = load_file(base_safetensors)
    lora = load_file(lora_safetensors)
    
    # 匹配 lora_A / lora_B 对
    lora_pairs = find_lora_pairs(lora)
    
    for prefix, a_key, b_key in lora_pairs:
        # 匹配对应的底模 key
        target_key = f"{prefix}.weight"
        if target_key not in base:
            continue
        
        lora_A = lora[a_key].to(device)
        lora_B = lora[b_key].to(device)
        delta = lora_B @ lora_A  # 等价于 ΔW
        
        base[target_key] = (base[target_key].to(device) + scale * delta).to(base[target_key].dtype)
    
    return base
```

### 4.3 训练命令构建

**Step 1 (standard LoRA training):**
```bash
python scripts/dev/anima_train_network.py \
  --config_file config/autosave/20240101-120000_step1.toml
```

TOML 内容:
```toml
pretrained_model_name_or_path = "path/to/anima-base-v1.0.safetensors"
train_data_dir = "/tmp/difflora_step1_dataset/"
output_dir = "/tmp/difflora_lora1/"
# ... 标准 Kohya 参数
```

**Step 2 (differential LoRA training):**
```bash
python scripts/dev/anima_train_network.py \
  --config_file config/autosave/20240101-120100_step2.toml
```

TOML 内容:
```toml
pretrained_model_name_or_path = "/tmp/difflora_merged/merged.safetensors"  # ← 已融合 LoRA1
train_data_dir = "/tmp/difflora_step2_dataset/"
output_dir = "/tmp/difflora_lora2/"
# ... 标准 Kohya 参数
```

### 4.4 数据流图

```
┌─────────────────────────────────────────────────────────────┐
│ 前端: Differential LoRA 页面                                  │
│ - 选择 folder_A（原风格图片目录）                              │
│ - 选择 folder_B（目标风格图片目录）                            │
│ - 配置训练参数 (lr, epochs, repeat, rank, trigger_word...)   │
│ - 配置标注参数 (auto_tag, tagger_model...)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ POST /api/differential-lora/run
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 后端处理器: handle_differential_lora_run()                   │
│                                                             │
│ 1. 配对图片 (同名匹配)                                       │
│ 2. 自动标注 (可选, 调用 mikazuki/tagger)                     │
│                                                             │
│ 3. 对每组配对 (顺序执行):                                    │
│    ├── Step 1: 训练 LoRA_A                                  │
│    │   ├── 创建临时数据集 (单图 + metadata.csv)              │
│    │   ├── 生成 TOML 配置                                   │
│    │   ├── 调用 process.run_train()                        │
│    │   └── 等待完成后获取 LoRA1 路径                        │
│    │                                                        │
│    ├── 合并: merge_lora_to_base.py                          │
│    │   └── 底模 + LoRA1 → merged.safetensors               │
│    │                                                        │
│    └── Step 2: 训练 LoRA_B (差分)                           │
│        ├── 创建临时数据集 (图B + cleaned_tags + trigger)    │
│        ├── 生成 TOML (pretrained=merged.safetensors)       │
│        ├── 调用 process.run_train()                        │
│        └── 保存差分 LoRA                                   │
│                                                             │
│ 4. 后处理:                                                   │
│    ├── 转换 ComfyUI 格式                                    │
│    ├── SVD 合并 (average_lora.py)                           │
│    └── 输出最终结果                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 详细实施计划

### 5.1 文件清单

| # | 文件路径 | 类型 | 说明 |
|---|---------|------|------|
| 1 | `tools/merge_lora_to_base.py` | 新建 | LoRA→底模 融合工具 |
| 2 | `tools/average_lora.py` | 移植 | SVD 合并（从 DiffSynth 移植） |
| 3 | `tools/convert_differential_to_comfyui.py` | 移植 | ComfyUI 格式转换 |
| 4 | `mikazuki/differential_lora/__init__.py` | 新建 | 模块初始化 |
| 5 | `mikazuki/differential_lora/task_runner.py` | 新建 | **核心**: 差分训练任务编排器 |
| 6 | `mikazuki/differential_lora/adapter.py` | 新建 | UI 配置 → TOML 转换 |
| 7 | `mikazuki/differential_lora/preprocess.py` | 新建 | 数据集准备 (配对、临时数据集) |
| 8 | `mikazuki/schema/differential-lora.ts` | 新建 | 前端表单 Schema |
| 9 | `mikazuki/app/differential_lora_api.py` | 新建 | API 路由 |
| 10 | `frontend/dist/lora/differential-lora.html` | 新建 | 前端训练页面 |
| 11 | `config/differential_lora.toml` | 新建 | 默认配置 |
| 12 | `tests/test_differential_lora.py` | 新建 | 测试 |

### 5.2 修改现有文件

| 文件 | 修改内容 |
|------|---------|
| `mikazuki/app/api.py` | 注册 Differential LoRA API 路由 |
| `mikazuki/app/application.py` | 注册 SPA 路由 |

### 5.3 分步实施

#### 阶段 1: 核心工具脚本

**Step 1.1: `tools/merge_lora_to_base.py`**
- 输入: base.safetensors, lora.safetensors, scale
- 输出: merged.safetensors
- 解析 Kohya 格式 LoRA key (lora_A / lora_B)
- 实现 ΔW = lora_B @ lora_A 展开和融合
- 支持 layer-by-layer 进度日志

**Step 1.2: `tools/average_lora.py`** (移植)
- 完全从 DiffSynth 复制，仅修改 import
- 依赖: safetensors, torch (已在 requirements.txt)
- SVD 合并逻辑完整不变
- 命令行接口: `python average_lora.py <dir> --method svd --rank 32`

**Step 1.3: `tools/convert_differential_to_comfyui.py`** (移植)
- 添加 `diffusion_model.` 前缀到所有 key
- 依赖: safetensors

#### 阶段 2: 后端核心逻辑

**Step 2.1: `mikazuki/differential_lora/task_runner.py`**

核心类: `DifferentialLoRARunner`

```python
class DifferentialLoRARunner:
    def __init__(self, config: dict):
        self.config = config
        self.folder_a = config["folder_a"]
        self.folder_b = config["folder_b"]
        self.output_dir = config["output_dir"]
        self.trigger_word = config["trigger_word"]
        # ...

    def run(self) -> list[dict]:
        """主入口: 执行完整差分训练流程"""
        pairs = self._pair_images()
        results = []
        for img_a, img_b in pairs:
            result = self._train_pair(img_a, img_b)
            results.append(result)
        self._postprocess(results)
        return results

    def _pair_images(self) -> list[tuple]:
        """匹配同名图片"""
        ...

    def _train_pair(self, img_a, img_b) -> dict:
        """训练一组配对"""
        # 1. 读取基础标签
        # 2. Step 1: 训练 LoRA A
        # 3. 合并 LoRA A → 底模
        # 4. Step 2: 训练 LoRA B (差分)
        ...

    def _run_step1(self, img_path, prompt, output_dir) -> Path:
        """Step 1: 标准 LoRA 训练"""
        # 创建临时数据集目录
        # 构建 metadata.csv
        # 生成 TOML 配置
        # 调用已有的 process.run_train() + 轮询等待
        ...

    def _run_step2(self, img_path, prompt, output_dir, lora1_path) -> Path:
        """Step 2: 差分 LoRA 训练 (以合并模型为底模)"""
        # 1. merge_lora_to_base(base, lora1) → merged
        # 2. 创建数据集
        # 3. 生成 TOML (pretrained=merged)
        # 4. process.run_train() + 轮询等待
        ...
```

**Step 2.2: `mikazuki/differential_lora/preprocess.py`**

```python
def create_single_image_dataset(image_path, prompt, output_dir):
    """创建单图训练数据集"""
    # 1. 复制图片到 dataset_dir
    # 2. 生成 metadata.csv: image,prompt
    # 3. 返回 dataset_dir

def pair_images(folder_a, folder_b):
    """匹配同名图片配对"""
    # 返回 [(img_a_path, img_b_path), ...]
```

**Step 2.3: `mikazuki/differential_lora/adapter.py`**

```python
def build_step1_toml(config, dataset_dir, output_dir) -> dict:
    """构建 Step 1 的 TOML 配置"""
    return {
        "pretrained_model_name_or_path": config["model_path"],
        "train_data_dir": dataset_dir,
        "output_dir": output_dir,
        "logging_dir": config["logging_dir"],
        "network_module": "networks.lora",
        "network_args": [
            f"conv_dim={config['conv_dim']}",
            f"conv_alpha={config['conv_alpha']}",
        ],
        "network_alpha": config["lora_rank"],
        "network_dim": config["lora_rank"],
        "learning_rate": config["learning_rate"],
        "resolution": config["resolution"],
        "train_batch_size": 1,
        "max_train_epochs": config["num_epochs"],
        "dataset_repeats": config["dataset_repeat"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "mixed_precision": config.get("mixed_precision", "bf16"),
        "save_every_n_epochs": config["num_epochs"],
        "save_precision": "fp16",
        # ... 更多参数映射
    }

def build_step2_toml(config, dataset_dir, output_dir, merged_model_path) -> dict:
    """构建 Step 2 的 TOML 配置 (使用合并后的底模)"""
    base = build_step1_toml(config, dataset_dir, output_dir)
    base["pretrained_model_name_or_path"] = merged_model_path
    return base
```

#### 阶段 3: API 端点

**Step 3.1: `mikazuki/app/differential_lora_api.py`**

```python
router = APIRouter(prefix="/api/differential-lora")

@router.post("/run")
async def run_differential_lora(request: Request):
    """启动差分 LoRA 训练"""
    config = await request.json()
    
    # 验证参数
    folder_a = config["folder_a"]
    folder_b = config["folder_b"]
    
    # 自动标注 (如果启用)
    if config.get("auto_tag", True):
        await run_tagger(folder_a, config["tagger_config"])
    
    # 启动训练 (后台任务)
    runner = DifferentialLoRARunner(config)
    # 使用 FastAPI BackgroundTasks 或独立的 async 循环
    
    return APIResponseSuccess(message="训练已启动", data={"task_id": task_id})

@router.get("/status/{task_id}")
async def get_differential_status(task_id: str):
    """查询差分训练进度"""
    ...

@router.get("/pairs/{task_id}")
async def get_pair_list(task_id: str):
    """获取图片配对列表 (预览用)"""
    ...
```

**Step 3.2: 注册路由 (修改 `mikazuki/app/api.py`)**

```python
from mikazuki.app.differential_lora_api import router as differential_lora_router
app.include_router(differential_lora_router)
```

#### 阶段 4: 前端 Schema

**Step 4.1: `mikazuki/schema/differential-lora.ts`**

```typescript
Schema.intersect([
    Schema.object({
        model_train_type: Schema.const("differential-lora").default("differential-lora").disabled(),
        folder_a: Schema.string().role('filepicker', { type: "folder" })
            .description("原风格图片目录 (folder A)"),
        folder_b: Schema.string().role('filepicker', { type: "folder" })
            .description("目标风格图片目录 (folder B)"),
        output_dir: Schema.string().role('filepicker', { type: "folder" })
            .default("./models/differential_lora")
            .description("输出目录"),
        tag_dir: Schema.string().role('filepicker', { type: "folder" })
            .description("标签目录 (默认与 folder A 相同)"),
    }).description("差分训练配置"),

    Schema.object({
        trigger_word: Schema.string()
            .default("Character_Splitting")
            .description("差分训练触发词"),
        remove_tokens: Schema.string()
            .description("从基础标签中删除的词 (逗号分隔)"),
    }).description("差分训练参数"),

    Schema.object({
        model_path: Schema.string().role('filepicker', { type: "model-file" })
            .default("./sd-models/anima/anima-base-v1.0.safetensors")
            .description("Anima 基础模型路径"),
        lora_rank: Schema.number().min(1).default(32)
            .description("LoRA rank"),
        learning_rate: Schema.string().default("1e-4")
            .description("学习率"),
        num_epochs: Schema.number().min(1).default(5)
            .description("训练轮数"),
        dataset_repeat: Schema.number().min(1).default(1000)
            .description("单图重复次数"),
        resolution: Schema.string().default("1024,1024")
            .description("训练分辨率"),
        gradient_accumulation_steps: Schema.number().min(1).default(1)
            .description("梯度累加步数"),
        mixed_precision: Schema.union(["no", "fp16", "bf16"]).default("bf16")
            .description("混合精度"),
    }).description("训练超参数"),

    Schema.object({
        auto_tag: Schema.boolean().default(true)
            .description("训练前自动打标"),
        tagger_model: Schema.union([
            "wd14-convnextv2-v2",
            "wd-eva02-large-tagger-v3",
            "wd-swinv2-v3",
            "wd-vit-v3",
            "wd14-swinv2-v2",
            "wd14-vit-v2",
            "wd14-moat-v2",
            "wd-vit-large-tagger-v3",
            "wd-convnext-v3",
            "cl_tagger_1_01",
        ]).default("wd14-convnextv2-v2")
            .description("标注模型"),
        tagger_threshold: Schema.number().min(0).max(1).default(0.35)
            .description("标签置信度阈值"),
        tagger_char_threshold: Schema.number().min(0).max(1).default(0.85)
            .description("角色标签阈值"),
        tagger_max_tags: Schema.number().min(0).default(0)
            .description("最大标签数 (0=不限)"),
        tagger_blacklist: Schema.string()
            .description("过滤标签 (逗号分隔)"),
    }).description("自动标注设置"),

    Schema.object({
        enable_tensorboard_log: Schema.boolean().default(true),
        sample_every: Schema.number().min(0).default(10000),
        sample_prompts: Schema.string().role('textarea'),
        sample_guidance_scale: Schema.number().default(1),
        sample_steps: Schema.number().default(10),
        sample_height: Schema.number().default(1024),
        sample_width: Schema.number().default(1024),
    }).description("采样与日志"),

    Schema.object({
        postprocess_svd: Schema.boolean().default(true)
            .description("SVD 合并所有差分 LoRA"),
        postprocess_comfyui: Schema.boolean().default(true)
            .description("转换为 ComfyUI 格式"),
        export_prompts_dir: Schema.string().role('filepicker', { type: "folder" })
            .description("导出 Step2 提示词 (不训练)"),
        keep_temp: Schema.boolean().default(false)
            .description("保留临时文件 (调试用)"),
    }).description("后处理"),
])
```

#### 阶段 5: 前端页面

参考现有 `anima-fast.html` 的模板结构。由于前端是 VuePress SPA，新页面只需:

1. 创建 `frontend/dist/lora/differential-lora.html` (复制现有 HTML 模板)
2. 更新 `sd-nav-i18n.js` 添加导航项 (如果需要)
3. Schema 由前端 JS 动态解析生成表单，无需额外 HTML 修改

#### 阶段 6: 配置默认值

`config/differential_lora.toml`:

```toml
[training]
model_path = "./sd-models/anima/anima-base-v1.0.safetensors"
lora_rank = 32
learning_rate = "1e-4"
num_epochs = 5
dataset_repeat = 1000
resolution = "1024,1024"
gradient_accumulation_steps = 1
mixed_precision = "bf16"

[differential]
trigger_word = "Character_Splitting"
remove_tokens = ""

[tagging]
auto_tag = true
model = "wd14-convnextv2-v2"
threshold = 0.35
char_threshold = 0.85
max_tags = 0

[output]
output_dir = "./models/differential_lora"
keep_temp = false

[logging]
enable_tensorboard = true
sample_every = 10000

[postprocess]
svd_merge = true
comfyui_convert = true
```

#### 阶段 7: 测试

```python
# tests/test_differential_lora.py

class TestMergeLoraToBase:
    def test_merge_simple_lora(self):
        """测试基础 LoRA 融合"""
        # 创建模拟 base + lora
        # 验证融合结果

    def test_merge_lora_key_matching(self):
        """测试 LoRA key 匹配逻辑"""

    def test_merge_scale(self):
        """测试不同 scale 参数"""

class TestDatasetPreprocess:
    def test_pair_images(self):
        """测试图片配对"""

    def test_create_single_image_dataset(self):
        """测试单图数据集创建"""

class TestDifferentialLoRAAPI:
    def test_run_endpoint(self):
        """测试 API /run 端点"""

    def test_pair_list_endpoint(self):
        """测试配对预览"""
```

---

## 6. 关键风险与解决方案

### 6.1 风险: 单图训练与 Kohya 数据加载兼容性

**问题**: Kohya 的 `train_network.py` 期望数据集有 `repeats` 子目录和 `.txt` caption 文件。单图训练的数据集结构需要匹配。

**解决**: 创建符合 Kohya 规范的最小数据集:
```
/tmp/difflora_dataset/
└── 1_img001/           # repeats 子目录
    ├── img001.jpg
    └── img001.txt      # caption
```

### 6.2 风险: 合并后模型大小

**问题**: 将 LoRA 融入底模后，合并文件的大小 = 底模大小 (Anima DiT ~2GB)。临时文件占用磁盘。

**解决**: 
- 使用 `keep_temp=false` 自动清理
- 合并时 in-place 修改副本，不加载原始底模（节省内存）

### 6.3 风险: 训练任务顺序执行阻塞 API

**问题**: 多组配对需要 Step1→Step2 串行执行，可能耗时较长。FastAPI 是异步的，不能长时间阻塞请求。

**解决**:
- 使用 FastAPI `BackgroundTasks` 或独立线程执行训练
- 通过任务状态 API 前端轮询进度
- 每组配对的进度实时上报

### 6.4 风险: Kohya LoRA key 前缀不一致

**问题**: 不同训练脚本生成的 LoRA key 前缀可能不同（如 `lora_unet_` vs `lora_te_` vs `diffusion_model.`）。

**解决**: 合并脚本自动检测常见前缀并适配。

### 6.5 风险: 训练脚本路径和模型路径

**问题**: Anima 训练需要特定格式的模型路径。

**解决**: 使用现有 `validate_model()` 函数验证，复用已有的路径解析逻辑。

---

## 7. 依赖关系

### 现有依赖（已安装）
```
torch, safetensors, accelerate, diffusers, toml
```

### 新增依赖（无需额外安装）
```
无 - 所有工具仅使用已有依赖
```

---

## 8. 执行优先级与时间估算

| 阶段 | 内容 | 优先级 | 预估时间 |
|------|------|--------|---------|
| 阶段1 | 合并/后处理工具脚本 | P0 | 1.5h |
| 阶段2 | 核心训练编排器 | P0 | 2h |
| 阶段3 | API 端点 + Schema | P1 | 1h |
| 阶段4 | 前端页面 | P1 | 0.5h |
| 阶段5 | 测试 | P2 | 1h |
| 总计 | | | ~6h |

---

## 9. 文件关系速查

```
tools/
  merge_lora_to_base.py          ← 独立工具 (LoRA → 底模融合)
  average_lora.py                ← 移植 (SVD 合并)
  convert_differential_to_comfyui.py  ← 移植 (格式转换)

mikazuki/
  differential_lora/
    __init__.py                  ← 模块导出
    task_runner.py               ← 核心编排器 (被 API 调用)
    adapter.py                   ← UI 配置 → TOML 转换
    preprocess.py                ← 数据集准备
  schema/
    differential-lora.ts         ← 前端表单 Schema
  app/
    differential_lora_api.py     ← REST API 路由

frontend/dist/lora/
  differential-lora.html         ← 前端训练页面

config/
  differential_lora.toml         ← 默认训练配置

tests/
  test_differential_lora.py      ← 单元/集成测试
```
