# AutoDL / 云 GPU 运维脚本

实现位于此目录；**仓库根目录保留同名转发器**，兼容旧文档与自定义镜像里写死的路径。

## 文件

| 文件 | 说明 |
|------|------|
| `apply_lora_next_anima_defaults.py` | AutoDL 路径下 tokenizer / 默认采样词等 |
| `start_lora_next.sh` | conda `lora-next` + 清端口 + 启动 GUI（6006） |
| `autostart_lora_gui.sh` / `restart_lora_gui.sh` | 后台拉起 / 重启 |
| `update_lora_gui.sh` | `git pull` + defaults + restart |

## 与根目录 `start_autodl.sh` 的区别

| 入口 | 场景 |
|------|------|
| **`/…/start_autodl.sh`（根目录，勿挪）** | 镜像开机绑定；薄脚本，端口 6006 |
| `scripts/autodl/start_lora_next.sh` | 已配置 conda 的完整运维链 |

编辑逻辑请改本目录；根目录 `start_lora_next.sh` 等仅 `exec` 转发。
