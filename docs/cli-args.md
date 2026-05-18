# 程序参数

`gui.py` 支持以下命令行参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--host` | str | `127.0.0.1` | 服务器主机名 |
| `--port` | int | `28000` | GUI 端口 |
| `--listen` | bool | `false` | 监听所有网卡（`0.0.0.0`） |
| `--skip-prepare-environment` | bool | `false` | 跳过环境准备 |
| `--disable-tensorboard` | bool | `false` | 禁用 TensorBoard |
| `--disable-tageditor` | bool | `false` | 禁用标签编辑器 |
| `--disable-train-monitor` | bool | `false` | 禁用训练监控页 |
| `--tensorboard-host` | str | `127.0.0.1` | TensorBoard 主机 |
| `--tensorboard-port` | int | `6006` | TensorBoard 端口 |
| `--train-monitor-port` | int | `6008` | 训练监控页端口 |
| `--localization` | str | | 界面语言 |
| `--dev` | bool | `false` | 开发者模式 |

## 示例

```bash
# 本地默认启动
python gui.py

# 监听所有网卡（远程访问）
python gui.py --listen

# AutoDL 环境（自定义端口）
python gui.py --port 6006 --listen --skip-prepare-environment --disable-tensorboard

# 禁用训练监控
python gui.py --disable-train-monitor
```
