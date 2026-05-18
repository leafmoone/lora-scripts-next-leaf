# Docker 部署

## 编译镜像

```bash
docker build -t lora-scripts-next:latest -f Dockerfile-for-Mainland-China .
```

## 运行容器

```bash
docker run --gpus all -p 28000:28000 -p 6006:6006 registry.cn-hangzhou.aliyuncs.com/go-to-mirror/akegarasu_lora-scripts:latest
```

亦可配合仓库内 `docker-compose.yaml` 使用。

## 注意事项

- 镜像体积较大，拉取请耐心等待
- GPU 与驱动问题请自行查阅 NVIDIA Container Toolkit 文档
- 端口映射：28000（GUI）、6006（TensorBoard）、6008（训练监控）
