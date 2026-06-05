#!/bin/bash
cd "$(dirname "$0")"

PORT=12346
TENSORBOARD_PORT=12348
MONITOR_PORT=12347

cleanup() {
    echo ""
    echo "正在停止服务..."
    # 杀掉所有相关子进程（孤儿进程）
    pkill -P $$ 2>/dev/null
    # 释放端口
    fuser -k ${PORT}/tcp ${TENSORBOARD_PORT}/tcp ${MONITOR_PORT}/tcp 2>/dev/null
    # 兜底：按进程名清理
    pkill -f "tensorboard.main.*${TENSORBOARD_PORT}" 2>/dev/null
    pkill -f "train_monitor/server.py" 2>/dev/null
    echo "服务已停止，端口已释放"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo "启动 GUI 服务..."
echo "  Frontend: http://0.0.0.0:${PORT}"
echo "  TensorBoard: http://0.0.0.0:${TENSORBOARD_PORT}"
echo "  Train Monitor: http://0.0.0.0:${MONITOR_PORT}"
echo "按 Ctrl+C 停止服务"
echo ""

python gui.py --port ${PORT} &
GUI_PID=$!
wait $GUI_PID
cleanup
