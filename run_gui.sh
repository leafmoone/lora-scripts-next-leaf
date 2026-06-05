#!/bin/bash
cd "$(dirname "$0")"
nohup python gui.py > /var/log/gui.log 2>&1 &
echo "GUI started, PID: $!"
echo "  Frontend: http://0.0.0.0:12345"
echo "  TensorBoard: http://0.0.0.0:12348"
echo "  Train Monitor: http://0.0.0.0:12347"
