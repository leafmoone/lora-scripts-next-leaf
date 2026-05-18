#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate lora-next

# AutoDL academic network acceleration. It is optional, so keep startup working
# on images where the helper is not installed.
if [[ -f /etc/network_turbo ]]; then
  source /etc/network_turbo
fi

cd "$SCRIPT_DIR"

export OMP_NUM_THREADS=1

"$SCRIPT_DIR/apply_lora_next_anima_defaults.py"

# Clean up old processes to avoid duplicate GUI/backend port listeners.
python - <<'PY'
import glob
import os
import signal

ports = {"60006", "6006", "6007", "6008", "28000", "28001"}
listen_inodes = set()

for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        lines = open(table, encoding="utf-8").read().splitlines()[1:]
    except FileNotFoundError:
        continue
    for line in lines:
        parts = line.split()
        local_address = parts[1]
        state = parts[3]
        inode = parts[9]
        port = str(int(local_address.rsplit(":", 1)[-1], 16))
        if state == "0A" and port in ports:
            listen_inodes.add(inode)

pids = set()
for fd_path in glob.glob("/proc/[0-9]*/fd/*"):
    try:
        target = os.readlink(fd_path)
    except OSError:
        continue
    if not target.startswith("socket:["):
        continue
    inode = target.removeprefix("socket:[").removesuffix("]")
    if inode not in listen_inodes:
        continue
    pid = int(fd_path.split("/")[2])
    if pid != os.getpid():
        pids.add(pid)

for pid in pids:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
PY

"$SCRIPT_DIR/start_lora_monitor.sh" > /root/lora_monitor.log 2>&1 &

python gui.py \
  --listen \
  --host 0.0.0.0 \
  --port 6006 \
  --skip-prepare-environment \
  --disable-tensorboard
