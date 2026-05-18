#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate lora-next

if [[ -f /etc/network_turbo ]]; then
  source /etc/network_turbo
fi

cd "$SCRIPT_DIR"
git pull
python "$SCRIPT_DIR/apply_lora_next_anima_defaults.py"

"$SCRIPT_DIR/restart_lora_gui.sh"
