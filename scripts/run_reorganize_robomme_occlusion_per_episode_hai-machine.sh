#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/yininghong/chenyuan/TTT-physics/repos/robomme_benchmark
FASTWAM_ROOT=/home/yininghong/chenyuan/TTT-physics/repos/FastWAM-TTT
DATA_ROOT=$FASTWAM_ROOT/data/robomme-occlusion
COLLECTION_SESSION=robomme_occlusion_lerobot_v2_20260709
LOG_PATH=$DATA_ROOT/reorganize_per_episode_20260709.log

while tmux has-session -t "$COLLECTION_SESSION" 2>/dev/null; do
    sleep 20
done

cd "$REPO_ROOT"
PYTHONUNBUFFERED=1 .venv/bin/python \
    scripts/reorganize_robomme_occlusion_per_episode_hai_machine.py \
    --data-root "$DATA_ROOT" \
    --case-types sequence swap reveal \
    2>&1 | tee "$LOG_PATH"

uv pip uninstall --python .venv/bin/python lerobot

.venv/bin/python - <<'PY'
import sys

sys.path.insert(0, "/home/yininghong/chenyuan/TTT-physics/repos/FastWAM-TTT/src")
from fastwam.datasets.lerobot.lerobot.lerobot_dataset import CODEBASE_VERSION

assert CODEBASE_VERSION == "v2.1", CODEBASE_VERSION
print(f"RoboMME collector LeRobot backend: {CODEBASE_VERSION}", flush=True)
PY
