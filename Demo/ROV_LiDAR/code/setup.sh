#!/usr/bin/env bash
set -euo pipefail
ROV_LIDAR_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROV_LIDAR_PYTHON="${ROV_LIDAR_PYTHON:-python3}"
ROV_LIDAR_CACHE="${ROV_LIDAR_WORK_DIR:-${HOME}/.cache/dam_damage_module/rov_lidar}"
ROV_LIDAR_CACHE="$("${ROV_LIDAR_PYTHON}" -B - "${ROV_LIDAR_CODE_DIR}" "${ROV_LIDAR_CACHE}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve().parent
cache = Path(sys.argv[2]).expanduser().resolve()
if cache == root or root in cache.parents:
    raise SystemExit('ROV_LIDAR_WORK_DIR must be outside ROV_LiDAR.')
print(cache)
PY
)"
mkdir -p -- "${ROV_LIDAR_CACHE}/python"
"${ROV_LIDAR_PYTHON}" -m pip install --upgrade --target "${ROV_LIDAR_CACHE}/python" -r "${ROV_LIDAR_CODE_DIR}/requirements.txt"
