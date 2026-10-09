#!/usr/bin/env bash
set -euo pipefail
ROV_LIDAR_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROV_LIDAR_PYTHON:-python3}" -B "${ROV_LIDAR_CODE_DIR}/process.py" "$@"
