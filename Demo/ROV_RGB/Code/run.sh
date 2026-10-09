#!/usr/bin/env bash
set -euo pipefail
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${ROV_RGB_PYTHON:-$code_dir/../.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then python_bin=python3; fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
exec "$python_bin" "$code_dir/process.py" "$@"
