#!/usr/bin/env bash
set -euo pipefail
viewer_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${ROV_RGB_PYTHON:-$viewer_code_dir/../../../.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then python_bin=python3; fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
"$python_bin" "$viewer_code_dir/build_viewer.py"
exec "$python_bin" "$viewer_code_dir/serve.py" "$@"
