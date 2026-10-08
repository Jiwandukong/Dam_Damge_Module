#!/usr/bin/env bash
set -euo pipefail
multibeam_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
multibeam_project_dir="$(dirname -- "$multibeam_code_dir")"
multibeam_work_dir="${MULTIBEAM_WORK_DIR:-$(dirname -- "$multibeam_project_dir")/unmodified_multibeam/Multibeam_work}"
multibeam_env_dir="$multibeam_work_dir/environment/.venv"
multibeam_requirements="$multibeam_code_dir/requirements.txt"
if [[ -f "$multibeam_code_dir/requirements-lock.txt" ]]; then
    multibeam_requirements="$multibeam_code_dir/requirements-lock.txt"
fi
if [[ ! -x "$multibeam_env_dir/bin/python" ]]; then
    python3 -m venv --without-pip "$multibeam_env_dir"
fi
if "$multibeam_env_dir/bin/python" -m pip --version >/dev/null 2>&1; then
    "$multibeam_env_dir/bin/python" -m pip install -r "$multibeam_requirements"
else
    python3 -m pip --python "$multibeam_env_dir/bin/python" install pip -r "$multibeam_requirements"
fi
export PYTHONDONTWRITEBYTECODE=1
"$multibeam_env_dir/bin/python" "$multibeam_code_dir/process.py" --check
