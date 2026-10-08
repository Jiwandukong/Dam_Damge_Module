#!/usr/bin/env bash
set -euo pipefail
multibeam_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
multibeam_project_dir="$(dirname -- "$multibeam_code_dir")"
multibeam_work_dir="${MULTIBEAM_WORK_DIR:-$(dirname -- "$multibeam_project_dir")/unmodified_multibeam/Multibeam_work}"
multibeam_python="${MULTIBEAM_PYTHON:-$multibeam_work_dir/environment/.venv/bin/python}"
if [[ ! -x "$multibeam_python" ]]; then
    printf '%s\n' '먼저 Code/setup.sh를 실행해 주세요. 설치한 Python은 MULTIBEAM_PYTHON으로 지정할 수 있습니다.' >&2
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$multibeam_python" "$multibeam_code_dir/process.py" "$@"
