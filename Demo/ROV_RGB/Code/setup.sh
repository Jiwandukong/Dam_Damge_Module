#!/usr/bin/env bash
set -euo pipefail
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$code_dir/../.venv"
"$code_dir/../.venv/bin/python" -m pip install -r "$code_dir/requirements.txt"
