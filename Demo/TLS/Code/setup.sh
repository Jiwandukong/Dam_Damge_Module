#!/usr/bin/env bash
set -euo pipefail
tls_code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 -m pip install --target "${TLS_WORK_DIR:-$HOME/.cache/dam_damage_module/tls_spalling}/python" -r "$tls_code_dir/requirements.txt"
