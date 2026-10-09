"""Portable module paths; generated files live in the user's cache."""
from pathlib import Path
import os
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache')))
WORK = Path(os.environ.get('ROV_RGB_WORK_DIR', str(CACHE/'dam_damage_module/rov_rgb'))).expanduser().resolve()
VIDEO = ROOT/'Data/source.mp4'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.new')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)
