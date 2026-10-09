"""Reuse grid assignments only when their transform and candidate positions match."""

import hashlib
import json
import math
from pathlib import Path


def load_grid_mapping(transform_path):
    path = Path(transform_path)
    cache = path.parent / 'grid_mapping.json'
    if not cache.exists():
        return None
    data = json.loads(cache.read_text())
    if data['transform_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
        return None
    model = Path(__file__).resolve().parents[2] / 'Dam_model/Daecheongdam/daecheongdam_regions_grid5m.gltf'
    if model.exists():
        if hashlib.sha256(model.read_bytes()).hexdigest() != data['model_sha256']:
            return None
        for uri, digest in data['buffer_sha256'].items():
            with (model.parent / uri).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    return None
    return {r['damage_id']: r for r in data['candidates']}


def assigned_grid(damage_id, xyz, mapping):
    if mapping is None:
        return '', ''
    record = mapping[damage_id]
    if math.dist(xyz, record['world_xyz']) > 1e-8:
        raise ValueError('Cached grid assignment differs from candidate XYZ; refresh the model mapping')
    return record['member_name'], record['grid_id']
