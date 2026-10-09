"""Restore the three survey inputs from the checksum-pinned GitHub Release."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK = Path(os.environ.get('ROV_LIDAR_WORK_DIR', Path.home()/'.cache/dam_damage_module/rov_lidar'))
ASSET = {
    'name': 'rov_lidar_data.zip',
    'url': 'https://github.com/Jiwandukong/Dam_Damge_Module/releases/download/rov-lidar-demo-v1/rov_lidar_data.zip',
    'bytes': 147095301,
    'sha256': 'f2ff79c5ff14a4ffac06ef75fa047dc92f4d9da2d4c5cb6b1da15ea4dab2eb7c',
    'members': [
        {'path': 'data/01_EYAS_translated.e57', 'bytes': 322881536,
         'sha256': '192dc65382fde4ea599e702873bee6bfee087a89034b8c5b9b231506e0cc425f'},
        {'path': 'data/01_EYAS_translated_segmented.ply', 'bytes': 447191845,
         'sha256': '7286f226d9fed59c503f7feb73291c809ff664eebed440a5fea0a01926cc93b6'},
        {'path': 'data/01_EYAS_translated_segmented_slab_zone.ply', 'bytes': 110700616,
         'sha256': '8aa34da2efafb941b47bfbb12905e1e8bc3cc8635d54a855cc093cd4c7069ce8'},
    ],
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def matches(path, expected):
    return path.is_file() and path.stat().st_size == expected['bytes'] and sha256(path) == expected['sha256']


def ensure_assets(work_dir=DEFAULT_WORK, root=ROOT):
    work_dir, root = Path(work_dir).expanduser().resolve(), Path(root).resolve()
    if work_dir == root or root in work_dir.parents:
        raise ValueError('Download cache must be outside ROV_LiDAR.')
    if all(matches(root/member['path'], member) for member in ASSET['members']):
        print('Input files already match the Release.', flush=True)
        return
    cache = work_dir/'downloads'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache/ASSET['name']
    if not matches(archive, ASSET):
        print(f"Downloading {ASSET['name']}...", flush=True)
        request = Request(ASSET['url'], headers={'User-Agent': 'ROV-LiDAR-Demo/1.0'})
        with tempfile.NamedTemporaryFile(dir=cache, suffix='.part', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with urlopen(request, timeout=60) as response:
                    shutil.copyfileobj(response, stream, length=4*1024*1024)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            if not matches(temporary, ASSET):
                raise RuntimeError('Release ZIP size or SHA-256 does not match.')
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as packed:
        expected_paths = {member['path'] for member in ASSET['members']}
        if set(packed.namelist()) != expected_paths or len(packed.infolist()) != len(expected_paths):
            raise RuntimeError('Unexpected members in the Release ZIP.')
        for member in ASSET['members']:
            destination = root/member['path']
            if matches(destination, member):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.download', delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    with packed.open(member['path']) as source:
                        shutil.copyfileobj(source, stream, length=4*1024*1024)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                if not matches(temporary, member):
                    raise RuntimeError(f"Extracted input does not match: {member['path']}")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
    print('Restored 3 input files; SHA-256 verified.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    ensure_assets(args.work_dir)


if __name__ == '__main__':
    main()
