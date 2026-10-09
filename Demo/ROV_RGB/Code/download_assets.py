#!/usr/bin/env python3
"""Download and verify the ROV checkpoint, Sunflicker cache and shared dam model."""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from urllib.request import Request, urlopen
import zipfile

from paths import ROOT, WORK, read_json, sha

ASSETS = read_json(ROOT/'Code/assets.json')['release_assets']


def download_verified(asset, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha(destination) == asset['sha256']:
        return destination
    print('Downloading Release asset: '+asset['filename'], flush=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.partial', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            request = Request(asset['url'], headers={'User-Agent': 'Dam-Damage-Module-ROV'})
            with urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, stream, length=8*1024*1024)
            stream.close()
            if sha(temporary) != asset['sha256']:
                raise ValueError('Release asset SHA256 mismatch: '+asset['filename'])
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def files_valid(directory, expected):
    return all((directory/name).is_file() and sha(directory/name) == digest
               for name, digest in expected.items())


def ensure_archive(scope, work=WORK):
    work = Path(work).resolve()
    asset = ASSETS[scope]
    expected = asset['files_sha256']
    if files_valid(work, expected):
        return
    archive = download_verified(asset, work/'downloads'/asset['filename'])
    with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory(dir=work) as temporary:
        stage = Path(temporary)
        names = set()
        for member in bundle.infolist():
            name = PurePosixPath(member.filename)
            if (name.is_absolute() or '..' in name.parts or '\\' in member.filename
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError('Invalid Release ZIP path: '+member.filename)
            if member.filename in names:
                raise ValueError('Duplicate Release ZIP entry: '+member.filename)
            names.add(member.filename)
        if not set(expected).issubset(names):
            raise ValueError('Release archive is missing required files')
        # Extract only the files declared in the checked-in manifest.
        for name in expected:
            destination = stage/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as source, destination.open('wb') as target:
                shutil.copyfileobj(source, target)
        if not files_valid(stage, expected):
            raise ValueError('Extracted Release file SHA256 mismatch')
        for name in expected:
            destination = work/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            (stage/name).replace(destination)


def ensure_model(work=WORK):
    asset = ASSETS['model']
    return download_verified(asset, Path(work)/'models'/asset['filename'])


def ensure_dam_model(work=WORK):
    ensure_archive('dam', work)
    return Path(work)/'Daecheongdam/daecheongdam_regions_grid5m.gltf'


def prepare_release_assets(args):
    if args.checkpoint is None:
        args.checkpoint = ensure_model(args.work_dir)
    if not args.input_preprocessed and not args.video.is_file():
        ensure_archive('preprocessing', args.work_dir)
    if args.demo_random_3d and args.dam_model is None:
        args.dam_model = ensure_dam_model(args.work_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['all', 'model', 'preprocessing', 'dam'], default='all')
    parser.add_argument('--work-dir', type=Path, default=WORK)
    args = parser.parse_args()
    if args.scope in {'all', 'model'}:
        print(ensure_model(args.work_dir))
    for scope in ['preprocessing', 'dam']:
        if args.scope in {'all', scope}:
            ensure_archive(scope, args.work_dir)
    print('Release assets verified: '+str(args.work_dir.resolve()))


if __name__ == '__main__':
    main()
