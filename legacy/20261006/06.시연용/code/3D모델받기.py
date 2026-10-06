#!/usr/bin/env python3
"""좌표·정량 계산에 사용하는 기존 대청댐 OBJ를 검증하여 받습니다."""

from __future__ import annotations

import argparse
from pathlib import Path

from demo512.release_download import restore_release


DEMO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = DEMO_ROOT / "02.시연모델/댐3D모델"
DEFAULT_MANIFEST = DEFAULT_DIRECTORY / "release_manifest.json"
DEFAULT_OUTPUT = DEFAULT_DIRECTORY / "daecheong_dam_epsg5186_zup.obj"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="기존 대청댐 3D 모델(OBJ) 다운로드. GPU·추가 Python 패키지는 필요하지 않습니다.",
        epilog="크기·SHA256 검사 후 저장하며, 다른 기존 파일은 덮어쓰지 않습니다.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="저장 경로 (기본: 02.시연모델/댐3D모델의 OBJ).")
    parser.add_argument("--parts-dir", type=Path,
                        help="이미 받은 OBJ가 있는 폴더. 지정하면 다운로드하지 않습니다.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="크기·SHA256·Release 주소가 기록된 배포 정보 파일.")
    parser.add_argument("--base-url", help="Release 다운로드 기본 주소를 직접 지정합니다.")
    args = parser.parse_args(argv)
    try:
        restore_release(args.manifest, args.output, parts_dir=args.parts_dir, base_url=args.base_url)
    except (OSError, ValueError) as error:
        parser.exit(1, f"3D 모델을 받을 수 없습니다: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
