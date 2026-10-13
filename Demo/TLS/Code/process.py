#!/usr/bin/env python3
"""Original E57 geometry + registered positions → candidates and mapped outputs."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'Code/config.json')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'Data')
    parser.add_argument('--output', type=Path, default=ROOT/'Output')
    parser.add_argument('--work-dir', type=Path, help='External intermediate/cache directory')
    parser.add_argument('--export-only', action='store_true', help='Export matching cached detections')
    args = parser.parse_args()
    if args.work_dir:
        os.environ['TLS_WORK_DIR'] = str(args.work_dir.expanduser().resolve())
    import runtime
    import numpy as np
    import local_plane_spalling as algorithm
    from export_results import export_results

    cfg = json.loads(args.config.read_text())
    parts = cfg['parts']
    if not parts or len(set(parts)) != len(parts) or not set(parts) <= {'dacheung_part1', 'dachung_part2'}:
        raise ValueError('parts must contain unique Part1/Part2 names')
    outward = np.asarray(cfg['outward_hint_xyz'], dtype=float)
    if outward.shape != (3,) or not np.isfinite(outward).all() or np.linalg.norm(outward) == 0:
        raise ValueError('A nonzero finite outward direction is required')
    intermediate = runtime.WORK/'local_plane_spalling'
    if args.export_only:
        saved = intermediate/'run_config.json'
        if not saved.is_file() or json.loads(saved.read_text()) != cfg:
            raise ValueError('Cached configuration differs. Run without --export-only first.')
    else:
        intermediate.mkdir(parents=True, exist_ok=True)
        # Write configuration last: interrupted runs cannot be exported as complete.
        (intermediate/'run_config.json').unlink(missing_ok=True)
        summaries = [algorithm.run_part(part, cfg, intermediate, args.data_dir) for part in parts]
        (intermediate/'run_summary.json').write_text(json.dumps({
            'method': 'overlapping_ring_RANSAC_local_planes', 'source': 'original_E57',
            'human_labels_used': False, 'outward_direction_verified': cfg['outward_direction_verified'],
            'parts': summaries}, ensure_ascii=False, indent=2)+'\n')
        (intermediate/'run_config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+'\n')
    export_results(cfg, intermediate, args.data_dir.resolve(), args.output.resolve())


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f'오류: {error}', file=sys.stderr)
        sys.exit(1)
