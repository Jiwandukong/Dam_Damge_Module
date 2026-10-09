#!/usr/bin/env python3
"""Update only demo position fields of existing damage CSVs; preserve inference."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
from datetime import datetime
import json
import shutil
from demo_mapping import MEMBER_COLUMNS, POSITION_COLUMNS, SpillwayWaterlineGridSampler
from result_export import write_results
from paths import ROOT, WORK, read_json, write_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'Output')
    parser.add_argument('--work-dir', type=Path, default=WORK)
    parser.add_argument('--dam-model', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--frame-step-m', type=float, default=.25)
    args = parser.parse_args()
    summary_path = args.work_dir/'inference_summary.json'
    summary = read_json(summary_path)
    sampler = SpillwayWaterlineGridSampler(args.dam_model or summary['demo_mapping']['model'], seed=args.seed)
    position_fields = set(POSITION_COLUMNS+MEMBER_COLUMNS+[
        'world_center_x_m', 'world_center_y_m', 'world_center_z_m',
        'member_name', 'section_name', 'grid_id', 'grid_guid', 'mapping_status'])
    tables = []; sources = {}; retained = {}; before_hashes = {}
    for kind in ['CRC', 'SPL']:
        path = args.output/'Result'/f'{kind}_result.csv'
        before_hashes[kind] = sha(path)
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle); fields = list(reader.fieldnames); rows = list(reader)
        for field in POSITION_COLUMNS+MEMBER_COLUMNS:
            if field not in fields:
                fields.append(field)
        assert len(rows) == summary['counts'][kind]
        for row in rows:
            source = (path.parent/row['source_image_path']).resolve()
            overlay = (path.parent/row['overlay_path']).resolve()
            if row['image'] in sources and sources[row['image']] != source:
                raise ValueError('Conflicting frame paths')
            sources[row['image']] = source
            for asset in [source, overlay]:
                if asset not in retained:
                    retained[asset] = sha(asset)
        tables.append((kind, path, fields, rows))
    sampler.prepare_frames(sources.values(), frame_step=args.frame_step_m)
    backup = args.work_dir/'Reports'/('position_remap_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True)
    shutil.copy2(summary_path, backup/summary_path.name)
    for kind, path, fields, rows in tables:
        shutil.copy2(path, backup/path.name)
        for row in rows:
            previous = row.copy()
            sampler.sample(row)
            assert all(row[k] == value for k, value in previous.items() if k not in position_fields)
    # Compute all positions before replacing either CSV.
    assert tables[0][2]==tables[1][2], 'Per-class CSV schemas differ'
    summary['final_result']=write_results(args.output,[r for _,_,_,rows in tables for r in rows],columns=tables[0][2])
    summary['demo_mapping'] = sampler.summary()
    summary['mapping_status'] = 'demo_random'
    write_json(summary_path, summary)
    assert all(sha(asset) == digest for asset, digest in retained.items())
    report = dict(observations=sum(len(t[3]) for t in tables), frames=len(sources),
        backup=str(backup), eligible_grid_ids=sampler.eligible_grid_ids,
        frame_group_count=len(sampler.frame_groups),
        multi_frame_group_count=sum(len(g['images']) > 1 for g in sampler.frame_groups),
        frame_step_m=sampler.frame_step, changed_columns=sorted(position_fields),
        non_position_csv_fields_unchanged=True, source_images_and_overlays_unchanged=True,
        before_csv_sha256=before_hashes, after_csv_sha256={k: sha(p) for k, p, _, _ in tables})
    write_json(args.work_dir/'Reports/spillway_frame_remap.json', report)
    layout_path = args.work_dir/'Reports/frame_group_layout.csv'
    with layout_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image', 'group_id', 'index', 'size', 'offset_x_m', 'center_local_x_m', 'z_m', 'grid_band'])
        writer.writeheader()
        for name, layout in sampler.frame_layout.items():
            writer.writerow(dict(image=name, **layout))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
