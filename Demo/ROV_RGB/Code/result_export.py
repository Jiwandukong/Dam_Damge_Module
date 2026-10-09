"""Write the final combined CSV and matching per-class CSVs."""
from collections import Counter
import csv
from pathlib import Path
from demo_mapping import MEMBER_COLUMNS, POSITION_COLUMNS
from paths import sha

RESULT_COLUMNS = [
    'image', 'damage_id', 'damage_type', 'damage_name_ko', 'pixel_nodes_json',
    'world_center_x_m', 'world_center_y_m', 'world_center_z_m', 'length_px', 'length_m',
    'width_px', 'width_m', 'area_m2', 'area_px', 'member_name', 'section_name',
    'grid_id', 'grid_guid', 'DRI', 'source_image_path', 'overlay_path',
    'mapping_status', 'measurement_status', 'pixel_center_x_px', 'pixel_center_y_px',
    'bbox_px_json', 'crop_origin_x_px', 'crop_origin_y_px', 'crop_size_px',
    'overlay_padding_px_json', 'result_origin',
] + POSITION_COLUMNS + MEMBER_COLUMNS


def write_results(output, rows, columns=RESULT_COLUMNS):
    rows = sorted(rows, key=lambda r: r['damage_id'])
    if len({r['damage_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate damage IDs in final results')
    if any(r['damage_type'] not in {'CRC', 'SPL'} for r in rows):
        raise ValueError('Unexpected damage class in final results')
    if len(columns) != len(set(columns)) or not set(MEMBER_COLUMNS).issubset(columns):
        raise ValueError('Result schema needs distinct columns including member IDs')
    if any(set(r)-set(columns) for r in rows):
        raise ValueError('Result schema would discard existing fields')
    for row in rows:
        if row.get('mapping_status')=='demo_random' and any(
                row.get(k) is None or str(row.get(k,''))==''
                for k in ['grid_id','grid_guid','member_id','member_node_id']):
            raise ValueError('Demo result needs grid and member IDs: '+row['damage_id'])
    folder = Path(output)/'Result'
    folder.mkdir(parents=True, exist_ok=True)
    exports = {'result.csv': rows}
    exports.update({f'{kind}_result.csv': [r for r in rows if r['damage_type']==kind]
                    for kind in ['CRC', 'SPL']})
    for name, selected in exports.items():
        temporary = folder/(name+'.new')
        with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader(); writer.writerows(selected)
    for name in exports:
        (folder/(name+'.new')).replace(folder/name)
    return dict(primary_csv=str((folder/'result.csv').resolve()),
                rows=len(rows), counts=dict(Counter(r['damage_type'] for r in rows)),
                columns=list(columns), csv_sha256={name: sha(folder/name) for name in exports})
