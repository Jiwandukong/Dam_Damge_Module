"""Export regular geometric candidates in the Demo point-cloud sensor layout."""
import csv
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import runtime
import laspy
import numpy as np
import matplotlib.pyplot as plt
from local_plane_spalling import load_source, write_csv

FIELDS = [
    'dataset', 'damage_id', 'source_damage_id', 'damage_type', 'damage_name_ko',
    'detection_status', 'world_center_x_m', 'world_center_y_m', 'world_center_z_m',
    'crs', 'grid_id', 'grid_guid', 'member_name', 'mapping_status',
    'coordinate_frame',
    'area_m2', 'mean_depth_m', 'p95_depth_m', 'max_depth_m', 'point_count',
    'member_guid', 'nearest_grid_distance_m', 'nearest_member_distance_m',
    'mapping_max_distance_m', 'grid_area_m2', 'grid_damage_area_m2',
    'grid_damage_ratio_pct', 'mapped_area_m2', 'unmapped_area_m2', 'mapped_grid_count',
    'source_data_path', 'registered_data_path', 'pointcloud_path', 'visualization_path',
]

def result_row(part, row, xyz, source_ids, data_dir, output):
    """Use the observed candidate centroid as the sole exported XYZ triplet."""
    centre = np.mean(xyz[source_ids], axis=0)
    damage_id = row['damage_id']
    return {
        'dataset': part, 'damage_id': damage_id, 'source_damage_id': int(row['candidate_id']),
        'damage_type': 'SPL', 'damage_name_ko': '박락 후보', 'detection_status': row['status'],
        **{'world_center_'+axis+'_m': float(value) for axis, value in zip('xyz', centre)},
        'crs': '', 'grid_id': '', 'grid_guid': '', 'member_name': '',
        'mapping_status': 'unregistered', 'coordinate_frame': 'source_E57_local',
        'area_m2': row['covered_area_m2'], 'mean_depth_m': row['mean_depth_m'],
        'p95_depth_m': row['p95_depth_m'], 'max_depth_m': row['max_depth_m'],
        'point_count': len(source_ids),
        'source_data_path': os.path.relpath(data_dir/(part+'.e57'), output/'Result').replace(os.sep, '/'),
        **{key: '' for key in FIELDS if key in {
            'member_guid', 'nearest_grid_distance_m', 'nearest_member_distance_m',
            'mapping_max_distance_m', 'grid_area_m2', 'grid_damage_area_m2',
            'grid_damage_ratio_pct', 'mapped_area_m2', 'unmapped_area_m2',
            'mapped_grid_count', 'registered_data_path'}},
        'pointcloud_path': 'SPL/'+damage_id+'.las',
        'visualization_path': '../Visualize/SPL/'+damage_id+'.png',
    }


def write_las(path, xyz, rgb, depth, candidate_id, source_ids):
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.scales = np.full(3, 0.000001)
    header.offsets = (xyz.min(axis=0)+xyz.max(axis=0))/2
    for name, dtype, description in [
        ('depth_m', np.float32, 'Inward geometric depth in metres'),
        ('candidate_id', np.int32, 'Region ID within source E57'),
        ('source_vertex_id', np.uint32, 'Zero-based original E57 record'),
    ]:
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype, description=description))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz.T
    cloud.red, cloud.green, cloud.blue = (rgb.astype(np.uint16)*257).T
    cloud['depth_m'] = depth
    cloud['candidate_id'] = np.full(len(xyz), candidate_id, dtype=np.int32)
    cloud['source_vertex_id'] = source_ids.astype(np.uint32)
    cloud.write(path)


def snapshot(path, damage_id, row, xyz, rgb, uvw, source_ids, context, depth, surface, cfg, registered=False):
    """Show the original surface, selected returns and the existing reference."""
    rng = np.random.default_rng(cfg['random_seed']+int(row['candidate_id']))
    shown = context if len(context) <= 40000 else rng.choice(context, 40000, replace=False)
    selected = source_ids if len(source_ids) <= 30000 else rng.choice(source_ids, 30000, replace=False)
    figure = plt.figure(figsize=(16, 9), dpi=240, layout='constrained')
    front = figure.add_subplot(1, 2, 1)
    front.scatter(uvw[shown, 0], uvw[shown, 1], c=rgb[shown]/255, s=2, linewidths=0)
    front.scatter(uvw[selected, 0], uvw[selected, 1], c='#ed4444', s=3, linewidths=0)
    front.set(title='Original RGB / red: selected candidate points', xlabel='Surface U (m)',
              ylabel='Surface V (m)', aspect='equal')
    front.grid(alpha=.15)

    view = figure.add_subplot(1, 2, 2, projection='3d')
    # The surface frame is orthonormal; no depth exaggeration or noise is applied.
    centre = np.mean(uvw[source_ids], axis=0)
    relative = uvw-centre
    view.scatter(*relative[shown].T, color='#9da7ad', s=1, alpha=.3, linewidths=0, depthshade=False)
    low = cfg['minimum_depth_m']*1000
    high = max(low+5, min(cfg['maximum_depth_m']*1000, float(row['max_depth_m'])*1000))
    colored = view.scatter(*relative[selected].T, c=depth[selected]*1000, cmap='YlOrRd',
                           vmin=low, vmax=high, s=5, linewidths=0, depthshade=False)
    origin = surface['uv_origin']; resolution = float(surface['resolution_m'])
    lower = uvw[context, :2].min(axis=0); upper = uvw[context, :2].max(axis=0)
    ix0, iy0 = np.maximum(0, np.floor((lower-origin)/resolution).astype(int))
    ix1, iy1 = np.minimum(np.array(surface['reference_w'].shape[::-1])-1,
                         np.ceil((upper-origin)/resolution).astype(int))
    xs = np.arange(ix0, ix1+1, max(1, (ix1-ix0)//45))
    ys = np.arange(iy0, iy1+1, max(1, (iy1-iy0)//45))
    xx, yy = np.meshgrid(xs, ys)
    if xx.size:
        reference = np.where(surface['valid_reference'][yy, xx], surface['reference_w'][yy, xx], np.nan)
        view.plot_surface(origin[0]+(xx+.5)*resolution-centre[0],
                          origin[1]+(yy+.5)*resolution-centre[1], reference-centre[2],
                          color='#369a72', alpha=.12, linewidth=0, rcount=46, ccount=46)
    bounds = relative[context].max(axis=0)-relative[context].min(axis=0)
    view.set_box_aspect(np.maximum(bounds, 1e-6))
    view.view_init(elev=28, azim=-62)
    # Centimetre-scale normal variation on a metre-scale face makes Z tick labels overlap.
    # Keep the physical aspect ratio and read depth from the color scale instead.
    view.set_zticks([])
    view.set(title='3D / green: supported local reference surface',
             xlabel='Relative U (m)', ylabel='Relative V (m)', zlabel='Relative W (m)')
    view.xaxis.labelpad = 12
    view.yaxis.labelpad = 12
    figure.colorbar(colored, ax=view, shrink=.6, pad=.08, label='Inward depth (mm)')
    figure.suptitle(f'{damage_id} | geometric spalling candidate | {row["covered_area_m2"]:.3f} m²', fontsize=20)
    coordinate_note = 'Geometry: original E57; location: registered E57.' if registered else 'TLS local coordinates.'
    figure.supxlabel(f'{len(source_ids):,} original points | cell mean depth {row["mean_depth_m"]*1000:.1f} mm | '
                       'Candidate status; manual verification required. '+coordinate_note, fontsize=11)
    figure.savefig(path, dpi=240)
    plt.close(figure)


def export_results(cfg, intermediate, data_dir, output):
    final_rows = []
    grid_allocation_count = 0
    mapping_cfg = cfg.get('mapping', {})
    mapper = None
    registration_info = {}
    if mapping_cfg.get('enabled', False):
        from model_mapping import ModelMapper
        from registered_source import load_registered, candidate_cell_positions
        mapping_root = Path(__file__).resolve().parents[1]
        mapper = ModelMapper(mapping_root/mapping_cfg['model_path'],
                             mapping_cfg['maximum_distance_m'], mapping_cfg['grid_area_m2'])
        print(f'MAPPING: {len(mapper.grids)} Grids; {len(mapper.members)} members; cutoff {mapper.maximum_distance:.2f}m', flush=True)
    # Complete the export in the external work directory before replacing final files.
    with tempfile.TemporaryDirectory(prefix='export-', dir=runtime.WORK) as temporary:
        stage = Path(temporary)
        (stage/'Result/SPL').mkdir(parents=True)
        (stage/'Visualize/SPL').mkdir(parents=True)
        for part in cfg['parts']:
            folder = intermediate/part
            summary = json.loads((folder/'summary.json').read_text())
            xyz, rgb, intensity, source_hash = load_source(part, data_dir)
            if source_hash != summary['source_sha256']:
                raise ValueError(f'{part}: source changed; rerun detection without --export-only')
            with np.load(folder/'point_candidates.npz') as saved:
                candidate = saved['candidate_id']; depth = saved['depth_m']
                basis = saved['frame_basis']; centre = saved['frame_center_xyz']
                point_cell = saved['point_cell'] if mapper else None
                assert not bool(saved['labels_are_ground_truth'])
            if len(candidate) != len(xyz):
                raise ValueError(f'{part}: cached point count differs from source')
            uvw = (xyz-centre)@basis.T
            with np.load(folder/'surface_grid.npz') as saved:
                surface = {key: saved[key] for key in saved.files}
            with (folder/'candidates.csv').open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            positions = xyz
            if mapper:
                registered_path = (mapping_root/mapping_cfg['registered_files'][part]).resolve()
                positions, registration_info[part] = load_registered(part, registered_path, xyz, rgb, intensity)
                regular_ids = [int(row['candidate_id']) for row in rows if row['status'] == 'spalling_candidate']
                cell_ids, cell_positions, cell_areas, cell_depths = candidate_cell_positions(
                    surface, point_cell, positions, regular_ids)
                cell_regions = surface['region_id'].ravel()[cell_ids]
            for raw in rows:
                if raw['status'] != 'spalling_candidate':
                    continue
                row = {key: value if key in {'damage_id', 'status', 'review_flags'} else float(value)
                       for key, value in raw.items()}
                source_ids = np.flatnonzero(candidate == int(row['candidate_id']))
                if len(source_ids) != int(row['point_count']) or not len(source_ids):
                    raise ValueError(f'{part}/{row["damage_id"]}: inconsistent cached point membership')
                damage_id = row['damage_id']
                if not re.fullmatch(r'TLS[12]-\d{4}', damage_id):
                    raise ValueError(f'Unexpected damage ID: {damage_id}')
                las_path = Path('Result/SPL')/(damage_id+'.las')
                png_path = Path('Visualize/SPL')/(damage_id+'.png')
                write_las(stage/las_path, positions[source_ids], rgb[source_ids], depth[source_ids],
                          int(row['candidate_id']), source_ids)
                lower = uvw[source_ids, :2].min(axis=0)-.4
                upper = uvw[source_ids, :2].max(axis=0)+.4
                context = np.flatnonzero(np.isfinite(uvw).all(axis=1) &
                    (uvw[:, 0] >= lower[0]) & (uvw[:, 0] <= upper[0]) &
                    (uvw[:, 1] >= lower[1]) & (uvw[:, 1] <= upper[1]) &
                    (np.abs(uvw[:, 2]-np.median(uvw[source_ids, 2])) < cfg['global_normal_distance_limit_m']))
                snapshot(stage/png_path, damage_id, row, xyz, rgb, uvw, source_ids, context, depth, surface, cfg,
                         registered=mapper is not None)
                result = result_row(part, row, positions, source_ids, data_dir, output)
                if mapper:
                    selected = cell_regions == int(row['candidate_id'])
                    allocations, unmapped = mapper.map_cells(part, damage_id, cell_positions[selected],
                                                            cell_areas[selected], cell_depths[selected])
                    mapped = sum(entry['grid_damage_area_m2'] for entry in allocations)
                    if not np.isclose(mapped+unmapped, float(row['covered_area_m2']), atol=1e-10, rtol=0):
                        raise ValueError(f'{damage_id}: Grid allocations do not conserve damage area')
                    center = positions[source_ids].mean(axis=0)
                    result.update(mapper.map_center(center), coordinate_frame='Dam_model_Z_up_XYZ',
                                  mapped_area_m2=mapped, unmapped_area_m2=unmapped,
                                  mapped_grid_count=len(allocations),
                                  registered_data_path=os.path.relpath(registered_path, output/'Result').replace(os.sep, '/'))
                    if allocations:
                        primary = next((entry for entry in allocations if entry['grid_id'] == result['grid_id']), None)
                        if primary is None and not result['grid_id']:
                            primary = max(allocations, key=lambda entry: entry['grid_damage_area_m2'])
                            result.update({key:primary[key] for key in ('grid_id','grid_guid','member_name','member_guid','grid_area_m2')})
                            result['mapping_status'] = 'partially_mapped_cells'
                        if primary:
                            result.update({key:primary[key] for key in ('grid_damage_area_m2','grid_damage_ratio_pct')})
                        else:
                            result.update(grid_damage_area_m2=0., grid_damage_ratio_pct=0.)
                        if unmapped > 1e-10 and result['mapping_status'] == 'mapped':
                            result['mapping_status'] = 'partially_mapped'
                    elif result['grid_id']:
                        result.update(grid_damage_area_m2=0., grid_damage_ratio_pct=0.,
                                      mapping_status='center_only_no_cells_within_distance')
                    grid_allocation_count += len(allocations)
                final_rows.append(result)
                print(f'EXPORT {damage_id}: {len(source_ids):,} points', flush=True)
        final_rows.sort(key=lambda row: row['damage_id'])
        write_csv(stage/'Result/SPL_result.csv', final_rows, FIELDS)
        if mapper:
            (intermediate/'mapping_summary.json').write_text(json.dumps({
                'model':str(mapper.path.resolve()), 'method':'nearest_point_to_filled_triangle',
                'grid_count':len(mapper.grids), 'member_count':len(mapper.members),
                'maximum_distance_m':mapper.maximum_distance, 'nominal_grid_area_m2':mapper.grid_area,
                'registration':registration_info, 'damage_rows':len(final_rows),
                'grid_allocations':grid_allocation_count,
                'mapped_damage_count':sum(bool(row['grid_id']) for row in final_rows),
                'member_mapped_damage_count':sum(bool(row['member_name']) for row in final_rows),
                'mapped_area_m2':sum(row['mapped_area_m2'] for row in final_rows),
                'unmapped_area_m2':sum(row['unmapped_area_m2'] for row in final_rows)},
                ensure_ascii=False, indent=2)+'\n')
        old_csv = output/'Result/SPL_result.csv'
        old_ids = set()
        if old_csv.exists():
            with old_csv.open(encoding='utf-8-sig') as stream:
                old_ids = {row['damage_id'] for row in csv.DictReader(stream)}
        files = [p for p in stage.rglob('*') if p.is_file()]
        files.sort(key=lambda p: p.suffix == '.csv')
        for source in files:
            target = output/source.relative_to(stage)
            target.parent.mkdir(parents=True, exist_ok=True)
            pending = target.with_suffix(target.suffix+'.tmp')
            shutil.copy2(source, pending)
            pending.replace(target)
        if (output/'Result/SPL_grid_result.csv').exists():
            retired = runtime.WORK/'retired_exports/Result/SPL_grid_result.csv'
            retired.parent.mkdir(parents=True, exist_ok=True)
            (output/'Result/SPL_grid_result.csv').replace(retired)
        # Retire only previously exported owned files; keep unrelated user files.
        for damage_id in old_ids-{row['damage_id'] for row in final_rows}:
            if not re.fullmatch(r'TLS[12]-\d{4}', damage_id):
                continue
            for subdir, suffix in [('Result/SPL', '.las'), ('Visualize/SPL', '.png')]:
                path = output/subdir/(damage_id+suffix)
                if path.exists():
                    retired = runtime.WORK/'retired_exports'/subdir/path.name
                    retired.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(retired)
    print(f'FINAL {len(final_rows)} candidates → {output}', flush=True)
