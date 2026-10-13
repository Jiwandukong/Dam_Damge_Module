"""Match registered coordinates to precise original E57 geometry by record ID."""
import json
from pathlib import Path

import runtime
import numpy as np
import pye57
from local_plane_spalling import sha256


def load_registered(part, path, original_xyz, original_rgb, original_intensity):
    path = Path(path)
    fingerprint = sha256(path)
    cache = runtime.WORK/(part+'_registered_source.npz')
    manifest = cache.with_suffix('.json')
    if cache.is_file() and manifest.is_file():
        info = json.loads(manifest.read_text())
        if info.get('registered_sha256') == fingerprint:
            with np.load(cache) as saved:
                xyz, rgb, intensity = saved['xyz'], saved['rgb'], saved['intensity']
        else:
            xyz = None
    else:
        xyz = None
    if xyz is None:
        with pye57.E57(str(path)) as source:
            if source.scan_count != 1:
                raise ValueError(f'{part}: registered E57 must contain one scan')
            header = source.get_header(0)
            data = source.read_scan_raw(0, ignore_unsupported_fields=True)
            xyz = np.column_stack([data['cartesian'+key] for key in 'XYZ'])
            if header.has_pose():
                xyz = source.to_global(xyz, header.rotation, header.translation)
            rgb = np.column_stack([data['color'+key] for key in ('Red', 'Green', 'Blue')])
            intensity = data.get('intensity', np.zeros(len(xyz), dtype=np.float32))
        np.savez(cache, xyz=xyz, rgb=rgb, intensity=intensity)
    if xyz.shape != original_xyz.shape or not np.isfinite(xyz).all():
        raise ValueError(f'{part}: registered point count/coordinates differ from original')
    if not np.array_equal(rgb, original_rgb) or not np.array_equal(intensity, original_intensity):
        raise ValueError(f'{part}: registered point order cannot be verified; RGB/intensity differ')
    # Validate record correspondence geometrically as well. The fit is only a
    # diagnostic; exports and mapping use the actual registered coordinates.
    p0, q0 = original_xyz.mean(axis=0), xyz.mean(axis=0)
    u, _, vt = np.linalg.svd((original_xyz-p0).T@(xyz-q0))
    rotation = vt.T@np.diag([1, 1, np.linalg.det(vt.T@u.T)])@u.T
    translation = q0-p0@rotation.T
    squared_sum, maximum = 0., 0.
    for start in range(0, len(xyz), 250000):
        residual = xyz[start:start+250000]-(original_xyz[start:start+250000]@rotation.T+translation)
        norm2 = np.sum(residual**2, axis=1)
        squared_sum += float(norm2.sum())
        maximum = max(maximum, float(np.sqrt(norm2.max())))
    rms = float(np.sqrt(squared_sum/len(xyz)))
    if rms > .10 or maximum > .30:
        raise ValueError(f'{part}: original/registered correspondence is inconsistent (RMS {rms:.3f}m, max {maximum:.3f}m)')
    info = {'registered_source':str(path), 'registered_sha256':fingerprint,
            'points':len(xyz), 'RGB_intensity_order_equal':True,
            'rigid_fit_rms_m':rms, 'rigid_fit_max_error_m':maximum,
            'position_source':'actual_registered_E57', 'measurement_source':'original_precise_E57',
            'diagnostic_rotation':rotation.tolist(), 'diagnostic_translation':translation.tolist()}
    manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n')
    print(f'REGISTERED {part}: {len(xyz):,} matched records; geometry-fit RMS={rms:.4f}m', flush=True)
    return xyz, info


def candidate_cell_positions(surface, point_cell, registered_xyz, candidate_ids):
    """Locate analysis cells by their matched registered measurements."""
    regions = surface['region_id'].ravel()
    selected_cells = np.flatnonzero(np.isin(regions, candidate_ids))
    lookup = np.full(len(regions), -1, dtype=np.int32)
    lookup[selected_cells] = np.arange(len(selected_cells))
    valid = point_cell >= 0
    source = np.flatnonzero(valid)
    local = lookup[point_cell[source]]
    source, local = source[local >= 0], local[local >= 0]
    counts = np.bincount(local, minlength=len(selected_cells))
    if np.any(counts == 0):
        raise ValueError('A candidate analysis cell has no matched registered measurements')
    positions = np.column_stack([np.bincount(local, weights=registered_xyz[source, axis],
                                             minlength=len(selected_cells))/counts for axis in range(3)])
    areas = surface['surface_area_factor'].ravel()[selected_cells]*float(surface['resolution_m'])**2
    depths = surface['depth_m'].ravel()[selected_cells]
    return selected_cells, positions, areas, depths
