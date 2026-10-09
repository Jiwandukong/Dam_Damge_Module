"""Align the metric survey frame to the dam face with constant along-line world Z."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from review_positions import MODEL, closest_on_triangle, load_triangles, sha

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'Code'))
from coordinate_mapping import DEFAULT_TRANSFORM, horizontal_frame, load_transform
from scan_geometry import dzt_scan_geometry

GRID_NAMES = [f'NOF_R_{i:04d}' for i in range(157, 164)]


def first_boundary(origin, direction, triangles):
    a = triangles[:, 0]
    e1, e2 = triangles[:, 1] - a, triangles[:, 2] - a
    h = np.cross(direction, e2)
    det = np.sum(e1 * h, axis=1)
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=np.abs(det) > 1e-12)
    s = origin - a
    u = inv * np.sum(s * h, axis=1)
    q = np.cross(s, e1)
    v = inv * (q @ direction)
    distance = inv * np.sum(e2 * q, axis=1)
    valid = (np.abs(det) > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (distance > 0)
    if not valid.any():
        raise ValueError('The horizontal dam-face ray does not meet the equipment mesh')
    return float(distance[valid].min())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proposal', type=Path, default=HERE / 'anchor_proposal.json')
    args = parser.parse_args()
    previous = json.loads(args.previous.read_text())
    if previous.get('model_anchors_applied'):
        raise ValueError('Selected endpoints are already applied. Use Code/fit_start_end.py with selected_anchors.json to refit them; rebuilding the provisional direction would displace the selected end.')
    if previous['scale'] != 1:
        raise ValueError('Fix distance units before aligning the survey direction')
    _, meshes = load_triangles(MODEL)
    grids = [m for m in meshes if m['name'] in GRID_NAMES]
    if set(m['name'] for m in grids) != set(GRID_NAMES):
        raise ValueError('The reference dam-face meshes are missing')
    triangles = np.concatenate([m['triangles'] for m in grids])
    area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(area_vectors, axis=1)
    old_normal = np.array(previous['R'])[:, 2]
    area_vectors[(area_vectors @ old_normal) < 0] *= -1
    normal = area_vectors.sum(axis=0)
    normal /= np.linalg.norm(normal)
    along = np.cross(normal, [0., 0., 1.])
    along /= np.linalg.norm(along)
    if along @ np.array(previous['R'])[:, 0] < 0:
        along *= -1
    along[2] = 0.0
    across = np.cross(normal, along)
    if across @ np.array(previous['R'])[:, 1] < 0:
        across *= -1
    rotation = horizontal_frame(along, across)
    normals = area_vectors / area[:, None]
    max_tangent_error = float(np.degrees(np.arcsin(np.clip(np.abs(normals @ along), 0, 1))).max())
    transform = {**previous, 'R': rotation, 'horizontal_z_locked': True,
        'provenance': 'Metric review draft. Rotation follows the area-weighted dam-face normal of NOF_R_0157~0163, with an exactly horizontal along-line axis. Scale=1 and the previous origin are retained; exact field endpoints remain unconfirmed.',
        'mesh_alignment': {'model_sha256': sha(MODEL), 'reference_grids': GRID_NAMES,
            'reference_normal_world': normal.tolist(), 'horizontal_direction_world': along.tolist(),
            'method': 'Area-weighted triangle normal; horizontal tangent is normal cross world Z',
            'max_triangle_horizontal_tangent_angle_deg': max_tangent_error,
            'absolute_origin_verified': False}}
    args.output.write_text(json.dumps(transform, ensure_ascii=False, indent=2) + '\n')
    load_transform(args.output)

    grid_triangles = np.concatenate([m['triangles'] for m in grids if m['name'] == 'NOF_R_0157'])
    grid_weights = np.linalg.norm(np.cross(grid_triangles[:, 1]-grid_triangles[:, 0],
                                           grid_triangles[:, 2]-grid_triangles[:, 0]), axis=1)
    centroid = np.average(grid_triangles.mean(axis=1), axis=0, weights=grid_weights)
    equipment = np.concatenate([m['triangles'] for m in meshes if m['name'] == '옥외_부대설비'])
    boundary_distance = first_boundary(centroid, along, equipment)
    scan = HERE.parent / 'Data/processed data/400MHz/400_LINE_001.DZT'
    measured_length = dzt_scan_geometry(scan)['total_distance_m']
    end = centroid + boundary_distance * along
    start = end - measured_length * along
    start[2] = end[2] = centroid[2]

    def point(xyz, mesh):
        return {'world_xyz': xyz.tolist(), 'gltf_xyz': [float(xyz[0]), float(xyz[2]), float(-xyz[1])], 'mesh': mesh}

    proposal = {'status': 'unconfirmed_model_boundary_proposal',
        'start': point(start, 'NOF_R_0157'), 'end': point(end, '옥외_부대설비'),
        'grid_surface_centroid': point(centroid, 'NOF_R_0157'),
        'reference_frequency_mhz': 400, 'reference_line_no': 1,
        'measured_length_m': measured_length, 'horizontal_z_locked': True,
        'endpoint_delta_z_m': 0.0, 'grid_centroid_to_equipment_boundary_m': boundary_distance,
        'start_distance_from_grid_surface_m': min(float(np.linalg.norm(closest_on_triangle(start, t)-start)) for t in grid_triangles),
        'model_sha256': sha(MODEL), 'horizontal_direction_world': along.tolist(),
        'method': 'Guide only: horizontal mesh-tangent ray at the NOF_R_0157 centroid height; first equipment boundary hit is the end; measured length is traced backwards for the start. Same world Z at both ends. No field anchor or equipment clearance is asserted.',
        'current_start_world_xyz': transform['t']}
    args.proposal.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'horizontal_direction_world': along.tolist(), 'endpoint_delta_z_m': 0,
        'max_triangle_horizontal_tangent_angle_deg': max_tangent_error,
        'grid_centroid_to_equipment_boundary_m': boundary_distance,
        'origin_retained': transform['t'] == previous['t']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
