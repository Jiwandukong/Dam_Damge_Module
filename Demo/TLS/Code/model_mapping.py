"""Nearest triangle mapping in the fixed Dam_model Z-up XYZ frame."""
import json
from pathlib import Path

import runtime
import numpy as np
from scipy.spatial import cKDTree


def node_matrix(node):
    if 'matrix' in node:
        return np.asarray(node['matrix'], dtype=float).reshape(4, 4, order='F')
    x, y, z, w = node.get('rotation', [0, 0, 0, 1])
    result = np.eye(4)
    result[:3, :3] = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y],
    ])@np.diag(node.get('scale', [1, 1, 1]))
    result[:3, 3] = node.get('translation', [0, 0, 0])
    return result


def triangle_distance_squared(points, triangles):
    """Distance to the filled triangle, including its edges and vertices."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, v = b-a, c-a, points-a
    aa = np.einsum('ij,ij->i', ab, ab)
    bb = np.einsum('ij,ij->i', ab, ac)
    cc = np.einsum('ij,ij->i', ac, ac)
    av = np.einsum('ij,ij->i', v, ab)
    cv = np.einsum('ij,ij->i', v, ac)
    determinant = aa*cc-bb*bb
    valid = determinant > np.finfo(float).eps*np.maximum(aa*cc, 1e-30)*16
    u = np.divide(cc*av-bb*cv, determinant, out=np.zeros_like(aa), where=valid)
    w = np.divide(aa*cv-bb*av, determinant, out=np.zeros_like(aa), where=valid)
    projection = a+u[:, None]*ab+w[:, None]*ac
    interior = valid & (u >= 0) & (w >= 0) & (u+w <= 1)
    result = np.where(interior, np.sum((projection-points)**2, axis=1), np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end-start
        length = np.einsum('ij,ij->i', edge, edge)
        t = np.clip(np.divide(np.einsum('ij,ij->i', points-start, edge), length,
                              out=np.zeros_like(length), where=length > 0), 0, 1)
        result = np.minimum(result, np.sum((start+t[:, None]*edge-points)**2, axis=1))
    return result


class TriangleIndex:
    def __init__(self, triangles, owners):
        self.origin = (triangles.min(axis=(0, 1))+triangles.max(axis=(0, 1)))/2
        self.triangles = triangles-self.origin
        self.owners = np.asarray(owners, dtype=np.int32)
        self.centres = self.triangles.mean(axis=1)
        self.tree = cKDTree(self.centres)
        self.radius = np.linalg.norm(self.triangles-self.centres[:, None], axis=2).max()
        self.lower = self.triangles.min(axis=1)
        self.upper = self.triangles.max(axis=1)

    def nearest(self, points, maximum_distance=None):
        """Exact nearest triangle; a cutoff bounds candidate search, not accuracy."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)-self.origin
        if not np.isfinite(points).all():
            raise ValueError('Mapping coordinates must be finite')
        owners = np.full(len(points), -1, dtype=np.int32)
        distances = np.full(len(points), np.inf)
        if maximum_distance is None:
            # AABB lower bounds certify the unrestricted nearest result. This
            # avoids assuming that a triangle's centroid is its nearest point.
            for i, point in enumerate(points):
                _, initial = self.tree.query(point, k=min(16, len(self.triangles)))
                initial = np.atleast_1d(initial)
                best = np.min(triangle_distance_squared(np.broadcast_to(point, (len(initial), 3)),
                                                       self.triangles[initial]))
                delta = np.maximum(np.maximum(self.lower-point, point-self.upper), 0)
                indices = np.flatnonzero(np.einsum('ij,ij->i', delta, delta) <= best+1e-12)
                d = triangle_distance_squared(np.broadcast_to(point, (len(indices), 3)), self.triangles[indices])
                j = int(np.argmin(d))
                owners[i] = self.owners[indices[j]]
                distances[i] = np.sqrt(d[j])
            return owners, distances
        limit = float(maximum_distance)
        if not np.isfinite(limit) or limit < 0:
            raise ValueError('Maximum mapping distance must be finite and nonnegative')
        # Every triangle within the cutoff has its centroid within cutoff plus
        # its radius, so this search cannot miss a valid triangle.
        for start in range(0, len(points), 128):
            chunk = points[start:start+128]
            candidates = self.tree.query_ball_point(chunk, limit+self.radius+1e-9)
            for j, indices in enumerate(candidates):
                if not indices:
                    continue
                indices = np.asarray(indices, dtype=np.int32)
                point = chunk[j]
                delta = np.maximum(np.maximum(self.lower[indices]-point, point-self.upper[indices]), 0)
                indices = indices[np.einsum('ij,ij->i', delta, delta) <= limit**2+1e-12]
                if not len(indices):
                    continue
                d = triangle_distance_squared(np.broadcast_to(point, (len(indices), 3)), self.triangles[indices])
                k = int(np.argmin(d))
                if d[k] <= limit**2+1e-12:
                    owners[start+j] = self.owners[indices[k]]
                    distances[start+j] = np.sqrt(d[k])
        return owners, distances


class ModelMapper:
    def __init__(self, path, maximum_distance=1.0, grid_area=25.0):
        self.path = Path(path)
        self.maximum_distance = float(maximum_distance)
        self.grid_area = float(grid_area)
        if not np.isfinite([self.maximum_distance, self.grid_area]).all() or self.maximum_distance < 0 or self.grid_area <= 0:
            raise ValueError('Invalid mapping distance or nominal Grid area')
        model = json.loads(self.path.read_text())
        buffers = [(self.path.parent/b['uri']).read_bytes() for b in model['buffers']]
        dtypes = {5120:'i1', 5121:'u1', 5122:'<i2', 5123:'<u2', 5125:'<u4', 5126:'<f4'}
        widths = {'SCALAR':1, 'VEC2':2, 'VEC3':3, 'VEC4':4}
        cache = {}

        def accessor(index):
            if index not in cache:
                a = model['accessors'][index]
                if 'sparse' in a:
                    raise ValueError('Sparse model accessors are unsupported')
                view = model['bufferViews'][a['bufferView']]
                dtype = np.dtype(dtypes[a['componentType']])
                width = widths[a['type']]
                cache[index] = np.ndarray((a['count'], width), dtype=dtype,
                    buffer=buffers[view['buffer']], offset=view.get('byteOffset', 0)+a.get('byteOffset', 0),
                    strides=(view.get('byteStride', width*dtype.itemsize), dtype.itemsize))
            return cache[index]

        grid_triangles, grid_owners, member_triangles, member_owners = [], [], [], []
        self.grids, self.members = [], []
        member_lookup = {}

        def walk(index, parent_transform, member=None):
            node = model['nodes'][index]
            transform = parent_transform@node_matrix(node)
            extras = node.get('extras', {})
            # Named groups with a GUID define the structural members. Leaf
            # mesh GUIDs identify their individual surfaces instead.
            if 'children' in node and extras.get('guid') and node.get('name'):
                member = (node['name'], extras['guid'])
            if 'mesh' in node:
                identity = member or (node.get('name', ''), extras.get('guid', ''))
                if identity not in member_lookup:
                    member_lookup[identity] = len(self.members)
                    self.members.append({'member_name':identity[0], 'member_guid':identity[1]})
                member_id = member_lookup[identity]
                grid = all(k in extras for k in ('face', 'row', 'col'))
                if grid:
                    grid_id = len(self.grids)
                    self.grids.append({'grid_id':node.get('name', ''), 'grid_guid':extras.get('guid', ''),
                                       **self.members[member_id], 'member_index':member_id,
                                       'model_grid_area_m2':extras.get('area', '')})
                for primitive in model['meshes'][node['mesh']]['primitives']:
                    if primitive.get('mode', 4) != 4:
                        raise ValueError('Expected triangulated model')
                    pos = accessor(primitive['attributes']['POSITION']).astype(float)
                    pos = pos@transform[:3, :3].T+transform[:3, 3]
                    pos = pos[:, [0, 2, 1]]*np.array([1, -1, 1])
                    indices = accessor(primitive['indices']).ravel() if 'indices' in primitive else np.arange(len(pos))
                    triangles = pos[indices].reshape(-1, 3, 3)
                    member_triangles.append(triangles)
                    member_owners.append(np.full(len(triangles), member_id, dtype=np.int32))
                    if grid:
                        grid_triangles.append(triangles)
                        grid_owners.append(np.full(len(triangles), grid_id, dtype=np.int32))
            for child in node.get('children', []):
                walk(child, transform, member)

        for root in model['scenes'][model.get('scene', 0)]['nodes']:
            walk(root, np.eye(4))
        if not grid_triangles or not member_triangles:
            raise ValueError('The model has no Grid/member triangles')
        self.grid_index = TriangleIndex(np.concatenate(grid_triangles), np.concatenate(grid_owners))
        self.member_index = TriangleIndex(np.concatenate(member_triangles), np.concatenate(member_owners))

    def map_center(self, center):
        grid, gd = self.grid_index.nearest([center])
        member, md = self.member_index.nearest([center])
        grid_ok, member_ok = gd[0] <= self.maximum_distance, md[0] <= self.maximum_distance
        result = {'grid_id':'', 'grid_guid':'', 'member_name':'', 'member_guid':'',
                  'nearest_grid_distance_m':float(gd[0]), 'nearest_member_distance_m':float(md[0]),
                  'mapping_status':'unmapped_distance', 'grid_area_m2':'',
                  'grid_damage_area_m2':'', 'grid_damage_ratio_pct':'',
                  'mapping_max_distance_m':self.maximum_distance}
        if member_ok:
            result.update(self.members[member[0]], mapping_status='member_only')
        if grid_ok:
            selected = self.grids[grid[0]]
            result.update(grid_id=selected['grid_id'], grid_guid=selected['grid_guid'],
                          grid_area_m2=self.grid_area, mapping_status='mapped')
        return result

    def map_cells(self, part, damage_id, positions, areas, depths):
        grid_ids, distances = self.grid_index.nearest(positions, self.maximum_distance)
        rows = []
        for grid in np.unique(grid_ids[grid_ids >= 0]):
            selected = grid_ids == grid
            area = float(areas[selected].sum())
            identity = self.grids[grid]
            rows.append({'dataset':part, 'damage_id':damage_id,
                         **{k:identity[k] for k in ('grid_id','grid_guid','member_name','member_guid')},
                         'grid_area_m2':self.grid_area, 'grid_damage_area_m2':area,
                         'grid_damage_ratio_pct':area/self.grid_area*100,
                         'mean_depth_m':float(np.average(depths[selected], weights=areas[selected])),
                         'max_depth_m':float(depths[selected].max()),
                         'assigned_cell_count':int(selected.sum()),
                         'mean_mapping_distance_m':float(np.average(distances[selected], weights=areas[selected])),
                         'max_mapping_distance_m':float(distances[selected].max())})
        return rows, float(areas[grid_ids < 0].sum())
