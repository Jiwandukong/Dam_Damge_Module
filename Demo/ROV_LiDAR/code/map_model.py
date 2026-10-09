"""Associate measured XYZ with native glTF grids; source coordinates are unchanged."""
from pathlib import Path
import json
import numpy as np
MAPPING_FIELDS = ["grid_id", "member_name"]

def node_matrix(node):
    if 'matrix' in node:
        return np.asarray(node['matrix'], dtype=np.float64).reshape(4, 4, order='F')
    x, y, z, w = node.get('rotation', [0, 0, 0, 1])
    rotation = np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w], [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w], [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation @ np.diag(node.get('scale', [1, 1, 1]))
    matrix[:3, 3] = node.get('translation', [0, 0, 0])
    return matrix

class GridMapper:

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.model = json.loads(self.path.read_text())
        model = self.model
        buffers = [(self.path.parent / b['uri']).read_bytes() for b in model['buffers']]
        cache = {}
        types = {5120: '<i1', 5121: '<u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
        widths = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}

        def accessor(index):
            if index not in cache:
                item = model['accessors'][index]
                if 'sparse' in item:
                    raise ValueError('Sparse model accessors are unsupported')
                view = model['bufferViews'][item['bufferView']]
                dtype = np.dtype(types[item['componentType']])
                width = widths[item['type']]
                cache[index] = np.ndarray((item['count'], width), dtype=dtype, buffer=buffers[view['buffer']], offset=view.get('byteOffset', 0) + item.get('byteOffset', 0), strides=(view.get('byteStride', width * dtype.itemsize), dtype.itemsize)).copy()
            return cache[index]
        self.grids = {}
        triangles, node_ids = ([], [])

        def walk(index, parent_matrix, parent_id):
            node = model['nodes'][index]
            matrix = parent_matrix @ node_matrix(node)
            extras = node.get('extras', {})
            if 'mesh' in node and all((k in extras for k in ('face', 'row', 'col'))):
                parent = model['nodes'][parent_id] if parent_id is not None else {}
                self.grids[index] = {'grid_id': node.get('name', ''), 'grid_guid': extras.get('guid', ''), 'member_id': parent.get('extras', {}).get('guid', ''), 'member_name': parent.get('name', ''), 'node_id': index, 'member_node_id': parent_id, 'face': extras['face'], 'row': extras['row'], 'col': extras['col']}
                for primitive in model['meshes'][node['mesh']]['primitives']:
                    if primitive.get('mode', 4) != 4:
                        raise ValueError('Native grids must use triangle geometry')
                    local = accessor(primitive['attributes']['POSITION']).astype(np.float64)
                    vertices = local @ matrix[:3, :3].T + matrix[:3, 3]
                    vertices = vertices[:, [0, 2, 1]]
                    vertices[:, 1] *= -1
                    indices = accessor(primitive['indices']).reshape(-1).astype(np.int64) if 'indices' in primitive else np.arange(len(vertices))
                    triangle = vertices[indices.reshape(-1, 3)]
                    triangles.append(triangle)
                    node_ids.append(np.full(len(triangle), index, dtype=np.int32))
            for child in node.get('children', []):
                walk(child, matrix, index)
        for root in model['scenes'][model.get('scene', 0)]['nodes']:
            walk(root, np.eye(4), None)
        if not triangles:
            raise ValueError('The supplied model has no native grid triangles')
        self.triangles = np.concatenate(triangles)
        self.node_ids = np.concatenate(node_ids)
        self.a = self.triangles[:, 0]
        self.ab = self.triangles[:, 1] - self.a
        self.ac = self.triangles[:, 2] - self.a
        self.denominator = self.ab[:, 0] * self.ac[:, 1] - self.ab[:, 1] * self.ac[:, 0]
        self.projectable = np.abs(self.denominator) > 1e-12

    def associate(self, center):
        point = np.asarray(center, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Damage center must contain three finite XYZ coordinates')
        offset = point[:2] - self.a[:, :2]
        denominator = np.where(self.projectable, self.denominator, 1)
        u = (offset[:, 0] * self.ac[:, 1] - offset[:, 1] * self.ac[:, 0]) / denominator
        v = (self.ab[:, 0] * offset[:, 1] - self.ab[:, 1] * offset[:, 0]) / denominator
        inside = self.projectable & (u >= -1e-08) & (v >= -1e-08) & (u + v <= 1 + 1e-08)
        candidates = np.flatnonzero(inside)
        if not len(candidates):
            return (dict.fromkeys(MAPPING_FIELDS, ''), {'status': 'outside_grid_XY_footprints', 'measured_center_xyz': point.tolist()})
        height = self.a[:, 2] + u * self.ab[:, 2] + v * self.ac[:, 2]
        order = np.lexsort((candidates, self.node_ids[candidates], np.abs(height[candidates] - point[2])))
        index = int(candidates[order[0]])
        grid = self.grids[int(self.node_ids[index])]
        fields = {name: grid[name] for name in MAPPING_FIELDS}
        detail = dict(grid, status='mapped_by_vertical_projection', measured_center_xyz=point.tolist(), projected_center_xyz=[float(point[0]), float(point[1]), float(height[index])], vertical_offset_m=float(point[2] - height[index]), triangle_index=index, barycentric=[float(1 - u[index] - v[index]), float(u[index]), float(v[index])])
        return (fields, detail)
