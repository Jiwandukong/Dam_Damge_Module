"""Demo frame groups on spillway waterline grids and the next lower row."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import re
import numpy as np

SPILLWAY_SECTION = 'SPW_여수로_교각_도수면'
POSITION_ORIGIN = 'synthetic_spillway_similar_frame_layout'
MEMBER_COLUMNS = ['member_id', 'member_node_id']
POSITION_COLUMNS = [
    'position_origin', 'mapping_seed', 'water_level_m', 'submergence_m',
    'surface_node_id', 'surface_mesh_id', 'surface_primitive_id',
    'surface_triangle_id', 'surface_barycentric_json',
    'frame_group_id', 'frame_group_index', 'frame_group_size',
    'frame_offset_x_m', 'surface_local_x_m',
]


def node_matrix(node):
    if 'matrix' in node:
        return np.asarray(node['matrix'], dtype=np.float64).reshape(4, 4, order='F')
    x, y, z, w = node.get('rotation', [0, 0, 0, 1])
    rotation = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation @ np.diag(node.get('scale', [1, 1, 1]))
    matrix[:3, 3] = node.get('translation', [0, 0, 0])
    return matrix


def clip_polygon(polygon, distance):
    """Clip XYZ and its original-triangle barycentric coordinates together."""
    output = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        a, b = float(distance(start[:3])), float(distance(end[:3]))
        if a >= 0:
            output.append(start)
        if (a >= 0) != (b >= 0):
            output.append(start + a / (a-b) * (end-start))
    return np.asarray(output, dtype=np.float64).reshape(-1, 6)


class SpillwayWaterlineGridSampler:
    def __init__(self, path, seed=42, clearance=.25, max_normal_z=2**-.5):
        self.path = Path(path).resolve()
        self.seed = int(seed)
        self.clearance = float(clearance)
        if self.clearance < 0 or not 0 <= max_normal_z < 1:
            raise ValueError('Invalid spillway sampling limits')
        raw = self.path.read_bytes()
        self.model_sha = hashlib.sha256(raw).hexdigest()
        g = json.loads(raw)
        buffers = [(self.path.parent/b['uri']).read_bytes() for b in g['buffers']]
        self.buffer_hashes = {b['uri']: hashlib.sha256(raw).hexdigest()
                              for b, raw in zip(g['buffers'], buffers)}
        cache = {}
        types = {5120:'<i1', 5121:'<u1', 5122:'<i2', 5123:'<u2', 5125:'<u4', 5126:'<f4'}
        widths = {'SCALAR':1, 'VEC2':2, 'VEC3':3, 'VEC4':4}

        def accessor(index):
            if index not in cache:
                a = g['accessors'][index]
                if 'sparse' in a:
                    raise ValueError('Sparse model accessor is unsupported')
                v = g['bufferViews'][a['bufferView']]
                dtype = np.dtype(types[a['componentType']]); width = widths[a['type']]
                cache[index] = np.ndarray((a['count'], width), dtype=dtype,
                    buffer=buffers[v['buffer']],
                    offset=v.get('byteOffset', 0)+a.get('byteOffset', 0),
                    strides=(v.get('byteStride', width*dtype.itemsize), dtype.itemsize)).copy()
            return cache[index]

        self.geometries = []

        def walk(index, parent, names, parent_id=None):
            node = g['nodes'][index]
            matrix = parent @ node_matrix(node)
            name = node.get('name', ''); ancestors = names + [name]
            section = SPILLWAY_SECTION if SPILLWAY_SECTION in ancestors else None
            water = name == '수면'
            extras = node.get('extras', {})
            wall = (section is not None and re.fullmatch(r'SPW_\d+', name) is not None
                    and extras.get('face') == 0 and bool(extras.get('guid')))
            if 'mesh' in node and (water or wall):
                for primitive_id, primitive in enumerate(g['meshes'][node['mesh']]['primitives']):
                    if primitive.get('mode', 4) != 4:
                        raise ValueError('Expected triangle geometry')
                    local = accessor(primitive['attributes']['POSITION']).astype(np.float64)
                    vertices = local @ matrix[:3, :3].T + matrix[:3, 3]
                    # Inverse of the supplied Blender -> glTF axis rotation.
                    vertices = vertices[:, [0, 2, 1]]; vertices[:, 1] *= -1
                    indices = (accessor(primitive['indices']).reshape(-1).astype(np.int64)
                               if 'indices' in primitive else np.arange(len(vertices)))
                    triangles = vertices[indices.reshape(-1, 3)]
                    self.geometries.append(dict(node_id=index, mesh_id=node['mesh'],
                        primitive_id=primitive_id, name=name, section=section, water=water,
                        guid=extras.get('guid', ''), grid_face=extras.get('face'),
                        member_id=g['nodes'][parent_id].get('extras', {}).get('guid', '') if parent_id is not None else '',
                        member_name=g['nodes'][parent_id].get('name', '') if parent_id is not None else '',
                        member_node_id=parent_id,
                        grid_row=extras.get('row'), grid_col=extras.get('col'), triangles=triangles))
            for child in node.get('children', []):
                walk(child, matrix, ancestors, index)

        for index in g['scenes'][g.get('scene', 0)]['nodes']:
            walk(index, np.eye(4), [])

        self.water = []
        for geometry in self.geometries:
            if not geometry['water']:
                continue
            for triangle in geometry['triangles']:
                normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
                norm = np.linalg.norm(normal)
                if norm <= 1e-10 or normal[2]/norm < .99:
                    continue  # Exclude the water volume's vertical and bottom faces.
                origin = triangle[0].copy()
                normal /= norm
                self.water.append(dict(triangle=triangle, normal=normal, origin=origin))
        if not self.water:
            raise ValueError('No upward water-surface triangles in supplied model')
        self.model_water_levels = sorted(set(round(float(w['triangle'][:, 2].mean()), 3)
                                             for w in self.water))
        downstream_level = min(float(w['triangle'][:, 2].mean()) for w in self.water)
        self.water = [w for w in self.water
                      if abs(float(w['triangle'][:, 2].mean())-downstream_level) < .05]

        self.fragments = []
        self.weights = []
        self.eligible_grids = {}
        submerged_candidates = []

        def append_fragments(geometry, water_id, candidates, band):
            water = self.water[water_id]
            triangles = geometry['triangles']
            for triangle_id, polygon in candidates:
                polygon = clip_polygon(polygon, lambda p, w=water:
                                       self.water_height(w, p)-p[2]-self.clearance)
                for i in range(1, len(polygon)-1):
                    fragment = polygon[[0, i, i+1]]
                    area = np.linalg.norm(np.cross(fragment[1, :3]-fragment[0, :3],
                                                   fragment[2, :3]-fragment[0, :3]))/2
                    if area <= 1e-8:
                        continue
                    self.fragments.append((geometry, int(triangle_id), water_id, fragment))
                    self.weights.append(float(area))
                    self.eligible_grids[geometry['name']] = dict(
                        node_id=geometry['node_id'], guid=geometry['guid'],
                        member_id=geometry['member_id'], member_name=geometry['member_name'],
                        member_node_id=geometry['member_node_id'],
                        face=geometry['grid_face'], row=geometry['grid_row'], col=geometry['grid_col'],
                        band=band, model_z_min_m=float(triangles[:, :, 2].min()),
                        model_z_max_m=float(triangles[:, :, 2].max()))

        for geometry in self.geometries:
            if geometry['water']:
                continue
            triangles = geometry['triangles']
            cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            norms = np.linalg.norm(cross, axis=1)
            good = (norms > 1e-10) & (np.abs(cross[:, 2]) <= max_normal_z*norms)
            for water_id, water in enumerate(self.water):
                candidates = []
                intersects_waterline = False
                for triangle_id in np.flatnonzero(good):
                    original = triangles[triangle_id]
                    surface = water['triangle']
                    if (np.any(original[:, :2].max(0) < surface[:, :2].min(0)) or
                        np.any(original[:, :2].min(0) > surface[:, :2].max(0))):
                        continue
                    distances = []
                    for start, end in zip(surface, np.roll(surface, -1, axis=0)):
                        edge = end-start
                        distances.append(lambda p, a=start, e=edge:
                                         e[0]*(p[1]-a[1])-e[1]*(p[0]-a[0]))
                    if any(np.all([distance(p) < 0 for p in original]) for distance in distances):
                        continue
                    polygon = np.column_stack((original, np.eye(3)))
                    for distance in distances:
                        polygon = clip_polygon(polygon, distance)
                        if len(polygon) < 3:
                            break
                    if len(polygon) < 3:
                        continue
                    depths = np.array([self.water_height(water, p[:3])-p[2] for p in polygon])
                    intersects_waterline |= depths.min() <= 1e-8 and depths.max() >= -1e-8
                    candidates.append((int(triangle_id), polygon))
                if not intersects_waterline:
                    submerged_candidates.append((geometry, water_id, candidates))
                    continue
                append_fragments(geometry, water_id, candidates, 'waterline')
        if not self.fragments:
            raise ValueError('No spillway grids cross downstream water with a submerged surface')
        self.waterline_grid_ids = sorted(self.eligible_grids)
        above_cells = {(v['row'], v['col']) for v in self.eligible_grids.values()}
        for geometry, water_id, candidates in submerged_candidates:
            if (geometry['grid_row']-1, geometry['grid_col']) in above_cells:
                append_fragments(geometry, water_id, candidates, 'lower')
        self.lower_grid_ids = sorted(k for k, v in self.eligible_grids.items() if v['band']=='lower')
        if not self.lower_grid_ids:
            raise ValueError('No submerged spillway grid row immediately below the waterline grids')
        self.eligible_grid_ids = sorted(self.eligible_grids)
        self.cumulative = np.cumsum(self.weights)
        self.frame_layout = None
        first = next(g for g in self.geometries if g['name'] == self.waterline_grid_ids[0])
        last = next(g for g in self.geometries if g['name'] == self.waterline_grid_ids[-1])
        self.axis_x = last['triangles'].reshape(-1, 3).mean(0)-first['triangles'].reshape(-1, 3).mean(0)
        self.axis_x[2] = 0
        self.axis_x /= np.linalg.norm(self.axis_x)
        if self.axis_x[0] < 0:
            self.axis_x *= -1
        # Preserve the previous origin, horizontal anchors and upper-row coordinates.
        vertices = np.vstack([fragment[:, :3] for geometry, _, _, fragment in self.fragments
                              if geometry['name'] in self.waterline_grid_ids])
        self.local_origin = vertices.mean(0)
        local_x = (vertices-self.local_origin) @ self.axis_x
        self.local_x_limits = (float(local_x.min()), float(local_x.max()))

    def prepare_frames(self, paths, frame_step=.25, component_span=.8):
        """Give similar frames a shared anchor and constant-height horizontal offsets."""
        from frame_grouping import GROUPING_PARAMETERS, group_similar_frames
        from PIL import Image
        if frame_step <= 0 or component_span < 0:
            raise ValueError('Frame step must be positive and component span nonnegative')
        paths = list(map(Path, paths))
        groups, comparisons = group_similar_frames(paths, self.seed)
        low, high = self.local_x_limits
        low += .5; high -= .5
        cell_width = (high-low)/len(groups)
        if cell_width <= component_span+.2:
            raise ValueError('Too many frame groups for the spillway display width')
        largest = max(len(g) for g in groups)
        self.frame_step = min(float(frame_step), (cell_width-component_span-.2)/max(largest-1, 1))
        self.component_span = float(component_span)
        self.frame_layout = {}
        self.frame_groups = []
        level = float(np.mean([w['triangle'][:, 2].mean() for w in self.water]))
        lower_height_min = max(self.eligible_grids[k]['model_z_min_m'] for k in self.lower_grid_ids)+.25
        lower_height_max = min(self.eligible_grids[k]['model_z_max_m'] for k in self.lower_grid_ids)-.25
        if lower_height_min >= lower_height_max:
            raise ValueError('Lower grid row has no common height range for frame groups')
        for group_index, names in enumerate(groups):
            group_id = f'G{group_index+1:03d}'
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
                f'{self.seed}:{group_id}:{names[0]}'.encode()).digest()[:8], 'little'))
            width = (len(names)-1)*self.frame_step+self.component_span
            jitter = max(0., (cell_width-width)/2-.1)
            anchor = low+(group_index+.5)*cell_width+rng.uniform(-jitter, jitter)
            band = 'lower' if group_index % 3 == 1 else 'waterline'
            height = (rng.uniform(lower_height_min, lower_height_max) if band=='lower'
                      else level-rng.uniform(.65, 1.25))
            self.frame_groups.append(dict(group_id=group_id, images=names,
                anchor_local_x_m=float(anchor), anchor_z_m=float(height), grid_band=band))
            for index, name in enumerate(names):
                offset = index*self.frame_step
                self.frame_layout[name] = dict(group_id=group_id, index=index,
                    size=len(names), offset_x_m=offset,
                    center_local_x_m=float(anchor+offset-(len(names)-1)*self.frame_step/2),
                    z_m=float(height), grid_band=band)
        self.frame_image_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        self.frame_image_widths = {}
        for path in paths:
            with Image.open(path) as image:
                self.frame_image_widths[path.name] = image.width
        self.frame_comparisons = comparisons
        self.grouping_parameters = GROUPING_PARAMETERS.copy()

    def surface_position(self, local_x, height):
        """Find the exact grid triangle at a wall-horizontal coordinate and fixed Z."""
        for geometry, triangle_id, water_id, fragment in self.fragments:
            xs = (fragment[:, :3]-self.local_origin) @ self.axis_x
            zs = fragment[:, 2]
            if not xs.min()-1e-9 <= local_x <= xs.max()+1e-9 or not zs.min()-1e-9 <= height <= zs.max()+1e-9:
                continue
            matrix = np.vstack((xs, zs, np.ones(3)))
            if abs(np.linalg.det(matrix)) < 1e-10:
                continue
            weights = np.linalg.solve(matrix, [local_x, height, 1.])
            if weights.min() >= -1e-9:
                barycentric = weights @ fragment[:, 3:]
                point = barycentric @ geometry['triangles'][triangle_id]
                return geometry, triangle_id, water_id, barycentric, point
        raise ValueError(f'Frame layout is outside eligible spillway grid surfaces: x={local_x}, z={height}')

    @staticmethod
    def water_height(water, point):
        n, origin = water['normal'], water['origin']
        return float(origin[2]-(n[0]*(point[0]-origin[0])+n[1]*(point[1]-origin[1]))/n[2])

    def sample(self, row):
        if self.frame_layout is None:
            raise ValueError('Call prepare_frames before assigning grouped display positions')
        frame = self.frame_layout[row['image']]
        # A bounded horizontal component offset keeps multiple damages from a frame together.
        nodes = np.asarray(json.loads(row['pixel_nodes_json']))
        pixel_x = float(nodes[:, 0].mean())
        local_x = frame['center_local_x_m']+self.component_span*(pixel_x/self.frame_image_widths[row['image']]-.5)
        geometry, triangle_id, water_id, barycentric, point = self.surface_position(local_x, frame['z_m'])
        level = self.water_height(self.water[water_id], point)
        assert level-point[2] >= self.clearance-1e-8
        assert geometry['name'] in self.eligible_grid_ids
        assert self.eligible_grids[geometry['name']]['band']==frame['grid_band']
        row.update(world_center_x_m=float(point[0]), world_center_y_m=float(point[1]),
                   world_center_z_m=float(point[2]), member_name=geometry['member_name'],
                   member_id=geometry['member_id'], member_node_id=geometry['member_node_id'],
                   section_name=geometry['section'], grid_id=geometry['name'],
                   grid_guid=geometry['guid'], mapping_status='demo_random',
                   position_origin=POSITION_ORIGIN, mapping_seed=self.seed,
                   water_level_m=level, submergence_m=level-float(point[2]),
                   surface_node_id=geometry['node_id'], surface_mesh_id=geometry['mesh_id'],
                   surface_primitive_id=geometry['primitive_id'], surface_triangle_id=triangle_id,
                   surface_barycentric_json=json.dumps(barycentric.tolist(), separators=(',', ':')),
                   frame_group_id=frame['group_id'], frame_group_index=frame['index'],
                   frame_group_size=frame['size'], frame_offset_x_m=frame['offset_x_m'],
                   surface_local_x_m=local_x)
        return row

    def summary(self):
        section_areas = {}
        grid_areas = {}
        for (geometry, _, _, _), area in zip(self.fragments, self.weights):
            section_areas[geometry['section']] = section_areas.get(geometry['section'], 0.)+area
            grid_areas[geometry['name']] = grid_areas.get(geometry['name'], 0.)+area
        result = dict(purpose='synthetic display positions; not measured damage locations',
            mapping_status='demo_random', position_origin=POSITION_ORIGIN,
            scope='spillway_waterline_and_next_lower_grid_row', seed=self.seed,
            model=str(self.path), model_sha256=self.model_sha, buffer_sha256=self.buffer_hashes,
            coordinate_system='EPSG:5186 X/Y, model altitude Z; glTF=(X,Z,-Y)',
            grid_selection='SPW discharge face 0; downstream waterline grids and same-column next lower row within water footprint',
            water_selection='lowest upward water-surface level in supplied model (downstream)',
            minimum_submergence_m=self.clearance, eligible_surface_area_m2=float(self.cumulative[-1]),
            clipped_triangle_fragments=len(self.fragments), eligible_section_areas_m2=section_areas,
            eligible_grid_count=len(self.eligible_grid_ids), eligible_grid_ids=self.eligible_grid_ids,
            eligible_grid_metadata=self.eligible_grids, eligible_grid_areas_m2=grid_areas,
            waterline_grid_ids=self.waterline_grid_ids, lower_grid_ids=self.lower_grid_ids,
            lower_row_group_allocation='every third frame group, starting with G002; keep each group in one row',
            selected_water_levels_m=sorted(set(round(float(w['triangle'][:, 2].mean()), 3) for w in self.water)),
            model_water_levels_m=self.model_water_levels,
            local_x_axis_world_xyz=self.axis_x.tolist(), local_origin_world_xyz=self.local_origin.tolist())
        if self.frame_layout is not None:
            result.update(layout_strategy='similar_frames_along_spillway_local_x',
                frame_step_m=self.frame_step, within_frame_horizontal_span_m=self.component_span,
                frame_group_count=len(self.frame_groups),
                multi_frame_group_count=sum(len(g['images']) > 1 for g in self.frame_groups),
                frame_groups=self.frame_groups, frame_layout=self.frame_layout,
                frame_image_sha256=self.frame_image_hashes,
                grouping_parameters=self.grouping_parameters, adjacent_frame_comparisons=self.frame_comparisons)
        return result
