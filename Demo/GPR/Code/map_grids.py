"""Assign GPR display points to the closest dam grid and its parent member."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve().parent
GPR = HERE.parent
MODEL = GPR.parent / "Dam_model/Daecheongdam/daecheongdam_regions_grid5m.gltf"
CSV = GPR / "Output/Result/ANM_result.csv"
DTYPES = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_triangles(path):
    model = json.loads(path.read_text())
    buffers = [(path.parent / b["uri"]).read_bytes() for b in model["buffers"]]
    cache = {}

    def accessor(index):
        if index not in cache:
            a = model["accessors"][index]
            if "sparse" in a:
                raise ValueError("Sparse accessor is unsupported")
            view = model["bufferViews"][a["bufferView"]]
            dtype = np.dtype(DTYPES[a["componentType"]])
            width = WIDTHS[a["type"]]
            cache[index] = np.ndarray((a["count"], width), dtype=dtype, buffer=buffers[view["buffer"]],
                offset=view.get("byteOffset", 0) + a.get("byteOffset", 0),
                strides=(view.get("byteStride", width * dtype.itemsize), dtype.itemsize)).copy()
        return cache[index]

    def local(node):
        if "matrix" in node:
            return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4, order="F")
        x, y, z, w = node.get("rotation", [0, 0, 0, 1])
        r = np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                      [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                      [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])
        matrix = np.eye(4)
        matrix[:3, :3] = r @ np.diag(node.get("scale", [1, 1, 1]))
        matrix[:3, 3] = node.get("translation", [0, 0, 0])
        return matrix

    meshes = []

    def walk(index, parent, parent_name):
        node = model["nodes"][index]
        world = parent @ local(node)
        name = node.get("name", "")
        if "mesh" in node:
            mesh = model["meshes"][node["mesh"]]
            for primitive in mesh["primitives"]:
                if primitive.get("mode", 4) != 4:
                    raise ValueError("Only triangle primitives are supported")
                positions = accessor(primitive["attributes"]["POSITION"]).astype(np.float64)
                positions = positions @ world[:3, :3].T + world[:3, 3]
                # glTF (X,Z,-Y) -> world (X,Y,Z), before float32 scene insertion.
                positions = positions[:, [0, 2, 1]] * [1, -1, 1]
                indices = accessor(primitive["indices"]).reshape(-1).astype(np.int64) if "indices" in primitive else np.arange(len(positions))
                triangles = positions[indices.reshape(-1, 3)]
                valid = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1) > 1e-12
                triangles = triangles[valid]
                meshes.append(dict(name=name or mesh.get("name", ""), parent=parent_name,
                    grid=bool(re.fullmatch(r"(NOF_[LR]|SPW)_\d+", name)), triangles=triangles))
        for child in node.get("children", []):
            walk(child, world, name or parent_name)

    for index in model["scenes"][model.get("scene", 0)]["nodes"]:
        walk(index, np.eye(4), "")
    return model, meshes


def closest_on_triangle(point, tri):
    # Refine the raycasting result using float64 world coordinates.
    a, b, c = tri
    ab, ac, ap = b-a, c-a, point-a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = point-b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1*d4-d3*d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + d1/(d1-d3)*ab
    cp = point-c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5*d2-d1*d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + d2/(d2-d6)*ac
    va = d3*d6-d5*d4
    if va <= 0 and d4-d3 >= 0 and d5-d6 >= 0:
        return b + (d4-d3)/(d4-d3+d5-d6)*(c-b)
    denominator = 1/(va+vb+vc)
    return a + vb*denominator*ab + vc*denominator*ac


def inspect(rows, meshes, kind, center):
    selected = [m for m in meshes if kind == "all" or m["grid"] == (kind == "grid")]
    triangles = np.concatenate([m["triangles"] for m in selected])
    owners = np.repeat(np.arange(len(selected)), [len(m["triangles"]) for m in selected])
    relative = (triangles-center).reshape(-1, 3).astype(np.float32)
    faces = np.arange(len(relative), dtype=np.uint32).reshape(-1, 3)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(relative), o3d.core.Tensor(faces))
    xyz = np.array([[float(row["world_center_"+a+"_m"]) for a in "xyz"] for row in rows])
    primitive_ids = scene.compute_closest_points(o3d.core.Tensor((xyz-center).astype(np.float32)))["primitive_ids"].numpy()
    details = []
    for row, point, primitive in zip(rows, xyz, primitive_ids):
        mesh = selected[int(owners[primitive])]
        nearest = closest_on_triangle(point, triangles[primitive])
        details.append(dict(damage_id=row["damage_id"], distance_m=float(np.linalg.norm(nearest-point)),
            nearest_node=mesh["name"], parent_node=mesh["parent"], nearest_point_xyz=nearest.tolist(),
            candidate_xyz=point.tolist(), delta_xyz=(nearest-point).tolist()))
    distances = [d["distance_m"] for d in details]
    return dict(triangle_count=len(triangles), min_distance_m=min(distances), median_distance_m=float(np.median(distances)),
                max_distance_m=max(distances), nearest_nodes=dict(Counter(d["nearest_node"] for d in details)), details=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV)
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("GPR_WORK_DIR", str(Path.home()/".cache/dam_damage_module/gpr"))) / "grid_mapping_report.json")
    parser.add_argument("--dam-model", type=Path, default=MODEL)
    parser.add_argument("--preview-transform", type=Path)
    parser.add_argument('--apply-grid-mapping',action='store_true',
                        help='Assign nearby grid triangles after user-selected model anchors are applied')
    parser.add_argument('--max-grid-distance-m',type=float,default=.10)
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.preview_transform:
        if args.apply_grid_mapping:
            parser.error('A preview cannot overwrite the result mapping')
        sys.path.insert(0, str(GPR / "Code"))
        from coordinate_mapping import candidate_world, load_transform
        transform = load_transform(args.preview_transform)
        for row in rows:
            for axis, value in zip("xyz", candidate_world(row["line_no"], row["x_m"], transform)):
                row["world_center_"+axis+"_m"] = value
    model, meshes = load_triangles(args.dam_model)
    center = np.array([243000.0, 431000.0, 0.0])
    report = dict(model_path=str(args.dam_model), model_sha256=sha(args.dam_model),
        buffer_sha256={b["uri"]:sha(args.dam_model.parent/b["uri"]) for b in model["buffers"]},
        candidate_csv_sha256=sha(args.csv),
        axis_convention="world XYZ = (glTF X, -glTF Z, glTF Y); equal coordinate frame assumed",
        preview_transform=str(args.preview_transform) if args.preview_transform else None,
        results={kind:inspect(rows, meshes, kind, center) for kind in ["grid", "surface", "all"]})
    if args.apply_grid_mapping:
        sys.path.insert(0,str(GPR/'Code'))
        from coordinate_mapping import DEFAULT_TRANSFORM, candidate_world, load_transform
        transform=load_transform()
        if not transform.get('model_anchors_applied') or transform['selected_anchors']['model_sha256']!=sha(args.dam_model):
            raise ValueError('Apply user-selected anchors from this model before assigning grid/member fields')
        if not np.isfinite(args.max_grid_distance_m) or args.max_grid_distance_m<=0:
            raise ValueError('Grid distance tolerance must be positive and finite')
        mappings=[]
        for row,detail in zip(rows,report['results']['grid']['details']):
            xyz=[float(row['world_center_'+axis+'_m']) for axis in 'xyz']
            if np.linalg.norm(np.array(xyz)-candidate_world(row['line_no'],row['x_m'],transform))>1e-8:
                raise ValueError('Export candidate XYZ with the current anchor transform first')
            assigned=detail['distance_m']<=args.max_grid_distance_m
            row['grid_id']=detail['nearest_node'] if assigned else ''
            row['member_name']=detail['parent_node'] if assigned else ''
            mappings.append(dict(damage_id=row['damage_id'],world_xyz=xyz,
                grid_id=row['grid_id'],member_name=row['member_name'],grid_distance_m=detail['distance_m']))
        mapping=dict(model_sha256=sha(args.dam_model),buffer_sha256=report['buffer_sha256'],
            transform_sha256=sha(DEFAULT_TRANSFORM),max_grid_distance_m=args.max_grid_distance_m,
            method='Nearest grid triangle for user-anchored surface projection; XYZ unchanged',
            mapped_count=sum(bool(r['grid_id']) for r in rows),candidates=mappings)
        destination=GPR/'Data/Calibration/grid_mapping.json'
        destination.write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n')
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8-sig',newline='',dir=args.csv.parent,delete=False) as stream:
            temporary=Path(stream.name)
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        temporary.replace(args.csv)
        report['candidate_csv_sha256']=sha(args.csv)
        report['grid_mapping_applied']=dict(mapped_count=mapping['mapped_count'],max_distance_m=args.max_grid_distance_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({kind:{key:value for key,value in result.items() if key != "details"}
                      for kind,result in report["results"].items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
