"""Fit selected LINE_001 anchors, preserving vertical spacing and horizontal Z."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from coordinate_mapping import DEFAULT_TRANSFORM, horizontal_frame, load_transform
from scan_geometry import dzt_scan_geometry

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT.parent / "Dam_model/Daecheongdam/daecheongdam_regions_grid5m.gltf"
FOLDERS = {400:"400MHz",900:"900MHz",1600:"1.6GHz",2600:"2.6GHz"}


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def unit(v):
    length=math.hypot(*v)
    if length<1e-9:
        raise ValueError("Endpoints or transverse direction are degenerate")
    return [x/length for x in v]


def fit(record, previous, measured_length, tolerance, fit_horizontal_endpoints=False):
    start=record["start"]["world_xyz"]
    end=record["end"]["world_xyz"]
    if len(start)!=3 or len(end)!=3 or not all(math.isfinite(x) for x in start+end):
        raise ValueError("Both endpoints must have finite world XYZ coordinates")
    difference=[b-a for a,b in zip(start,end)]
    if abs(difference[2]) > 1e-8:
        raise ValueError("Horizontal endpoints must have the same world Z. Select the end on the mesh intersection at the start height.")
    difference[2] = 0.0
    selected_length=math.hypot(*difference)
    if not fit_horizontal_endpoints and abs(selected_length-measured_length)>tolerance:
        raise ValueError(f"Selected distance {selected_length:.3f}m differs from measured distance {measured_length:.3f}m. Recheck endpoint locations/reference scan; scale remains 1.")
    along=unit(difference)
    mesh_along=[row[0] for row in previous["R"]]
    mesh_angle=math.degrees(math.acos(max(-1.0,min(1.0,dot(along,mesh_along)))))
    if mesh_angle > .2:
        raise ValueError(f"Selected direction differs from the horizontal mesh direction by {mesh_angle:.3f} degrees. Recheck endpoint locations; the line must follow the dam face.")
    old_across=[row[1] for row in previous["R"]]
    rotation=horizontal_frame(along if fit_horizontal_endpoints else mesh_along,old_across)
    ratio=selected_length/measured_length if fit_horizontal_endpoints else 1.0
    predicted_end=[start[i]+rotation[i][0]*measured_length*ratio for i in range(3)]
    residual=math.dist(predicted_end,end)
    if residual > tolerance:
        raise ValueError(f"End point is {residual:.3f}m from the measured horizontal mesh-parallel line. Recheck endpoints.")
    return {**previous,"scale":1.0,"R":rotation,
            "t":start,"y0":0.0,"calibration_status":"selected_anchor_fit" if fit_horizontal_endpoints else "start_end_review",
            "provenance":"User-selected model endpoints applied. Horizontal along-scan distances use an explicit endpoint interpolation ratio; vertical station spacing and vertical scan distances stay in metres. World Z is constant along horizontal lines. Field survey/CRS accuracy is not established by model picks." if fit_horizontal_endpoints else "Review fit to user-selected LINE_001 start/end. Scale=1; mesh-derived horizontal direction retained; along-line world Z locked.",
            "selected_anchors":record,"selected_anchor_distance_m":selected_length,
            "reference_measured_length_m":measured_length,"endpoint_residual_m":residual,
            "selected_mesh_direction_angle_deg":mesh_angle,"horizontal_z_locked":True,
            "horizontal_distance_scale":ratio,
            "selected_minus_measured_length_m":selected_length-measured_length,
            "vertical_station_spacing_preserved":True,
            "model_anchors_applied":True,"field_absolute_placement_verified":False,
            "transverse_axis_verified":False,"anchor_plane_xy_m":[0.0,0.0]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True,help="Review transform output; choose an explicit destination")
    parser.add_argument("--previous",type=Path,default=DEFAULT_TRANSFORM)
    parser.add_argument("--length-tolerance-m",type=float,default=.10)
    parser.add_argument('--fit-horizontal-endpoints',action='store_true',
        help='Fit both selected endpoints with a horizontal-only interpolation ratio; preserve vertical station spacing and lengths')
    args=parser.parse_args()
    record=json.loads(args.anchors.read_text())
    if not record.get("start") or not record.get("end"):
        raise ValueError("Select both actual endpoints before fitting")
    if record["model_sha256"]!=hashlib.sha256(MODEL.read_bytes()).hexdigest():
        raise ValueError("Selected points came from a different model")
    if int(record["reference_line_no"])!=1:
        raise ValueError("This endpoint fit expects the horizontal LINE_001 reference")
    freq=int(record["reference_frequency_mhz"])
    scan=ROOT / "Data/processed data" / FOLDERS[freq] / f"{freq}_LINE_001.DZT"
    measured=dzt_scan_geometry(scan)["total_distance_m"]
    if not math.isfinite(args.length_tolerance_m) or args.length_tolerance_m<=0:
        raise ValueError("Specify a positive length tolerance")
    transform=fit(record,load_transform(args.previous),measured,args.length_tolerance_m,args.fit_horizontal_endpoints)
    args.output.write_text(json.dumps(transform,ensure_ascii=False,indent=2)+"\n")
    load_transform(args.output)
    print(f"Review transform saved: {args.output}; vertical metric scale=1; horizontal endpoint ratio={transform['horizontal_distance_scale']:.9f}; endpoint residual={transform['endpoint_residual_m']:.4f}m")


if __name__=="__main__":
    main()
