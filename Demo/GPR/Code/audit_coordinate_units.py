"""Audit physical distances against independent DZX spacing and the 3D layout."""

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from coordinate_mapping import (load_transform, candidate_world, display_plane_position,
                                plane_position, LINE_X_POSITIONS)
from scan_geometry import dzt_scan_geometry

ROOT = Path(__file__).resolve().parents[1]


def main():
    transform = load_transform()
    scans = []
    for path in sorted((ROOT / "Data/processed data").glob("*/*.DZT")):
        metadata = dzt_scan_geometry(path)
        xml = ET.parse(path.with_suffix(".DZX")).getroot()
        ns = {"d": xml.tag.split("}")[0].strip("{")}
        properties = xml.find("d:GlobalProperties", ns)
        unit = properties.find("d:horizontalUnit", ns).text
        factor = {"m": 1.0, "cm": 0.01}[unit]
        spacing = float(properties.find("d:unitsPerScan", ns).text) * factor
        if not math.isclose(spacing, metadata["dx_m"], rel_tol=2e-5, abs_tol=1e-7):
            raise AssertionError(f"DZT metric spacing disagrees with DZX unitsPerScan: {path}")
        freq, _, line = path.stem.split("_")
        p0 = candidate_world(line, 0, transform)
        p1 = candidate_world(line, metadata['total_distance_m'], transform)
        model_length = math.dist(p0, p1)
        ratio = transform.get('horizontal_distance_scale', 1.0) if int(line) <= 3 else 1.0
        if not math.isclose(model_length, metadata['total_distance_m']*ratio, rel_tol=0, abs_tol=1e-8):
            raise AssertionError('3D length differs from the documented horizontal endpoint ratio or vertical metric length')
        delta_z = p1[2] - p0[2]
        if int(line) <= 3 and delta_z != 0:
            raise AssertionError("A horizontal survey line changes world Z")
        distances = [float(e.text) for e in xml.findall("d:File/d:Profile/d:WayPt/d:distance", ns)]
        declared_end = (distances[-1]-distances[0])*factor
        scans.append(dict(source=path.relative_to(ROOT).as_posix(), freq_mhz=int(freq),line_no=int(line),
            **metadata, dzx_display_unit=unit, dzx_spacing_m=spacing, model_line_length_m=model_length,
            start_world_xyz=p0, end_world_xyz=p1, endpoint_delta_z_m=delta_z,
            horizontal_endpoint_ratio=ratio, model_minus_measured_length_m=model_length-metadata['total_distance_m'],
            dzx_declared_distance_m=declared_end,
            dzx_distance_field_consistent=math.isclose(declared_end, metadata["total_distance_m"], abs_tol=metadata["dx_m"]*1.1)))
    assert len(scans) == 48
    with (ROOT / "Output/Result/ANM_result.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    max_error = 0
    max_measured_distance_change = 0
    for a, b in itertools.combinations(rows, 2):
        plane_a = display_plane_position(a['line_no'], a['x_m'], transform)
        plane_b = display_plane_position(b['line_no'], b['x_m'], transform)
        world_a = [float(a["world_center_"+axis+"_m"]) for axis in "xyz"]
        world_b = [float(b["world_center_"+axis+"_m"]) for axis in "xyz"]
        max_error = max(max_error, abs(math.dist(plane_a, plane_b)-math.dist(world_a, world_b)))
        measured_distance = math.dist(plane_position(a['line_no'], a['x_m']), plane_position(b['line_no'], b['x_m']))
        max_measured_distance_change = max(max_measured_distance_change, abs(measured_distance-math.dist(world_a,world_b)))
    if max_error > 1e-8:
        raise AssertionError('The 3D candidate layout differs from the documented endpoint interpolation')
    stations = sorted(LINE_X_POSITIONS.items())
    station_errors = [abs(math.dist(candidate_world(a,0,transform),candidate_world(b,0,transform))-(xb-xa))
                      for (a,xa),(b,xb) in itertools.combinations(stations,2)]
    if max(station_errors) > 1e-8:
        raise AssertionError('Vertical station spacing changed')
    anchor_residuals = None
    if transform.get('model_anchors_applied'):
        record = transform['selected_anchors']
        anchor_residuals = [math.dist(candidate_world(1,d,transform),record[role]['world_xyz'])
                            for role,d in [('start',0),('end',transform['reference_measured_length_m'])]]
        if max(anchor_residuals) > 1e-8:
            raise AssertionError('Reference endpoints differ from the supplied anchors')
    csv_path = ROOT / "Output/Result/ANM_result.csv"
    with csv_path.open("rb") as stream:
        csv_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(calibration_status=transform["calibration_status"], scale=transform["scale"],
        legacy_scale=transform["legacy_scale"], distance_source="DZT scans-per-metre; N traces have N-1 intervals",
        csv_sha256=csv_sha, candidate_count=len(rows), scan_count=len(scans),
        max_pairwise_distance_error_m=max_error,
        pairwise_error_reference='Endpoint-interpolated layout; fixed metre vertical stations',
        max_world_vs_measured_pairwise_distance_change_m=max_measured_distance_change,
        horizontal_distance_scale=transform.get('horizontal_distance_scale',1.0),
        vertical_station_spacing_preserved=True, max_vertical_station_spacing_error_m=max(station_errors),
        model_anchors_applied=bool(transform.get('model_anchors_applied')),
        reference_anchor_residuals_m=anchor_residuals,
        horizontal_z_locked=True,
        max_horizontal_endpoint_delta_z_m=max(abs(s['endpoint_delta_z_m']) for s in scans if s['line_no']<=3),
        absolute_placement_verified=False, depth_converted=False,
        assumed_origin_world_xyz=transform["t"],
        dzx_inconsistent_distance_fields=[s["source"] for s in scans if not s["dzx_distance_field_consistent"]],
        scans=scans)
    destination = ROOT / "Data/Calibration/unit_review.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "scans"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
