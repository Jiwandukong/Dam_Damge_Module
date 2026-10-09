"""Apply user-selected delivery exclusions while retaining original detections."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

REMOVED_CSV_FIELDS = {"max_depth_m", "median_depth_m", "pointcloud_selection", "analysis_method", "result_origin",
                      "grid_id", "grid_guid", "member_id", "analysis_resolution_m"}


def exclusions():
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    return config.get("excluded_output_damages", [])


def excluded_damage(dataset, kind, source_id, rules=None):
    rules = exclusions() if rules is None else rules
    return next((rule for rule in rules if rule["dataset"] == dataset
                 and rule["damage_type"] == kind and str(rule["source_damage_id"]) == str(source_id)), None)


def apply_exclusions(output):
    """Remove configured damages and retired delivery columns from result CSVs."""
    output = Path(output).resolve()
    rules = exclusions()
    removed = []
    for kind in ("SC", "DP"):
        path = output / "Result" / f"{kind}_result.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rejected = [row for row in rows if excluded_damage(row["dataset"], row["damage_type"], row["source_damage_id"], rules)]
        retained_fields = [name for name in fields if name not in REMOVED_CSV_FIELDS]
        if not rejected and retained_fields == fields:
            continue
        files = []
        for row in rejected:
            for column in ("pointcloud_path", "visualization_path", "overlay_path"):
                if row.get(column):
                    target = (path.parent / row[column]).resolve()
                    if output not in target.parents:
                        raise ValueError(f"Excluded artifact path escapes Output: {target}")
                    files.append(target)
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=retained_fields)
            writer.writeheader()
            writer.writerows({name: row[name] for name in retained_fields} for row in rows if row not in rejected)
        os.replace(temporary, path)
        for target in files:
            if target.is_file():
                target.unlink()
        removed.extend(rejected)
    return removed
