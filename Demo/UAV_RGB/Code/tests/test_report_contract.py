"""The demo's 16-column writer must not depend on production display changes."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo512 import report_contract as report


def example_row():
    row = dict.fromkeys(report.PRIMARY_COLUMNS)
    row.update(image="source.JPG", damage_id="D000001", damage_type="CRC",
               damage_name_ko="균열", length_px=2.5, width_px=1.0,
               source_image_path="원본사진/source.JPG",
               tile_original_paths_json='["512원본타일/tile.png"]',
               tile_overlay_paths_json='["512라벨오버레이/tile.png"]')
    return row


def test_demo_schema_matches_the_exact_16_column_writer():
    schema = json.loads(report.SCHEMA_PATH.read_text(encoding="utf-8"))
    assert len(report.PRIMARY_COLUMNS) == 16
    assert schema["required"] == report.PRIMARY_COLUMNS
    assert list(schema["properties"]) == report.PRIMARY_COLUMNS
    assert schema["additionalProperties"] is False


def test_csv_keeps_utf8_bom_column_order_and_image_links(tmp_path):
    path = tmp_path / "nested/results.csv"
    row = example_row()
    report.write_csv(path, [row], report.PRIMARY_COLUMNS)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == report.PRIMARY_COLUMNS
        actual = next(reader)
        assert next(reader, None) is None
    assert actual["width_m"] == ""
    assert actual["length_px"] == "2.5"
    for name in report.PRIMARY_COLUMNS[-3:]:
        assert actual[name] == row[name]


def test_workbook_preserves_existing_sheet_values_and_formatting(tmp_path):
    path = tmp_path / "nested/results.xlsx"
    row = example_row()
    report.write_workbook(path, rows=[row])
    workbook = load_workbook(path)
    try:
        assert workbook.sheetnames == ["Damage_Details"]
        sheet = workbook.active
        values = list(sheet.values)
        assert list(values[0]) == report.PRIMARY_COLUMNS
        assert list(values[1]) == [row[name] for name in report.PRIMARY_COLUMNS]
        assert sheet.freeze_panes == "A2"
        assert sheet.auto_filter.ref == "A1:P2"
        assert sheet["A1"].fill.fgColor.rgb == "005595AF"
        assert sheet["A1"].font.bold is True
        assert sheet["A1"].font.color.rgb == "00FFFFFF"
        assert sheet["A1"].alignment.horizontal == "center"
        assert sheet["E2"].alignment.wrap_text is None or sheet["E2"].alignment.wrap_text is False
    finally:
        workbook.close()


def test_json_helpers_preserve_numpy_scalars_arrays_and_paths():
    value = {"flag": np.bool_(True), "count": np.int64(3),
             "scale": np.float32(0.5), "points": np.array([[1, 2]]),
             "path": Path("relative/file.png")}
    expected = {"flag": True, "count": 3, "scale": 0.5,
                "points": [[1, 2]], "path": "relative/file.png"}
    assert report.json_ready(value) == expected
    assert json.loads(report._excel_value(value)) == expected
