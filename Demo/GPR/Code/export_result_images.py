"""Export pixel-aligned preprocessed B-scans and their line-endpoint map.

This restores the producer for the delivered ``result_image`` directory.  It
uses the exact preprocessing function used by the rule-based detector, so the
grayscale image is also the unmarked base of each anomaly overlay.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from detect_anomalies_rulebased import FILE_RX, list_pairs, preprocess, to_png_uint8
from gpr_gssi_io import load_gssi_dzt_dxt


CSV_COLUMNS = [
    "freq_mhz",
    "line_no",
    "image",
    "width",
    "height",
    "start_px_x",
    "start_px_y",
    "end_px_x",
    "end_px_y",
    "X0",
    "Y0",
    "Z0",
    "X1",
    "Y1",
    "Z1",
]


def export(folders, lines3d_csv: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    lines3d = pd.read_csv(lines3d_csv)
    rows = []

    for folder in folders:
        pairs = list_pairs(folder)
        if not pairs:
            continue
        first_name = next(iter(pairs.values()))["dzt"]
        match = FILE_RX.match(os.path.basename(first_name))
        freq = int(match.group("freq")) if match else None
        if freq is None:
            raise ValueError(f"Could not infer frequency from {first_name}")

        out_freq = os.path.join(out_dir, str(freq))
        os.makedirs(out_freq, exist_ok=True)
        for line_no in sorted(pairs):
            pair = pairs[line_no]
            data, dt, _dx, meta = load_gssi_dzt_dxt(pair["dzt"], pair["dzx"])
            image = to_png_uint8(preprocess(data, dt, freq))
            out_png = os.path.join(out_freq, f"LINE_{line_no:03d}.png")
            image.save(out_png)

            line_row = lines3d[
                (lines3d["freq_mhz"] == freq) & (lines3d["line_no"] == line_no)
            ]
            if len(line_row) != 1:
                raise ValueError(
                    f"Expected one 3D mapping row for {freq} MHz LINE_{line_no:03d}; "
                    f"found {len(line_row)}"
                )
            mapped = line_row.iloc[0]
            width, height = image.size
            rows.append(
                {
                    "freq_mhz": freq,
                    "line_no": line_no,
                    "image": f"\\{freq}\\LINE_{line_no:03d}.png",
                    "width": width,
                    "height": height,
                    "start_px_x": 0,
                    "start_px_y": 0,
                    "end_px_x": width - 1,
                    "end_px_y": 0,
                    "X0": mapped["X0"],
                    "Y0": mapped["Y0"],
                    "Z0": mapped["Z0"],
                    "X1": mapped["X1"],
                    "Y1": mapped["Y1"],
                    "Z1": mapped["Z1"],
                }
            )

    out_csv = os.path.join(out_dir, "image_pixel_to_3d_map.csv")
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--lines3d", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    export(args.folders, args.lines3d, args.out)


if __name__ == "__main__":
    main()
