r"""
gpr_batch_plane.py
-------------------
Batch export of preprocessed B-scan images by frequency folders and
extraction of start/end coordinates per survey line for plane assembly.

Inputs
------
- Root folders per frequency, each containing 12 pairs:
  <FREQ>_LINE_###.DZT + <FREQ>_LINE_###.DZX
  Example directories:
    Data/processed data/1.6GHz
    Data/processed data/2.6GHz
    Data/processed data/400MHz
    Data/processed data/900MHz

Outputs
-------
- Preprocessed images (PNG) under <out_dir>/<freq>/<LINE_###>_preproc.png
- A CSV listing start/end (x,y) coordinates per line under <out_dir>/line_endpoints.csv

Notes
-----
- Line layout (from provided drawing):
  LINE 1,2,3 run along x-direction (transverse). y positions: 0.0, 0.8, 1.2 m
  LINE 4..12 run along y-direction (longitudinal). x positions (m):
    {4:1.0, 5:4.0, 6:5.5, 7:9.0, 8:12.0, 9:15.0, 10:18.0, 11:21.0, 12:25.0}
  Raw distances along a line are taken from DZX (or GPRPy profile) to set endpoints.
"""

import os
import re
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

from gpr_gssi_io import load_gssi_dzt_dxt
from coordinate_mapping import LINE_Y_OFFSETS, LINE_X_POSITIONS
from gpr_preproc_demo import select_preset
from gpr_anomaly_pipeline import (
    dewow, background_subtract, agc_gain, bandpass,
)


FOLDER_FREQ_MAP = {
    "1.6GHz": 1600,
    "2.6GHz": 2600,
    "400MHz": 400,
    "900MHz": 900,
}

FILE_RX = re.compile(r"^(?P<freq>\d+)_LINE_(?P<line>\d{3})\.(?P<ext>DZT|DZX)$", re.IGNORECASE)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_pairs(folder: str):
    """Return dict line_no -> {"dzt":path, "dzx":path} for valid pairs in folder."""
    tmp = {}
    for name in os.listdir(folder):
        m = FILE_RX.match(name)
        if not m:
            continue
        line_no = int(m.group("line"))
        ext = m.group("ext").upper()
        d = tmp.setdefault(line_no, {})
        d[ext.lower()] = os.path.join(folder, name)
    # keep only full pairs
    return {k: v for k, v in tmp.items() if "dzt" in v and "dzx" in v}


def preprocess_and_save(data: np.ndarray, dt: float, dx: float, meta: dict,
                        out_png: str, dewow_ns: float, agc_ns: float,
                        band: tuple):
    """Apply simple preprocessing and save a PNG."""
    # time-zero shift is skipped here (can be added if needed); assume input already close
    step = dewow(data, dt, window_ns=dewow_ns)
    step = background_subtract(step)
    step = agc_gain(step, dt, window_ns=agc_ns)
    step = bandpass(step, dt, band[0], band[1], order=4)

    # Plot with time up (0 at top) and distance along x
    t_ns = np.arange(step.shape[0]) * dt * 1e9
    extent = [0, step.shape[1] * dx, t_ns[-1], t_ns[0]]

    plt.figure(figsize=(12, 6))
    vmax = np.percentile(np.abs(step), 99)
    plt.imshow(step, cmap="gray", aspect="auto", origin="upper",
               extent=extent, vmin=-vmax, vmax=vmax)
    plt.xlabel("Distance [m]")
    plt.ylabel("Time [ns]")
    plt.title("Preprocessed B-scan")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def compute_endpoints(line_no: int, meta: dict):
    """Return (x0,y0,x1,y1,length_m) for a line using layout rules and meta distances."""
    total = float(meta.get("total_distance_m", 0.0))
    if line_no in (1, 2, 3):
        y = LINE_Y_OFFSETS[line_no]
        return 0.0, y, total, y, total
    else:
        x = LINE_X_POSITIONS.get(line_no)
        return x, 0.0, x, total, total


def process_folder(folder: str, out_dir: str):
    freq_label = os.path.basename(folder)
    freq_mhz = FOLDER_FREQ_MAP.get(freq_label)
    if freq_mhz is None:
        # try infer from files if folder label not standard
        freq_mhz = None

    print(f"Processing folder: {folder} (freq={freq_mhz})")
    pairs = list_pairs(folder)
    if not pairs:
        print("No valid pairs found.")
        return []

    # Prepare output
    out_freq = os.path.join(out_dir, str(freq_mhz) if freq_mhz else freq_label)
    ensure_dir(out_freq)

    # endpoints collector
    rows = []

    # sort by line number
    for line_no in sorted(pairs.keys()):
        dzt = pairs[line_no]["dzt"]
        dzx = pairs[line_no]["dzx"]
        data, dt, dx, meta = load_gssi_dzt_dxt(dzt, dzx)
        meta = dict(meta or {})
        if freq_mhz is not None:
            meta["freq_mhz"] = freq_mhz

        # select presets
        from gpr_preproc_demo import DefaultConfig  # lazy import to reuse
        cfg = DefaultConfig()
        dewow_ns, agc_ns, band, used = select_preset(meta.get("freq_mhz"), cfg)

        # save PNG
        name = f"LINE_{line_no:03d}_preproc.png"
        out_png = os.path.join(out_freq, name)
        preprocess_and_save(data, dt, dx, meta, out_png, dewow_ns, agc_ns, band)

        # endpoints
        x0, y0, x1, y1, length_m = compute_endpoints(line_no, meta)
        rows.append({
            "folder": freq_label,
            "freq_mhz": meta.get("freq_mhz"),
            "line_no": line_no,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "length_m": length_m,
            # Preserve the legacy Windows-style path stored in the delivered
            # line_endpoints.csv, even when reproduction runs on Linux.
            "png": os.path.relpath(out_png, out_dir).replace("/", "\\").replace(os.path.sep, "\\")
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="Frequency folders to process")
    ap.add_argument("--out", default="./out_export", help="Output directory")
    args = ap.parse_args()

    ensure_dir(args.out)
    all_rows = []
    for folder in args.folders:
        all_rows.extend(process_folder(folder, args.out))

    # write CSV
    csv_path = os.path.join(args.out, "line_endpoints.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "folder", "freq_mhz", "line_no", "x0", "y0", "x1", "y1", "length_m", "png"
        ])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()


