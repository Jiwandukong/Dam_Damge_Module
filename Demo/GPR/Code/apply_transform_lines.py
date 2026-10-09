"""
apply_transform_lines.py
------------------------
Apply the survey pose and optional horizontal endpoint interpolation to
line_endpoints.csv and export 3D start/end coordinates per line.
Optionally, save a simple 3D visualization of the 12 lines.

Usage
-----
python Code/apply_transform_lines.py \
  --plane_csv Output/result_batch/line_endpoints.csv \
  --transform Data/Calibration/transform.json --out Output/lines3d_out \
  --plot
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from coordinate_mapping import load_transform, map_line_plane
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plane_csv", required=True)
    ap.add_argument("--transform", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load transform
    T = load_transform(args.transform)

    # Load plane endpoints
    df = pd.read_csv(args.plane_csv)

    rows = []
    for _, r in df.iterrows():
        x0 = float(r["x0"])
        y0p = float(r["y0"])
        x1 = float(r["x1"])
        y1p = float(r["y1"])

        p0 = map_line_plane(r['line_no'], x0, y0p, T)
        p1 = map_line_plane(r['line_no'], x1, y1p, T)

        rows.append({
            "folder": r.get("folder"),
            "freq_mhz": r.get("freq_mhz"),
            "line_no": int(r.get("line_no")),
            "X0": p0[0], "Y0": p0[1], "Z0": p0[2],
            "X1": p1[0], "Y1": p1[1], "Z1": p1[2],
        })

    out_csv = os.path.join(args.out, "lines_3d.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    if args.plot:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        for r in rows:
            X = [r["X0"], r["X1"]]
            Y = [r["Y0"], r["Y1"]]
            Z = [r["Z0"], r["Z1"]]
            ax.plot(X, Y, Z, linewidth=2)
            ax.text(X[0], Y[0], Z[0], f"L{int(r['line_no']):02d}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("GPR Lines (Transformed to Model)")
        plt.tight_layout()
        out_png = os.path.join(args.out, "lines_3d.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print("Saved:", out_png)


if __name__ == "__main__":
    main()


