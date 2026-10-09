"""
rectify_tiepoints.py
--------------------
Given 4 approximate 3D points from the model corresponding to two vertical
feature lines at x = x1 and x = x2 on the GPR-derived plane, rectify them to a
perfect rectangle with constraints z1=z3 and z2=z4, then compute a 2D-plane ->
3D rigid metre-to-metre transform usable as tie mapping.

Inputs
------
1) plane CSV (created by gpr_batch_plane.py): line_endpoints.csv
   - columns: folder,freq_mhz,line_no,x0,y0,x1,y1,length_m,png
   - Retained as provenance; line lengths never determine vertical tie extent.
2) points CSV with four world XYZ rows in metres
   (order: (x1,y0),(x1,y1),(x2,y0),(x2,y1))
3) x1, x2 and explicit --plane-y1 in metres

Outputs
-------
- corrected_tie_points.csv  (rectified Q1'..Q4')
- transform.json            (s, R, t for mapping [x,y,0] -> 3D)

Usage
-----
python Code/rectify_tiepoints.py --plane_csv Output/result_batch/line_endpoints.csv \
       --points_csv Data/Calibration/tie_points.csv --x1 5.9 --x2 8.1 \
       --plane-y1 2.0 --out /path/to/work/tie_out
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from coordinate_mapping import horizontal_frame


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Rigid metre-to-metre fit from src -> dst; never shrink the survey."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    C = (Y.T @ X) / len(src)
    U, _, Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    # The rectified rectangle has a horizontal x axis. Remove numerical tilt
    # explicitly so every point on each horizontal scan keeps the same world Z.
    R = np.array(horizontal_frame(R[:, 0], R[:, 1]))
    s = 1.0
    t = mu_d - s * (R @ mu_s)
    return float(s), R.astype(float), t.astype(float)


def rectify_rectangle(Q: np.ndarray):
    """Rectify 4 points into a right rectangle in 3D basis (u,v,n).

    Input Q with order: [Q1(x1,0), Q2(x1,L), Q3(x2,0), Q4(x2,L)].
    Enforce: parallel verticals, orthogonal axes, and equal heights per row via
    z1=z3 (use their mean), z2=z4 (use their mean).
    """
    p1, p2, p3, p4 = Q
    # Solve the horizontal/orthogonal constraint together. Editing corner Z
    # after constructing a rectangle can destroy orthogonality and reintroduce tilt.
    rotation = np.array(horizontal_frame((p3-p1+p4-p2)*0.5, (p2-p1+p4-p3)*0.5))
    u, v = rotation[:, 0], rotation[:, 1]
    L_model = 0.5 * (np.dot(p2-p1, v) + np.dot(p4-p3, v))
    W_model = 0.5 * (np.dot(p3 - p1, u) + np.dot(p4 - p2, u))

    o = Q.mean(axis=0) - 0.5 * (W_model*u + L_model*v)
    Qr = np.vstack([
        o,
        o + L_model * v,
        o + W_model * u,
        o + W_model * u + L_model * v
    ])

    return Qr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plane_csv", required=True, help="line_endpoints.csv path")
    ap.add_argument("--points_csv", required=True, help="4 points CSV path (x,y,z)")
    ap.add_argument("--x1", type=float, required=True)
    ap.add_argument("--x2", type=float, required=True)
    ap.add_argument("--y0", type=float, default=0.0, help="Plane start y (e.g., -0.3)")
    ap.add_argument("--plane-y1", type=float, required=True,
                    help="Actual plane y coordinate of upper tie points, in metres")
    ap.add_argument("--max-residual-m", type=float, default=0.10,
                    help="Reject incompatible tie points instead of fitting a scale")
    ap.add_argument("--out", default="./tie_out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Horizontal line length is not the vertical calibration extent.
    y0 = float(args.y0)
    L = float(args.plane_y1)
    if not np.isfinite([args.x1, args.x2, y0, L, args.max_residual_m]).all() or L <= y0 or args.x2 <= args.x1 or args.max_residual_m <= 0:
        ap.error("Require finite x2>x1, plane-y1>y0 and a positive residual tolerance")

    # Load 4 approx points
    pts = pd.read_csv(args.points_csv)
    Q = pts[["x", "y", "z"]].values.astype(float)
    if Q.shape != (4, 3):
        raise ValueError("points_csv must contain exactly 4 rows with columns x,y,z")
    if not np.isfinite(Q).all():
        raise ValueError("Tie-point world coordinates must be finite")

    # Rectify
    Qcorr = rectify_rectangle(Q)

    # Build plane 3D points (z=0 plane coords)
    x1, x2 = float(args.x1), float(args.x2)
    P3 = np.array([[x1, y0, 0.0], [x1, L, 0.0], [x2, y0, 0.0], [x2, L, 0.0]], dtype=float)

    # Similarity transform
    s, R, t = umeyama(P3, Qcorr)
    residuals = np.linalg.norm(P3 @ R.T + t - Q, axis=1)
    if residuals.max() > args.max_residual_m:
        raise ValueError(f"Tie-point dimensions disagree with the measured metre layout: max residual {residuals.max():.4f} m. Check point order, dimensions and units.")
    # apply_transform_lines adds y0 to local measured plane y values.
    raw_t = t + R @ np.array([0.0, y0, 0.0])

    # Save corrected tie points
    out_pts = os.path.join(args.out, "corrected_tie_points.csv")
    pd.DataFrame(Qcorr, columns=["x", "y", "z"]).to_csv(out_pts, index=False)

    # Save transform
    out_js = os.path.join(args.out, "transform.json")
    with open(out_js, "w", encoding="utf-8") as f:
        json.dump({
            "scale": s,
            "R": R.tolist(),
            "t": raw_t.tolist(),
            "x1": x1, "x2": x2, "y0": 0.0, "tie_y0": y0, "L": L,
            "span": L - y0,
            "plane_csv": os.path.abspath(args.plane_csv),
            "points_csv": os.path.abspath(args.points_csv),
            "calibration_status": "tie_point_fit_review",
            "horizontal_z_locked": True,
            "plane_units": "m", "world_units": "m",
            "max_tie_residual_m": float(residuals.max())
        }, f, indent=2)

    print("Saved:", out_pts)
    print("Saved:", out_js)


if __name__ == "__main__":
    main()


