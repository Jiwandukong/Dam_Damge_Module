"""
plot_anomalies_3d.py
--------------------
Quick 3D visualization for anomalies from anomalies_rulebased.csv

Usage
-----
python Code/plot_anomalies_3d.py --csv Output/anomaly_rb/anomalies_rulebased.csv \
  --out Output/anomaly_rb --color t_ns

Options
-------
- --color: which column to color by. One of: t_ns, freq_mhz, line_no. Default: t_ns
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--color", default="t_ns", choices=["t_ns", "freq_mhz", "line_no"]) 
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    if not set(["X","Y","Z"]).issubset(df.columns):
        raise ValueError("CSV must contain X,Y,Z columns")

    # If freq_mhz available, export per-frequency plots
    if "freq_mhz" in df.columns:
        for f in sorted(df["freq_mhz"].dropna().unique()):
            dff = df[df["freq_mhz"] == f]
            if dff.empty:
                continue
            X = dff["X"].values.astype(float)
            Y = dff["Y"].values.astype(float)
            Z = dff["Z"].values.astype(float)
            C = dff[args.color].values if args.color in dff.columns else np.linspace(0,1,len(dff))

            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(X, Y, Z, c=C, cmap="viridis", s=8, alpha=0.9)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Anomaly Points (Rule-based) - {int(f)} MHz")
            cb = plt.colorbar(sc, pad=0.1)
            cb.set_label(args.color)
            plt.tight_layout()
            out_png = os.path.join(args.out, f"anomalies_3d_{int(f)}.png")
            plt.savefig(out_png, dpi=150)
            plt.close()
            print("Saved:", out_png)
    else:
        # single combined
        X = df["X"].values.astype(float)
        Y = df["Y"].values.astype(float)
        Z = df["Z"].values.astype(float)
        C = df[args.color].values if args.color in df.columns else np.linspace(0,1,len(df))

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X, Y, Z, c=C, cmap="viridis", s=8, alpha=0.9)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Anomaly Points (Rule-based)")
        cb = plt.colorbar(sc, pad=0.1)
        cb.set_label(args.color)
        plt.tight_layout()
        out_png = os.path.join(args.out, "anomalies_3d.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print("Saved:", out_png)


if __name__ == "__main__":
    main()


