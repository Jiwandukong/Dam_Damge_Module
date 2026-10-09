"""
detect_anomalies_rulebased.py
-----------------------------
Conservative, rule-based anomaly detection for GPR B-scans.
Finds only high-confidence candidates and paints markers on PNGs, also exports
CSV with 2D/3D coordinates via lines_3d.csv (from apply_transform_lines.py).

Usage
-----
python Code/detect_anomalies_rulebased.py \
  --folders "Data/processed data/1.6GHz" "Data/processed data/2.6GHz" \
  --lines3d Output/lines3d_out/lines_3d.csv --out Output/anomaly_rb

Notes
-----
- Uses same preprocessing as data-only export (time-zero, dewow, bg, AGC, BP).
- Conservative masks:
  * Hyperbola-like scatter: scattering_density > 0.92
  * High-attenuation pocket: attenuation_alpha > 0.9 AND envelope low (<0.5)
  * Excludes near-surface (t < 0.5 ns) to avoid direct wave.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from overlay_markers import DEFAULT_MARKER_HALF_SIZE, DEFAULT_MARKER_WIDTH, render_markers

from gpr_gssi_io import load_gssi_dzt_dxt
from gpr_preproc_demo import select_preset, DefaultConfig
from gpr_anomaly_pipeline import (
    dewow, background_subtract, agc_gain, bandpass,
    time_zero_pick, apply_time_zero,
    analytic_envelope, robust_unit, attenuation_alpha,
    coherence_across_traces, scattering_density,
)

FILE_RX = re.compile(r"^(?P<freq>\d+)_LINE_(?P<line>\d{3})\.(?P<ext>DZT|DZX)$", re.IGNORECASE)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# Thresholds (rolled back to baseline; previous experimental values kept here for reference)
# Baseline: incoh_scat=0.72, incoh_att=0.62, env_high=0.65, env_low=0.45
# Experimental (discarded): incoh_scat=0.78/0.93, incoh_att=0.68/0.80, env_high=0.60, env_low=0.48
SCA_THRESH = 0.985
ATT_THRESH = 0.965
INCOH_SCAT_THRESH = 0.72
INCOH_ATT_THRESH = 0.62
ENV_HIGH_THRESH = 0.65
ENV_LOW_THRESH = 0.45


def list_pairs(folder: str):
    tmp = {}
    for name in os.listdir(folder):
        m = FILE_RX.match(name)
        if not m:
            continue
        line_no = int(m.group("line"))
        ext = m.group("ext").upper()
        d = tmp.setdefault(line_no, {})
        d[ext.lower()] = os.path.join(folder, name)
    return {k: v for k, v in tmp.items() if "dzt" in v and "dzx" in v}


def preprocess(data: np.ndarray, dt: float, freq_mhz: int):
    cfg = DefaultConfig()
    dewow_ns, agc_ns, band, _ = select_preset(freq_mhz, cfg)
    idx0 = time_zero_pick(data, method="autopick")
    x = apply_time_zero(data, idx0)
    x = dewow(x, dt, window_ns=dewow_ns)
    x = background_subtract(x)
    x = agc_gain(x, dt, window_ns=agc_ns)
    x = bandpass(x, dt, band[0], band[1], order=4)
    return x


def detect_conservative(x: np.ndarray, dt: float, *, incoh_win: int=9, tmask_top_ns: float=0.8, tmask_bottom_frac: float=0.8, min_area: int=45, incoh_q: float=None):
    """Return list of (row,col,score,type) for conservative anomalies."""
    env_n, incoh_n, att, sca_n, tmask = compute_rule_features(x, dt, incoh_win=incoh_win, tmask_top_ns=tmask_top_ns, tmask_bottom_frac=tmask_bottom_frac)

    # Determine incoherence thresholds (fixed vs quantile)
    if incoh_q is not None:
        # compute per-image threshold from masked region
        incoh_vals = incoh_n[tmask, :].ravel()
        th_incoh_scat = float(np.quantile(incoh_vals, incoh_q))
        th_incoh_att  = th_incoh_scat
    else:
        th_incoh_scat = INCOH_SCAT_THRESH
        th_incoh_att  = INCOH_ATT_THRESH

    # mid-conservative rules (between previous two)
    # slightly more conservative on scatter threshold
    mask_scat = (sca_n > SCA_THRESH) & (incoh_n > th_incoh_scat) & (env_n > ENV_HIGH_THRESH) & tmask[:, None]
    mask_att  = (att > ATT_THRESH)  & (env_n < ENV_LOW_THRESH) & (incoh_n > th_incoh_att) & tmask[:, None]
    mask = mask_scat | mask_att

    return mask_to_detections(mask, sca_n, att, min_area=min_area)


def compute_rule_features(x: np.ndarray, dt: float, *, incoh_win: int=9, tmask_top_ns: float=0.8, tmask_bottom_frac: float=0.8):
    """Compute normalized feature maps and time/depth mask used by rules."""
    env = analytic_envelope(x)
    env_n = robust_unit(env, axis=1)
    coh = coherence_across_traces(x, win_traces=incoh_win)
    incoh_n = robust_unit(1.0 - coh)
    att = attenuation_alpha(env, dt, tmin_ns=2.0, tmax_ns=200.0, window_ns=40.0)
    sca = scattering_density(env, z_thresh=3.0, max_filter_size=(3,3), smooth_sigma=1.0)
    sca_n = robust_unit(sca)

    nt = x.shape[0]
    t = np.arange(nt) * dt * 1e9
    tmask_top = t >= float(tmask_top_ns)
    tmask_bottom = (np.arange(nt) < int(float(tmask_bottom_frac)*nt))
    tmask = tmask_top & tmask_bottom
    return env_n, incoh_n, att, sca_n, tmask


def mask_to_detections(mask: np.ndarray, sca_n: np.ndarray, att: np.ndarray, *, min_area: int=45):
    """Convert boolean mask to detections using CC filtering and centroids."""
    from scipy.ndimage import label, find_objects
    lbl, num = label(mask.astype(np.uint8))
    res = []
    for i in range(1, num+1):
        sl = find_objects(lbl==i)
        if not sl:
            continue
        sl = sl[0]
        r0, r1 = sl[0].start, sl[0].stop
        c0, c1 = sl[1].start, sl[1].stop
        area = (r1 - r0) * (c1 - c0)
        if area < int(min_area):
            continue
        sub = lbl[sl] == i
        rr, cc = np.where(sub)
        r_cent = r0 + rr.mean()
        c_cent = c0 + cc.mean()
        scat_score = float(sca_n[int(round(r_cent)), int(round(c_cent))])
        att_score  = float(att[int(round(r_cent)), int(round(c_cent))])
        atype = "scatter" if scat_score >= att_score else "attenuation"
        score = max(scat_score, att_score)
        res.append((r_cent, c_cent, score, atype))
    return res


def to_png_uint8(arr: np.ndarray) -> Image.Image:
    vmax = np.percentile(np.abs(arr), 99)
    vmax = float(max(vmax, 1e-9))
    norm = np.clip(arr / vmax, -1.0, 1.0)
    gray = ((norm * 0.5 + 0.5) * 255.0).astype(np.uint8)
    return Image.fromarray(gray)


def process(
    folders,
    lines3d_csv: str,
    out_dir: str,
    ablation: bool = False,
    incoh_win: int = 9,
    tmask_top_ns: float = 0.8,
    tmask_bottom: float = 0.8,
    min_area: int = 45,
    incoh_q: float = None,
    marker_half_size: int = DEFAULT_MARKER_HALF_SIZE,
    marker_width: int = DEFAULT_MARKER_WIDTH,
):
    ensure_dir(out_dir)
    lines3d = pd.read_csv(lines3d_csv)

    rows = []
    for folder in folders:
        pairs = list_pairs(folder)
        if not pairs:
            continue
        anyname = next(iter(pairs.values()))["dzt"]
        m = FILE_RX.match(os.path.basename(anyname))
        freq = int(m.group("freq")) if m else None
        out_freq = os.path.join(out_dir, str(freq) if freq else os.path.basename(folder))
        ensure_dir(out_freq)

        for line_no in sorted(pairs.keys()):
            dzt = pairs[line_no]["dzt"]
            dzx = pairs[line_no]["dzx"]
            data, dt, dx, meta = load_gssi_dzt_dxt(dzt, dzx)
            fm = freq or meta.get("freq_mhz")
            x = preprocess(data, dt, fm)
            # detections
            dets = detect_conservative(
                x,
                dt,
                incoh_win=incoh_win,
                tmask_top_ns=tmask_top_ns,
                tmask_bottom_frac=tmask_bottom,
                min_area=min_area,
                incoh_q=incoh_q,
            )

            # render overlay on data-only image
            depth_cut = int(0.8 * x.shape[0])
            img = render_markers(
                to_png_uint8(x), [(c, r) for r, c, _score, _type in dets if r < depth_cut],
                marker_half_size, marker_width,
            )
            # save overlay
            out_png = os.path.join(out_freq, f"LINE_{line_no:03d}_anomaly.png")
            img.save(out_png)
            # relative path formatting: \anomaly_rb\<freq>\LINE_xxx_anomaly.png
            base_out = os.path.basename(out_dir.rstrip("/\\"))
            # Preserve the legacy CSV representation used by the delivered
            # outputs, independent of the host operating system.
            rel = os.path.relpath(out_png, out_dir).replace('/', '\\').replace(os.path.sep, '\\')
            rel_img = "\\" + base_out + "\\" + rel

            # 3D mapping via start/end interpolation
            line_row = lines3d[(lines3d["freq_mhz"]==fm)&(lines3d["line_no"]==line_no)]
            if len(line_row)==1:
                lr = line_row.iloc[0]
                P0 = np.array([lr["X0"], lr["Y0"], lr["Z0"]], dtype=float)
                P1 = np.array([lr["X1"], lr["Y1"], lr["Z1"]], dtype=float)
                length_m = float(meta.get("total_distance_m", x.shape[1]*dx))
                for (r,c,score,atype) in dets:
                    if r >= depth_cut:
                        continue
                    cx = int(round(float(c)))
                    cy = int(round(float(r)))
                    # keep sub-pixel precision in CSV; use ints only for overlay drawing
                    x_m = float(c) * dx
                    frac = np.clip(x_m / max(length_m,1e-9), 0.0, 1.0)
                    P = P0 + frac*(P1-P0)
                    rows.append({
                        "freq_mhz": fm, "line_no": line_no,
                        "x_px": float(c), "y_px": float(r),
                        "x_m": x_m, "t_ns": float(r)*dt*1e9,
                        "X": P[0], "Y": P[1], "Z": P[2],
                        "image": rel_img,
                    })

            # Ablation overlays (optional): stepwise masks and final
            if ablation:
                env_n, incoh_n, att, sca_n, tmask = compute_rule_features(
                    x,
                    dt,
                    incoh_win=incoh_win,
                    tmask_top_ns=tmask_top_ns,
                    tmask_bottom_frac=tmask_bottom,
                )
                masks = {
                    # 단독 지표
                    "scat": (sca_n > SCA_THRESH) & tmask[:, None],
                    "incoh": (incoh_n > INCOH_SCAT_THRESH) & tmask[:, None],
                    "envelope": (env_n > ENV_HIGH_THRESH) & tmask[:, None],
                    # 세 지표 AND 조합
                    "full": (sca_n > SCA_THRESH) & (incoh_n > INCOH_SCAT_THRESH) & (env_n > ENV_HIGH_THRESH) & tmask[:, None],
                }

                abl_dir = os.path.join(out_freq, "ablation")
                ensure_dir(abl_dir)
                for name, msk in masks.items():
                    dets_abl = mask_to_detections(msk, sca_n, att)
                    imga = to_png_uint8(x).convert('RGB')
                    dra = ImageDraw.Draw(imga)
                    depth_cut = int(0.8 * x.shape[0])
                    cross_half_a = 5
                    line_width_a = 2
                    if (fm == 1600) and (line_no in (1, 2, 3)):
                        cross_half_a = 8
                        line_width_a = 3
                    for (r, c, _s, _t) in dets_abl:
                        cx, cy = int(round(float(c))), int(round(float(r)))
                        if cy >= depth_cut:
                            continue
                        dra.line([(cx - cross_half_a, cy), (cx + cross_half_a, cy)], fill=(255,0,0), width=line_width_a)
                        dra.line([(cx, cy - cross_half_a), (cx, cy + cross_half_a)], fill=(255,0,0), width=line_width_a)
                    out_png_a = os.path.join(abl_dir, f"LINE_{line_no:03d}_{name}.png")
                    imga.save(out_png_a)

    if rows:
        df = pd.DataFrame(rows)
        out_csv = os.path.join(out_dir, "anomalies_rulebased.csv")
        df.to_csv(out_csv, index=False)
        print("Saved:", out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folders", nargs="+", required=True)
    ap.add_argument("--lines3d", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablation", action="store_true", help="Save stepwise overlay images for feature ablation")
    ap.add_argument("--incoh_win", type=int, default=9, help="Coherence window (traces)")
    ap.add_argument("--tmask_top_ns", type=float, default=0.8, help="Exclude t < this ns")
    ap.add_argument("--tmask_bottom", type=float, default=0.8, help="Exclude bottom (1-f) fraction")
    ap.add_argument("--min_area", type=int, default=45, help="Min connected area")
    ap.add_argument("--incoh_q", type=float, default=None, help="If set (0-1), use per-image incoherence quantile threshold")
    ap.add_argument("--marker-half-size", type=int, default=DEFAULT_MARKER_HALF_SIZE)
    ap.add_argument("--marker-width", type=int, default=DEFAULT_MARKER_WIDTH)
    args = ap.parse_args()
    if args.marker_half_size < 1 or args.marker_width < 1:
        ap.error("Marker size and width must be positive")
    process(
        args.folders,
        args.lines3d,
        args.out,
        ablation=args.ablation,
        incoh_win=args.incoh_win,
        tmask_top_ns=args.tmask_top_ns,
        tmask_bottom=args.tmask_bottom,
        min_area=args.min_area,
        incoh_q=args.incoh_q,
        marker_half_size=args.marker_half_size,
        marker_width=args.marker_width,
    )


if __name__ == "__main__":
    main()
