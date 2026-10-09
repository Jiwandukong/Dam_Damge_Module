"""
gpr_preproc_demo.py
-------------------
Step-by-step preprocessing & visualization for a single GSSI DZT (+DXT) file.

Implements the front-half of a standard concrete/dam GPR pipeline:
time-zero -> dewow -> background subtraction -> AGC -> band-pass,
plus quick attribute maps (envelope).

References in practice:
- ASTM D6432-19 standard guide for surface GPR processing order
- Daniels (2004) GPR 2nd ed. (dewow/background/gain/band-pass usage)
- Hugenschmidt (2002), Solla et al. (2014) for concrete structures

Usage
-----
python gpr_preproc_demo.py /path/to/file.DZT [/path/to/file.DXT] --freq 1600
"""

import argparse, os
import numpy as np
import matplotlib.pyplot as plt

from gpr_gssi_io import load_gssi_dzt_dxt
from gpr_anomaly_pipeline import GPRBScan, GPRPipeline, DefaultConfig

# Frequency-specific presets (ASTM D6432 관행 + 현장 경험 기반)
# 키: 안테나 중심 주파수 [MHz]
# 값: dewow/AGC 윈도우(ns), band-pass (Nyquist 분수)
FREQUENCY_PRESETS = {
    2600: {"dewow": 8.0,  "agc": 8.0,  "band": (0.05, 0.95)},
    1600: {"dewow": 12.0, "agc": 12.0, "band": (0.05, 0.90)},
    900:  {"dewow": 25.0, "agc": 20.0, "band": (0.04, 0.85)},
    400:  {"dewow": 60.0, "agc": 30.0, "band": (0.03, 0.65)},
}

def select_preset(freq_mhz: int, fallback_cfg: DefaultConfig):
    try:
        f = int(freq_mhz)
    except Exception:
        f = None
    if f is not None and len(FREQUENCY_PRESETS) > 0:
        nearest = min(FREQUENCY_PRESETS.keys(), key=lambda k: abs(k - f))
        P = FREQUENCY_PRESETS.get(f, FREQUENCY_PRESETS[nearest])
        return (
            float(P.get("dewow", fallback_cfg.dewow_window_ns)),
            float(P.get("agc", fallback_cfg.agc_window_ns)),
            tuple(P.get("band", fallback_cfg.bandpass_fraction)),
            int(nearest if f not in FREQUENCY_PRESETS else f),
        )
    return fallback_cfg.dewow_window_ns, fallback_cfg.agc_window_ns, fallback_cfg.bandpass_fraction, None

def plot_bscan(B, title, dt, dx, vmax=None, cmap="gray", meta=None):
    plt.figure(figsize=(12,6))
    
    print(f"플롯 데이터 shape: {B.shape}")
    
    # GPRPy GUI와 동일한 방향으로 표시
    # 데이터가 (n_times, n_traces) 형태이므로:
    # - 가로축: n_traces (거리)
    # - 세로축: n_times (시간)
    
    # GPRPy의 실제 시간 범위 사용
    if meta and "gprpy_time_range" in meta:
        time_start, time_end = meta["gprpy_time_range"]
        print(f"GPRPy 시간 범위 사용: {time_start:.2f}ns ~ {time_end:.2f}ns")
    else:
        time_start, time_end = 0, B.shape[1]*dt*1e9
        print(f"계산된 시간 범위 사용: {time_start:.2f}ns ~ {time_end:.2f}ns")
    
    if meta and "distance_range" in meta:
        x_start, x_end = meta["distance_range"]
        extent = [x_start, x_end, time_start, time_end]  # x[m], t[ns] - 시간이 위에서 아래로
    else:
        extent = [0, B.shape[1]*dx, time_start, time_end]  # x[m], t[ns] - 시간이 위에서 아래로
    
    B_display = B
    
    # imshow에서 origin='upper' 사용하여 GPR 표준 방향으로 표시 (0ns가 상단)
    plt.imshow(B_display, aspect='auto', extent=extent, cmap=cmap, 
               vmin=-vmax if vmax else None, vmax=vmax, origin='upper')
    plt.xlabel("Distance [m]")
    plt.ylabel("Time [ns]")
    plt.title(title)
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dzt")
    ap.add_argument("dxt", nargs="?", default=None)
    ap.add_argument("--freq", type=int, default=None, help="Antenna center frequency in MHz (optional override)")
    ap.add_argument("--out", type=str, default="./out_demo", help="Output directory")
    args = ap.parse_args()

    data, dt, dx, meta = load_gssi_dzt_dxt(args.dzt, args.dxt)
    if args.freq is not None:
        meta["freq_mhz"] = args.freq

    # 0) RAW
    plot_bscan(data, "RAW DZT", dt, dx, vmax=np.percentile(np.abs(data), 99), meta=meta)

    # 1) Preprocess using the robust pipeline
    cfg = DefaultConfig(save_npz=True, save_json_log=True)
    # 주파수별 프리셋 선택
    dewow_win, agc_win, band_frac, used_freq = select_preset(meta.get("freq_mhz"), cfg)
    print(f"[Preset] freq={meta.get('freq_mhz')} MHz -> dewow={dewow_win} ns, agc={agc_win} ns, band={band_frac}")

    # 파이프라인 설정에 반영 (파이프라인 내부도 방어적으로 동작하지만 여기서 우선 적용)
    cfg.dewow_window_ns = dewow_win
    cfg.agc_window_ns = agc_win
    cfg.bandpass_fraction = band_frac

    pipe = GPRPipeline(cfg)
    b = GPRBScan(data=data, dt=dt, dx=dx, meta={**meta, "line_id": os.path.basename(args.dzt)})
    pre = pipe.preprocess(b)

    # visualize key steps (we can re-apply pieces here for teaching clarity)
    from gpr_anomaly_pipeline import dewow, background_subtract, agc_gain, bandpass, analytic_envelope, robust_unit

    raw = data.copy()
    t0_idx = pre["time_zero_idx"]
    # replicate steps
    step1 = raw
    if t0_idx>0:
        pad = np.zeros((t0_idx, raw.shape[1]), dtype=raw.dtype)
        step1 = np.vstack([raw[t0_idx:], pad])

    step2 = dewow(step1, dt, window_ns=cfg.dewow_window_ns)
    step3 = background_subtract(step2) if cfg.background_removal else step2
    step4 = agc_gain(step3, dt, window_ns=cfg.agc_window_ns)
    lowf, highf = cfg.bandpass_fraction
    step5 = bandpass(step4, dt, lowf, highf, order=4)
    env = analytic_envelope(step5)
    env_n = robust_unit(env, axis=1)

    # 2) Plots
    vmax = np.percentile(np.abs(raw), 99)
    plot_bscan(step1, f"After time-zero shift (idx={t0_idx})", dt, dx, vmax=vmax, meta=meta)
    plot_bscan(step2, "Dewow", dt, dx, vmax=np.percentile(np.abs(step2), 99), meta=meta)
    plot_bscan(step3, "Background removal", dt, dx, vmax=np.percentile(np.abs(step3), 99), meta=meta)
    plot_bscan(step4, "AGC", dt, dx, vmax=np.percentile(np.abs(step4), 99), meta=meta)
    plot_bscan(step5, f"Band-pass ({lowf:.2f}-{highf:.2f} of Nyquist)", dt, dx, vmax=np.percentile(np.abs(step5), 99), meta=meta)
    plot_bscan(env_n, "Envelope (normalized, higher=brighter)", dt, dx, vmax=np.percentile(env_n, 99), cmap="viridis", meta=meta)

    # 3) Save minimal outputs
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "preproc_steps.npz"),
                        step_timezero=step1, dewow=step2, bg_removed=step3,
                        agc=step4, bandpass=step5, envelope=env_n, dt=dt, dx=dx, meta=meta)
    print("Saved:", os.path.join(args.out, "preproc_steps.npz"))

if __name__ == "__main__":
    main()
