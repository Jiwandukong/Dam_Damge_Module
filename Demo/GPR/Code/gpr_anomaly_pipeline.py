"""
GPR Anomaly Pipeline (multi-line x multi-frequency)
===================================================
Goal
----
Given multiple B-scans (12 lines) acquired with multiple antennas (e.g., 2.6/1.6/0.9/0.4 GHz),
this module performs *repeatable* preprocessing and produces *evidence-based* anomaly maps
indicative of voids/delamination/moisture ingress/corrosion-prone zones and crack-like scattering.

Design principles are aligned with widely-used practices and literature:
- ASTM D6432-19: Standard Guide for Using the Surface Ground Penetrating Radar Method for Subsurface Investigation
- Daniels, D. J. (2004). Ground Penetrating Radar (2nd ed.). IET.
- Hugenschmidt, J. (2000, 2002); Solla et al. (2014 review): GPR for concrete/bridge deck condition assessment
- Maierhofer et al. (2010): NDT of concrete – GPR processing/interpretation
The implemented steps (time-zero, dewow, background removal, gain, bandpass, migration-optional,
attribute analysis, attenuation estimation, scattering density, robust anomaly scoring) reflect
common procedures reported in those works.

Inputs
------
This module is format-agnostic. Load your B-scan into a numpy array of shape (n_times, n_traces)
per line/frequency and pass sampling and geometry metadata.

Minimal requirements per B-scan:
- data: np.ndarray (n_times, n_traces), dtype float32/float64
- dt:   sample interval in seconds (e.g., 0.5e-9 for 0.5 ns)
- dx:   trace spacing in meters (optional but recommended; default=1.0)
- meta: dict with optional fields:
        {"freq_mhz": 1600, "line_id": "LINE_1", "antenna_id": "1.6GHz",
         "time_zero_ns": None, "velocity_m_per_ns": None}

Outputs
-------
- Preprocessed B-scan
- Feature maps: envelope (energy), instantaneous frequency, attenuation alpha,
  scattering density (hyperbola-like), coherence (background similarity)
- Anomaly score map (0..1), and per-depth "risk profile"
- Artifacts saved under an output folder you choose (npz/csv/png)

Usage (pseudo)
--------------
from gpr_anomaly_pipeline import GPRBScan, GPRPipeline, DefaultConfig

bscan = GPRBScan(data, dt=0.5e-9, dx=0.02, meta={"freq_mhz": 1600, "line_id":"LINE_1"})
pipe = GPRPipeline(DefaultConfig())
result = pipe.run(bscan, out_dir="./out/LINE_1_1600MHz")

# result is a dict with keys:
#   ["preprocessed", "envelope", "inst_freq", "atten_alpha", "scatter_density",
#    "coherence", "anomaly", "risk_profile_depth", "logs"]

Notes
-----
- Migration is optional and requires a representative velocity (m/ns) or a v(z) profile.
- Depth conversion is intentionally separated from XY mapping; keep it in TWT domain if unsure.
- Multi-frequency fusion is provided as a simple weighted-combine helper.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import os
import json
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter, uniform_filter1d, sobel, maximum_filter, label
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest

# -----------------------------
# Configuration
# -----------------------------

@dataclass
class DefaultConfig:
    # Preprocessing
    time_zero_method: str = "autopick"   # ["autopick", "given", "none"]
    time_zero_ns: Optional[float] = None # if given
    dewow_window_ns: float = 40.0        # typical 10-80 ns for concrete
    background_removal: bool = True      # subtract trace-mean across x
    agc_window_ns: float = 20.0          # automatic gain control window
    bandpass_fraction: Tuple[float,float] = (0.15, 0.9)  # as fraction of Nyquist (safety net)

    # Attribute params
    inst_freq_smooth_sigma: float = 1.0
    scatter_lof_sigma: float = 1.0       # smoothing before scatter detection
    scatter_peak_prominence: float = 2.5 # z-score threshold for peaks
    scatter_max_filter: Tuple[int,int] = (3,3)  # neighborhood for local maxima

    # Attenuation estimation (per-trace log-linear fit on envelope vs time)
    atten_tmin_ns: float = 20.0
    atten_tmax_ns: float = 200.0
    atten_window_ns: float = 40.0        # rolling window for local alpha

    # Robust anomaly score
    anomaly_weights: Dict[str,float] = None  # set in __post_init__
    isolation_forest: bool = True
    isolation_fraction: float = 0.02     # expected anomalies
    random_state: int = 42

    # Saving
    save_npz: bool = True
    save_png: bool = False               # set True to save quicklook images
    save_json_log: bool = True

    def __post_init__(self):
        if self.anomaly_weights is None:
            # weights sum is arbitrary; score will be normalized
            self.anomaly_weights = {
                "envelope": 1.0,         # high energy (after normalization) as indicator
                "inst_freq": 0.6,        # abnormal instantaneous frequency
                "atten_alpha": 0.8,      # high attenuation (moisture/steel corrosion proxy)
                "scatter_density": 1.2,  # hyperbola-like scatterers (cracks/voids/targets)
                "coherence": 0.6         # low coherence regions among traces
            }

# Antenna-specific suggested band limits (Hz fractions relative to Nyquist).
# You can override with bandpass_fraction.
SUGGESTED_BANDS = {
    # These are typical; fine-tune with your dt.
    2600: (0.05, 0.95),
    1600: (0.05, 0.9),
    # Keep these identical to the values used to create the delivered
    # rule-based outputs (see gpr_preproc_demo.FREQUENCY_PRESETS).
    900:  (0.04, 0.85),
    400:  (0.03, 0.65),
}

# Frequency-aware presets (ns windows and fractional bandpass),
# inspired by ASTM D6432 guidance and common field practice.
FREQUENCY_PRESETS = {
    2600: {
        "dewow_window_ns": 8.0,
        "agc_window_ns": 8.0,
        "bandpass_fraction": SUGGESTED_BANDS[2600],
    },
    1600: {
        "dewow_window_ns": 12.0,
        "agc_window_ns": 12.0,
        "bandpass_fraction": SUGGESTED_BANDS[1600],
    },
    900: {
        "dewow_window_ns": 25.0,
        "agc_window_ns": 20.0,
        "bandpass_fraction": SUGGESTED_BANDS[900],
    },
    400: {
        "dewow_window_ns": 60.0,
        "agc_window_ns": 30.0,
        "bandpass_fraction": SUGGESTED_BANDS[400],
    },
}

def _select_preset_for_frequency(freq_mhz: int, cfg: "DefaultConfig"):
    """Select preprocessing params for the given frequency.

    If exact match not found, choose the nearest available preset.
    Fallback to config defaults when presets are missing.
    """
    try:
        f = int(freq_mhz)
    except Exception:
        f = None

    if f is not None and len(FREQUENCY_PRESETS) > 0:
        nearest = min(FREQUENCY_PRESETS.keys(), key=lambda k: abs(k - f))
        preset = FREQUENCY_PRESETS.get(f, FREQUENCY_PRESETS[nearest])
        band = preset.get("bandpass_fraction", cfg.bandpass_fraction)
        dewow_ns = float(preset.get("dewow_window_ns", cfg.dewow_window_ns))
        agc_ns = float(preset.get("agc_window_ns", cfg.agc_window_ns))
        return band, dewow_ns, agc_ns

    return cfg.bandpass_fraction, cfg.dewow_window_ns, cfg.agc_window_ns

# -----------------------------
# Data structure
# -----------------------------

@dataclass
class GPRBScan:
    data: np.ndarray             # (n_times, n_traces)
    dt: float                    # sample interval [s]
    dx: float = 1.0              # trace spacing [m]
    meta: Optional[Dict] = None  # {"freq_mhz":..., "line_id":..., "antenna_id":..., "time_zero_ns":..., "velocity_m_per_ns":...}

    def copy(self):
        return GPRBScan(self.data.copy(), self.dt, self.dx, dict(self.meta) if self.meta else None)

    @property
    def nt(self):
        return int(self.data.shape[0])

    @property
    def nx(self):
        return int(self.data.shape[1])

    @property
    def time_ns(self):
        return np.arange(self.nt) * self.dt * 1e9

# -----------------------------
# Utility functions
# -----------------------------

def _butter_bandpass(low, high, fs, order=4):
    b,a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return b,a

def bandpass(data, dt, low_frac, high_frac, order=4):
    fs = 1.0/dt
    b,a = butter(order, [low_frac*(fs/2), high_frac*(fs/2)], btype="band", fs=fs)
    return filtfilt(b,a, data, axis=0, padlen=min(200, data.shape[0]-1))

def dewow(data, dt, window_ns=40.0):
    """Dewow by subtracting running mean (low-frequency drift)."""
    if window_ns is None or window_ns <= 0:
        return data
    w = max(1, int((window_ns*1e-9)/dt))
    mean_trend = uniform_filter1d(data, size=w, axis=0, mode="nearest")
    return data - mean_trend

def background_subtract(data):
    """Subtract average trace across x (removes horizontal banding from direct wave)."""
    mean_trace = np.mean(data, axis=1, keepdims=True)
    return data - mean_trace

def agc_gain(data, dt, window_ns=20.0, eps=1e-6):
    """Simple AGC: divide by running RMS in time."""
    w = max(1, int((window_ns*1e-9)/dt))
    sq = data**2
    rms = np.sqrt(uniform_filter1d(sq, size=w, axis=0, mode="nearest")) + eps
    return data / rms

def time_zero_pick(data, method="autopick"):
    """Auto-pick time-zero as the first strong onset in average trace derivative."""
    if method != "autopick":
        return 0
    avg = np.mean(data, axis=1)
    g = np.gradient(avg)
    # choose first index above 3 * MAD of derivative
    mad = median_abs_deviation(g, scale="normal") + 1e-9
    idx = np.argmax(g > 3.0*mad)
    return int(idx)

def apply_time_zero(data, idx0):
    if idx0 <= 0:
        return data
    pad = np.zeros((idx0, data.shape[1]), dtype=data.dtype)
    truncated = data[idx0:]
    return np.vstack([truncated, pad])

def analytic_envelope(data):
    """Hilbert transform envelope (magnitude)."""
    analytic = hilbert(data, axis=0)
    return np.abs(analytic)

def instantaneous_frequency(data, dt):
    """Compute instantaneous frequency from analytic signal derivative."""
    analytic = hilbert(data, axis=0)
    phase = np.unwrap(np.angle(analytic), axis=0)
    inst_freq = np.gradient(phase, dt, axis=0) / (2*np.pi)  # Hz
    return inst_freq

def coherence_across_traces(data, win_traces=9):
    """Simple coherence proxy: 1 - normalized local variance across traces."""
    # local mean and variance across traces in a sliding window
    k = win_traces
    if k < 3:
        return np.ones_like(data, dtype=float)
    pad = k//2
    # cumulative sums for fast moving window statistics
    csum = np.cumsum(data, axis=1)
    csum2 = np.cumsum(data**2, axis=1)
    # windowed sums
    def wsum(arr):
        left = np.pad(arr[:, :-k], ((0,0),(k,0)), mode='edge')
        return arr - left
    S1 = wsum(csum)
    S2 = wsum(csum2)
    n = float(k)
    mean = S1 / n
    var = (S2 - (S1**2)/n) / max(1.0, (n-1.0))
    std = np.sqrt(np.maximum(var, 0.0))
    # normalize per-time row
    std_norm = std / (np.mean(std, axis=1, keepdims=True) + 1e-6)
    coh = 1.0 - np.clip(std_norm, 0.0, 2.0)/2.0
    return coh

def scattering_density(envelope, z_thresh=2.5, max_filter_size=(3,3), smooth_sigma=1.0):
    """Hyperbola-like scattering density: count of local envelope peaks (z-scored)."""
    # robust z-score per-depth row
    med = np.median(envelope, axis=1, keepdims=True)
    try:
        mad = median_abs_deviation(envelope, axis=1, scale="normal", keepdims=True)
    except TypeError:
        mad0 = median_abs_deviation(envelope, axis=1, scale="normal")
        mad = np.expand_dims(mad0, axis=1)
    mad = mad + 1e-9
    z = (envelope - med) / mad
    z = gaussian_filter(z, sigma=smooth_sigma)
    # local maxima
    neigh = maximum_filter(z, size=max_filter_size)
    peaks = (z == neigh) & (z > z_thresh)
    # density by box filter
    density = gaussian_filter(peaks.astype(float), sigma=1.0)
    return density

def attenuation_alpha(envelope, dt, tmin_ns=20.0, tmax_ns=200.0, window_ns=40.0):
    """
    Estimate attenuation coefficient alpha (1/s) locally by linear fit to log(envelope) vs time.
    Returns a map same size as envelope.
    """
    t = np.arange(envelope.shape[0]) * dt
    tmin = int(max(0, (tmin_ns*1e-9)/dt))
    tmax = int(min(envelope.shape[0], (tmax_ns*1e-9)/dt))
    w = max(3, int((window_ns*1e-9)/dt))
    loge = np.log(envelope + 1e-9)
    alpha = np.zeros_like(envelope)
    # rolling linear regression y = a + b t -> alpha ~= -b
    for x in range(envelope.shape[1]):
        y = loge[tmin:tmax, x]
        # moving window fit via cumulative sums for speed
        n = len(y)
        if n < w: 
            continue
        tvec = t[:n]
        # precompute sums
        Sx = uniform_filter1d(tvec, size=w, mode="nearest") * w
        Sy = uniform_filter1d(y, size=w, mode="nearest") * w
        Sxx = uniform_filter1d(tvec**2, size=w, mode="nearest") * w
        Sxy = uniform_filter1d(tvec*y, size=w, mode="nearest") * w
        denom = (w*Sxx - Sx**2) + 1e-12
        b = (w*Sxy - Sx*Sy) / denom
        # place result into center region
        alpha_local = -b  # attenuation ~ negative slope
        alpha[tmin:tmin+len(alpha_local), x] = alpha_local
    # normalize alpha to 0..1 for anomaly combination
    a = alpha.copy()
    a -= np.nanmin(a)
    if np.nanmax(a) > 0:
        a /= np.nanmax(a)
    return a

def robust_unit(z, axis=None):
    """Robust 0..1 scaling using median/MAD."""
    med = np.nanmedian(z, axis=axis, keepdims=True)
    # Support SciPy versions without keepdims in median_abs_deviation
    try:
        mad = median_abs_deviation(z, axis=axis, scale="normal", keepdims=True)
    except TypeError:
        mad0 = median_abs_deviation(z, axis=axis, scale="normal")
        if axis is None:
            mad = mad0
        else:
            mad = np.expand_dims(mad0, axis=axis)
    mad = mad + 1e-9
    zz = (z - med) / (3.0*mad)  # ~±3 MAD -> ~±1
    zz = np.clip(zz, -1.0, 3.0)
    zz -= np.nanmin(zz)
    if np.nanmax(zz) > 0:
        zz /= np.nanmax(zz)
    return zz

# -----------------------------
# Pipeline
# -----------------------------

class GPRPipeline:
    def __init__(self, cfg: DefaultConfig):
        self.cfg = cfg

    def preprocess(self, b: GPRBScan) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        data = np.asarray(b.data, dtype=np.float32)

        # Time-zero
        if cfg.time_zero_method == "given" and b.meta and b.meta.get("time_zero_ns") is not None:
            idx0 = int((b.meta["time_zero_ns"]*1e-9)/b.dt)
        elif cfg.time_zero_method == "autopick":
            idx0 = time_zero_pick(data, method="autopick")
        else:
            idx0 = 0
        data = apply_time_zero(data, idx0)

        # Frequency-aware parameter selection (dewow/AGC/bandpass)
        lowhigh, dewow_win_ns, agc_win_ns = _select_preset_for_frequency(
            b.meta.get("freq_mhz") if b.meta else None, cfg
        )

        # Dewow
        data = dewow(data, b.dt, window_ns=dewow_win_ns)

        # Background removal
        if cfg.background_removal:
            data = background_subtract(data)

        # AGC
        data = agc_gain(data, b.dt, window_ns=agc_win_ns)

        # Bandpass
        lowf, highf = lowhigh
        data = bandpass(data, b.dt, lowf, highf, order=4)

        try:
            fmsg = b.meta.get("freq_mhz") if b.meta else None
            print(
                f"[Preprocess] freq={fmsg} MHz | dewow={dewow_win_ns} ns, "
                f"agc={agc_win_ns} ns, band={lowf:.2f}-{highf:.2f} x Nyquist"
            )
        except Exception:
            pass

        return {"preprocessed": data, "time_zero_idx": idx0}

    def attributes(self, pre: Dict[str, np.ndarray], b: GPRBScan) -> Dict[str, np.ndarray]:
        data = pre["preprocessed"]
        env = analytic_envelope(data)
        instf = instantaneous_frequency(data, b.dt)
        instf = gaussian_filter(instf, sigma=self.cfg.inst_freq_smooth_sigma)

        # Normalize features to 0..1 robustly
        env_n = robust_unit(env, axis=1)
        instf_n = robust_unit(np.abs(instf), axis=1)

        # Scattering density (hyperbola-like
        sca = scattering_density(env, z_thresh=self.cfg.scatter_peak_prominence,
                                  max_filter_size=self.cfg.scatter_max_filter,
                                  smooth_sigma=self.cfg.scatter_lof_sigma)
        sca_n = robust_unit(sca)

        # Attenuation alpha
        att = attenuation_alpha(env, b.dt,
                                tmin_ns=self.cfg.atten_tmin_ns,
                                tmax_ns=self.cfg.atten_tmax_ns,
                                window_ns=self.cfg.atten_window_ns)

        # Coherence (low coherence is suspicious) -> invert
        coh = coherence_across_traces(data, win_traces=9)
        incoh = robust_unit(1.0 - coh)

        return {
            "envelope": env_n,
            "inst_freq": instf_n,
            "scatter_density": sca_n,
            "atten_alpha": att,
            "coherence": incoh,
        }

    def anomaly_map(self, feats: Dict[str,np.ndarray]) -> np.ndarray:
        # weighted sum of normalized features
        W = self.cfg.anomaly_weights
        score = np.zeros_like(next(iter(feats.values())))
        for k,w in W.items():
            if k in feats:
                score += w * feats[k]
        # normalize 0..1
        score = robust_unit(score)
        return score

    def refine_with_isolation_forest(self, score: np.ndarray) -> np.ndarray:
        # Optional unsupervised refinement at pixel-level using IsolationForest
        if not self.cfg.isolation_forest:
            return score
        X = score.reshape(-1,1)
        iso = IsolationForest(n_estimators=200,
                              contamination=self.cfg.isolation_fraction,
                              random_state=self.cfg.random_state)
        iso.fit(X)
        s = -iso.decision_function(X)   # higher -> more anomalous
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s.reshape(score.shape)

    def risk_profile_per_depth(self, score: np.ndarray) -> np.ndarray:
        # mean risk along traces for each time/depth
        return np.mean(score, axis=1)

    def run(self, b: GPRBScan, out_dir: Optional[str]=None) -> Dict[str, np.ndarray]:
        os.makedirs(out_dir or ".", exist_ok=True)
        pre = self.preprocess(b)
        feats = self.attributes(pre, b)
        score = self.anomaly_map(feats)
        score_ref = self.refine_with_isolation_forest(score)
        risk_depth = self.risk_profile_per_depth(score_ref)

        result = {
            "preprocessed": pre["preprocessed"],
            "envelope": feats["envelope"],
            "inst_freq": feats["inst_freq"],
            "scatter_density": feats["scatter_density"],
            "atten_alpha": feats["atten_alpha"],
            "coherence": feats["coherence"],
            "anomaly": score_ref,
            "risk_profile_depth": risk_depth,
            "logs": {
                "time_zero_idx": int(pre["time_zero_idx"]),
                "config": self.cfg.__dict__,
                "meta": b.meta if b.meta else {}
            }
        }
        if out_dir is not None:
            np.savez_compressed(os.path.join(out_dir, "result_maps.npz"), **{
                k: v for k,v in result.items() if isinstance(v, np.ndarray)
            })
            if self.cfg.save_json_log:
                with open(os.path.join(out_dir, "log.json"), "w", encoding="utf-8") as f:
                    json.dump(result["logs"], f, indent=2, ensure_ascii=False)
        return result


# -----------------------------
# Multi-frequency fusion
# -----------------------------

def fuse_multifreq(scores: List[np.ndarray], weights: Optional[List[float]]=None,
                   method: str="weighted_mean") -> np.ndarray:
    """
    Fuse multiple anomaly score maps (same shape) into a single map.
    - "max": element-wise maximum (conservative, highlights any anomaly)
    - "weighted_mean": default; weights favor high-frequency near-surface and low-frequency deeper.
    """
    S = np.stack(scores, axis=0)  # (n_f, nt, nx)
    if method == "max":
        return np.max(S, axis=0)
    if weights is None:
        # default weights: higher freq gets larger near top; simple depth weighting
        n_f, nt, nx = S.shape
        w = np.ones((n_f, nt, 1), dtype=float)
        # linearly decay weight with depth index for high freq
        for i in range(n_f):
            decay = 1.0 - 0.7*(np.arange(nt)/max(1,nt-1))
            w[i,:,0] = decay**(i+1)  # higher index -> stronger decay
        W = w / (w.sum(axis=0, keepdims=True) + 1e-9)
    else:
        W = np.array(weights, dtype=float).reshape(-1,1,1)
        W = W / (W.sum() + 1e-9)
    F = (S * W).sum(axis=0)
    # normalize
    return (F - F.min()) / (F.max() - F.min() + 1e-9)


# -----------------------------
# Example stub (does not run by default)
# -----------------------------

if __name__ == "__main__":
    import numpy as np
    # Example synthetic B-scan for quick smoke test (no real units)
    nt, nx = 500, 200
    dt = 0.5e-9   # 0.5 ns
    data = np.random.normal(0, 1e-3, size=(nt,nx)).astype(np.float32)

    # inject a fake hyperbola-like scattering region
    for x0 in [50, 120, 170]:
        for x in range(nx):
            t0 = 60 + int(0.15*(x-x0)**2)  # parabolic curve (toy)
            if 0 <= t0 < nt:
                data[t0, x] += 1.0

    b = GPRBScan(data=data, dt=dt, dx=0.02, meta={"freq_mhz": 1600, "line_id":"LINE_1"})
    pipe = GPRPipeline(DefaultConfig(save_npz=True, save_png=False))
    out_dir = "./demo_LINE1_1600MHz"
    out = pipe.run(b, out_dir=out_dir)
    print(f"Saved demo outputs to {out_dir}")
