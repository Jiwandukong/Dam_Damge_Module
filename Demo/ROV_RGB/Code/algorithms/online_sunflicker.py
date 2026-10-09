"""Online sunflicker removal pipeline (CPU-only)."""

from __future__ import annotations

import csv
import json
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LOG_FIELDS = [
    "frame_index",
    "basename",
    "alignment_status",
    "valid_history_frames",
    "mean_before",
    "mean_after",
    "temporal_model_status",
]
DEBUG_LOG_FIELDS = [
    "frame_index",
    "basename",
    "Y_min",
    "Y_max",
    "Y_mean",
    "Y_std",
    "B_min",
    "B_max",
    "B_mean",
    "B_std",
    "pred_min",
    "pred_max",
    "pred_mean",
    "pred_std",
    "L_pos_min",
    "L_pos_max",
    "L_pos_mean",
    "L_pos_std",
    "delta_min",
    "delta_max",
    "delta_mean",
    "delta_std",
    "R_pos_mean",
    "R_pos_std",
    "R_pos_max",
    "scale_raw",
    "scale_clipped",
    "mean_correction_bright_pre_fallback",
    "mean_correction_bright_post_fallback",
    "bright_pixels_delta_ge_1_ratio",
    "bright_pixels_delta_ge_2_ratio",
    "bright_pixels_delta_ge_5_ratio",
    "mean_abs_correction",
    "mean_correction_bright",
    "mean_correction_dark",
    "changed_ratio",
    "changed_dark_ratio",
    "changed_in_dark_over_changed",
    "mask_coverage",
    "warn_overdark",
    "warn_ineffective",
]


@dataclass
class SunflickerConfig:
    project_root: Path
    input_video: Path
    output_root: Path
    source_dataset_dir: Path
    debug_subset_dir: Path

    target_source: str = "dataset"  # dataset, video
    window_radius: int = 5
    window_type: str = "centered"  # centered, causal

    history: int = 5  # backward compatibility; used when target_source=video and window_radius not set by caller
    align_method: str = "ecc"  # ecc, orb_homography, none
    channel: str = "gray"  # gray, lab_l
    lp_sigma: float = 21.0
    latent_dim: int = 8
    prediction_model: str = "ar1"  # ar1, ema, linear
    correction: str = "bright_only_additive"  # bright_only_additive, additive, multiplicative
    correction_strength: float = 0.8

    export_video: bool = False
    export_all_frames: bool = False
    export_dataset: bool = True
    export_debug_subset: bool = False

    debug_every: int = 0
    max_frames: int = -1
    fps_output: float = -1.0
    codec: str = "mp4v"
    frame_name_pattern: str = "frame_{idx:06d}"
    output_video_name: Optional[str] = None

    min_valid_observations: int = 2
    model_history: int = 10
    downsample_factor: int = 8
    num_workers: int = 1
    parallel_mode: str = "none"  # none, chunk
    show_progress: bool = True

    def validate(self) -> None:
        if self.target_source not in {"dataset", "video"}:
            raise ValueError("target_source must be one of: dataset, video")
        if self.window_radius < 1:
            raise ValueError("window_radius must be >= 1")
        if self.window_type not in {"centered", "causal"}:
            raise ValueError("window_type must be one of: centered, causal")
        if self.history < 1:
            raise ValueError("history must be >= 1")
        if self.align_method not in {"ecc", "orb_homography", "none"}:
            raise ValueError("align_method must be one of: ecc, orb_homography, none")
        if self.channel not in {"gray", "lab_l"}:
            raise ValueError("channel must be one of: gray, lab_l")
        if self.lp_sigma <= 0:
            raise ValueError("lp_sigma must be > 0")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.prediction_model not in {"ar1", "ema", "linear"}:
            raise ValueError("prediction_model must be one of: ar1, ema, linear")
        if self.correction not in {"bright_only_additive", "additive", "multiplicative"}:
            raise ValueError("correction must be one of: bright_only_additive, additive, multiplicative")
        if not (0.0 <= self.correction_strength <= 2.0):
            raise ValueError("correction_strength must be in [0.0, 2.0]")
        if self.max_frames == 0 or self.max_frames < -1:
            raise ValueError("max_frames must be -1 or >= 1")
        if self.downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1")
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if self.parallel_mode not in {"none", "chunk"}:
            raise ValueError("parallel_mode must be one of: none, chunk")


class FrameCache:
    def __init__(self, video_path: Path, max_items: int = 64) -> None:
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        self.cache: Dict[int, np.ndarray] = {}
        self.order: List[int] = []
        self.max_items = max_items

    def close(self) -> None:
        self.cap.release()
        self.cache.clear()
        self.order.clear()

    def get(self, idx: int) -> Optional[np.ndarray]:
        if idx in self.cache:
            if idx in self.order:
                self.order.remove(idx)
            self.order.append(idx)
            return self.cache[idx].copy()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None

        self.cache[idx] = frame
        self.order.append(idx)
        if len(self.order) > self.max_items:
            oldest = self.order.pop(0)
            self.cache.pop(oldest, None)
        return frame.copy()


def ensure_dirs(output_root: Path) -> Dict[str, Path]:
    paths = {
        "videos": output_root / "videos",
        "frames_all": output_root / "frames_all",
        "dataset_out": output_root / "dataset_0.2_sunflicker_removed",
        "frames_debug_subset": output_root / "frames_debug_subset",
        "debug": output_root / "debug",
        "logs": output_root / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def make_basename(pattern: str, idx: int) -> str:
    return pattern.format(idx=idx)


def parse_frame_index_from_stem(stem: str) -> Optional[int]:
    parts = stem.split("_")
    if not parts:
        return None
    tail = parts[-1]
    if not tail.isdigit():
        return None
    return int(tail)


def scan_image_basenames(folder: Path) -> Set[str]:
    out: Set[str] = set()
    if not folder.exists():
        return out
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            out.add(f.stem)
    return out


def scan_dataset_targets(dataset_dir: Path, max_targets: int = -1) -> List[Tuple[int, str, str]]:
    targets: List[Tuple[int, str, str]] = []
    if not dataset_dir.exists():
        return targets

    for f in sorted(dataset_dir.iterdir()):
        if not (f.is_file() and f.suffix.lower() in IMAGE_EXTS):
            continue
        idx = parse_frame_index_from_stem(f.stem)
        if idx is None:
            continue
        targets.append((idx, f.stem, f.suffix.lower()))

    targets.sort(key=lambda x: x[0])
    if max_targets > 0:
        targets = targets[:max_targets]
    return targets


def get_local_window_indices(target_idx: int, frame_count: int, radius: int, window_type: str) -> List[int]:
    if window_type == "causal":
        start = max(0, target_idx - radius)
        end = target_idx
    else:
        start = max(0, target_idx - radius)
        end = min(frame_count - 1, target_idx + radius)
    return list(range(start, end + 1))


def extract_luminance(frame_bgr: np.ndarray, channel: str) -> np.ndarray:
    if channel == "gray":
        lum = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        lum = lab[:, :, 0]
    return lum.astype(np.float32)


def apply_corrected_luminance(frame_bgr: np.ndarray, corrected_lum: np.ndarray, channel: str) -> np.ndarray:
    corrected_u8 = np.clip(corrected_lum, 0, 255).astype(np.uint8)
    if channel == "gray":
        orig = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        delta = corrected_lum - orig
        out = frame_bgr.astype(np.float32)
        out[:, :, 0] += delta
        out[:, :, 1] += delta
        out[:, :, 2] += delta
        return np.clip(out, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = corrected_u8
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def align_neighbor_to_target(
    target_lum: np.ndarray,
    neighbor_lum: np.ndarray,
    method: str,
    orb_nfeatures: int = 1200,
    orb_ransac_thresh: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    h, w = target_lum.shape

    if method == "none":
        return neighbor_lum.copy(), np.ones((h, w), dtype=np.uint8), True, "none"

    if method == "ecc":
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
        try:
            cc, warp = cv2.findTransformECC(
                target_lum,
                neighbor_lum,
                warp,
                cv2.MOTION_AFFINE,
                criteria,
                None,
                1,
            )
            aligned = cv2.warpAffine(
                neighbor_lum,
                warp,
                (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            valid = cv2.warpAffine(
                np.ones((h, w), dtype=np.uint8),
                warp,
                (w, h),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            return aligned, valid, True, f"ecc_cc={cc:.5f}"
        except cv2.error as e:
            return (
                neighbor_lum.copy(),
                np.ones((h, w), dtype=np.uint8),
                False,
                f"ecc_fail:{str(e).splitlines()[0][:120]}",
            )

    orb = cv2.ORB_create(nfeatures=orb_nfeatures)
    tgt_u8 = np.clip(target_lum, 0, 255).astype(np.uint8)
    nbr_u8 = np.clip(neighbor_lum, 0, 255).astype(np.uint8)
    kp1, des1 = orb.detectAndCompute(tgt_u8, None)
    kp2, des2 = orb.detectAndCompute(nbr_u8, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return neighbor_lum.copy(), np.ones((h, w), dtype=np.uint8), False, "orb_fail:not_enough_keypoints"

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des2, des1, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return neighbor_lum.copy(), np.ones((h, w), dtype=np.uint8), False, "orb_fail:not_enough_matches"

    src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier = cv2.findHomography(src, dst, cv2.RANSAC, orb_ransac_thresh)
    if H is None:
        return neighbor_lum.copy(), np.ones((h, w), dtype=np.uint8), False, "orb_fail:homography_none"

    aligned = cv2.warpPerspective(
        neighbor_lum,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.warpPerspective(
        np.ones((h, w), dtype=np.uint8),
        H,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    inliers = int(inlier.sum()) if inlier is not None else 0
    return aligned, valid, True, f"orb_ok:inliers={inliers}/{len(good)}"


def masked_background_median(
    aligned_lums: Sequence[np.ndarray],
    valid_masks: Sequence[np.ndarray],
    fallback: np.ndarray,
    min_valid: int,
) -> np.ndarray:
    stack = np.stack(aligned_lums, axis=0).astype(np.float32)
    mask = np.stack(valid_masks, axis=0).astype(bool)
    stack_nan = np.where(mask, stack, np.nan)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(stack_nan, axis=0)
    count = np.sum(mask, axis=0)
    bad = (count < min_valid) | np.isnan(med)
    med[bad] = fallback[bad]
    return med.astype(np.float32)


def estimate_illumination_field(target_lum: np.ndarray, background: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(target_lum - background, (0, 0), sigmaX=sigma, sigmaY=sigma)


def downsample_field(field: np.ndarray, factor: int) -> np.ndarray:
    h, w = field.shape
    ds_h = max(8, h // factor)
    ds_w = max(8, w // factor)
    return cv2.resize(field, (ds_w, ds_h), interpolation=cv2.INTER_AREA)


def upsample_field(field_ds: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    h, w = out_shape
    return cv2.resize(field_ds, (w, h), interpolation=cv2.INTER_CUBIC)


def pca_project(data: np.ndarray, latent_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(axis=0, keepdims=True)
    centered = data - mean
    if centered.shape[0] < 2:
        k = min(latent_dim, centered.shape[1])
        comp = np.eye(centered.shape[1], dtype=np.float32)[:k]
        z = centered @ comp.T
        return z.astype(np.float32), comp.astype(np.float32), mean.astype(np.float32)

    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    k = max(1, min(latent_dim, vt.shape[0]))
    comp = vt[:k]
    z = centered @ comp.T
    return z.astype(np.float32), comp.astype(np.float32), mean.astype(np.float32)


def predict_latent(z_hist: np.ndarray, model: str) -> Tuple[np.ndarray, str]:
    t = z_hist.shape[0]
    if t == 0:
        return np.zeros((z_hist.shape[1],), dtype=np.float32), "empty"
    if t == 1:
        return z_hist[-1].copy(), "cold_start"

    if model == "ema":
        alpha = 0.6
        z = z_hist[0].copy()
        for i in range(1, t):
            z = alpha * z_hist[i] + (1.0 - alpha) * z
        return z.astype(np.float32), "ema"

    if model == "ar1":
        x = z_hist[:-1]
        y = z_hist[1:]
        num = np.sum(x * y, axis=0)
        den = np.sum(x * x, axis=0) + 1e-6
        a = np.clip(num / den, -1.2, 1.2)
        pred = a * z_hist[-1]
        return pred.astype(np.float32), "ar1"

    if t < 3:
        return z_hist[-1].copy(), "linear_fallback"
    xs = np.arange(t, dtype=np.float32)
    x0 = xs.mean()
    xv = xs - x0
    den = np.sum(xv * xv) + 1e-6
    slopes = ((z_hist - z_hist.mean(axis=0)) * xv[:, None]).sum(axis=0) / den
    return (z_hist[-1] + slopes).astype(np.float32), "linear"


def predict_illumination_field(illum_history_ds: Sequence[np.ndarray], latent_dim: int, model: str) -> Tuple[np.ndarray, str]:
    if len(illum_history_ds) == 0:
        raise RuntimeError("illum_history_ds is empty")

    sample = illum_history_ds[-1]
    ds_h, ds_w = sample.shape
    data = np.stack([x.reshape(-1) for x in illum_history_ds], axis=0).astype(np.float32)
    z, comp, mean = pca_project(data, latent_dim=latent_dim)
    z_pred, status = predict_latent(z, model)
    vec_pred = mean.reshape(-1) + comp.T @ z_pred
    return vec_pred.reshape(ds_h, ds_w).astype(np.float32), status


def _arr_stats(x: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
    }


def _robust_spread(x: np.ndarray) -> float:
    p90 = float(np.percentile(x, 90))
    p10 = float(np.percentile(x, 10))
    return max(0.0, p90 - p10)


def validate_correction_behavior(target_lum: np.ndarray, background: np.ndarray, delta: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    eps = 1e-6
    changed = delta > 1e-3
    dark_region = target_lum <= background
    bright_region = ~dark_region

    changed_ratio = float(np.mean(changed))
    changed_dark_ratio = float(np.mean(changed & dark_region))
    changed_in_dark_over_changed = float(np.sum(changed & dark_region) / (np.sum(changed) + eps))
    mean_corr_bright = float(np.mean(delta[bright_region])) if np.any(bright_region) else 0.0
    mean_corr_dark = float(np.mean(delta[dark_region])) if np.any(dark_region) else 0.0
    mean_abs_corr = float(np.mean(np.abs(delta)))
    mask_coverage = float(np.mean(mask > 0.1))

    warn_overdark = 1.0 if changed_in_dark_over_changed > 0.35 else 0.0
    warn_ineffective = 1.0 if mean_abs_corr < 0.3 else 0.0

    return {
        "mean_abs_correction": mean_abs_corr,
        "mean_correction_bright": mean_corr_bright,
        "mean_correction_dark": mean_corr_dark,
        "changed_ratio": changed_ratio,
        "changed_dark_ratio": changed_dark_ratio,
        "changed_in_dark_over_changed": changed_in_dark_over_changed,
        "mask_coverage": mask_coverage,
        "warn_overdark": warn_overdark,
        "warn_ineffective": warn_ineffective,
    }


def correct_luminance(
    target_lum: np.ndarray,
    background: np.ndarray,
    pred_illum: np.ndarray,
    correction: str,
    strength: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    eps = 1e-6
    residual = target_lum - background
    r_pos = np.maximum(residual, 0.0)
    l_pos = np.maximum(pred_illum, 0.0)

    tau = float(np.percentile(r_pos, 20))
    p95 = float(np.percentile(r_pos, 95))
    mask = np.clip((r_pos - tau) / (max(p95 - tau, 1e-3)), 0.0, 1.0)

    gate_soft = 0.3 + 0.7 * mask
    mask_active = mask > 0.15
    if np.any(mask_active):
        r90 = float(np.percentile(r_pos[mask_active], 90))
        l90 = float(np.percentile(l_pos[mask_active], 90))
    else:
        r90 = float(np.percentile(r_pos, 90))
        l90 = float(np.percentile(l_pos, 90))
    scale_raw = r90 / (l90 + eps)
    scale = float(np.clip(scale_raw, 1.0, 8.0))
    s = float(np.clip(strength, 0.0, 2.0))

    if correction == "bright_only_additive":
        delta_base = s * scale * l_pos * gate_soft
        delta_base = np.minimum(delta_base, r_pos)
        mean_corr_bright_pre = float(np.mean(delta_base[mask_active])) if np.any(mask_active) else 0.0

        delta = delta_base
        # Ineffective-correction fallback: auto-boost if too weak in bright mask.
        if mean_corr_bright_pre < 2.0:
            boost = float(np.clip(2.0 / (mean_corr_bright_pre + eps), 1.0, 2.5))
            gate_fallback = np.where(mask > 0.05, 1.0, 0.0).astype(np.float32)
            delta = np.maximum(delta, s * scale * boost * l_pos * gate_fallback)
            delta = np.minimum(delta, r_pos)
        mean_corr_bright_post = float(np.mean(delta[mask_active])) if np.any(mask_active) else 0.0
        corrected = np.clip(target_lum - delta, 0, 255)
    elif correction == "additive":
        mean_corr_bright_pre = 0.0
        mean_corr_bright_post = 0.0
        delta = s * pred_illum
        corrected = np.clip(target_lum - delta, 0, 255)
    else:
        mean_corr_bright_pre = 0.0
        mean_corr_bright_post = 0.0
        mean_t = float(np.mean(target_lum))
        denom = np.clip(1.0 + (s * pred_illum / (mean_t + eps)), 0.5, 2.0)
        corrected = target_lum / denom
        corrected *= mean_t / (float(np.mean(corrected)) + eps)
        corrected = np.clip(corrected, 0, 255)
        delta = np.clip(target_lum - corrected, 0, 255)

    abs_diff = np.abs(target_lum - corrected)
    diag_arrays = {
        "residual": residual,
        "l_pos": l_pos,
        "mask": mask,
        "delta": delta,
        "abs_diff": abs_diff,
    }

    stats: Dict[str, float] = {}
    stats.update(_arr_stats(target_lum, "Y"))
    stats.update(_arr_stats(background, "B"))
    stats.update(_arr_stats(pred_illum, "pred"))
    stats.update(_arr_stats(l_pos, "L_pos"))
    stats.update(_arr_stats(delta, "delta"))
    stats["R_pos_mean"] = float(np.mean(r_pos))
    stats["R_pos_std"] = float(np.std(r_pos))
    stats["R_pos_max"] = float(np.max(r_pos))
    stats["scale_raw"] = float(scale_raw)
    stats["scale_clipped"] = float(scale)
    stats["mean_correction_bright_pre_fallback"] = float(mean_corr_bright_pre)
    stats["mean_correction_bright_post_fallback"] = float(mean_corr_bright_post)
    if np.any(mask_active):
        bright_delta = delta[mask_active]
        stats["bright_pixels_delta_ge_1_ratio"] = float(np.mean(bright_delta >= 1.0))
        stats["bright_pixels_delta_ge_2_ratio"] = float(np.mean(bright_delta >= 2.0))
        stats["bright_pixels_delta_ge_5_ratio"] = float(np.mean(bright_delta >= 5.0))
    else:
        stats["bright_pixels_delta_ge_1_ratio"] = 0.0
        stats["bright_pixels_delta_ge_2_ratio"] = 0.0
        stats["bright_pixels_delta_ge_5_ratio"] = 0.0
    stats.update(validate_correction_behavior(target_lum, background, delta, mask))
    return corrected.astype(np.float32), diag_arrays, stats


def save_debug(
    debug_dir: Path,
    basename: str,
    target_lum: np.ndarray,
    background: np.ndarray,
    illum_recent: np.ndarray,
    pred_illum: np.ndarray,
    residual: np.ndarray,
    l_pos: np.ndarray,
    mask: np.ndarray,
    delta: np.ndarray,
    corrected_lum: np.ndarray,
    abs_diff: np.ndarray,
    target_bgr: np.ndarray,
    corrected_bgr: np.ndarray,
    model_status: str,
) -> None:
    cv2.imwrite(str(debug_dir / f"{basename}_01_target_lum.png"), np.clip(target_lum, 0, 255).astype(np.uint8))
    cv2.imwrite(str(debug_dir / f"{basename}_02_background.png"), np.clip(background, 0, 255).astype(np.uint8))
    cv2.imwrite(str(debug_dir / f"{basename}_03_residual.png"), np.clip(residual + 128.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(debug_dir / f"{basename}_04_pred_illum.png"), np.clip(pred_illum + 128.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(debug_dir / f"{basename}_05_L_pos.png"), np.clip(l_pos + 128.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(debug_dir / f"{basename}_06_mask.png"), np.clip(mask * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(debug_dir / f"{basename}_07_delta.png"), np.clip(delta, 0, 255).astype(np.uint8))
    cv2.imwrite(str(debug_dir / f"{basename}_08_corrected_lum.png"), np.clip(corrected_lum, 0, 255).astype(np.uint8))
    cv2.imwrite(str(debug_dir / f"{basename}_09_abs_diff.png"), np.clip(abs_diff, 0, 255).astype(np.uint8))
    cv2.imwrite(str(debug_dir / f"{basename}_10_side_by_side_{model_status}.png"), np.hstack([target_bgr, corrected_bgr]))


def process_target_frame(
    cfg: SunflickerConfig,
    cache: FrameCache,
    target_idx: int,
    frame_count: int,
    width: int,
    height: int,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, float]],
    str,
    int,
    int,
]:
    target_bgr = cache.get(target_idx)
    if target_bgr is None:
        return None, None, None, None, None, None, "target_missing", 0, 0

    target_lum = extract_luminance(target_bgr, cfg.channel)

    window_indices = get_local_window_indices(target_idx, frame_count, cfg.window_radius, cfg.window_type)
    aligned_lums: List[np.ndarray] = [target_lum]
    valid_masks: List[np.ndarray] = [np.ones((height, width), dtype=np.uint8)]
    illum_hist_ds: Deque[np.ndarray] = deque(maxlen=max(2, cfg.model_history))

    align_ok = 0
    align_fail = 0

    for idx in window_indices:
        if idx == target_idx:
            continue
        fr_bgr = cache.get(idx)
        if fr_bgr is None:
            continue
        fr_lum = extract_luminance(fr_bgr, cfg.channel)
        aligned, valid, ok, _ = align_neighbor_to_target(target_lum, fr_lum, cfg.align_method)
        aligned_lums.append(aligned)
        valid_masks.append(valid)
        if ok:
            align_ok += 1
        else:
            align_fail += 1

    background = masked_background_median(
        aligned_lums,
        valid_masks,
        fallback=target_lum,
        min_valid=cfg.min_valid_observations,
    )

    # Build local illumination history from aligned local clip only.
    for aligned in aligned_lums:
        illum = estimate_illumination_field(aligned, background, sigma=cfg.lp_sigma)
        illum_hist_ds.append(downsample_field(illum, factor=cfg.downsample_factor))

    pred_ds, model_status = predict_illumination_field(
        list(illum_hist_ds),
        latent_dim=cfg.latent_dim,
        model=cfg.prediction_model,
    )
    pred_illum = upsample_field(pred_ds, out_shape=(height, width))
    corrected_lum, corr_diag, corr_stats = correct_luminance(
        target_lum,
        background,
        pred_illum,
        cfg.correction,
        cfg.correction_strength,
    )
    corrected_bgr = apply_corrected_luminance(target_bgr, corrected_lum, cfg.channel)

    if align_ok == 0 and align_fail == 0:
        align_status = "no_history"
    elif align_fail == 0:
        align_status = "success"
    elif align_ok == 0:
        align_status = "failure"
    else:
        align_status = "partial"

    if corr_stats.get("warn_overdark", 0.0) > 0.5:
        model_status = f"{model_status}|warn_overdark"
    if corr_stats.get("warn_ineffective", 0.0) > 0.5:
        model_status = f"{model_status}|warn_weak"

    return (
        corrected_bgr,
        target_lum,
        background,
        pred_illum,
        corr_diag,
        corr_stats,
        f"{model_status}|{align_status}",
        len(aligned_lums) - 1,
        align_fail,
    )


def split_chunks(items: Sequence[Tuple[int, str, str]], n_chunks: int) -> List[List[Tuple[int, str, str]]]:
    total = len(items)
    if total == 0:
        return []
    n_chunks = max(1, min(n_chunks, total))
    base = total // n_chunks
    rem = total % n_chunks
    out: List[List[Tuple[int, str, str]]] = []
    s = 0
    for i in range(n_chunks):
        k = base + (1 if i < rem else 0)
        e = s + k
        out.append(list(items[s:e]))
        s = e
    return out


def _process_chunk_worker(
    cfg: SunflickerConfig,
    chunk_id: int,
    chunk_items: Sequence[Tuple[int, str, str]],
    frame_count: int,
    width: int,
    height: int,
    out_paths: Dict[str, str],
    debug_subset_basenames: Set[str],
) -> Dict[str, object]:
    out = {k: Path(v) for k, v in out_paths.items()}
    cache = FrameCache(cfg.input_video, max_items=max(64, (2 * cfg.window_radius + 1) * 4))

    chunk_csv = out["logs"] / f"sunflicker_chunk_{chunk_id:03d}.csv"
    csv_file = chunk_csv.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=LOG_FIELDS)
    csv_writer.writeheader()
    chunk_debug_csv = out["logs"] / f"sunflicker_debug_chunk_{chunk_id:03d}.csv"
    debug_csv_file = chunk_debug_csv.open("w", newline="", encoding="utf-8")
    debug_csv_writer = csv.DictWriter(debug_csv_file, fieldnames=DEBUG_LOG_FIELDS)
    debug_csv_writer.writeheader()

    result = {
        "chunk_id": chunk_id,
        "csv_path": str(chunk_csv),
        "debug_csv_path": str(chunk_debug_csv),
        "processed_count": 0,
        "exported_all_count": 0,
        "exported_dataset_count": 0,
        "copied_json_count": 0,
        "exported_debug_subset_count": 0,
    }

    try:
        iterator: Sequence[Tuple[int, str, str]] | tqdm = chunk_items
        if cfg.show_progress:
            iterator = tqdm(
                chunk_items,
                desc=f"Chunk {chunk_id:02d}",
                position=chunk_id + 1,
                leave=True,
                dynamic_ncols=True,
            )
        for n, (target_idx, stem, ext) in enumerate(iterator):
            if target_idx < 0 or target_idx >= frame_count:
                continue

            corrected_bgr, target_lum, background, pred_illum, corr_diag, corr_stats, status, valid_hist, align_fail = process_target_frame(
                cfg, cache, target_idx, frame_count, width, height
            )
            if corrected_bgr is None or target_lum is None or background is None or pred_illum is None or corr_diag is None or corr_stats is None:
                continue

            basename = stem
            filename = f"{basename}{ext}"

            if cfg.export_all_frames:
                cv2.imwrite(str(out["frames_all"] / filename), corrected_bgr)
                result["exported_all_count"] += 1

            if cfg.export_dataset and cfg.target_source == "dataset":
                cv2.imwrite(str(out["dataset_out"] / filename), corrected_bgr)
                result["exported_dataset_count"] += 1
                src_json = cfg.source_dataset_dir / f"{basename}.json"
                if src_json.exists():
                    dst_json = out["dataset_out"] / f"{basename}.json"
                    if not dst_json.exists():
                        shutil.copy2(src_json, dst_json)
                        result["copied_json_count"] += 1

            if cfg.export_debug_subset and basename in debug_subset_basenames:
                cv2.imwrite(str(out["frames_debug_subset"] / filename), corrected_bgr)
                result["exported_debug_subset_count"] += 1

            if cfg.debug_every > 0 and (target_idx % cfg.debug_every == 0):
                illum_recent = estimate_illumination_field(target_lum, background, sigma=cfg.lp_sigma)
                corrected_lum = extract_luminance(corrected_bgr, cfg.channel)
                orig = cache.get(target_idx)
                save_debug(
                    out["debug"],
                    basename,
                    target_lum,
                    background,
                    illum_recent,
                    pred_illum,
                    corr_diag["residual"],
                    corr_diag["l_pos"],
                    corr_diag["mask"],
                    corr_diag["delta"],
                    corrected_lum,
                    corr_diag["abs_diff"],
                    orig if orig is not None else corrected_bgr,
                    corrected_bgr,
                    status,
                )
                debug_row = {"frame_index": target_idx, "basename": filename}
                debug_row.update(corr_stats)
                debug_csv_writer.writerow(debug_row)

            csv_writer.writerow(
                {
                    "frame_index": target_idx,
                    "basename": filename,
                    "alignment_status": "failure"
                    if align_fail == valid_hist and valid_hist > 0
                    else "success"
                    if align_fail == 0
                    else "partial"
                    if valid_hist > 0
                    else "no_history",
                    "valid_history_frames": valid_hist,
                    "mean_before": float(np.mean(target_lum)),
                    "mean_after": float(np.mean(extract_luminance(corrected_bgr, cfg.channel))),
                    "temporal_model_status": status,
                }
            )
            result["processed_count"] += 1
    finally:
        cache.close()
        csv_file.close()
        debug_csv_file.close()

    return result


def run_online_sunflicker(cfg: SunflickerConfig) -> Dict[str, object]:
    cfg.validate()
    if not cfg.input_video.exists():
        raise FileNotFoundError(f"input video not found: {cfg.input_video}")

    out = ensure_dirs(cfg.output_root)
    debug_subset_basenames = scan_image_basenames(cfg.debug_subset_dir)

    cap_meta = cv2.VideoCapture(str(cfg.input_video))
    if not cap_meta.isOpened():
        raise RuntimeError(f"cannot open input video: {cfg.input_video}")
    frame_count = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = float(cap_meta.get(cv2.CAP_PROP_FPS))
    width = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_meta.release()

    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("invalid video metadata")

    if cfg.target_source == "dataset":
        dataset_targets = scan_dataset_targets(cfg.source_dataset_dir, max_targets=cfg.max_frames)
        work_items = dataset_targets
    else:
        process_count = frame_count if cfg.max_frames < 0 else min(frame_count, cfg.max_frames)
        work_items = [(i, make_basename(cfg.frame_name_pattern, i), ".jpg") for i in range(process_count)]

    writer = None
    if cfg.export_video:
        out_name = cfg.output_video_name or f"{cfg.input_video.stem}_sunflicker_removed.mp4"
        fps_out = cfg.fps_output if cfg.fps_output > 0 else (fps_in if fps_in > 0 else 30.0)
        fourcc = cv2.VideoWriter_fourcc(*cfg.codec)
        writer = cv2.VideoWriter(str(out["videos"] / out_name), fourcc, fps_out, (width, height))
        if not writer.isOpened():
            raise RuntimeError("cannot initialize video writer")

    run_cfg_path = out["logs"] / "run_config.json"
    run_cfg_path.write_text(
        json.dumps(
            {
                "project_root": str(cfg.project_root),
                "input_video": str(cfg.input_video),
                "output_root": str(cfg.output_root),
                "source_dataset_dir": str(cfg.source_dataset_dir),
                "debug_subset_dir": str(cfg.debug_subset_dir),
                "target_source": cfg.target_source,
                "window_radius": cfg.window_radius,
                "window_type": cfg.window_type,
                "history": cfg.history,
                "align_method": cfg.align_method,
                "channel": cfg.channel,
                "lp_sigma": cfg.lp_sigma,
                "latent_dim": cfg.latent_dim,
                "prediction_model": cfg.prediction_model,
                "correction": cfg.correction,
                "correction_strength": cfg.correction_strength,
                "export_video": cfg.export_video,
                "export_all_frames": cfg.export_all_frames,
                "export_dataset": cfg.export_dataset,
                "export_debug_subset": cfg.export_debug_subset,
                "debug_every": cfg.debug_every,
                "max_frames": cfg.max_frames,
                "fps_output": cfg.fps_output,
                "codec": cfg.codec,
                "frame_name_pattern": cfg.frame_name_pattern,
                "min_valid_observations": cfg.min_valid_observations,
                "model_history": cfg.model_history,
                "downsample_factor": cfg.downsample_factor,
                "num_workers": cfg.num_workers,
                "parallel_mode": cfg.parallel_mode,
                "show_progress": cfg.show_progress,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = out["logs"] / "sunflicker_log.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=LOG_FIELDS)
    csv_writer.writeheader()
    debug_csv_path = out["logs"] / "sunflicker_debug_metrics.csv"
    debug_csv_file = debug_csv_path.open("w", newline="", encoding="utf-8")
    debug_csv_writer = csv.DictWriter(debug_csv_file, fieldnames=DEBUG_LOG_FIELDS)
    debug_csv_writer.writeheader()

    summary = {
        "frame_count": frame_count,
        "target_count": len(work_items),
        "processed_count": 0,
        "exported_all_count": 0,
        "exported_dataset_count": 0,
        "copied_json_count": 0,
        "exported_debug_subset_count": 0,
        "csv_log": str(csv_path),
        "debug_metrics_log": str(debug_csv_path),
        "run_config": str(run_cfg_path),
    }

    try:
        if cfg.parallel_mode == "chunk" and cfg.num_workers > 1:
            if writer is not None:
                writer.release()
                writer = None
                raise ValueError(
                    "export_video is not supported with parallel_mode=chunk. Use parallel_mode=none for video export."
                )

            chunks = split_chunks(work_items, cfg.num_workers)
            out_paths = {k: str(v) for k, v in out.items()}
            results: List[Dict[str, object]] = []
            bar = tqdm(total=len(chunks), desc="Sunflicker chunks", disable=not cfg.show_progress)
            try:
                with ProcessPoolExecutor(max_workers=cfg.num_workers) as ex:
                    futures = [
                        ex.submit(
                            _process_chunk_worker,
                            cfg,
                            i,
                            chunk,
                            frame_count,
                            width,
                            height,
                            out_paths,
                            debug_subset_basenames,
                        )
                        for i, chunk in enumerate(chunks)
                    ]
                    for fut in as_completed(futures):
                        results.append(fut.result())
                        bar.update(1)
            finally:
                bar.close()

            for r in results:
                summary["processed_count"] += int(r["processed_count"])
                summary["exported_all_count"] += int(r["exported_all_count"])
                summary["exported_dataset_count"] += int(r["exported_dataset_count"])
                summary["copied_json_count"] += int(r["copied_json_count"])
                summary["exported_debug_subset_count"] += int(r["exported_debug_subset_count"])

            for r in sorted(results, key=lambda x: int(x["chunk_id"])):
                with Path(str(r["csv_path"])).open("r", newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        csv_writer.writerow(row)
                with Path(str(r["debug_csv_path"])).open("r", newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        debug_csv_writer.writerow(row)
        else:
            cache = FrameCache(cfg.input_video, max_items=max(64, (2 * cfg.window_radius + 1) * 4))
            try:
                iterator = tqdm(work_items, desc="Sunflicker targets", disable=not cfg.show_progress)
                for n, (target_idx, stem, ext) in enumerate(iterator):
                    if target_idx < 0 or target_idx >= frame_count:
                        continue

                    corrected_bgr, target_lum, background, pred_illum, corr_diag, corr_stats, status, valid_hist, align_fail = process_target_frame(
                        cfg, cache, target_idx, frame_count, width, height
                    )
                    if corrected_bgr is None or target_lum is None or background is None or pred_illum is None or corr_diag is None or corr_stats is None:
                        continue

                    basename = stem
                    filename = f"{basename}{ext}"

                    if writer is not None:
                        writer.write(corrected_bgr)

                    if cfg.export_all_frames:
                        cv2.imwrite(str(out["frames_all"] / filename), corrected_bgr)
                        summary["exported_all_count"] += 1

                    if cfg.export_dataset and cfg.target_source == "dataset":
                        cv2.imwrite(str(out["dataset_out"] / filename), corrected_bgr)
                        summary["exported_dataset_count"] += 1
                        src_json = cfg.source_dataset_dir / f"{basename}.json"
                        if src_json.exists():
                            dst_json = out["dataset_out"] / f"{basename}.json"
                            if not dst_json.exists():
                                shutil.copy2(src_json, dst_json)
                                summary["copied_json_count"] += 1

                    if cfg.export_debug_subset and basename in debug_subset_basenames:
                        cv2.imwrite(str(out["frames_debug_subset"] / filename), corrected_bgr)
                        summary["exported_debug_subset_count"] += 1

                    if cfg.debug_every > 0 and (n % cfg.debug_every == 0):
                        illum_recent = estimate_illumination_field(target_lum, background, sigma=cfg.lp_sigma)
                        corrected_lum = extract_luminance(corrected_bgr, cfg.channel)
                        orig = cache.get(target_idx)
                        save_debug(
                            out["debug"],
                            basename,
                            target_lum,
                            background,
                            illum_recent,
                            pred_illum,
                            corr_diag["residual"],
                            corr_diag["l_pos"],
                            corr_diag["mask"],
                            corr_diag["delta"],
                            corrected_lum,
                            corr_diag["abs_diff"],
                            orig if orig is not None else corrected_bgr,
                            corrected_bgr,
                            status,
                        )
                        debug_row = {"frame_index": target_idx, "basename": filename}
                        debug_row.update(corr_stats)
                        debug_csv_writer.writerow(debug_row)

                    csv_writer.writerow(
                        {
                            "frame_index": target_idx,
                            "basename": filename,
                            "alignment_status": "failure"
                            if align_fail == valid_hist and valid_hist > 0
                            else "success"
                            if align_fail == 0
                            else "partial"
                            if valid_hist > 0
                            else "no_history",
                            "valid_history_frames": valid_hist,
                            "mean_before": float(np.mean(target_lum)),
                            "mean_after": float(np.mean(extract_luminance(corrected_bgr, cfg.channel))),
                            "temporal_model_status": status,
                        }
                    )
                    summary["processed_count"] += 1
            finally:
                cache.close()
    finally:
        csv_file.close()
        debug_csv_file.close()
        if writer is not None:
            writer.release()

    return summary
