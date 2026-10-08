
import os
import math
import numpy as np
import pandas as pd
import laspy
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree, Delaunay, ConvexHull
from scipy.interpolate import UnivariateSpline
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed

# ============================================================
# CONFIG
# ============================================================
SLAB_LAS_FP = "/home/gunwoo/multibeam/multibeam/0318_slab/output_open3d_patch_A/A_align_filled_slab_zone_open3d_patch_final_final2.las"
OUT_DIR = "/home/gunwoo/multibeam/multibeam/0318_slab/local_damage_quant_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- slicing / healthy zone ----------
SLICE_WIDTH_M = 0.25
TRIM_Q_PROFILE = (0.10, 0.90)
MIN_POINTS_PER_SLICE = 40
HEALTHY_Q_SLOPE = 0.80
HEALTHY_Q_CURV = 0.80
HEALTHY_Q_ROUGH = 0.80
HEALTHY_MIN_DENSITY_Q = 0.10
MIN_HEALTHY_RUN_SLICES = 5
PREFER_EDGE_RUNS = False

# ---------- robust plane ----------
PLANE_MAX_ITERS = 8
PLANE_MAD_K = 2.8
PLANE_MIN_KEEP_RATIO = 0.55

# ---------- reference profile ----------
PROFILE_SMOOTHING = None   # None이면 자동, 숫자를 주면 spline smoothing factor로 사용
PROFILE_MIN_HEALTHY_SLICES = 12

# ---------- local geometric features ----------
K_NEIGHBORS = 30
CHUNK_SIZE = 10000
N_JOBS = 30
COMPUTE_NORMAL_VARIATION = False   # True면 한 번 더 계산하므로 느려진다

# ---------- distress candidate thresholds ----------
RESIDUAL_MAD_K = 2.5
CURVATURE_Q = 0.85
ROUGHNESS_Q = 0.85
PLANARITY_Q_MAX = 0.80      # 너무 비평면인 곳을 강조할 때 사용
MIN_DAMAGE_DEPTH_M = 0.005  # 최소 음의 local residual (m)
DISTRESS_MODE = "negative"  # "negative" 또는 "absolute"

# ---------- clustering ----------
DBSCAN_EPS_M = 0.10
DBSCAN_MIN_SAMPLES = 25
MIN_PATCH_POINTS = 50
MAX_TRI_EDGE_M = 0.25

# ---------- plotting ----------
FIG_DPI = 150
POINT_PLOT_SAMPLE = 200000
RANDOM_SEED = 42



# ============================================================
# HELPERS
# ============================================================

def fit_pca_axes(points_xyz):
    """
    3D PCA로 major/minor/normal 축을 추정한다.
    반환:
        centroid: (3,)
        axes: (3,3) = [major, minor, normal] column-wise
        eigvals_desc: descending eigenvalues
    """
    centroid = points_xyz.mean(axis=0)
    centered = points_xyz - centroid
    cov = centered.T @ centered / max(len(points_xyz) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major = eigvecs[:, 0]
    minor = eigvecs[:, 1]
    normal = eigvecs[:, 2]

    # normal을 +Z 방향으로 통일
    if normal[2] < 0:
        normal = -normal

    # right-handed basis 보정
    minor = np.cross(normal, major)
    minor /= np.linalg.norm(minor) + 1e-12
    major = np.cross(minor, normal)
    major /= np.linalg.norm(major) + 1e-12

    axes = np.column_stack([major, minor, normal])
    return centroid, axes, eigvals


def to_local_coords(points_xyz, origin_xyz, axes):
    """
    global XYZ -> local (s, t, h)
    s: major-axis direction
    t: minor-axis direction
    h: normal direction
    """
    return (points_xyz - origin_xyz) @ axes


def contiguous_runs(mask_bool):
    """True 구간의 [start, end] 인덱스 목록 반환."""
    runs = []
    start = None
    for i, v in enumerate(mask_bool):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask_bool) - 1))
    return runs


def trimmed_stat(x, q_low=0.10, q_high=0.90):
    """trimmed median, trimmed std 계산."""
    if len(x) == 0:
        return np.nan, np.nan
    lo, hi = np.quantile(x, [q_low, q_high])
    xt = x[(x >= lo) & (x <= hi)]
    if len(xt) == 0:
        return np.nan, np.nan
    return np.median(xt), np.std(xt)


def slice_profile_stats(local_coords, slice_width, min_points=30, trim_q=(0.10, 0.90)):
    """
    장축 s를 따라 슬라이스를 만들고 각 슬라이스별 representative height, roughness, density를 계산한다.
    반환 DataFrame columns:
      slice_id, s_center, n_points, z_med, z_std, s_min, s_max
    """
    s = local_coords[:, 0]
    h = local_coords[:, 2]

    s_min = np.min(s)
    s_max = np.max(s)
    edges = np.arange(s_min, s_max + slice_width, slice_width)
    if len(edges) < 2:
        edges = np.array([s_min, s_max + slice_width])

    slice_id = np.digitize(s, edges) - 1
    records = []
    for sid in range(len(edges) - 1):
        mask = (slice_id == sid)
        n = int(mask.sum())
        ss = s[mask]
        hh = h[mask]
        if n < min_points:
            z_med, z_std = np.nan, np.nan
        else:
            z_med, z_std = trimmed_stat(hh, trim_q[0], trim_q[1])
        records.append({
            "slice_id": sid,
            "s_center": 0.5 * (edges[sid] + edges[sid + 1]),
            "n_points": n,
            "z_med": z_med,
            "z_std": z_std,
            "s_min": edges[sid],
            "s_max": edges[sid + 1],
        })
    prof = pd.DataFrame(records)

    z = prof["z_med"].to_numpy(dtype=float)
    ds = slice_width
    slope = np.full(len(prof), np.nan)
    curv = np.full(len(prof), np.nan)
    for i in range(1, len(prof) - 1):
        if np.isfinite(z[i - 1]) and np.isfinite(z[i]) and np.isfinite(z[i + 1]):
            slope[i] = abs((z[i + 1] - z[i - 1]) / (2.0 * ds))
            curv[i] = abs((z[i + 1] - 2.0 * z[i] + z[i - 1]) / (ds ** 2))

    prof["slope_abs"] = slope
    prof["curv_abs"] = curv
    return prof


def select_healthy_slices(profile_df,
                          q_slope=0.70,
                          q_curv=0.70,
                          q_rough=0.70,
                          q_density_min=0.20,
                          min_run=6,
                          prefer_edges=True):
    """
    기울기·곡률·거칠기·밀도 조건으로 healthy slice를 자동 선택한다.
    반환:
        updated profile_df (healthy_candidate, healthy_final 포함)
        selected_runs: [(start,end), ...]
    """
    df = profile_df.copy()

    finite_slope = df["slope_abs"].dropna()
    finite_curv = df["curv_abs"].dropna()
    finite_rough = df["z_std"].dropna()
    finite_density = df["n_points"].dropna()

    T_slope = finite_slope.quantile(q_slope) if len(finite_slope) else np.inf
    T_curv = finite_curv.quantile(q_curv) if len(finite_curv) else np.inf
    T_rough = finite_rough.quantile(q_rough) if len(finite_rough) else np.inf
    T_density = finite_density.quantile(q_density_min) if len(finite_density) else 0

    cond = (
        df["z_med"].notna() &
        (df["n_points"] >= T_density) &
        (df["slope_abs"].fillna(np.inf) <= T_slope) &
        (df["curv_abs"].fillna(np.inf) <= T_curv) &
        (df["z_std"].fillna(np.inf) <= T_rough)
    )
    df["healthy_candidate"] = cond

    runs = contiguous_runs(cond.to_numpy())
    runs = [(a, b) for a, b in runs if (b - a + 1) >= min_run]

    if prefer_edges and len(runs) > 2:
        # 양 끝에 가까운 run을 우선 선택
        left_run = min(runs, key=lambda x: x[0])
        right_run = max(runs, key=lambda x: x[1])
        runs = [left_run] if left_run == right_run else [left_run, right_run]

    final_mask = np.zeros(len(df), dtype=bool)
    for a, b in runs:
        final_mask[a:b+1] = True
    df["healthy_final"] = final_mask

    thresholds = {
        "T_slope": T_slope,
        "T_curv": T_curv,
        "T_rough": T_rough,
        "T_density": T_density,
    }
    return df, runs, thresholds


def fit_pca_plane(points_xyz):
    """Orthogonal TLS plane via PCA."""
    centroid = points_xyz.mean(axis=0)
    centered = points_xyz - centroid
    cov = centered.T @ centered / max(len(points_xyz) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    normal = eigvecs[:, -1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    if normal[2] < 0:
        normal = -normal
    return centroid, normal, eigvals, eigvecs


def signed_point_plane_distance(points_xyz, centroid, normal):
    return (points_xyz - centroid) @ normal


def robust_tls_plane(points_xyz,
                     max_iters=8,
                     mad_k=2.8,
                     min_keep_ratio=0.55):
    """
    iterative MAD-trimmed orthogonal plane fitting.
    반환:
        centroid, normal, keep_mask, history
    """
    keep = np.ones(len(points_xyz), dtype=bool)
    history = []

    for it in range(max_iters):
        pts_k = points_xyz[keep]
        centroid, normal, eigvals, eigvecs = fit_pca_plane(pts_k)
        dist = signed_point_plane_distance(points_xyz, centroid, normal)
        dist_k = dist[keep]

        med = np.median(dist_k)
        mad = 1.4826 * np.median(np.abs(dist_k - med))
        if mad < 1e-12:
            history.append({"iter": it, "n_keep": int(keep.sum()), "mad": mad, "stopped": True})
            break

        new_keep = np.abs(dist - med) <= (mad_k * mad)
        if new_keep.mean() < min_keep_ratio:
            # 너무 많이 버리는 상황 방지: 중앙 quantile trim으로 완화
            qlo, qhi = np.quantile(dist, [0.10, 0.90])
            new_keep = (dist >= qlo) & (dist <= qhi)

        history.append({"iter": it, "n_keep": int(new_keep.sum()), "mad": float(mad), "stopped": False})

        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    centroid, normal, eigvals, eigvecs = fit_pca_plane(points_xyz[keep])
    return centroid, normal, keep, pd.DataFrame(history)


def build_plane_basis(normal, major_hint):
    """
    normal과 major axis hint를 이용해 plane 내부 basis u,v를 만든다.
    u는 major_hint를 plane에 투영한 방향, v는 normal x u.
    """
    u = major_hint - np.dot(major_hint, normal) * normal
    nu = np.linalg.norm(u)
    if nu < 1e-12:
        # fallback: normal과 수직인 임의축 사용
        if abs(normal[2]) < 0.9:
            tmp = np.array([0.0, 0.0, 1.0])
        else:
            tmp = np.array([1.0, 0.0, 0.0])
        u = np.cross(tmp, normal)
        nu = np.linalg.norm(u)
    u = u / (nu + 1e-12)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v


def project_to_plane_frame(points_xyz, plane_centroid, u, v, normal):
    d = points_xyz - plane_centroid
    s = d @ u
    t = d @ v
    h = d @ normal
    return np.column_stack([s, t, h])


def fit_reference_profile(slice_df, healthy_mask, smoothing=None):
    """
    healthy slice의 z_med를 이용해 장축 reference profile spline을 적합한다.
    반환:
      spline_function(s)
    """
    use = slice_df[healthy_mask & slice_df["z_med"].notna()].copy()
    if len(use) < 4:
        return None

    x = use["s_center"].to_numpy(dtype=float)
    y = use["z_med"].to_numpy(dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # 중복 x 방지
    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]
    if len(xu) < 4:
        return None

    # smoothing factor 자동 설정: healthy residual variance 기반
    if smoothing is None:
        if len(yu) > 1:
            s_val = len(yu) * np.var(yu)
        else:
            s_val = 0.0
    else:
        s_val = smoothing

    k = min(3, len(xu) - 1)
    spline = UnivariateSpline(xu, yu, k=k, s=s_val)
    return spline


def _feature_batch(points_xyz, idx_batch, tree, k_neighbors):
    dists, nbr_idx = tree.query(points_xyz[idx_batch], k=k_neighbors, workers=1)
    normals = np.zeros((len(idx_batch), 3), dtype=float)
    curvature = np.full(len(idx_batch), np.nan, dtype=float)
    roughness = np.full(len(idx_batch), np.nan, dtype=float)
    planarity = np.full(len(idx_batch), np.nan, dtype=float)

    for i, inds in enumerate(nbr_idx):
        nbrs = points_xyz[inds]
        c = nbrs.mean(axis=0)
        X = nbrs - c
        cov = X.T @ X / max(len(nbrs) - 1, 1)
        evals, evecs = np.linalg.eigh(cov)  # ascending
        evals = np.maximum(evals, 0.0)
        l0, l1, l2 = evals[0], evals[1], evals[2]
        n = evecs[:, 0]
        normals[i] = n / (np.linalg.norm(n) + 1e-12)
        curvature[i] = l0 / (l0 + l1 + l2 + 1e-12)
        roughness[i] = math.sqrt(l0)
        planarity[i] = (l1 - l0) / (l2 + 1e-12)

    return idx_batch, normals, curvature, roughness, planarity


def compute_local_features(points_xyz, k_neighbors=30, chunk_size=10000, n_jobs=-1):
    """
    각 점의 local normal, curvature, roughness, planarity를 계산한다.
    대용량을 고려해 chunk + joblib 병렬로 구현한다.
    """
    n = len(points_xyz)
    tree = cKDTree(points_xyz)
    batches = [np.arange(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_feature_batch)(points_xyz, batch, tree, k_neighbors) for batch in batches
    )

    normals = np.zeros((n, 3), dtype=float)
    curvature = np.full(n, np.nan, dtype=float)
    roughness = np.full(n, np.nan, dtype=float)
    planarity = np.full(n, np.nan, dtype=float)

    for idx_batch, nrm, curv, rough, plan in results:
        normals[idx_batch] = nrm
        curvature[idx_batch] = curv
        roughness[idx_batch] = rough
        planarity[idx_batch] = plan

    return normals, curvature, roughness, planarity


def _normal_var_batch(points_xyz, normals, idx_batch, tree, k_neighbors):
    _, nbr_idx = tree.query(points_xyz[idx_batch], k=k_neighbors, workers=1)
    out = np.full(len(idx_batch), np.nan, dtype=float)
    for i, inds in enumerate(nbr_idx):
        n0 = normals[idx_batch[i]]
        nn = normals[inds]
        dots = np.clip(np.abs(nn @ n0), -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))
        out[i] = np.median(angles)
    return idx_batch, out


def compute_normal_variation(points_xyz, normals, k_neighbors=30, chunk_size=10000, n_jobs=-1):
    tree = cKDTree(points_xyz)
    n = len(points_xyz)
    batches = [np.arange(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_normal_var_batch)(points_xyz, normals, batch, tree, k_neighbors) for batch in batches
    )
    out = np.full(n, np.nan, dtype=float)
    for idx_batch, vals in results:
        out[idx_batch] = vals
    return out


def robust_threshold_negative(x, mad_k=2.5):
    med = np.median(x[np.isfinite(x)])
    mad = 1.4826 * np.median(np.abs(x[np.isfinite(x)] - med))
    thr = med - mad_k * mad
    return thr, med, mad


def triangle_area_2d(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def compute_patch_metrics(local_sth, depth_pos, labels, slab_bbox_st, max_tri_edge=0.25):
    """
    local_sth: (N,3) = [s,t,h]
    depth_pos: (N,) 0 이상, 파손 깊이
    labels: DBSCAN label (-1 제외)
    slab_bbox_st: dict with s_min, s_max, t_min, t_max

    반환 DataFrame: patch별 area/depth/volume/edge relation
    """
    records = []
    unique_labels = sorted([lb for lb in np.unique(labels) if lb != -1])

    for lb in unique_labels:
        mask = (labels == lb)
        pts = local_sth[mask]
        dep = depth_pos[mask]
        if len(pts) < MIN_PATCH_POINTS:
            continue

        st = pts[:, :2]
        s = st[:, 0]
        t = st[:, 1]

        area = np.nan
        volume = np.nan
        perimeter = np.nan
        tri_count = 0

        try:
            if len(st) >= 3:
                hull = ConvexHull(st)
                hull_area = float(hull.volume)   # 2D에서 volume=area
                hull_perim = float(hull.area)    # 2D에서 area=perimeter
            else:
                hull_area = np.nan
                hull_perim = np.nan

            tri = Delaunay(st)
            area_sum = 0.0
            vol_sum = 0.0
            tri_count = 0
            for simp in tri.simplices:
                p1, p2, p3 = st[simp[0]], st[simp[1]], st[simp[2]]
                e12 = np.linalg.norm(p1 - p2)
                e23 = np.linalg.norm(p2 - p3)
                e31 = np.linalg.norm(p3 - p1)
                if max(e12, e23, e31) > max_tri_edge:
                    continue
                a = triangle_area_2d(p1, p2, p3)
                if a <= 0:
                    continue
                mean_dep = float(np.mean(dep[simp]))
                area_sum += a
                vol_sum += a * mean_dep
                tri_count += 1

            if tri_count > 0:
                area = area_sum
                volume = vol_sum
                perimeter = hull_perim
            else:
                area = hull_area
                volume = hull_area * float(np.mean(dep)) if np.isfinite(hull_area) else np.nan
                perimeter = hull_perim
        except Exception:
            # fallback: convex hull 기반 근사
            if len(st) >= 3:
                hull = ConvexHull(st)
                area = float(hull.volume)
                perimeter = float(hull.area)
                volume = area * float(np.mean(dep))

        s_min, s_max = slab_bbox_st["s_min"], slab_bbox_st["s_max"]
        t_min, t_max = slab_bbox_st["t_min"], slab_bbox_st["t_max"]
        dist_to_edges = np.vstack([
            s - s_min,
            s_max - s,
            t - t_min,
            t_max - t,
        ])
        min_edge_dist = float(np.min(dist_to_edges))
        edge_touch = bool(min_edge_dist <= DBSCAN_EPS_M)

        records.append({
            "patch_id": int(lb),
            "n_points": int(mask.sum()),
            "centroid_s": float(np.mean(s)),
            "centroid_t": float(np.mean(t)),
            "s_min": float(np.min(s)),
            "s_max": float(np.max(s)),
            "t_min": float(np.min(t)),
            "t_max": float(np.max(t)),
            "max_depth_m": float(np.max(dep)),
            "mean_depth_m": float(np.mean(dep)),
            "p95_depth_m": float(np.quantile(dep, 0.95)),
            "patch_area_m2": float(area) if np.isfinite(area) else np.nan,
            "patch_volume_m3": float(volume) if np.isfinite(volume) else np.nan,
            "patch_perimeter_m": float(perimeter) if np.isfinite(perimeter) else np.nan,
            "tri_count": int(tri_count),
            "min_edge_dist_m": min_edge_dist,
            "edge_touch": edge_touch,
        })

    if len(records) == 0:
        return pd.DataFrame(columns=[
            "patch_id", "n_points", "centroid_s", "centroid_t", "s_min", "s_max", "t_min", "t_max",
            "max_depth_m", "mean_depth_m", "p95_depth_m", "patch_area_m2", "patch_volume_m3",
            "patch_perimeter_m", "tri_count", "min_edge_dist_m", "edge_touch"
        ])
    return pd.DataFrame(records).sort_values(["patch_volume_m3", "patch_area_m2"], ascending=False)



# ============================================================
# 1) LOAD POINT CLOUD
# ============================================================
las = laspy.read(SLAB_LAS_FP)
pts = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
pts = pts[np.isfinite(pts).all(axis=1)]

print(f"[INFO] total slab points: {len(pts):,}")

# initial slab axes from PCA
centroid0, axes0, eigvals0 = fit_pca_axes(pts)
local0 = to_local_coords(pts, centroid0, axes0)

print("[INFO] initial eigenvalues:", eigvals0)
print("[INFO] major axis:", axes0[:, 0])
print("[INFO] minor axis:", axes0[:, 1])
print("[INFO] normal axis:", axes0[:, 2])



# ============================================================
# 2) SLICE PROFILE + HEALTHY ZONE DETECTION
# ============================================================
profile_df = slice_profile_stats(
    local0,
    slice_width=SLICE_WIDTH_M,
    min_points=MIN_POINTS_PER_SLICE,
    trim_q=TRIM_Q_PROFILE
)

profile_df, healthy_runs, healthy_thr = select_healthy_slices(
    profile_df,
    q_slope=HEALTHY_Q_SLOPE,
    q_curv=HEALTHY_Q_CURV,
    q_rough=HEALTHY_Q_ROUGH,
    q_density_min=HEALTHY_MIN_DENSITY_Q,
    min_run=MIN_HEALTHY_RUN_SLICES,
    prefer_edges=PREFER_EDGE_RUNS,
)

print("[INFO] healthy thresholds:", healthy_thr)
print("[INFO] healthy runs:", healthy_runs)

# point-level healthy slice mask
s0 = local0[:, 0]
healthy_point_mask = np.zeros(len(pts), dtype=bool)
for _, row in profile_df[profile_df["healthy_final"]].iterrows():
    healthy_point_mask |= (s0 >= row["s_min"]) & (s0 < row["s_max"])

print(f"[INFO] healthy points: {healthy_point_mask.sum():,}")

profile_csv = os.path.join(OUT_DIR, "01_slice_profile.csv")
profile_df.to_csv(profile_csv, index=False)
print("[SAVE]", profile_csv)



# ============================================================
# 3) ROBUST SURROGATE REFERENCE PLANE
# ============================================================
pts_healthy = pts[healthy_point_mask]
plane_centroid, plane_normal, plane_keep_mask, plane_hist = robust_tls_plane(
    pts_healthy,
    max_iters=PLANE_MAX_ITERS,
    mad_k=PLANE_MAD_K,
    min_keep_ratio=PLANE_MIN_KEEP_RATIO,
)

print("[INFO] reference plane centroid:", plane_centroid)
print("[INFO] reference plane normal:", plane_normal)
print(f"[INFO] final healthy inliers used for plane: {plane_keep_mask.sum():,} / {len(plane_keep_mask):,}")

a_u, a_v = build_plane_basis(plane_normal, axes0[:, 0])
plane_axes = np.column_stack([a_u, a_v, plane_normal])
local_plane = project_to_plane_frame(pts, plane_centroid, a_u, a_v, plane_normal)

# global residual = point-to-reference-plane signed distance
residual_global = local_plane[:, 2]

plane_hist_csv = os.path.join(OUT_DIR, "02_plane_fit_history.csv")
plane_hist.to_csv(plane_hist_csv, index=False)
print("[SAVE]", plane_hist_csv)



# ============================================================
# 4) REFERENCE PROFILE ALONG MAJOR AXIS
# ============================================================
profile_plane = slice_profile_stats(
    local_plane,
    slice_width=SLICE_WIDTH_M,
    min_points=MIN_POINTS_PER_SLICE,
    trim_q=TRIM_Q_PROFILE
)
profile_plane["healthy_final"] = profile_df["healthy_final"].values[:len(profile_plane)]

if profile_plane["healthy_final"].sum() >= PROFILE_MIN_HEALTHY_SLICES:
    spline_ref = fit_reference_profile(
        profile_plane,
        healthy_mask=profile_plane["healthy_final"].to_numpy(),
        smoothing=PROFILE_SMOOTHING,
    )
else:
    spline_ref = None

s_plane = local_plane[:, 0]
if spline_ref is None:
    reference_profile_h = np.zeros(len(pts), dtype=float)
    print("[WARN] spline reference profile was not fitted. local residual = global residual")
else:
    reference_profile_h = spline_ref(s_plane)

residual_local = residual_global - reference_profile_h

profile_plane["ref_profile_h"] = np.nan
if spline_ref is not None:
    valid = profile_plane["s_center"].notna()
    profile_plane.loc[valid, "ref_profile_h"] = spline_ref(profile_plane.loc[valid, "s_center"].to_numpy())

profile_plane_csv = os.path.join(OUT_DIR, "03_plane_frame_profile.csv")
profile_plane.to_csv(profile_plane_csv, index=False)
print("[SAVE]", profile_plane_csv)



# ============================================================
# 5) LOCAL SURFACE FEATURES
# ============================================================
normals, curvature, roughness, planarity = compute_local_features(
    pts,
    k_neighbors=K_NEIGHBORS,
    chunk_size=CHUNK_SIZE,
    n_jobs=N_JOBS,
)

if COMPUTE_NORMAL_VARIATION:
    normal_variation = compute_normal_variation(
        pts,
        normals,
        k_neighbors=K_NEIGHBORS,
        chunk_size=CHUNK_SIZE,
        n_jobs=N_JOBS,
    )
else:
    normal_variation = np.full(len(pts), np.nan, dtype=float)

print("[INFO] feature calculation completed")



# ============================================================
# 6) DISTRESS CANDIDATES
# ============================================================
res_thr, res_med, res_mad = robust_threshold_negative(residual_local, mad_k=RESIDUAL_MAD_K)
curv_thr = np.nanquantile(curvature, CURVATURE_Q)
rough_thr = np.nanquantile(roughness, ROUGHNESS_Q)
plan_thr = np.nanquantile(planarity, PLANARITY_Q_MAX)

if DISTRESS_MODE == "negative":
    depth_pos = np.maximum(-residual_local, 0.0)
    cond_residual = (residual_local <= min(res_thr, -MIN_DAMAGE_DEPTH_M))
else:
    depth_pos = np.abs(residual_local)
    cond_residual = (np.abs(residual_local) >= max(abs(res_thr), MIN_DAMAGE_DEPTH_M))

cond_geom = (
    (curvature >= curv_thr) |
    (roughness >= rough_thr) |
    (planarity <= plan_thr)
)

candidate_mask = cond_residual & cond_geom & np.isfinite(depth_pos)

print("[INFO] residual threshold:", res_thr)
print("[INFO] curvature threshold:", curv_thr)
print("[INFO] roughness threshold:", rough_thr)
print("[INFO] planarity threshold max:", plan_thr)
print(f"[INFO] candidate points: {candidate_mask.sum():,}")



# ============================================================
# 7) PATCH CLUSTERING
# ============================================================
labels = np.full(len(pts), -1, dtype=int)

if candidate_mask.sum() > 0:
    X_cluster = local_plane[candidate_mask, :2]  # s, t
    db = DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_SAMPLES)
    cand_labels = db.fit_predict(X_cluster)
    labels[candidate_mask] = cand_labels

    # small clusters 제거
    valid_labels = []
    for lb in np.unique(cand_labels):
        if lb == -1:
            continue
        if np.sum(cand_labels == lb) >= MIN_PATCH_POINTS:
            valid_labels.append(lb)

    remap = {old: new for new, old in enumerate(sorted(valid_labels))}
    labels2 = np.full(len(pts), -1, dtype=int)
    mask_c = candidate_mask.copy()
    tmp = labels[mask_c]
    tmp2 = np.array([remap.get(lb, -1) for lb in tmp], dtype=int)
    labels2[mask_c] = tmp2
    labels = labels2

print("[INFO] valid patch labels:", sorted([lb for lb in np.unique(labels) if lb != -1]))



# ============================================================
# 8) PATCH METRICS
# ============================================================
slab_bbox_st = {
    "s_min": float(np.min(local_plane[:, 0])),
    "s_max": float(np.max(local_plane[:, 0])),
    "t_min": float(np.min(local_plane[:, 1])),
    "t_max": float(np.max(local_plane[:, 1])),
}

patch_df = compute_patch_metrics(
    local_plane,
    depth_pos=depth_pos,
    labels=labels,
    slab_bbox_st=slab_bbox_st,
    max_tri_edge=MAX_TRI_EDGE_M,
)

patch_csv = os.path.join(OUT_DIR, "04_damage_patch_metrics.csv")
patch_df.to_csv(patch_csv, index=False)
print("[SAVE]", patch_csv)
print(patch_df.head(20))



# ============================================================
# 9) POINT-LEVEL OUTPUTS
# ============================================================
point_df = pd.DataFrame({
    "x": pts[:, 0],
    "y": pts[:, 1],
    "z": pts[:, 2],
    "s": local_plane[:, 0],
    "t": local_plane[:, 1],
    "h_to_plane": local_plane[:, 2],
    "residual_global_m": residual_global,
    "reference_profile_h_m": reference_profile_h,
    "residual_local_m": residual_local,
    "depth_pos_m": depth_pos,
    "curvature": curvature,
    "roughness": roughness,
    "planarity": planarity,
    "normal_x": normals[:, 0],
    "normal_y": normals[:, 1],
    "normal_z": normals[:, 2],
    "normal_variation_deg": normal_variation,
    "healthy_slice_point": healthy_point_mask,
    "candidate_point": candidate_mask,
    "patch_id": labels,
})

point_csv = os.path.join(OUT_DIR, "05_point_level_damage_features_1.csv")
point_df.to_csv(point_csv, index=False)
print("[SAVE]", point_csv)



# ============================================================
# 10) SUMMARY
# ============================================================
summary = {
    "n_points_total": int(len(pts)),
    "n_points_healthy": int(healthy_point_mask.sum()),
    "n_candidate_points": int(candidate_mask.sum()),
    "n_valid_patches": int(len(patch_df)),
    "plane_centroid_x": float(plane_centroid[0]),
    "plane_centroid_y": float(plane_centroid[1]),
    "plane_centroid_z": float(plane_centroid[2]),
    "plane_normal_x": float(plane_normal[0]),
    "plane_normal_y": float(plane_normal[1]),
    "plane_normal_z": float(plane_normal[2]),
    "residual_global_rmse_m": float(np.sqrt(np.nanmean(residual_global ** 2))),
    "residual_local_rmse_m": float(np.sqrt(np.nanmean(residual_local ** 2))),
    "residual_threshold_m": float(res_thr),
    "curvature_threshold": float(curv_thr),
    "roughness_threshold": float(rough_thr),
    "planarity_threshold_max": float(plan_thr),
}
summary_df = pd.DataFrame([summary])
summary_csv = os.path.join(OUT_DIR, "06_summary.csv_1")
summary_df.to_csv(summary_csv, index=False)
print("[SAVE]", summary_csv)
summary_df.T



# ============================================================
# 11) VISUALIZATION
# ============================================================
rng = np.random.default_rng(RANDOM_SEED)
if len(pts) > POINT_PLOT_SAMPLE:
    idx_plot = rng.choice(len(pts), size=POINT_PLOT_SAMPLE, replace=False)
else:
    idx_plot = np.arange(len(pts))

fig = plt.figure(figsize=(18, 14), dpi=FIG_DPI)

gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.9])
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])

# (a) profile
ax1.plot(profile_plane["s_center"], profile_plane["z_med"], color="0.4", lw=1.5, label="Slice median profile")
if spline_ref is not None:
    s_dense = np.linspace(np.nanmin(profile_plane["s_center"]), np.nanmax(profile_plane["s_center"]), 1000)
    ax1.plot(s_dense, spline_ref(s_dense), color="red", lw=2.0, label="Reference profile")
healthy_pf = profile_plane[profile_plane["healthy_final"]]
ax1.scatter(healthy_pf["s_center"], healthy_pf["z_med"], s=20, color="green", label="Healthy slices")
ax1.set_title("Longitudinal profile and healthy-zone reference")
ax1.set_xlabel("s along slab major axis (m)")
ax1.set_ylabel("h to reference plane (m)")
ax1.legend()
ax1.grid(True, alpha=0.2)

# (b) global residual map
v1 = np.nanquantile(np.abs(residual_global), 0.98)
sc2 = ax2.scatter(local_plane[idx_plot, 0], local_plane[idx_plot, 1], c=residual_global[idx_plot], s=1,
                  cmap="RdBu_r", vmin=-v1, vmax=v1)
ax2.set_title("Global residual map")
ax2.set_xlabel("s (m)")
ax2.set_ylabel("t (m)")
ax2.set_aspect("equal", adjustable="box")
plt.colorbar(sc2, ax=ax2, label="m")

# (c) local residual map
v2 = np.nanquantile(np.abs(residual_local), 0.98)
sc3 = ax3.scatter(local_plane[idx_plot, 0], local_plane[idx_plot, 1], c=residual_local[idx_plot], s=1,
                  cmap="RdBu_r", vmin=-v2, vmax=v2)
ax3.set_title("Local detrended residual map")
ax3.set_xlabel("s (m)")
ax3.set_ylabel("t (m)")
ax3.set_aspect("equal", adjustable="box")
plt.colorbar(sc3, ax=ax3, label="m")

# (d) patch map
plot_labels = labels[idx_plot]
sc4 = ax4.scatter(local_plane[idx_plot, 0], local_plane[idx_plot, 1], c=plot_labels, s=2, cmap="tab20")
ax4.set_title("Damage patch clustering (DBSCAN)")
ax4.set_xlabel("s (m)")
ax4.set_ylabel("t (m)")
ax4.set_aspect("equal", adjustable="box")

# (e) depth histogram
ax5.hist(depth_pos[depth_pos > 0], bins=100, color="0.4", alpha=0.85)
ax5.axvline(MIN_DAMAGE_DEPTH_M, color="red", linestyle="--", label="Min depth config")
ax5.set_title("Positive damage depth distribution")
ax5.set_xlabel("depth_pos (m)")
ax5.set_ylabel("count")
ax5.legend()

plt.tight_layout()
fig_fp = os.path.join(OUT_DIR, "07_damage_quantification_overview.png")
plt.savefig(fig_fp, bbox_inches="tight")
plt.show()
print("[SAVE]", fig_fp)

# ------------------------------------------------------------
# Damage / non-damage overlay on extracted slab
# ------------------------------------------------------------
damage_mask_plot = labels[idx_plot] >= 0
nondamage_mask_plot = ~damage_mask_plot

fig2, ax6 = plt.subplots(1, 1, figsize=(8, 8), dpi=FIG_DPI)
ax6.scatter(
    local_plane[idx_plot[nondamage_mask_plot], 0],
    local_plane[idx_plot[nondamage_mask_plot], 1],
    c="lightgray",
    s=1,
    label="Non-damage",
)
ax6.scatter(
    local_plane[idx_plot[damage_mask_plot], 0],
    local_plane[idx_plot[damage_mask_plot], 1],
    c="crimson",
    s=2,
    label="Damage",
)
ax6.set_title("Extracted slab: damage vs non-damage")
ax6.set_xlabel("s (m)")
ax6.set_ylabel("t (m)")
ax6.set_aspect("equal", adjustable="box")
ax6.legend(markerscale=4)

plt.tight_layout()
fig2_fp = os.path.join(OUT_DIR, "08_damage_vs_nondamage_overlay.png")
plt.savefig(fig2_fp, bbox_inches="tight")
plt.show()
print("[SAVE]", fig2_fp)
