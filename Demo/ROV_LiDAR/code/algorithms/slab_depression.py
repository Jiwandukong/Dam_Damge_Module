"""Original slab depression analysis: kNN, PCA UV, core plane, MAD/DBSCAN and automatic grid."""
from pathlib import Path
import argparse,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib
matplotlib.use('Agg')
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(os.environ.get('ROV_LIDAR_WORK_DIR',str(Path.home()/'.cache/dam_damage_module/rov_lidar')))

def analyze(input_las, output_dir):
    import matplotlib.pyplot as plt
    plt.show=lambda *args,**kwargs:plt.close('all')
    # BEGIN ORIGINAL ANALYSIS
    # -*- coding: utf-8 -*-

    from pathlib import Path

    import laspy
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt

    from shapely.geometry import box
    from shapely.ops import unary_union, transform as shp_transform
    from scipy.ndimage import label
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors


    # =========================================================
    # 사용자 설정
    # =========================================================
    INPUT_LAS = str(input_las)
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    CRS = "EPSG:5186"

    # outlier 제거
    KNN_K = 16
    OUTLIER_Z = 3.0

    # 기준 평면
    CORE_RATIO = 0.80  # 중앙 80% 영역으로 reference plane

    # residual grid
    CELL_SIZE = None   # None이면 자동 추정

    # damage candidate threshold
    DAMAGE_K = 3.0     # median - DAMAGE_K * 1.4826 * MAD

    # clustering
    DBSCAN_EPS = None  # None이면 자동
    DBSCAN_MIN_SAMPLES = 10

    # depression patch threshold
    DEPRESSION_K = 3.0
    MIN_PATCH_AREA = 0.01  # m^2

    # edge metrics
    EDGE_BAND_RATIO = 0.08
    EDGE_SIGMA_SCALE = 1.0

    # plotting
    MAX_PLOT_POINTS = 150000


    # =========================================================
    # 기본 함수
    # =========================================================
    def robust_mad(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        sigma = 1.4826 * (mad + 1e-12)
        return med, sigma


    def load_las_xyz(path):
        las = laspy.read(path)
        xyz = np.column_stack([las.x, las.y, las.z]).astype(float)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        return xyz


    def remove_knn_outliers(xyz, k=16, z_thresh=3.0):
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(xyz)
        dists, _ = nbrs.kneighbors(xyz)

        score = dists[:, 1:].mean(axis=1)
        med, sigma = robust_mad(score)
        z = np.abs((score - med) / sigma)

        keep = z <= z_thresh
        return xyz[keep], keep, score


    def build_local_uv(xyz):
        """
        XY 기준 PCA
        - u: 긴 축
        - v: 짧은 축
        """
        xy = xyz[:, :2]
        center_xy = xy.mean(axis=0)

        pca = PCA(n_components=2)
        pca.fit(xy - center_xy)
        comp = pca.components_.copy()

        uv = (xy - center_xy) @ comp.T

        if np.ptp(uv[:, 0]) < np.ptp(uv[:, 1]):
            comp = comp[[1, 0], :]
            uv = (xy - center_xy) @ comp.T

        u = uv[:, 0]
        v = uv[:, 1]
        z = xyz[:, 2]

        meta = {
            "center_xy_x": float(center_xy[0]),
            "center_xy_y": float(center_xy[1]),
            "ux_x": float(comp[0, 0]),
            "ux_y": float(comp[0, 1]),
            "vx_x": float(comp[1, 0]),
            "vx_y": float(comp[1, 1]),
            "u_min": float(np.min(u)),
            "u_max": float(np.max(u)),
            "v_min": float(np.min(v)),
            "v_max": float(np.max(v)),
            "L": float(np.max(u) - np.min(u)),
            "W": float(np.max(v) - np.min(v)),
        }
        return u, v, z, meta


    def fit_reference_plane_from_core(u, v, z, core_ratio=0.80):
        """
        슬래브 중앙부(core)만으로 기준 평면 적합
        z = a*u + b*v + c
        """
        umin, umax = np.min(u), np.max(u)
        vmin, vmax = np.min(v), np.max(v)

        du = (umax - umin) * (1 - core_ratio) / 2.0
        dv = (vmax - vmin) * (1 - core_ratio) / 2.0

        core_mask = (
            (u >= umin + du) & (u <= umax - du) &
            (v >= vmin + dv) & (v <= vmax - dv)
        )

        if core_mask.sum() < 50:
            raise ValueError("core 영역 점 수가 너무 적다. CORE_RATIO를 키워라.")

        X_core = np.column_stack([u[core_mask], v[core_mask]])
        reg = LinearRegression()
        reg.fit(X_core, z[core_mask])

        a, b = reg.coef_
        c = reg.intercept_

        X_all = np.column_stack([u, v])
        z_hat = reg.predict(X_all)
        residual = z - z_hat

        normal = np.array([-a, -b, 1.0], dtype=float)
        normal /= np.linalg.norm(normal)

        rmse_core = float(np.sqrt(np.mean((z[core_mask] - reg.predict(X_core)) ** 2)))
        tilt_deg = float(np.degrees(np.arctan(np.sqrt(a * a + b * b))))

        return {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "normal_u": float(normal[0]),
            "normal_v": float(normal[1]),
            "normal_z": float(normal[2]),
            "tilt_deg": tilt_deg,
            "rmse_core": rmse_core,
            "core_mask": core_mask,
            "z_hat": z_hat,
            "residual": residual,
        }


    def make_grid(u, v, values, cell_size):
        umin, umax = float(np.min(u)), float(np.max(u))
        vmin, vmax = float(np.min(v)), float(np.max(v))

        u_edges = np.arange(umin, umax + cell_size, cell_size)
        v_edges = np.arange(vmin, vmax + cell_size, cell_size)

        nu = len(u_edges) - 1
        nv = len(v_edges) - 1

        iu = np.clip(np.digitize(u, u_edges) - 1, 0, nu - 1)
        iv = np.clip(np.digitize(v, v_edges) - 1, 0, nv - 1)

        grid_sum = np.zeros((nu, nv), dtype=float)
        grid_cnt = np.zeros((nu, nv), dtype=float)

        np.add.at(grid_sum, (iu, iv), values)
        np.add.at(grid_cnt, (iu, iv), 1)

        grid_mean = np.full((nu, nv), np.nan, dtype=float)
        mask = grid_cnt > 0
        grid_mean[mask] = grid_sum[mask] / grid_cnt[mask]

        u_centers = (u_edges[:-1] + u_edges[1:]) / 2.0
        v_centers = (v_edges[:-1] + v_edges[1:]) / 2.0

        return {
            "grid": grid_mean,
            "count": grid_cnt,
            "u_edges": u_edges,
            "v_edges": v_edges,
            "u_centers": u_centers,
            "v_centers": v_centers,
            "iu": iu,
            "iv": iv,
        }


    def classify_damage_points(residual, k=3.0):
        med, sigma = robust_mad(residual)
        threshold = med - k * sigma
        damage_mask = residual < threshold
        return damage_mask, threshold


    def cluster_damage_points(u, v, damage_mask, eps=None, min_samples=10):
        xy = np.column_stack([u[damage_mask], v[damage_mask]])
        if len(xy) == 0:
            return np.full(len(u), -1, dtype=int), np.nan

        if eps is None:
            nbrs = NearestNeighbors(n_neighbors=min(6, len(xy)))
            nbrs.fit(xy)
            dists, _ = nbrs.kneighbors(xy)
            # 최근접거리 median의 3배 정도
            nn = dists[:, 1] if dists.shape[1] > 1 else np.full(len(xy), 0.1)
            eps = max(3.0 * np.median(nn), 0.05)

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels_small = db.fit_predict(xy)

        labels_all = np.full(len(u), -1, dtype=int)
        labels_all[damage_mask] = labels_small

        return labels_all, float(eps)


    def save_grid_csv(grid_pack, out_csv):
        rows = []
        for i, uu in enumerate(grid_pack["u_centers"]):
            for j, vv in enumerate(grid_pack["v_centers"]):
                rows.append({
                    "u": float(uu),
                    "v": float(vv),
                    "residual_mean": float(grid_pack["grid"][i, j]) if np.isfinite(grid_pack["grid"][i, j]) else np.nan,
                    "count": float(grid_pack["count"][i, j]),
                })
        pd.DataFrame(rows).to_csv(out_csv, index=False)


    def detect_depressions(grid_pack, meta, cell_size, k=3.0, min_patch_area=0.01):
        """
        residual grid에서 실제 depression patch 추출
        """
        grid = grid_pack["grid"]
        vals = grid[np.isfinite(grid)]

        med, sigma = robust_mad(vals)
        thr = med - k * sigma

        dep_mask = np.isfinite(grid) & (grid < thr)
        labels, nlab = label(dep_mask)

        u_centers = grid_pack["u_centers"]
        v_centers = grid_pack["v_centers"]
        U, V = np.meshgrid(u_centers, v_centers, indexing="ij")

        band = min(meta["L"], meta["W"]) * EDGE_BAND_RATIO
        edge_zone = (
            (U <= meta["u_min"] + band) |
            (U >= meta["u_max"] - band) |
            (V <= meta["v_min"] + band) |
            (V >= meta["v_max"] - band)
        )

        rows = []
        total_area = 0.0
        total_volume = 0.0

        for pid in range(1, nlab + 1):
            m = (labels == pid)
            if not np.any(m):
                continue

            area = float(np.sum(m) * cell_size * cell_size)
            if area < min_patch_area:
                continue

            vals_patch = grid[m]
            vals_patch = vals_patch[np.isfinite(vals_patch)]

            max_depth = float(-np.min(vals_patch))
            mean_depth = float(-np.mean(vals_patch))
            volume_loss = float(np.sum(np.maximum(0.0, -vals_patch)) * cell_size * cell_size)

            total_area += area
            total_volume += volume_loss

            rows.append({
                "patch_id": int(pid),
                "n_cells": int(np.sum(m)),
                "area": area,
                "max_depth": max_depth,
                "mean_depth": mean_depth,
                "volume_loss": volume_loss,
                "u_min": float(np.min(U[m])),
                "u_max": float(np.max(U[m])),
                "v_min": float(np.min(V[m])),
                "v_max": float(np.max(V[m])),
                "centroid_u": float(np.mean(U[m])),
                "centroid_v": float(np.mean(V[m])),
                "touches_edge": bool(np.any(edge_zone[m])),
            })

        dep_df = pd.DataFrame(rows)
        dep_sum = {
            "threshold": float(thr),
            "depression_total_area": float(total_area),
            "depression_total_volume": float(total_volume),
            "n_patches": int(len(dep_df)),
        }
        return dep_df, dep_mask, dep_sum


    def compute_edge_loss_metrics(grid_pack, meta, cell_size, edge_band_ratio=0.08, sigma_scale=1.0):
        grid = grid_pack["grid"]
        u_centers = grid_pack["u_centers"]
        v_centers = grid_pack["v_centers"]

        U, V = np.meshgrid(u_centers, v_centers, indexing="ij")
        valid = np.isfinite(grid)

        vals = grid[valid]
        med, sigma = robust_mad(vals)
        thr = min(0.0, med - sigma_scale * sigma)

        band_width = max(min(meta["L"], meta["W"]) * edge_band_ratio, cell_size * 4)

        masks = {
            "left":   U <= meta["u_min"] + band_width,
            "right":  U >= meta["u_max"] - band_width,
            "bottom": V <= meta["v_min"] + band_width,
            "top":    V >= meta["v_max"] - band_width,
        }

        rows = []
        cell_area = cell_size * cell_size

        for name, m in masks.items():
            mask = m & valid
            vals_edge = grid[mask]
            neg = vals_edge[vals_edge < thr]

            rows.append({
                "edge": name,
                "band_width": float(band_width),
                "threshold": float(thr),
                "n_valid_cells": int(np.sum(mask)),
                "n_negative_cells": int(len(neg)),
                "negative_area": float(len(neg) * cell_area),
                "negative_volume": float(np.sum(-neg) * cell_area) if len(neg) > 0 else 0.0,
                "mean_residual": float(np.nanmean(vals_edge)) if len(vals_edge) else np.nan,
                "min_residual": float(np.nanmin(vals_edge)) if len(vals_edge) else np.nan,
            })

        return pd.DataFrame(rows)


    def save_depression_patches_gpkg(grid_pack, dep_mask, meta, cell_size, out_gpkg, crs="EPSG:5186"):
        """
        dep_mask -> 실제 depression patch polygon(gpkg)
        """
        labels, nlab = label(dep_mask)
        grid = grid_pack["grid"]
        u_edges = grid_pack["u_edges"]
        v_edges = grid_pack["v_edges"]

        cx = meta["center_xy_x"]
        cy = meta["center_xy_y"]
        ux_x = meta["ux_x"]
        ux_y = meta["ux_y"]
        vx_x = meta["vx_x"]
        vx_y = meta["vx_y"]

        def uv_to_xy_transform(u, v, z=None):
            u = np.asarray(u)
            v = np.asarray(v)
            x = cx + u * ux_x + v * vx_x
            y = cy + u * ux_y + v * vx_y
            if z is None:
                return x, y
            return x, y, z

        rows = []
        cell_area = cell_size * cell_size

        for pid in range(1, nlab + 1):
            m = (labels == pid)
            if not np.any(m):
                continue

            boxes = []
            ii, jj = np.where(m)
            for i, j in zip(ii, jj):
                b = box(u_edges[i], v_edges[j], u_edges[i + 1], v_edges[j + 1])
                boxes.append(b)

            geom_uv = unary_union(boxes)
            geom_xy = shp_transform(uv_to_xy_transform, geom_uv)

            vals = grid[m]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue

            area = float(np.sum(m) * cell_area)
            max_depth = float(-np.min(vals))
            mean_depth = float(-np.mean(vals))
            volume_loss = float(np.sum(np.maximum(0.0, -vals)) * cell_area)

            rows.append({
                "patch_id": int(pid),
                "area": area,
                "max_depth": max_depth,
                "mean_depth": mean_depth,
                "volume_loss": volume_loss,
                "geometry": geom_xy
            })

        gdf = gpd.GeoDataFrame(rows, crs=crs)
        gdf["centroid_x"] = gdf.geometry.centroid.x
        gdf["centroid_y"] = gdf.geometry.centroid.y
        gdf.to_file(out_gpkg, driver="GPKG")
        return gdf


    def sample_idx(n, max_n=150000, seed=42):
        if n <= max_n:
            return np.arange(n)
        rng = np.random.default_rng(seed)
        return rng.choice(n, size=max_n, replace=False)


    # =========================================================
    # 실행
    # =========================================================
    print("[1] load LAS")
    xyz = load_las_xyz(INPUT_LAS)
    print("  raw points:", len(xyz))

    print("[2] outlier removal")
    xyz_clean, keep_mask, outlier_score = remove_knn_outliers(xyz, k=KNN_K, z_thresh=OUTLIER_Z)
    print("  cleaned points:", len(xyz_clean))

    print("[3] local UV")
    u, v, z, meta = build_local_uv(xyz_clean)
    print("  L =", meta["L"])
    print("  W =", meta["W"])

    print("[4] reference plane")
    plane = fit_reference_plane_from_core(u, v, z, core_ratio=CORE_RATIO)
    residual = plane["residual"]
    print("  tilt_deg  =", plane["tilt_deg"])
    print("  rmse_core =", plane["rmse_core"])

    print("[5] damage candidate")
    damage_mask, damage_thr = classify_damage_points(residual, k=DAMAGE_K)
    labels_all, dbscan_eps = cluster_damage_points(u, v, damage_mask, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    print("  damage threshold =", damage_thr)
    print("  DBSCAN eps       =", dbscan_eps)

    print("[6] build cleaned_df")
    cleaned_df = pd.DataFrame({
        "x": xyz_clean[:, 0],
        "y": xyz_clean[:, 1],
        "z": xyz_clean[:, 2],
        "u": u,
        "v": v,
        "z_hat": plane["z_hat"],
        "residual": residual,
        "damage_candidate": damage_mask.astype(int),
        "damage_cluster_id": labels_all,
    })

    print("[7] grid")
    if CELL_SIZE is None:
        cell_size = max(min(meta["L"], meta["W"]) / 120.0, 1e-4)
    else:
        cell_size = CELL_SIZE
    grid_pack = make_grid(u, v, residual, cell_size)
    print("  cell_size =", cell_size)

    print("[8] depression patches")
    dep_df, dep_mask, dep_sum = detect_depressions(
        grid_pack=grid_pack,
        meta=meta,
        cell_size=cell_size,
        k=DEPRESSION_K,
        min_patch_area=MIN_PATCH_AREA
    )
    print("  n_patches =", dep_sum["n_patches"])
    print("  total_area =", dep_sum["depression_total_area"])
    print("  total_volume =", dep_sum["depression_total_volume"])

    print("[9] edge metrics")
    edge_df = compute_edge_loss_metrics(
        grid_pack=grid_pack,
        meta=meta,
        cell_size=cell_size,
        edge_band_ratio=EDGE_BAND_RATIO,
        sigma_scale=EDGE_SIGMA_SCALE
    )

    print("[10] save csv")
    cleaned_df.to_csv(OUTPUT_DIR / "residual_points.csv", index=False)
    cleaned_df[cleaned_df["damage_candidate"] == 1].to_csv(OUTPUT_DIR / "damage_candidate_points.csv", index=False)
    save_grid_csv(grid_pack, OUTPUT_DIR / "residual_grid.csv")
    dep_df.to_csv(OUTPUT_DIR / "depression_patches.csv", index=False)
    edge_df.to_csv(OUTPUT_DIR / "edge_metrics.csv", index=False)

    # damage cluster summary
    cluster_rows = []
    valid_clusters = sorted([cid for cid in cleaned_df["damage_cluster_id"].unique() if cid >= 0])
    for cid in valid_clusters:
        grp = cleaned_df[cleaned_df["damage_cluster_id"] == cid]
        cluster_rows.append({
            "cluster_id": int(cid),
            "n_points": int(len(grp)),
            "u_min": float(grp["u"].min()),
            "u_max": float(grp["u"].max()),
            "v_min": float(grp["v"].min()),
            "v_max": float(grp["v"].max()),
            "mean_residual": float(grp["residual"].mean()),
            "min_residual": float(grp["residual"].min()),
            "max_residual": float(grp["residual"].max()),
        })
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df.to_csv(OUTPUT_DIR / "damage_clusters.csv", index=False)

    # slab metrics
    slab_metrics = pd.DataFrame([{
        "n_points_clean": len(cleaned_df),
        "slab_length": meta["L"],
        "slab_width": meta["W"],
        "cell_size": cell_size,
        "plane_a": plane["a"],
        "plane_b": plane["b"],
        "plane_c": plane["c"],
        "plane_normal_u": plane["normal_u"],
        "plane_normal_v": plane["normal_v"],
        "plane_normal_z": plane["normal_z"],
        "tilt_deg": plane["tilt_deg"],
        "plane_rmse_core": plane["rmse_core"],
        "damage_threshold": float(damage_thr),
        "damage_dbscan_eps": float(dbscan_eps) if np.isfinite(dbscan_eps) else np.nan,
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_min": float(np.min(residual)),
        "residual_max": float(np.max(residual)),
        "depression_threshold": dep_sum["threshold"],
        "depression_total_area": dep_sum["depression_total_area"],
        "depression_total_volume": dep_sum["depression_total_volume"],
        "n_depression_patches": dep_sum["n_patches"],
    }])
    slab_metrics.to_csv(OUTPUT_DIR / "slab_metrics.csv", index=False)

    print("[11] save depression_patches.gpkg")
    depression_gpkg_path = OUTPUT_DIR / "depression_patches.gpkg"
    gdf_dep = save_depression_patches_gpkg(
        grid_pack=grid_pack,
        dep_mask=dep_mask,
        meta=meta,
        cell_size=cell_size,
        out_gpkg=depression_gpkg_path,
        crs=CRS
    )
    print(gdf_dep[["patch_id", "area", "max_depth", "mean_depth", "volume_loss"]])
    print("  saved:", depression_gpkg_path)

    # =========================================================
    # 시각화 (원래 기능 유지)
    # =========================================================
    print("[12] make figures")

    idx = sample_idx(len(cleaned_df), MAX_PLOT_POINTS)

    # 1) residual scatter
    fig, ax = plt.subplots(figsize=(11, 4))
    sc = ax.scatter(cleaned_df["u"].values[idx], cleaned_df["v"].values[idx],
                    c=cleaned_df["residual"].values[idx], s=1, cmap="RdBu_r")
    ax.set_title("Residual Scatter")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="Residual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_scatter_uv.png", bbox_inches="tight")
    plt.show()

    # 2) damage overlay scatter
    fig, ax = plt.subplots(figsize=(11, 4))
    normal_mask = cleaned_df["damage_candidate"].values == 0
    damage_mask_plot = cleaned_df["damage_candidate"].values == 1

    idx_normal = sample_idx(np.sum(normal_mask), min(MAX_PLOT_POINTS, max(np.sum(normal_mask), 1)))
    idx_damage = sample_idx(np.sum(damage_mask_plot), min(MAX_PLOT_POINTS, max(np.sum(damage_mask_plot), 1)))

    u_normal = cleaned_df.loc[normal_mask, "u"].values[idx_normal] if np.sum(normal_mask) else np.array([])
    v_normal = cleaned_df.loc[normal_mask, "v"].values[idx_normal] if np.sum(normal_mask) else np.array([])
    u_damage = cleaned_df.loc[damage_mask_plot, "u"].values[idx_damage] if np.sum(damage_mask_plot) else np.array([])
    v_damage = cleaned_df.loc[damage_mask_plot, "v"].values[idx_damage] if np.sum(damage_mask_plot) else np.array([])

    if len(u_normal):
        ax.scatter(u_normal, v_normal, s=1, alpha=0.4, label="Normal")
    if len(u_damage):
        ax.scatter(u_damage, v_damage, s=2, alpha=0.9, label="Damage Candidate")
    ax.set_title("Damage Candidate Overlay")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_aspect("equal")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "damage_candidate_overlay.png", bbox_inches="tight")
    plt.show()

    # 3) residual heatmap
    fig, ax = plt.subplots(figsize=(11, 4))
    extent = [
        grid_pack["u_edges"][0], grid_pack["u_edges"][-1],
        grid_pack["v_edges"][0], grid_pack["v_edges"][-1]
    ]
    im = ax.imshow(grid_pack["grid"].T, origin="lower", extent=extent, aspect="equal", cmap="RdBu_r")
    ax.set_title("Residual Heatmap")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    plt.colorbar(im, ax=ax, label="Mean Residual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_heatmap.png", bbox_inches="tight")
    plt.show()

    # 4) depression patch overlay
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(grid_pack["grid"].T, origin="lower", extent=extent, aspect="equal", cmap="RdBu_r")
    U, V = np.meshgrid(grid_pack["u_centers"], grid_pack["v_centers"], indexing="ij")
    if dep_mask is not None and np.any(dep_mask):
        ax.contour(U, V, dep_mask.astype(int), levels=[0.5], linewidths=1.2)
    if len(dep_df):
        for _, row in dep_df.iterrows():
            uc = 0.5 * (row["u_min"] + row["u_max"])
            vc = 0.5 * (row["v_min"] + row["v_max"])
            ax.text(uc, vc, str(int(row["patch_id"])), fontsize=8, ha="center", va="center")
    ax.set_title("Depression Patches")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    plt.colorbar(im, ax=ax, label="Mean Residual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "depression_patch_overlay.png", bbox_inches="tight")
    plt.show()

    # 5) edge metrics
    if len(edge_df):
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(edge_df))
        ax.bar(x - 0.2, edge_df["negative_area"], width=0.4, label="Negative Area")
        ax.bar(x + 0.2, edge_df["negative_volume"], width=0.4, label="Negative Volume")
        ax.set_xticks(x)
        ax.set_xticklabels(edge_df["edge"].tolist())
        ax.set_title("Edge Loss Metrics")
        ax.set_xlabel("Edge")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "edge_loss_metrics.png", bbox_inches="tight")
        plt.show()

    print("[DONE]")
    print("output:", OUTPUT_DIR)
    # END ORIGINAL ANALYSIS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=WORK/'analysis/slab_depression/slab_extraction/slab_extracted.las')
    parser.add_argument('--output-dir',type=Path,default=WORK/'analysis/slab_depression/depression')
    args=parser.parse_args()
    analyze(args.input,args.output_dir)
if __name__=='__main__':main()
