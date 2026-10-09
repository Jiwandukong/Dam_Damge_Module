"""Original Open3D slab extraction; original thresholds are below in the analysis body."""
from pathlib import Path
import sys,argparse,tempfile,json,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from point_cloud_io import prepare_las
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(os.environ.get('ROV_LIDAR_WORK_DIR',str(Path.home()/'.cache/dam_damage_module/rov_lidar')))

def extract(input_las, output_dir):
    import matplotlib.pyplot as plt
    def save_overview(*args,**kwargs):
        for number in plt.get_fignums():
            plt.figure(number).savefig(output_dir/'overview.png',dpi=150)
        plt.close('all')
    plt.show=save_overview
    # BEGIN ORIGINAL ANALYSIS
    # Original notebook cell 2

    import os
    import time
    from collections import deque

    import laspy
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import open3d as o3d

    from scipy.spatial import cKDTree
    from scipy.ndimage import binary_fill_holes, binary_closing, binary_erosion, label
    from shapely.geometry import Point


    # Original notebook cell 4
    LAS_FP = str(input_las)
    OUT_DIR = str(output_dir)
    os.makedirs(OUT_DIR, exist_ok=True)

    PLOT_SAMPLE_N = 200_000
    POINT_PLOT_SAMPLE = PLOT_SAMPLE_N
    RANDOM_SEED = 42
    FIG_DPI = 150

    # Point-cloud preparation
    OUTLIER_NB_NEIGHBORS = 30
    OUTLIER_STD_RATIO = 2.0
    VOXEL_SIZE_M = 0.20

    # Local geometry scales
    NORMAL_RADIUS_M = 0.70
    NORMAL_MAX_NN = 50
    GEOM_RADIUS_M = 0.80
    GEOM_MAX_NN = 60

    # Seed thresholds
    PLANARITY_MIN = 0.30
    SURFACE_VARIATION_MAX = 0.040
    LOCAL_RESIDUAL_MAX_M = 0.10
    VERTICAL_NORMAL_MIN = 0.85

    # RANSAC dominant plane
    RANSAC_DIST_THR_M = 0.12
    RANSAC_N = 3
    RANSAC_ITERS = 6000 #6000

    # Region growing thresholds
    GROW_RADIUS_M = 0.4
    GROW_PLANE_DIST_THR_M = 0.15
    GROW_NORMAL_ANGLE_DEG = 12.0
    GROW_PLANARITY_MIN = 0.15
    GROW_SURFACE_VARIATION_MAX = 0.030
    GROW_LOCAL_RESIDUAL_MAX_M = 0.30

    # Patch filtering
    PATCH_MIN_POINTS = 100
    PATCH_MIN_AREA_M2 = 10.0
    PATCH_MAX_MEAN_RESIDUAL_M = 0.12
    PATCH_MIN_MEAN_PLANARITY = 0.25
    PATCH_MAX_MEAN_SURFACE_VARIATION = 0.040
    PATCH_MIN_VERTICAL_NORMAL = 0.90
    PATCH_MAX_ZSPAN_M = 3.0

    # Coplanar patch merging
    MERGE_MAX_NORMAL_ANGLE_DEG = 4.0
    MERGE_MAX_OFFSET_M = 0.1
    MERGE_MAX_GAP_M = 0.1

    # Final slab acceptance after patch merging
    FINAL_MIN_AREA_M2 = 50.0
    FINAL_MAX_MEAN_RESIDUAL_M = 0.2
    FINAL_MIN_MEAN_PLANARITY = 0.3
    FINAL_MAX_MEAN_SURFACE_VARIATION = 0.040
    FINAL_MIN_VERTICAL_NORMAL = 0.90
    FINAL_MAX_ZSPAN_M = 3.0

    # Final slab zone regularization in XY space
    ZONE_GRID_M = 0.3
    ZONE_CLOSE_ITERS = 2
    FILL_ZONE_HOLES = True
    KEEP_ONLY_LARGEST_ZONE = True

    # Transfer label back to full-resolution cloud
    FULL_NORMAL_RADIUS_M = 0.45
    FULL_NORMAL_MAX_NN = 50

    # asymmetric signed-distance transfer
    TRANSFER_POS_DIST_M = 0.08   # slab plane 위쪽은 엄격
    TRANSFER_NEG_DIST_M = 0.80   # slab plane 아래쪽은 넓게 허용 (depression 보존)
    TRANSFER_MIN_VERTICAL_NORMAL = 0.72

    # final Z-outlier cleanup on filled slab zone
    ENABLE_Z_OUTLIER_FILTER = True
    Z_OUTLIER_METHOD = "iqr"   # "iqr" or "quantile"
    Z_OUTLIER_SIDE = "upper"   # "upper" or "both"
    Z_IQR_K = 1.5
    Z_LOWER_Q = 0.0
    Z_UPPER_Q = 0.995

    # Original notebook cell 6

    plt.rcParams['figure.figsize'] = (8, 8)
    plt.rcParams['axes.grid'] = True


    def sample_points_idx(n, max_n=PLOT_SAMPLE_N, seed=42):
        if n <= max_n:
            return np.arange(n)
        rng = np.random.default_rng(seed)
        return rng.choice(n, size=max_n, replace=False)


    def plot_xy(points_xyz, title, mask=None, max_n=PLOT_SAMPLE_N, color_mask_by_z=False):
        idx = sample_points_idx(len(points_xyz), max_n=max_n)
        pts = points_xyz[idx]
        fig, ax = plt.subplots(figsize=(8, 8))
        if mask is None:
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=1, cmap='viridis')
            plt.colorbar(sc, ax=ax, label='Z')
        else:
            m = mask[idx]
            ax.scatter(pts[~m, 0], pts[~m, 1], c='lightgray', s=1, label='non-slab')
            if color_mask_by_z:
                sc = ax.scatter(pts[m, 0], pts[m, 1], c=pts[m, 2], s=2, cmap='viridis', label='slab')
                plt.colorbar(sc, ax=ax, label='Z')
            else:
                ax.scatter(pts[m, 0], pts[m, 1], c='tab:orange', s=1, label='slab')
            ax.legend(markerscale=6)
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        plt.show()


    def make_o3d_cloud(points_xyz):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)
        return pcd


    def pca_geometry(neighbors_xyz):
        c = neighbors_xyz.mean(axis=0)
        centered = neighbors_xyz - c
        cov = centered.T @ centered / max(len(neighbors_xyz) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        l1, l2, l3 = eigvals
        eps = 1e-12
        normal = eigvecs[:, -1]
        normal = normal / (np.linalg.norm(normal) + eps)
        if normal[2] < 0:
            normal = -normal

        planarity = (l2 - l3) / max(l1, eps)
        linearity = (l1 - l2) / max(l1, eps)
        surface_variation = l3 / max(l1 + l2 + l3, eps)

        d = -np.dot(normal, c)
        residual = np.abs(neighbors_xyz @ normal + d)
        local_residual = float(np.sqrt(np.mean(residual ** 2)))
        return normal, planarity, linearity, surface_variation, local_residual


    def compute_local_geometry(points_xyz, radius_m, max_nn):
        tree = cKDTree(points_xyz)
        normals = np.full((len(points_xyz), 3), np.nan, dtype=float)
        planarity = np.full(len(points_xyz), np.nan, dtype=float)
        linearity = np.full(len(points_xyz), np.nan, dtype=float)
        surface_variation = np.full(len(points_xyz), np.nan, dtype=float)
        local_residual = np.full(len(points_xyz), np.nan, dtype=float)

        tic = time.perf_counter()
        for i, p in enumerate(points_xyz):
            nbr_idx = tree.query_ball_point(p, r=radius_m)
            if len(nbr_idx) < 6:
                continue
            if len(nbr_idx) > max_nn:
                nbr_idx = nbr_idx[:max_nn]
            nbr_xyz = points_xyz[nbr_idx]
            n, pty, lin, svar, lres = pca_geometry(nbr_xyz)
            normals[i] = n
            planarity[i] = pty
            linearity[i] = lin
            surface_variation[i] = svar
            local_residual[i] = lres
        print(f"[TIME] compute_local_geometry: {time.perf_counter() - tic:.2f} s")
        return tree, normals, planarity, linearity, surface_variation, local_residual


    def point_to_plane_distance(points_xyz, plane_model):
        a, b, c, d = plane_model
        num = np.abs(a * points_xyz[:, 0] + b * points_xyz[:, 1] + c * points_xyz[:, 2] + d)
        den = np.sqrt(a * a + b * b + c * c) + 1e-12
        return num / den


    def signed_point_to_plane_distance(points_xyz, plane_model):
        a, b, c, d = plane_model
        num = a * points_xyz[:, 0] + b * points_xyz[:, 1] + c * points_xyz[:, 2] + d
        den = np.sqrt(a * a + b * b + c * c) + 1e-12
        return num / den


    def normal_angle_deg(normals, ref_normal):
        ref = ref_normal / (np.linalg.norm(ref_normal) + 1e-12)
        dots = np.clip(normals @ ref, -1.0, 1.0)
        return np.degrees(np.arccos(np.abs(dots)))


    def region_grow(points_xyz, tree, seed_mask, normals, planarity, surface_variation, local_residual, plane_model,
                    grow_radius_m, grow_plane_dist_thr_m, grow_normal_angle_deg,
                    grow_planarity_min, grow_surface_variation_max, grow_local_residual_max_m):
        grown = np.zeros(len(points_xyz), dtype=bool)
        q = deque(np.where(seed_mask)[0].tolist())
        grown[seed_mask] = True

        ref_normal = np.asarray(plane_model[:3], dtype=float)
        ref_normal = ref_normal / (np.linalg.norm(ref_normal) + 1e-12)

        tic = time.perf_counter()
        while q:
            idx = q.popleft()
            nbr_idx = tree.query_ball_point(points_xyz[idx], r=grow_radius_m)
            if not nbr_idx:
                continue
            nbr_idx = np.asarray(nbr_idx, dtype=int)
            nbr_idx = nbr_idx[~grown[nbr_idx]]
            if len(nbr_idx) == 0:
                continue

            d_plane = point_to_plane_distance(points_xyz[nbr_idx], plane_model)
            ang = normal_angle_deg(normals[nbr_idx], ref_normal)
            ok = (
                np.isfinite(planarity[nbr_idx])
                & np.isfinite(surface_variation[nbr_idx])
                & np.isfinite(local_residual[nbr_idx])
                & (d_plane <= grow_plane_dist_thr_m)
                & (ang <= grow_normal_angle_deg)
                & (planarity[nbr_idx] >= grow_planarity_min)
                & (surface_variation[nbr_idx] <= grow_surface_variation_max)
                & (local_residual[nbr_idx] <= grow_local_residual_max_m)
            )
            accepted = nbr_idx[ok]
            if len(accepted) == 0:
                continue
            grown[accepted] = True
            q.extend(accepted.tolist())
        print(f"[TIME] region_grow: {time.perf_counter() - tic:.2f} s")
        return grown


    def connected_components_from_mask(points_xyz, tree, mask, radius_m):
        active_idx = np.where(mask)[0]
        active_set = set(active_idx.tolist())
        visited = np.zeros(len(points_xyz), dtype=bool)
        components = []
        for start in active_idx:
            if visited[start]:
                continue
            q = deque([int(start)])
            visited[start] = True
            comp = []
            while q:
                i = q.popleft()
                comp.append(i)
                nbr_idx = tree.query_ball_point(points_xyz[i], r=radius_m)
                for j in nbr_idx:
                    if j in active_set and not visited[j]:
                        visited[j] = True
                        q.append(int(j))
            components.append(np.asarray(comp, dtype=int))
        return components


    def fit_plane_least_squares(points_xyz):
        X = np.column_stack([points_xyz[:, 0], points_xyz[:, 1], np.ones(len(points_xyz))])
        y = points_xyz[:, 2]
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b, c0 = coef
        # z = ax + by + c0 -> ax + by - z + c0 = 0
        plane_model = np.array([a, b, -1.0, c0], dtype=float)
        plane_model[:3] /= np.linalg.norm(plane_model[:3]) + 1e-12
        plane_model[3] /= np.linalg.norm(np.array([a, b, -1.0])) + 1e-12
        return plane_model


    def patch_stats(points_xyz, component_idx, normals, planarity, surface_variation, local_residual, voxel_size):
        pts = points_xyz[component_idx]
        plane_model = fit_plane_least_squares(pts)
        d_plane = point_to_plane_distance(pts, plane_model)
        mean_normal = np.nanmean(normals[component_idx], axis=0)
        mean_normal /= np.linalg.norm(mean_normal) + 1e-12
        area_m2 = len(component_idx) * (voxel_size ** 2)
        z_span_m = float(np.max(pts[:, 2]) - np.min(pts[:, 2]))

        return {
            'indices': component_idx,
            'n_points': int(len(component_idx)),
            'area_m2': float(area_m2),
            'plane_model': plane_model,
            'mean_abs_residual_m': float(np.nanmean(d_plane)),
            'rmse_residual_m': float(np.sqrt(np.nanmean(d_plane ** 2))),
            'mean_planarity': float(np.nanmean(planarity[component_idx])),
            'mean_surface_variation': float(np.nanmean(surface_variation[component_idx])),
            'mean_abs_nz': float(abs(mean_normal[2])),
            'centroid_x': float(np.mean(pts[:, 0])),
            'centroid_y': float(np.mean(pts[:, 1])),
            'centroid_z': float(np.mean(pts[:, 2])),
            'z_span_m': z_span_m,
        }


    def point_to_plane_signed_distance(points_xyz, plane_model):
        a, b, c, d = plane_model
        den = np.sqrt(a * a + b * b + c * c) + 1e-12
        return (a * points_xyz[:, 0] + b * points_xyz[:, 1] + c * points_xyz[:, 2] + d) / den


    def plane_offset_abs(plane_a, plane_b, ref_xy):
        xa, ya = ref_xy
        za = -(plane_a[0] * xa + plane_a[1] * ya + plane_a[3]) / (plane_a[2] + 1e-12)
        zb = -(plane_b[0] * xa + plane_b[1] * ya + plane_b[3]) / (plane_b[2] + 1e-12)
        return abs(za - zb)


    def centroid_gap(patch_a, patch_b):
        dx = patch_a['centroid_x'] - patch_b['centroid_x']
        dy = patch_a['centroid_y'] - patch_b['centroid_y']
        return float(np.hypot(dx, dy))


    def union_find(n):
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
        return parent, find, union



    def build_xy_occupancy(points_xy, cell_size):
        xmin, ymin = points_xy.min(axis=0)
        xmax, ymax = points_xy.max(axis=0)
        ncols = int(np.ceil((xmax - xmin) / cell_size)) + 1
        nrows = int(np.ceil((ymax - ymin) / cell_size)) + 1
        cols = np.floor((points_xy[:, 0] - xmin) / cell_size).astype(int)
        rows = np.floor((points_xy[:, 1] - ymin) / cell_size).astype(int)
        occ = np.zeros((nrows, ncols), dtype=bool)
        occ[rows, cols] = True
        return occ, xmin, ymin, cell_size


    def largest_component_2d(mask):
        lab, nlab = label(mask)
        if nlab == 0:
            return mask
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        return lab == np.argmax(counts)


    def occupancy_to_point_mask(points_xy, occ, xmin, ymin, cell_size):
        cols = np.floor((points_xy[:, 0] - xmin) / cell_size).astype(int)
        rows = np.floor((points_xy[:, 1] - ymin) / cell_size).astype(int)
        inside = (
            (rows >= 0) & (rows < occ.shape[0]) &
            (cols >= 0) & (cols < occ.shape[1])
        )
        mask = np.zeros(len(points_xy), dtype=bool)
        mask[inside] = occ[rows[inside], cols[inside]]
        return mask


    # Original notebook cell 8

    tic = time.perf_counter()
    las = laspy.read(LAS_FP)
    points_full = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    print('[INFO] full points:', len(points_full))
    print(f"[TIME] load_las: {time.perf_counter() - tic:.2f} s")


    # Original notebook cell 9
    plot_xy(points_full, 'Full LAS (sampled, colored by Z)')

    # Original notebook cell 11
    full_pcd = make_o3d_cloud(points_full)

    tic = time.perf_counter()
    work_pcd = full_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE_M)
    points_work = np.asarray(work_pcd.points)
    print('[INFO] working points:', len(points_work))
    print(f"[TIME] voxel_down_sample: {time.perf_counter() - tic:.2f} s")


    # Original notebook cell 12
    plot_xy(points_work, f'Working cloud (voxel={VOXEL_SIZE_M:.2f} m)')

    # Original notebook cell 14

    tic = time.perf_counter()
    work_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=NORMAL_MAX_NN)
    )
    work_pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    print(f"[TIME] estimate_normals: {time.perf_counter() - tic:.2f} s")

    work_tree, normals, planarity, linearity, surface_variation, local_residual = compute_local_geometry(
        points_work,
        radius_m=GEOM_RADIUS_M,
        max_nn=GEOM_MAX_NN,
    )

    abs_nz = np.abs(normals[:, 2])
    geom_df = pd.DataFrame({
        'planarity': planarity,
        'surface_variation': surface_variation,
        'local_residual': local_residual,
        'abs_nz': abs_nz,
    })
    geom_df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])


    # Original notebook cell 15

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].hist(planarity[np.isfinite(planarity)], bins=80, color='tab:blue')
    axes[0].axvline(PLANARITY_MIN, color='red', linestyle='--')
    axes[0].set_title('Planarity')

    axes[1].hist(surface_variation[np.isfinite(surface_variation)], bins=80, color='tab:green')
    axes[1].axvline(SURFACE_VARIATION_MAX, color='red', linestyle='--')
    axes[1].set_title('Surface variation')

    axes[2].hist(local_residual[np.isfinite(local_residual)], bins=80, color='tab:orange')
    axes[2].axvline(LOCAL_RESIDUAL_MAX_M, color='red', linestyle='--')
    axes[2].set_title('Local residual')

    axes[3].hist(abs_nz[np.isfinite(abs_nz)], bins=80, color='tab:purple')
    axes[3].axvline(VERTICAL_NORMAL_MIN, color='red', linestyle='--')
    axes[3].set_title('|normal_z|')
    plt.show()


    # Original notebook cell 17

    mask_seed_candidate = (
        np.isfinite(planarity)
        & np.isfinite(surface_variation)
        & np.isfinite(local_residual)
        & np.isfinite(abs_nz)
        & (planarity >= PLANARITY_MIN)
        & (surface_variation <= SURFACE_VARIATION_MAX)
        & (local_residual <= LOCAL_RESIDUAL_MAX_M)
        & (abs_nz >= VERTICAL_NORMAL_MIN)
    )

    print('[INFO] seed candidate points:', int(mask_seed_candidate.sum()))
    plot_xy(points_work, 'Seed candidates', mask=mask_seed_candidate)


    # Original notebook cell 19

    seed_points = points_work[mask_seed_candidate]
    seed_pcd = make_o3d_cloud(seed_points)

    tic = time.perf_counter()
    plane_model, inliers_local = seed_pcd.segment_plane(
        distance_threshold=RANSAC_DIST_THR_M,
        ransac_n=RANSAC_N,
        num_iterations=RANSAC_ITERS,
    )
    print(f"[TIME] segment_plane: {time.perf_counter() - tic:.2f} s")
    print('[INFO] plane_model [a, b, c, d]:', plane_model)
    print('[INFO] seed inliers:', len(inliers_local))

    seed_mask = np.zeros(len(points_work), dtype=bool)
    seed_idx = np.where(mask_seed_candidate)[0][np.asarray(inliers_local, dtype=int)]
    seed_mask[seed_idx] = True
    plot_xy(points_work, 'RANSAC seed plane', mask=seed_mask)


    # Original notebook cell 21

    grown_mask = region_grow(
        points_xyz=points_work,
        tree=work_tree,
        seed_mask=seed_mask,
        normals=normals,
        planarity=planarity,
        surface_variation=surface_variation,
        local_residual=local_residual,
        plane_model=plane_model,
        grow_radius_m=GROW_RADIUS_M,
        grow_plane_dist_thr_m=GROW_PLANE_DIST_THR_M,
        grow_normal_angle_deg=GROW_NORMAL_ANGLE_DEG,
        grow_planarity_min=GROW_PLANARITY_MIN,
        grow_surface_variation_max=GROW_SURFACE_VARIATION_MAX,
        grow_local_residual_max_m=GROW_LOCAL_RESIDUAL_MAX_M,
    )
    print('[INFO] grown points:', int(grown_mask.sum()))
    plot_xy(points_work, 'Region-grown slab candidates', mask=grown_mask)


    # Original notebook cell 23

    components = connected_components_from_mask(points_work, work_tree, grown_mask, radius_m=GROW_RADIUS_M)
    component_sizes = np.array([len(c) for c in components], dtype=int)
    print('[INFO] n_components:', len(components))
    print(pd.Series(component_sizes).describe())


    # Original notebook cell 24
    patches = []
    for comp in components:
        if len(comp) < PATCH_MIN_POINTS:
            continue

        stats = patch_stats(
            points_xyz=points_work,
            component_idx=comp,
            normals=normals,
            planarity=planarity,
            surface_variation=surface_variation,
            local_residual=local_residual,
            voxel_size=VOXEL_SIZE_M,
        )

        if stats['area_m2'] < PATCH_MIN_AREA_M2:
            continue
        if stats['mean_abs_residual_m'] > PATCH_MAX_MEAN_RESIDUAL_M:
            continue
        if stats['mean_planarity'] < PATCH_MIN_MEAN_PLANARITY:
            continue
        if stats['mean_surface_variation'] > PATCH_MAX_MEAN_SURFACE_VARIATION:
            continue
        if stats['mean_abs_nz'] < PATCH_MIN_VERTICAL_NORMAL:
            continue
        if stats['z_span_m'] > PATCH_MAX_ZSPAN_M:
            continue

        patches.append(stats)

    patch_df = pd.DataFrame([
        {
            'patch_id': i,
            'n_points': p['n_points'],
            'area_m2': p['area_m2'],
            'mean_abs_residual_m': p['mean_abs_residual_m'],
            'mean_planarity': p['mean_planarity'],
            'mean_surface_variation': p['mean_surface_variation'],
            'mean_abs_nz': p['mean_abs_nz'],
            'z_span_m': p['z_span_m'],
            'centroid_x': p['centroid_x'],
            'centroid_y': p['centroid_y'],
            'centroid_z': p['centroid_z'],
        }
        for i, p in enumerate(patches)
    ])
    patch_df.sort_values('area_m2', ascending=False).reset_index(drop=True)

    # Original notebook cell 26
    parent, find, union = union_find(len(patches))

    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            na = np.asarray(patches[i]['plane_model'][:3])
            nb = np.asarray(patches[j]['plane_model'][:3])
            ang = normal_angle_deg(na.reshape(1, 3), nb)[0]
            gap = centroid_gap(patches[i], patches[j])
            ref_xy = (
                (patches[i]['centroid_x'] + patches[j]['centroid_x']) / 2.0,
                (patches[i]['centroid_y'] + patches[j]['centroid_y']) / 2.0,
            )
            offset = plane_offset_abs(patches[i]['plane_model'], patches[j]['plane_model'], ref_xy)

            if ang <= MERGE_MAX_NORMAL_ANGLE_DEG and offset <= MERGE_MAX_OFFSET_M and gap <= MERGE_MAX_GAP_M:
                union(i, j)

    merged_groups = {}
    for i in range(len(patches)):
        root = find(i)
        merged_groups.setdefault(root, []).append(i)

    merged_records = []
    for gid, members in merged_groups.items():
        idx_all = np.concatenate([patches[m]['indices'] for m in members])
        pts = points_work[idx_all]
        plane_model = fit_plane_least_squares(pts)
        d_plane = point_to_plane_distance(pts, plane_model)
        mean_normal = np.nanmean(normals[idx_all], axis=0)
        mean_normal /= np.linalg.norm(mean_normal) + 1e-12
        z_span_m = float(np.max(pts[:, 2]) - np.min(pts[:, 2]))

        merged_records.append({
            'group_id': gid,
            'members': members,
            'indices': idx_all,
            'n_points': int(len(idx_all)),
            'area_m2': float(len(idx_all) * (VOXEL_SIZE_M ** 2)),
            'plane_model': plane_model,
            'mean_abs_residual_m': float(np.nanmean(d_plane)),
            'rmse_residual_m': float(np.sqrt(np.nanmean(d_plane ** 2))),
            'mean_planarity': float(np.nanmean(planarity[idx_all])),
            'mean_surface_variation': float(np.nanmean(surface_variation[idx_all])),
            'mean_abs_nz': float(abs(mean_normal[2])),
            'centroid_x': float(np.mean(pts[:, 0])),
            'centroid_y': float(np.mean(pts[:, 1])),
            'centroid_z': float(np.mean(pts[:, 2])),
            'z_span_m': z_span_m,
        })

    merged_df = pd.DataFrame([
        {
            'group_id': g['group_id'],
            'n_members': len(g['members']),
            'n_points': g['n_points'],
            'area_m2': g['area_m2'],
            'mean_abs_residual_m': g['mean_abs_residual_m'],
            'mean_planarity': g['mean_planarity'],
            'mean_surface_variation': g['mean_surface_variation'],
            'mean_abs_nz': g['mean_abs_nz'],
            'z_span_m': g['z_span_m'],
            'centroid_x': g['centroid_x'],
            'centroid_y': g['centroid_y'],
            'centroid_z': g['centroid_z'],
        }
        for g in merged_records
    ]).sort_values('area_m2', ascending=False).reset_index(drop=True)

    merged_df

    # Original notebook cell 28
    final_groups = []
    for g in merged_records:
        if g['area_m2'] < FINAL_MIN_AREA_M2:
            continue
        if g['mean_abs_residual_m'] > FINAL_MAX_MEAN_RESIDUAL_M:
            continue
        if g['mean_planarity'] < FINAL_MIN_MEAN_PLANARITY:
            continue
        if g['mean_surface_variation'] > FINAL_MAX_MEAN_SURFACE_VARIATION:
            continue
        if g['mean_abs_nz'] < FINAL_MIN_VERTICAL_NORMAL:
            continue
        if g['z_span_m'] > FINAL_MAX_ZSPAN_M:
            continue
        final_groups.append(g)

    final_mask_work_points = np.zeros(len(points_work), dtype=bool)
    for g in final_groups:
        final_mask_work_points[g['indices']] = True

    print('[INFO] final slab groups:', len(final_groups))
    print('[INFO] final slab working points:', int(final_mask_work_points.sum()))
    plot_xy(points_work, 'Final slab points on working cloud', mask=final_mask_work_points)

    # Convert point-wise slab result into a slab zone in XY space
    slab_zone_pts = points_work[final_mask_work_points][:, :2]
    occ, xmin_zone, ymin_zone, zone_cell = build_xy_occupancy(slab_zone_pts, ZONE_GRID_M)

    for _ in range(ZONE_CLOSE_ITERS):
        occ = binary_closing(occ)

    if FILL_ZONE_HOLES:
        occ = binary_fill_holes(occ)

    if KEEP_ONLY_LARGEST_ZONE:
        occ = largest_component_2d(occ)

    final_mask_work_zone = occupancy_to_point_mask(points_work[:, :2], occ, xmin_zone, ymin_zone, zone_cell)
    print('[INFO] final slab zone points on working cloud:', int(final_mask_work_zone.sum()))
    plot_xy(points_work, 'Final filled slab zone on working cloud', mask=final_mask_work_zone)

    # pick one representative slab plane for transfer to full cloud
    if len(final_groups) == 0:
        raise RuntimeError("No final slab group survived filtering.")

    best_group = max(final_groups, key=lambda g: g['area_m2'])
    final_slab_plane = best_group['plane_model']
    print('[INFO] representative slab plane:', final_slab_plane)
    print('[INFO] representative slab area_m2:', best_group['area_m2'])

    # Original notebook cell 30
    # ============================================================
    # Transfer filled slab zone to full cloud
    # - compatible version: keeps old variable names for visualization
    # - preserve depression
    # - remove gate with asymmetric signed-distance filter
    # ============================================================

    # 1) XY filled slab zone on full cloud
    mask_full_xy = occupancy_to_point_mask(points_full[:, :2], occ, xmin_zone, ymin_zone, zone_cell)
    print('[INFO] slab zone points on full cloud (XY filled zone):', int(mask_full_xy.sum()))

    # 2) estimate normals on full cloud
    tic = time.perf_counter()
    full_pcd_for_normal = make_o3d_cloud(points_full)
    full_pcd_for_normal.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=FULL_NORMAL_RADIUS_M,
            max_nn=FULL_NORMAL_MAX_NN
        )
    )
    full_pcd_for_normal.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    normals_full = np.asarray(full_pcd_for_normal.normals)
    abs_nz_full = np.abs(normals_full[:, 2])
    print(f"[TIME] full-cloud estimate_normals: {time.perf_counter() - tic:.2f} s")

    # 3) plane distances
    signed_full_plane = point_to_plane_signed_distance(points_full, final_slab_plane)
    dist_full_plane = np.abs(signed_full_plane)   # 기존 변수명 호환용

    # 4) final mask
    # 위(+): 엄격하게 제한 -> gate 제거
    # 아래(-): 넓게 허용 -> depression 보존
    mask_full_slab = (
        mask_full_xy
        & np.isfinite(abs_nz_full)
        & (signed_full_plane <= TRANSFER_POS_DIST_M)
        & (signed_full_plane >= -TRANSFER_NEG_DIST_M)
        & (abs_nz_full >= TRANSFER_MIN_VERTICAL_NORMAL)
    )

    print('[INFO] final measured slab points on full cloud:', int(mask_full_slab.sum()))

    # ------------------------------------------------------------
    # 5) basic XY visualization (기존 helper 호환)
    # ------------------------------------------------------------
    plot_xy(points_full, 'Extracted Concrete Slab Zone', mask=mask_full_xy)
    plot_xy(points_full, 'Transferred slab on full cloud', mask=mask_full_slab)

    # ------------------------------------------------------------
    # 6) extracted slab points / dataframe
    # - visualization/export uses filled slab zone
    # - transferred slab mask is kept separately for diagnostics
    # - optional Z-outlier cleanup is applied only at the end
    # ------------------------------------------------------------
    mask_full_filled_final = mask_full_xy.copy()
    z_low = np.nan
    z_high = np.nan

    if ENABLE_Z_OUTLIER_FILTER and mask_full_xy.sum() > 0:
        z_filled = points_full[mask_full_xy, 2]
        if Z_OUTLIER_METHOD == "iqr":
            q1, q3 = np.quantile(z_filled, [0.25, 0.75])
            iqr = q3 - q1
            z_low = q1 - Z_IQR_K * iqr
            z_high = q3 + Z_IQR_K * iqr
        elif Z_OUTLIER_METHOD == "quantile":
            z_low, z_high = np.quantile(z_filled, [Z_LOWER_Q, Z_UPPER_Q])
        else:
            raise ValueError(f"Unsupported Z_OUTLIER_METHOD: {Z_OUTLIER_METHOD}")

        if Z_OUTLIER_SIDE == "upper":
            keep_z = z_filled <= z_high
        elif Z_OUTLIER_SIDE == "both":
            keep_z = (z_filled >= z_low) & (z_filled <= z_high)
        else:
            raise ValueError(f"Unsupported Z_OUTLIER_SIDE: {Z_OUTLIER_SIDE}")

        mask_full_filled_final[mask_full_xy] = keep_z
        print(f"[INFO] Z-outlier cleanup on filled zone: kept {int(keep_z.sum())} / {len(keep_z)} points")
        if Z_OUTLIER_SIDE == "upper":
            print(f"[INFO] Z keep upper bound: <= {z_high:.3f} m")
        else:
            print(f"[INFO] Z keep range: [{z_low:.3f}, {z_high:.3f}] m")

    slab_points_full = points_full[mask_full_filled_final]

    df_slab_xyz = pd.DataFrame({
        "x": slab_points_full[:, 0],
        "y": slab_points_full[:, 1],
        "z": slab_points_full[:, 2],
        "signed_plane_dist_m": signed_full_plane[mask_full_filled_final],
        "abs_plane_dist_m": np.abs(signed_full_plane[mask_full_filled_final]),
        "abs_nz": abs_nz_full[mask_full_filled_final],
        "is_transferred_slab": mask_full_slab[mask_full_filled_final].astype(np.uint8),
    })

    print(df_slab_xyz[["z", "signed_plane_dist_m", "abs_plane_dist_m", "abs_nz"]].describe(
        percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
    ))

    # ------------------------------------------------------------
    # 7) Z-colored visualization
    # ------------------------------------------------------------
    sample_n = min(POINT_PLOT_SAMPLE, len(slab_points_full))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(slab_points_full), size=sample_n, replace=False) if len(slab_points_full) > sample_n else np.arange(len(slab_points_full))

    plt.figure(figsize=(8, 6), dpi=FIG_DPI)
    sc = plt.scatter(
        slab_points_full[idx, 0],
        slab_points_full[idx, 1],
        c=slab_points_full[idx, 2],
        s=1,
        cmap="viridis"
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Filled slab zone points colored by Z (after Z-outlier cleanup)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(sc, label="Z (m)")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 8) signed plane distance visualization
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 6), dpi=FIG_DPI)
    lim = np.quantile(df_slab_xyz["abs_plane_dist_m"], 0.98) if len(df_slab_xyz) > 0 else 1.0
    sc = plt.scatter(
        slab_points_full[idx, 0],
        slab_points_full[idx, 1],
        c=df_slab_xyz["signed_plane_dist_m"].to_numpy()[idx],
        s=1,
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Filled slab zone points colored by signed plane distance (after Z-outlier cleanup)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(sc, label="Signed plane distance (m)")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 9) histogram
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=FIG_DPI)
    plt.hist(df_slab_xyz["z"], bins=100, color="gray", alpha=0.8)
    plt.xlabel("Z (m)")
    plt.ylabel("Count")
    plt.title("Z distribution of filled slab zone points after cleanup")
    plt.tight_layout()
    plt.show()

    # Original notebook cell 32
    filled_zone_idx = np.where(mask_full_filled_final)[0]

    las_slab = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    las_slab.header.offsets = las.header.offsets
    las_slab.header.scales = las.header.scales
    las_slab.x = las.x[filled_zone_idx]
    las_slab.y = las.y[filled_zone_idx]
    las_slab.z = las.z[filled_zone_idx]

    las_out = os.path.join(OUT_DIR, 'slab_extracted.las')
    las_slab.write(las_out)
    print('[SAVE]', las_out)

    export_sample_idx = filled_zone_idx
    if len(export_sample_idx) > POINT_PLOT_SAMPLE:
        rng = np.random.default_rng(RANDOM_SEED)
        export_sample_idx = rng.choice(export_sample_idx, size=POINT_PLOT_SAMPLE, replace=False)

    las_crs = las.header.parse_crs()

    gdf_pts = gpd.GeoDataFrame(
        {'is_filled_slab_zone': np.ones(len(export_sample_idx), dtype=np.uint8)},
        geometry=[Point(x, y) for x, y in zip(las.x[export_sample_idx], las.y[export_sample_idx])],
        crs=las_crs,
    )
    pts_out = os.path.join(OUT_DIR, 'slab_extracted_sample.gpkg')
    gdf_pts.to_file(pts_out, driver='GPKG')
    print('[SAVE]', pts_out)
    # END ORIGINAL ANALYSIS
    patch_df.to_csv(output_dir/'patches.csv',index=True)
    merged_df.to_csv(output_dir/'merged_groups.csv',index=True)
    geom_df.describe(percentiles=[.05,.25,.5,.75,.95]).to_csv(output_dir/'geometry_summary.csv',index=True)
    summary={'full_points':len(points_full),'working_points':len(points_work),'final_groups':len(final_groups),'filled_zone_points':len(slab_points_full),'representative_plane':final_slab_plane.tolist()}
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/01_EYAS_translated_segmented_slab_zone.ply')
    parser.add_argument('--output-dir',type=Path,default=WORK/'analysis/slab_depression/slab_extraction')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='lidar-slab-') as directory:
        input_las=prepare_las(args.input,Path(directory))
        print(json.dumps(extract(input_las,args.output_dir),indent=2))
if __name__=='__main__':main()
