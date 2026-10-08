import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Geometry column does not contain geometry")

# ===== 경로 설정 =====
input_gpkg = r"/home/gunwoo/multibeam/multibeam/1012/results/0913_multibeam_A_align_posterior_order1a_res0.5_fillgap.gpkg"
tif_dir    = r"/home/gunwoo/multibeam/multibeam/1012/tif"

# 파일명에서 '_fillgap' 제거 후 동일 이름의 TIFF 자동 매칭
stem       = os.path.splitext(os.path.basename(input_gpkg))[0].replace("_fillgap", "")
tif_path   = os.path.join(tif_dir, stem + ".tif")

# CSV는 동일 경로/파일명으로 저장
output_csv = os.path.splitext(input_gpkg)[0] + ".csv"

# ===== 보조 함수: 래스터에서 z 샘플링 =====
def sample_z_at_points(dataset, xy_list):
    """(x, y) 리스트에 대해 z값 샘플링"""
    nodata = dataset.nodata
    zs = []
    for val in dataset.sample(xy_list):
        z = float(val[0])
        if nodata is not None and np.isfinite(nodata) and np.isclose(z, nodata):
            zs.append(np.nan)
        else:
            zs.append(z)
    return zs

# ===== 1) GPKG 읽기 =====
gdf = gpd.read_file(input_gpkg)

# scour_id 없으면 생성
if "scour_id" not in gdf.columns:
    gdf["scour_id"] = [f"scour_{i+1}" for i in range(len(gdf))]

# ===== 2) 래스터 열기 =====
with rasterio.open(tif_path) as src:
    # ===== 3) 폴리곤별 노드 z 및 중심점 z 추출 =====
    nodes_xyz_list = []
    centroid_xyz_list = []

    for geom in gdf.geometry:
        if geom.is_empty:
            nodes_xyz_list.append([])
            centroid_xyz_list.append((np.nan, np.nan, np.nan))
            continue

        # exterior 좌표 (x, y)
        exterior_coords = list(geom.exterior.coords)

        # z값 샘플링
        zs = sample_z_at_points(src, exterior_coords)

        # 이중 튜플 [(x, y, z), ...]
        nodes_xyz = [(x, y, (z if np.isfinite(z) else np.nan)) for (x, y), z in zip(exterior_coords, zs)]
        nodes_xyz_list.append(nodes_xyz)

        # 중심점 (x, y, z)
        c = geom.centroid
        cz = sample_z_at_points(src, [(c.x, c.y)])[0]
        centroid_xyz_list.append((c.x, c.y, cz if np.isfinite(cz) else np.nan))

# ===== 4) 면적 계산 보완 =====
if "area_m2" not in gdf.columns:
    gdf["area_m2"] = gdf.geometry.area

# ===== 5) 최종 DataFrame 구성 =====
cols_base = ["scour_id", "area_m2", "mean_depth", "max_depth", "median_depth"]
cols_existing = [c for c in cols_base if c in gdf.columns]

df_out = pd.DataFrame(gdf[cols_existing])
df_out = df_out.rename(columns={"area_m2": "area"})

df_out["centroid"] = centroid_xyz_list   # (x, y, z) 튜플
df_out["nodes_xyz"] = nodes_xyz_list     # [(x,y,z), ...]

# geometry 컬럼은 제외

# ===== 6) CSV 저장 =====
df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"✅ CSV 저장 완료: {output_csv}")
print(f"📊 총 폴리곤 개수: {len(df_out)}개")
print(df_out.head(2))