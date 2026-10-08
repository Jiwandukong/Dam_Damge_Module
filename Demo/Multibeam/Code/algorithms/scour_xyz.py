import geopandas as gpd
import rasterio
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Geometry column does not contain geometry")

# ===== 경로 설정 =====
input_gpkg = r"C:\Users\scsi\Desktop\오세훈\SCSI\2025-2\환경부\1012\results\0913_multibeam_A_align_posterior_order1a_res1.0_fillgap.gpkg"
tif_dir    = r"C:\Users\scsi\Desktop\오세훈\SCSI\2025-2\환경부\1012\tif"

# '_fillgap' 제거 후 동일 이름의 TIFF 자동 매칭
stem       = os.path.splitext(os.path.basename(input_gpkg))[0].replace("_fillgap", "")
tif_path   = os.path.join(tif_dir, stem + ".tif")

# XYZ 출력 파일 (CloudCompare용)
output_xyz = os.path.splitext(input_gpkg)[0] + "_cloud.xyz"

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

# ===== 2) 래스터 열기 및 좌표 샘플링 =====
xyz_points = []

with rasterio.open(tif_path) as src:
    for geom in gdf.geometry:
        if geom.is_empty:
            continue

        # exterior 노드 (x, y)
        exterior_coords = list(geom.exterior.coords)
        zs = sample_z_at_points(src, exterior_coords)

        # (x, y, z) 모두 모으기
        for (x, y), z in zip(exterior_coords, zs):
            if np.isfinite(z):
                xyz_points.append((x, y, z))

        # 중심점도 함께 추가
        c = geom.centroid
        cz = sample_z_at_points(src, [(c.x, c.y)])[0]
        if np.isfinite(cz):
            xyz_points.append((c.x, c.y, cz))

# ===== 3) XYZ 파일 저장 =====
with open(output_xyz, "w", encoding="utf-8") as f:
    for x, y, z in xyz_points:
        f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

print(f"✅ CloudCompare용 XYZ 저장 완료: {output_xyz}")
print(f"📊 총 좌표 개수: {len(xyz_points)}개")