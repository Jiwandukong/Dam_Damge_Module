import numpy as np
import pandas as pd
import os
import rasterio
from rasterio.transform import from_origin
import laspy
from numba_cube import run_cube_gridding
from joblib import Parallel, delayed
from tqdm import tqdm

# ===== Step 1: 파일 및 경로 설정 =====
las_file = r"C:\Users\scsi\Desktop\오세훈\SCSI\2025-2\환경부\0913_multibeam_align\0913_multibeam_B_align.las"

# 파일명 추출
las_filename = os.path.splitext(os.path.basename(las_file))[0]

# 결과 저장 디렉토리 설정
base_output_dir = r"C:\Users\scsi\Desktop\오세훈\SCSI\2025-2\환경부\1012"
os.makedirs(base_output_dir, exist_ok=True)

# ===== Step 2: LAS 파일 로드 =====
las = laspy.read(las_file)
crs_obj = las.header.parse_crs()

if crs_obj is None:
    print("⚠️ LAS에 CRS 정보가 없습니다. EPSG:5186 (Korea 2000 / Central Belt)로 기본 지정합니다.")
    crs_str = "EPSG:5186"
else:
    crs_str = crs_obj.to_string()  # 예: 'EPSG:5186' 또는 WKT

# LAS 좌표를 numpy 배열로 변환 (중요!)
easting  = np.asarray(las.x, dtype=np.float64)
northing = np.asarray(las.y, dtype=np.float64)
depth    = np.asarray(las.z, dtype=np.float64)  # 양수 그대로 사용

# ===== Step 3: 데이터프레임 구성 및 불확실성 계산 =====
df = pd.DataFrame({"easting": easting, "northing": northing, "depth": depth})

# (필요시 값 확인)
# print(df.head())

depth_precision = 0.01
heave_accuracy = 0.02
pitch_roll_error_deg = 0.02
heading_error_deg = 0.02
swath_angle_deg = 70

pitch_roll_error_rad = np.radians(pitch_roll_error_deg)
heading_error_rad = np.radians(heading_error_deg)
swath_angle_rad = np.radians(swath_angle_deg)

mean_depth = df["depth"].mean()

df["tvu"] = np.sqrt(
    depth_precision**2 +
    heave_accuracy**2 +
    (mean_depth * np.tan(pitch_roll_error_rad))**2
)

df["thu"] = mean_depth * np.tan(heading_error_rad) / np.cos(swath_angle_rad)

# ===== Step 4: 데이터 범위 계산 =====
min_easting, max_easting = df["easting"].agg(["min", "max"])
min_northing, max_northing = df["northing"].agg(["min", "max"])

easting  = df["easting"].to_numpy()
northing = df["northing"].to_numpy()
depth    = df["depth"].to_numpy()
thu      = df["thu"].to_numpy()
tvu      = df["tvu"].to_numpy()

# ===== Step 5: 해상도 및 방법 목록 =====
resolutions = [0.1, 0.3, 0.5, 1.0]
methods = ['predicted', 'prior', 'local', 'posterior']
iho_order = 'order1a'

# ===== GeoTIFF 저장 함수 =====
def save_as_tiff(grid_data, output_path, resolution, min_easting, max_northing, crs_str):
    """ grid 데이터를 GeoTIFF로 저장 """
    try:
        numrows, numcols = grid_data.shape
        transform = from_origin(min_easting, max_northing, resolution, resolution)

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=numrows, width=numcols,
            count=1, dtype=np.float32,
            crs=crs_str,
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(grid_data, 1)

        print(f"📂 TIFF 저장 완료: {output_path}")

    except Exception as e:
        print(f"⚠️ TIFF 저장 오류: {str(e)}")

# ===== CUBE 실행 및 TIFF 저장 =====
def process_cube(resolution, method):
    try:
        numcols = int(np.ceil((max_easting  - min_easting ) / resolution))
        numrows = int(np.ceil((max_northing - min_northing) / resolution))
        numcols = max(numcols, 1)
        numrows = max(numrows, 1)

        depth_grid, uncertainty_grid, _, _ = run_cube_gridding(
            depth, thu, tvu, easting, northing,
            numcols, numrows, min_easting, max_northing,
            method=method, iho_order=iho_order,
            grid_resolution_x=resolution, grid_resolution_y=resolution
        )

        # 파일명 및 경로 지정
        tiff_filename_out = f"{las_filename}_{method}_{iho_order}_res{resolution:.1f}.tif"
        tiff_filepath = os.path.join(base_output_dir, tiff_filename_out)

        # GeoTIFF 저장 (깊이 그리드)
        save_as_tiff(depth_grid, tiff_filepath, resolution, min_easting, max_northing, crs_str)

        # (선택) 불확실성 그리드도 저장하고 싶다면 아래 주석 해제
        # unc_filename_out = f"{las_filename}_{method}_{iho_order}_res{resolution:.1f}_uncertainty.tif"
        # unc_filepath = os.path.join(base_output_dir, unc_filename_out)
        # save_as_tiff(uncertainty_grid, unc_filepath, resolution, min_easting, max_northing, crs_str)

        return f"✅ 저장 완료: {tiff_filepath}"

    except Exception as e:
        return f"⚠️ 오류 발생 ({method}, {resolution}m): {str(e)}"

