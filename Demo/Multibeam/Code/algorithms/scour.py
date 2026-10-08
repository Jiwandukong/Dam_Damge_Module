import rasterio
import numpy as np
import geopandas as gpd
import cv2
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
from rasterio import features

def load_raster(tiff_file):
    """GeoTIFF 파일을 로드하여 픽셀 데이터를 반환"""
    with rasterio.open(tiff_file) as src:
        tiff_data = src.read(1)  # 첫 번째 밴드 로드
        transform = src.transform  # 픽셀 → 지리 좌표 변환 정보
        nodata_value = src.nodata  # NoData 값 확인
        crs = src.crs  # 좌표계 정보 (EPSG 코드 포함)

        # 해상도 (m/px) 계산 (절대값 사용)
        resolution_x, resolution_y = abs(src.res[0]), abs(src.res[1])  # 항상 양수 변환

        # NoData 값 처리 (NaN으로 변환)
        if nodata_value is not None:
            tiff_data[tiff_data == nodata_value] = np.nan

    return tiff_data, transform, crs, resolution_x, resolution_y


def polygonize_raster(tiff_data, transform, threshold_value):
    """특정 값 이상의 픽셀만 선택하여 폴리곤으로 변환"""
    binary_mask = tiff_data >= threshold_value  # 지정된 임계값 이상인 픽셀 선택
    
    # OpenCV를 사용하여 컨투어 검출
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    polygons = []
    areas = []
    for c in contours:
        if len(c) > 4:  # 최소 4점 이상만 폴리곤화
            pixel_coords = c[:, 0, :]
            geo_coords = [rasterio.transform.xy(transform, row, col) 
                          for col, row in pixel_coords]
            
            # 폴리곤 생성
            polygon = Polygon(geo_coords)

            # 반시계 방향(올바른 방향)으로 보정
            if polygon.area < 0:
                polygon = orient(polygon, sign=1.0)  # 반시계 방향으로 정렬

            polygons.append(polygon)
            areas.append(polygon.area)

    return polygons, areas


def filter_polygons(polygons, areas, min_area_m2):
    """면적 기반 필터링을 적용하여 작은 폴리곤 제거"""
    filtered_polygons = []
    filtered_areas = []

    for polygon, area_m2 in zip(polygons, areas):
        if area_m2 >= min_area_m2:
            filtered_polygons.append(polygon)
            filtered_areas.append(area_m2)

    return filtered_polygons, filtered_areas


def calculate_damage_ratio(tiff_data, resolution_x, resolution_y, filtered_areas):
    """전체 손상률(%) 계산"""
    total_valid_pixels = np.count_nonzero(~np.isnan(tiff_data))  # NoData 제외한 유효 픽셀 개수
    total_valid_area = total_valid_pixels * resolution_x * resolution_y  # 유효 영역의 실제 면적
    damaged_area = sum(filtered_areas)  # 폴리곤들의 총 면적

    damage_ratio = (damaged_area / total_valid_area) * 100 if total_valid_area > 0 else 0
    return damage_ratio, total_valid_area, damaged_area


def calculate_polygon_depths(tiff_data, polygons, transform, threshold_value):
    """
    polygon별로 rasterize 마스크를 만든 뒤,
    해당 폴리곤 내부 픽셀값에 대해 mean, max, median을 구한다.
    """
    height, width = tiff_data.shape
    depths = []

    for polygon in polygons:
        # 폴리곤을 rasterize하여 0/1 마스크 생성
        mask = features.rasterize(
            [(polygon, 1)],  # (geometry, value)
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=False,  # True로 하면 경계 부근 픽셀까지 포함
            dtype=np.uint8
        )

        # 마스크가 1인 지점의 픽셀값만 추출
        valid_values = tiff_data[mask == 1]
        valid_values = valid_values[~np.isnan(valid_values)]  # NoData 제거

        if len(valid_values) > 0:
            # threshold_value만큼 빼기
            depth_values = valid_values - threshold_value
            mean_depth = np.mean(depth_values)
            max_depth = np.max(depth_values)
            median_depth = np.median(depth_values)
        else:
            mean_depth, max_depth, median_depth = np.nan, np.nan, np.nan

        depths.append((mean_depth, max_depth, median_depth))

    return depths


def save_to_gpkg(polygons, areas, depths, crs, output_gpkg):
    """폴리곤 데이터를 GPKG로 저장"""
    gdf = gpd.GeoDataFrame(
        {
            'area_m2': areas,
            'mean_depth': [d[0] for d in depths],
            'max_depth': [d[1] for d in depths],
            'median_depth': [d[2] for d in depths],
            'geometry': polygons
        }, 
        crs=crs
    )
    gdf.to_file(output_gpkg, driver="GPKG")
    print(f"✅ 벡터화 완료: {output_gpkg} (손상 깊이 포함)")
    print(f"📝 총 폴리곤 개수: {len(gdf)}")
    if len(areas) > 0:
        print(f"📏 필터링된 최소 면적: {min(areas)} m²")
        print(f"📏 필터링된 최대 면적: {max(areas)} m²")
    else:
        print("📏 폴리곤이 없습니다.")


