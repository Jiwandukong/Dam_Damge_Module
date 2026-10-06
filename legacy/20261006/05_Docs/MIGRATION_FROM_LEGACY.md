# Migration from legacy project

새 저장소에서 제거한 경로:

- DeepLab 학습·추론 및 confidence raster
- LAS point-cloud Ray/XYZ-map/cache
- GeoTIFF, orthophoto homography, GCP height interpolation
- 좌표가 없는 GLB mapping
- July SAM3 checkpoint 상수와 1024/768 overlapping tiling

새 기준:

- 탐지: August 512 query-balanced SAM3
- mapping: `dam - Cloud.obj` 기반 EPSG:5186/Z-up mesh Ray
- 정량화: 손상 위치별 local mesh GSD
- 산출: 손상별 Excel + mask/overlay/CSV/JSON
