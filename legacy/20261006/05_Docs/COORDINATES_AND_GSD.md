# Coordinates and rough GSD

## 1. 카메라 Ray

원본 pixel `(u,v)`와 XMP의 보정 초점거리 `f`, 광학중심 `(cx,cy)`로 pinhole ray를 만듭니다.

```text
d_camera = normalize([(u-cx)/f, (v-cy)/f, 1])
d_world  = R(yaw,pitch,roll) · d_camera
ray(t)   = camera_center_EPSG5186 + t · d_world
```

카메라 위·경도는 WGS84에서 EPSG:5186으로 변환하며 고도는 MRK `Ellh`를 우선 사용합니다. Ray와 `dam - Cloud.obj` 삼각형의 첫 교차점을 손상 위치로 사용합니다.

## 2. 국부 3D pixel scale

원근 영상은 한 장 전체에 하나의 고정 GSD를 적용하지 않습니다. 최소면적 회전 사각형 중심과 가장 가까운 **해당 손상 component 내부 pixel**을 기준점으로 고정한 뒤, 사각형의 장축과 단축 방향으로 각각 양쪽 pixel을 표본화하고 네 Ray의 mesh hit를 사용합니다. 이 내부점 snap은 U자형처럼 오목한 손상의 빈 공간에서 GSD를 재는 문제를 막습니다.

```text
gsd_length = |P_length+ - P_length-| / pixel_span_length
gsd_width  = |P_width+  - P_width-|  / pixel_span_width
gsd_area   = |ΔP_length × ΔP_width| / (span_length · span_width)
```

보고 값은 다음과 같습니다.

```text
CRC length_m = length_px · gsd_length
CRC width_m  = width_px  · gsd_width
DLM area_m2  = area_px   · gsd_area
SPL area_m2  = area_px   · gsd_area
```

CRC의 길이·폭은 skeleton/centerline 또는 mesh geodesic이 아니라 component를 감싸는 회전 사각형의 장·단축입니다. 굽거나 분기된 균열에서는 실제 균열 경로 길이와 차이가 날 수 있습니다.

샘플 임무의 영상 중심 sanity check는 X 약 `0.616–0.636 mm/px`, Y 약 `0.754–0.800 mm/px`, 표면 면적 약 `0.465–0.509 mm²/px`였습니다. 실제 산출물은 이 전역 범위를 복사하지 않고 손상별 local Ray로 다시 계산합니다.

## 3. 정확도 표기

현재 결과는 다음 이유로 정밀 측량값이 아닌 근사값입니다.

- OBJ 내부에는 CRS가 없고 EPSG:5186/Z-up은 외부 계약입니다.
- MRK 타원체고와 OBJ Z의 수직 datum 일치가 검증되지 않았습니다.
- XMP `DewarpData`는 보존하지만 lens distortion을 Ray에 적용하지 않습니다.
- UAV가 벽면에 최대한 정면에 가깝게 촬영했다는 운용 가정을 사용합니다.
- 손상 경계와 class는 SAM3 예측입니다.
- 국부 stencil의 네 Ray가 메시의 날카로운 불연속을 가로지르는지에 대한 면-normal 연속성 검사는 아직 적용하지 않습니다.

현재 사용자용 CSV와 Excel은 PDF 예시와 같은 13개 열만 제공합니다. Ray 또는 국부 GSD 계산이 실패하면 관련 좌표·m·m² 값이 빈칸이 되며, 빈칸은 0으로 해석하면 안 됩니다.
