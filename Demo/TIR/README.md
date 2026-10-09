# TIR 손상 판독 결과

대청댐의 동일 촬영 위치 20곳을 네 시간대로 관찰한 TIR(열화상)·RGB 원본 영상 80쌍과 수동 손상 판독 결과입니다. 사람이 두 영상을 함께 검토해 작성한 손상 polygon을 댐 3D 모델의 대표점·부재·격자에 연결하고, Overlay 이미지와 최종 CSV로 제공합니다.

현재 결과는 손상 polygon **64건**, 라벨이 있는 촬영 위치 **16곳**입니다. 같은 위치의 네 시간대 라벨을 각각 기록하므로, 16개 위치에 시간대별 라벨 4건씩 대응합니다.

## 폴더 구성

| 경로 | 역할 |
|---|---|
| `rawdata/TIR/` | Radiometric 정보를 포함한 원본 FLIR JPG 80장. 네 시간대 하위 폴더에 각각 20장씩 저장합니다. |
| `rawdata/RGB/` | TIR와 쌍을 이루는 원본 RGB JPG 80장. TIR와 동일한 시간대 구성을 사용합니다. |
| `output/overlay/daybreak/` | 새벽 시간대의 Overlay PNG 20장. |
| `output/overlay/morning/` | 오전 시간대의 Overlay PNG 20장. |
| `output/overlay/afternoon/` | 오후 시간대의 Overlay PNG 20장. |
| `output/overlay/evening/` | 저녁 시간대의 Overlay PNG 20장. |
| [output/tir_damage_results.csv](output/tir_damage_results.csv) | 손상 64건의 식별자, 시간대, polygon, 3D 대표점, 부재·grid, 원본·Overlay 경로를 담은 최종 결과. |

원본 JPG와 기존 Overlay PNG의 이미지 데이터는 유지하고, 시간대별 폴더에 정리했습니다. 결과 CSV 하나에서 라벨별 TIR·RGB·Overlay와 3D 매핑 정보를 함께 확인할 수 있습니다.

## 시간대와 결과 수량

| 시간대 | 의미 | 기존 구분 | TIR·RGB 쌍 | Overlay | 손상 polygon |
|---|---|---|---:|---:|---:|
| `daybreak` | 새벽 | T1 | 20 | 20 | 16 |
| `morning` | 오전 | T2 | 20 | 20 | 16 |
| `afternoon` | 오후 | T3 | 20 | 20 | 16 |
| `evening` | 저녁 | T4 | 20 | 20 | 16 |
| **합계** | | | **80** | **80** | **64** |

시간대는 기존 T1~T4 자료의 구분을 바꾼 명칭입니다. 실제 촬영 날짜나 시각을 새로 추정한 값은 아닙니다.

전체 80장을 사람이 검토했으며, 64장에는 손상 polygon이 각각 1개 있습니다. 나머지 16장은 명확한 손상 polygon을 표시하지 않은 검토 영상으로, 원본과 Overlay를 함께 보존합니다. CSV는 polygon이 있는 64장만 수록합니다. Polygon이 없는 영상이나 표시 영역 바깥을 정상 콘크리트로 확정한 것은 아닙니다.

## Overlay 확인

Overlay는 640×480 PNG입니다. 라벨이 있는 영상에는 수동으로 작성한 손상 polygon의 반투명 표시와 윤곽선이 있으며, 나머지 영상은 polygon 표시가 없는 열화상 시각화입니다.

파일명은 `<Image_ID>_<location_key>_<time_id>_overlay.png` 형식입니다. 예를 들어 `TIRDEMO005_MAIN_006_daybreak_overlay.png`는 `MAIN_006` 위치의 새벽 영상입니다. 같은 `location_key`의 네 시간대 영상을 나란히 확인하면 동일 위치의 표시 영역을 비교할 수 있습니다.

열화상 배경은 영상별 Min-Max 정규화를 적용한 JET 색상입니다. 색상과 0~255 값은 상대적인 시각화 값이므로 실제 온도(°C)로 해석할 수 없으며, 서로 다른 영상의 같은 색상이 동일한 절대온도를 뜻하지 않습니다.

## 결과 CSV 컬럼

| 컬럼 | 설명 |
|---|---|
| `defect_object_id` | 수동 손상 polygon의 고유 ID. `TIRDFT000001`~`TIRDFT000064`입니다. |
| `Image_ID` | 해당 TIR·RGB 영상 쌍의 ID. 전체 영상 ID `TIRDEMO001`~`TIRDEMO080` 중 라벨이 있는 영상에 대응합니다. |
| `location_key` | 촬영 위치 식별자. 같은 위치의 네 시간대가 동일한 값을 공유합니다. |
| `time_id` | `daybreak`, `morning`, `afternoon`, `evening` 중 해당 시간대. |
| `X` | Blender에서 모델 표면을 수동 선택해 얻은 3D 대표점의 world X 좌표. |
| `Y` | 동일 대표점의 world Y 좌표. |
| `Z` | 동일 대표점의 world Z 좌표. Z축이 위쪽인 Blender 좌표계 기준입니다. |
| `member_name` | 연결된 3D 모델의 실제 부재 객체명. 상위 구획인 `Section_Name`과 구분합니다. |
| `Section_Name` | 수동 매핑 자료에 기록된 원본 GLTF의 상위 그룹명. |
| `grid_id` | 연결된 공식 격자 객체명. 현재 `Official_Grid_ID`와 동일하며, 격자가 없는 부재는 빈값입니다. |
| `Official_Grid_ID` | 수동 매핑에서 확인한 원본 GLTF의 공식 격자명. 격자가 없는 부재는 빈값입니다. |
| `Node_Image_XY` | 수동 손상 경계의 영상 픽셀 좌표 배열 `[[x,y],…]`. 영상 왼쪽 위가 원점이며, 2D polygon 꼭짓점을 보존합니다. |
| `Overlay_Relative_Path` | 해당 손상 표시 PNG의 상대경로. `output/overlay/<time_id>/...` 형식입니다. |
| `TIR_Relative_Path` | 해당 원본 TIR JPG의 상대경로. `rawdata/TIR/<time_id>/...` 형식입니다. |
| `RGB_Relative_Path` | 대응 원본 RGB JPG의 상대경로. `rawdata/RGB/<time_id>/...` 형식입니다. |

세 경로 컬럼은 **이 README가 있는 TIR 폴더**를 기준으로 해석합니다. 예: `output/overlay/daybreak/TIRDEMO005_MAIN_006_daybreak_overlay.png`, `rawdata/TIR/daybreak/1-6.jpg`, `rawdata/RGB/daybreak/1-6 (2).jpg`.

CSV는 UTF-8이며 polygon 배열의 쉼표·따옴표는 CSV quoting으로 보호합니다. `Node_Image_XY`는 CSV parser로 읽은 뒤 JSON 배열로 해석할 수 있습니다.

## 3D 매핑 기준

`X`, `Y`, `Z`는 촬영 위치별 수동 대표점입니다. 같은 `location_key`의 네 시간대가 동일 좌표를 공유하며, 영상의 polygon은 `Node_Image_XY`에 별도로 보존합니다. 대표점 매핑은 polygon 꼭짓점별 3D 투영이나 정밀한 손상 면적·치수 산출을 제공하지 않습니다.

| 연결 항목 | 결과 |
|---|---|
| 대표점이 있는 라벨 위치 | 16곳, 시간대별 총 64행 |
| 부재 연결 | 64행 모두 `member_name` 포함 |
| 격자 연결 | 12곳, 총 48행에 `grid_id` 포함 |
| 격자가 없는 부재 연결 | 4곳, 총 16행의 `grid_id`·`Official_Grid_ID`는 빈값 |

부재명은 모델의 `좌안_사력댐_제체`, `우안_비월류부`, `여수로_피어_01`을 그대로 사용합니다. 격자가 없는 부재에 연결된 행도 부재·대표점 정보를 가지며, 가까운 격자 번호를 임의로 배정하지 않았습니다.

XYZ는 원본 Blender 좌표를 유지합니다. 좌표계를 새로 검증하거나 CloudCompare와의 높이 차이를 보정한 결과는 아니므로 다른 시스템에서 사용할 때 좌표 기준을 확인해야 합니다. 같은 모델을 glTF 좌표로 직접 비교하는 경우 축 변환은 Blender `(X,Y,Z)` → glTF `(X,Z,-Y)`입니다.

## 수동 처리와 Code 미제공 사유

처리 과정에는 TIR·RGB 대조 판독, 손상 polygon 라벨링, Blender 모델 표면의 대표점 선택, 부재·공식 격자 확인 등 사람이 직접 수행한 단계가 포함됩니다. 데이터에 이러한 **manual process가 많아 전체 처리 과정의 자동화가 불가능하므로 `Code` 폴더를 제공하지 않습니다.**

이 Demo는 완료된 수동 판독 결과를 시각적으로 확인하고 영상·3D 매핑 관계를 검토하는 자료입니다. 새 데이터에 대해서는 동일한 수동 판독과 매핑 검토가 필요합니다.
