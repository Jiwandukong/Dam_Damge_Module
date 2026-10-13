# TIR 손상 판독 결과

대청댐 TIR·RGB 대응 영상 280쌍과 사람 수동 판독 결과입니다. 기존 Demo80의 20곳·80쌍에 신규 50개 후보 위치·200쌍을 추가했습니다. 총 촬영 위치 후보는 **70곳**, 수동 defect polygon은 **280개**, 손상 표시 영상은 **264장**입니다. RGB/TIR를 검토해 작성한 2D polygon, Blender 댐체 표면의 대표 XYZ, 부재·격자 연결을 Overlay와 결과 CSV로 제공합니다. AI 자동 추론 결과가 아닙니다.

## 폴더 구성

| 경로 | 역할 |
|---|---|
| `rawdata/TIR/` | FLIR Radiometric 원본 JPG 280장. 각 명목 구간 폴더에 70장씩 저장합니다. |
| `rawdata/RGB/` | 대응 RGB 원본 JPG 280장. TIR와 같은 구간으로 구분합니다. |
| `output/overlay/daybreak/` | daybreak 구간 Overlay PNG 70장. |
| `output/overlay/morning/` | morning 구간 Overlay PNG 70장. |
| `output/overlay/afternoon/` | afternoon 구간 Overlay PNG 70장. |
| `output/overlay/evening/` | evening 구간 Overlay PNG 70장. |
| [output/tir_damage_results.csv](output/tir_damage_results.csv) | 손상 polygon 280행의 ID, 구간, 2D 경계, 대표 XYZ, 부재·격자 및 영상 경로. |

원본과 Overlay는 기존 검증된 파일의 바이트를 유지했습니다. Radiometric JPG를 JET PNG로 대체하지 않았습니다. 현재 게시 형식인 소문자 `rawdata`, `output/overlay`와 상대경로 체계를 유지했습니다. ZIP을 저장소 루트에서 압축 해제하면 `Demo/TIR/` 아래에 배치됩니다.

## 시간대와 결과 수량

| 시간대 폴더 | 명목 명칭 | 기존 구분 | TIR·RGB 쌍 | Overlay | 손상 polygon |
|---|---|---|---:|---:|---:|
| `daybreak` | 새벽 | T1 | 70 | 70 | 70 |
| `morning` | 오전 | T2 | 70 | 70 | 70 |
| `afternoon` | 오후 | T3 | 70 | 70 | 70 |
| `evening` | 저녁 | T4 | 70 | 70 | 70 |
| **합계** | | | **280** | **280** | **280** |

이 명칭은 기존 GitHub의 T1~T4 폴더 대응 규칙입니다. 실제 촬영 시각·날짜나 해당 시간대의 일치 여부를 새롭게 확정한 것은 아닙니다.

전체 영상 중 **264장은 POSITIVE**, **16장은 REVIEWED_ZERO**입니다. 기존 검수 무손상 영상 16장의 원본과 Overlay를 유지했고 신규 200장에는 모두 하나 이상의 polygon이 있습니다. CSV는 영상당 한 행이 아니라 **polygon당 한 행**입니다. 동일 영상에 여러 polygon이 있으면 Image_ID와 영상 경로가 반복됩니다. Polygon 280개는 촬영 구간별 라벨의 합계이며 서로 다른 물리 손상 280개를 뜻하지 않습니다. REVIEWED_ZERO와 polygon 외부 영역을 정상 콘크리트로 확정하지 않습니다.

## Overlay 확인

Overlay는 640×480 PNG입니다. 손상 표시 영상에는 기존 사람이 작성한 polygon의 반투명 표시와 윤곽선이 있고, REVIEWED_ZERO는 polygon을 추가하지 않은 JET 시각화입니다. 기존 80장과 신규 200장의 검증된 Overlay를 다시 렌더링하지 않고 복사했습니다.

파일명은 `<Image_ID>_<location_key>_<time_id>_overlay.png` 규칙을 따릅니다. 예: `TIRDEMO005_MAIN_006_daybreak_overlay.png`. 기존 게시 파일명·경로는 유지하고 신규 파일에도 확정 Image_ID와 구간을 사용했습니다.

JET는 각 영상의 Min-Max에 따른 상대 온도 시각화입니다. 색상이나 0~255 값을 실제 섭씨 온도로 읽을 수 없고, 다른 영상의 같은 색상이 같은 절대온도를 뜻하지 않습니다. 기존 온도 분포 REVIEW 16개 영상도 유지했습니다. REVIEW를 손상 확정이나 추출 오류로 자동 해석하지 않습니다. RGB와 TIR는 대응 영상이지만 픽셀 단위 정합이나 완전히 동일한 시야가 보장되지는 않습니다.

## 결과 CSV 컬럼

현재 GitHub에 게시된 **16개 컬럼의 이름과 순서**를 그대로 사용했습니다. 기존 64개 행과 헤더는 원격 CSV 원문을 보존했고 신규 216개 행만 추가했습니다.

| 컬럼 | 설명 |
|---|---|
| `defect_object_id` | 수동 polygon 객체 ID. 기존 TIRDFT000001~TIRDFT000064를 유지하고 신규 TIRDFT000065~TIRDFT000280을 연결합니다. |
| `damage_name_ko` | 기존 게시 규칙의 `손상후보`. 새로운 세부 손상 종류를 추정하지 않습니다. |
| `Image_ID` | 영상 쌍의 고유 ID. 기존 TIRDEMO001~080을 유지하며 전체 TIRDEMO001~280을 사용합니다. |
| `location_key` | 촬영 위치 후보 ID. 정밀한 물리 위치 정합 완료를 뜻하지 않습니다. |
| `time_id` | daybreak, morning, afternoon, evening 중 해당 명목 촬영 구간. |
| `X` | 수동 Blender 대표점의 기존 World X. |
| `Y` | 같은 대표점의 기존 World Y. |
| `Z` | 같은 대표점의 기존 World Z. Blender는 Z가 위쪽입니다. |
| `member_name` | 저장된 클릭 Mesh 또는 저장 구획에 연결되는 원본 모델의 실제 구조 부재명. |
| `Section_Name` | 저장된 매핑 기록의 GLTF 상위 구획 그룹명. 부재명과 구분합니다. |
| `grid_id` | 원본 GLTF 격자 객체명. Official_Grid_ID와 동일하며 NON_GRID는 공란입니다. |
| `Official_Grid_ID` | 기존 격자 연결 필드. 원본 모델·저장 기록의 실제 격자명을 사용하고 임의 번호를 부여하지 않습니다. |
| `Node_Image_XY` | 원래 LabelMe polygon 정점 배열 [[x,y],…]. 값과 정점 순서를 보존했습니다. |
| `Overlay_Relative_Path` | `output/overlay/<time_id>/...`의 해당 Overlay 경로. |
| `TIR_Relative_Path` | `rawdata/TIR/<time_id>/...`의 원본 Radiometric JPG 경로. |
| `RGB_Relative_Path` | `rawdata/RGB/<time_id>/...`의 대응 RGB JPG 경로. |

경로 3개는 이 README가 있는 **Demo/TIR 폴더 기준**입니다. CSV가 있는 output 폴더 기준이 아닙니다. CSV는 UTF-8이며 배열의 쉼표·따옴표는 CSV quoting으로 보호됩니다. Node_Image_XY는 CSV parser로 읽은 뒤 JSON 배열로 해석합니다.

## 3D 매핑 기준

대표 XYZ는 기존 20곳 + 신규 DIRECT 44곳 + 신규 APPROXIMATE 6곳, 총 **70곳**입니다. XYZ 값을 재계산·보정하지 않았습니다. 대표점은 댐체 표면의 시연용 위치이며 카메라 위치나 개별 polygon 꼭짓점의 정밀 3D 좌표가 아닙니다. DIRECT도 독립 측량 검증을 뜻하지 않습니다.

| 연결 항목 | 결과 |
|---|---|
| 대표 XYZ가 있는 전체 후보 위치 | 70곳 |
| CSV에 라벨이 있는 위치 | 66곳, polygon 280행 |
| 부재 연결 | 280행 모두 member_name 포함 |
| 격자 연결 | 라벨 위치 45곳, 192행 |
| 격자가 없는 부재 연결 | 라벨 위치 21곳, 88행; 격자 필드 공란 |

부재명은 원본 모델의 `좌안_사력댐_제체`, `우안_비월류부`, `여수로_피어_01`을 사용합니다. 신규 Grid명은 저장된 GLTF/Blender 연결을 계승한 값이며 검증 이력을 임의로 VERIFIED로 변경하지 않았습니다. 격자가 없는 부재에 최근접 격자 번호를 붙이지 않았습니다.

**근사 대표점 6곳**: MAIN_029, MAIN_042, SECOND_014, SECOND_016, SECOND_017, SECOND_019. 이 위치의 polygon도 동일 CSV에 정상적으로 포함했습니다(24행). 사람의 직접 표면 클릭으로 저장한 시연용 근사 XYZ이며 APPROXIMATE / PENDING_LOCATION_REVIEW 이력은 원본 감사 자료에 유지합니다. 네 구간의 실제 동일 물리 위치가 정밀하게 확정됐다는 의미는 아닙니다.

Blender World 좌표를 그대로 유지했습니다. CloudCompare 높이 보정이나 CRS 재투영은 수행하지 않았습니다. 기존 설명의 Blender `(X,Y,Z)`와 glTF `(X,Z,-Y)` 축 관계는 참고 규칙이며, 이 CSV의 좌표를 변환한 것은 아닙니다. 다른 시스템에 연결할 때 모델 좌표 기준을 확인해야 합니다.

## 수동 처리와 Code 미제공 사유

사람의 RGB/TIR 대조, polygon 작성, Blender 표면 대표점 선택 및 부재·격자 확인을 포함하는 결과 자료입니다. 새 영상의 판독과 매핑에는 사람 검토가 필요합니다. 이 Demo는 완료 결과의 열람·연결 확인을 위한 패키지로 Code 폴더를 포함하지 않습니다. 개발 코드·Blender 장면·학습 모델·Excel·감사 중간 자료도 제외했습니다. 기존 동부 전달 Excel과 상세 온도·라벨·감사 자료는 별도 보존했습니다.
