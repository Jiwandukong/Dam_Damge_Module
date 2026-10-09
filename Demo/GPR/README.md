# GPR 이상 후보 표출 초안

대청댐 GSSI GPR 48개 측선에서 규칙 기반으로 검출한 이상 후보 126개를
표출하기 위한 자료입니다. 후보별 ID·3D 대표점·영상 위치·Overlay 경로를
CSV로 전달합니다. `ANM`은 GPR 이상 후보를 뜻하며 확정된 결함 종류가 아닙니다.

## 폴더 구성

```text
GPR/
├── Data/
│   ├── processed data/{400MHz,900MHz,1.6GHz,2.6GHz}/
│   └── Calibration/transform.json
├── Code/
├── MappingReview/   # Dam_model에서 GPR 위치 확인용 웹뷰어
├── Output/
│   ├── Overlay/ANM/{400,900,1600,2600}/LINE_001_anomaly.png ...
│   └── Result/ANM_result.csv
└── README.md
```

| 경로 | 내용 |
|---|---|
| `Data/processed data/` | 주파수별 DZT/DZX 12쌍, 총 48쌍 |
| `Data/Calibration/` | 지정 시작·끝점, 현재 변환·격자 매핑, 이전 변환·위치, 거리 검토 결과 |
| `Code/` | 전처리·규칙 기반 검출·3D 측선 변환·표출용 내보내기·검증 코드 |
| `Output/Overlay/ANM/` | 측선별 Overlay PNG 48장 |
| `Output/Result/ANM_result.csv` | 후보별 표출 정보 126행, UTF-8 BOM CSV |
| `MappingReview/` | 보정 XYZ·측선·이전 위치·최근접 격자·Overlay를 확인하는 웹뷰어 |

`Output`에는 Overlay와 결과 CSV만 둡니다. 전처리 영상, 측선 좌표표,
픽셀 대응표, 중간 후보표는 별도 작업 폴더에 저장합니다.

## Overlay

빨간 십자가 후보 중심을 표시합니다. 후보가 없는 측선도 영상은 보존합니다.
기존 B-scan의 픽셀 크기와 배경을 유지하므로 CSV의 영상 좌표가 그대로 대응합니다.
하나의 측선에 후보가 여러 개 있으면 여러 CSV 행이 같은 Overlay를 공유합니다.

기본 십자 반길이는 기존 5px에서 10px로, 선 두께는 2px에서 3px로 키웠습니다.
가로·세로 비율이 4 이상인 영상은 `영상 폭 / 1000`에 비례해 마커를 확대합니다.
현재 네 주파수의 `LINE_001~003`, 총 12장이 대상이며 나머지 36장은 유지합니다.
400·900MHz의 가로로 긴 영상도 같은 규칙으로 확대합니다.
예를 들어 폭 2,948px 영상은 반길이 29px, 두께 9px입니다.
예를 들어 폭 7,553px 영상은 반길이 76px, 두께 23px입니다.
폭 10,614px 영상은 반길이 106px, 두께 32px입니다.
긴 영상을 2,048px 폭으로 축소해 보면 십자 길이는 약 41px입니다.

## 결과 CSV

| 컬럼 | 의미 |
|---|---|
| `image` | 원본 DZT 파일명 |
| `damage_id` | 후보 ID, `G000001`~`G000126` |
| `damage_type` | `ANM` |
| `damage_name_ko` | `GPR 이상 후보` |
| `pixel_nodes_json` | B-scan의 후보 중심 한 점 `[[x,y]]`; 단위 px, 왼쪽 위가 원점 |
| `world_center_x_m`, `world_center_y_m`, `world_center_z_m` | 지정한 모델 시작·끝점에 맞춘 측선 표면 투영 XYZ, 단위 m |
| `member_name`, `grid_id` | 지정 모델 기준점과 최근접 격자 표면으로 매핑한 부재·격자; 현재 126행 모두 지정 |
| `freq_mhz` | 안테나 주파수: 400, 900, 1600, 2600MHz |
| `line_no` | 측선 번호, 1~12 |
| `x_m` | 후보 픽셀의 스캔 좌표 ÷ DZT scans-per-metre; 측선 시작점으로부터의 거리(m) |
| `t_ns` | 왕복시간(ns) |
| `source_data_path` | 원본 DZT의 상대경로 |
| `overlay_path` | 해당 측선 Overlay의 상대경로 |
| `mapping_status` | `line_surface_projection_anchor_fit`: 지정한 모델 시작·끝점을 적용한 표면 투영 |

경로는 모두 **CSV가 있는 `Output/Result` 폴더 기준**이며 `/`를 사용합니다.
예: `../Overlay/ANM/400/LINE_001_anomaly.png`.
후보 ID는 주파수·측선 순으로 부여합니다.
`pixel_nodes_json`은 후보 중심 한 점이며 손상 경계 polygon이 아닙니다.

| 주파수 | 후보 수 | Overlay 수 |
|---:|---:|---:|
| 400MHz | 41 | 12 |
| 900MHz | 42 | 12 |
| 1600MHz | 26 | 12 |
| 2600MHz | 17 | 12 |
| 합계 | 126 | 48 |

XYZ는 원자료 거리 `x_m`과 지정 시작·끝점으로 보간한 표면 투영 좌표입니다.
`t_ns`로 내부 깊이를 계산한 좌표가 아닙니다. 원자료의 CRS는 확인되지 않았습니다.
기존 전처리, 임계값, 후보 수·ID·영상 중심·왕복시간과 Overlay는 유지했습니다.
`x_m`은 DZT의 실제 측정 거리이며 표시 위치를 보간한 뒤에도 유지합니다.

이전 변환의 배율 `0.082692956`은 26.67m 측선을 약 2.21m로 축소했습니다.
현재 회전·세로 거리 배율은 `1.0`이며 DZT의 실제 스캔 간격을 사용합니다.
N개 스캔의 측선 길이는 `(N-1)/scans_per_metre`입니다.
400MHz LINE_001의 기록 길이는 약 26.664m이고 지정 두 점 사이는 약 26.408m입니다.
가로 측선의 진행 거리만 `0.990400411`의 비율로 보간해 기준 측선의 양 끝을 정확히 맞춥니다.
이 비율을 다른 가로 측선에도 적용하며 주파수·측선별 원래 촬영 길이 차이를 유지합니다.
세로 측선 위치 x=1,4,5.5,9,12,15,18,21,25m는 이 비율로 줄이지 않습니다.
세로 길이와 가로 측선 사이 간격도 유지합니다. LINE_004는 모델에서도 약 1.773m입니다.
방향은 mesh 수평 방향과 약 0.006도 차이인 지정 두 점의 수평 방향입니다.
가로 측선 `LINE_001~003`은 mesh의 수평 방향을 따르고 각각 시작·끝의 Z가 같습니다.
세로 측선의 높이 변화와 측선 간 간격은 유지합니다.
현재 원점은 지정한 `NOF_R_0157` 시작점입니다. 400MHz LINE_001의 끝은 지정한
`NOF_R_0162` 점과 일치하고 두 점의 Z는 66.832106811m입니다.

## 모델에서 위치 확인

`GPR` 옆에 `Dam_model/Daecheongdam/`이 있는 현재 폴더 구성에서 실행합니다.

```bash
python3 MappingReview/serve.py
```

브라우저에서 <http://127.0.0.1:8771/>에 접속합니다. **댐 전체**로 위치를
확인하고, **GPR 범위** 또는 **선택점 확대**로 후보를 살펴봅니다.
주파수·측선 필터와 후보 목록을 이용하면 겹치는 후보를 따로 확인할 수 있습니다.
후보를 선택하면 재산정한 XYZ, 최근접 격자와 거리, 해당 측선 Overlay가 나타납니다.
**측선 표시**로 실제 배치를, **이전 위치 비교**로 축소되었던 위치를 확인합니다.
**시작·끝 위치 확인**의 자동 제안선과 지정점은 같은 Z를 유지합니다.
끝점을 클릭하면 시작점과 같은 높이의 mesh 표면에 맞추며 높이 차이를 표시합니다.

후보 126개를 `NOF_R_0157~0162`와 그 상위 부재에 매핑했습니다.
mesh 표면과 후보의 최대 거리는 약 1.6cm이며 XYZ를 최근접점으로 옮기지는 않습니다.
이는 지정한 모델 좌표 기준의 표출 매핑이며 현장 좌표계의 측량 검증을 뜻하지 않습니다.
뷰어 자료는 `MappingReview`에 두므로 `Output`의 Overlay·CSV 구성은 유지됩니다.
자세한 자료 구성과 재생성 방법은 [MappingReview/README.md](MappingReview/README.md)에 있습니다.

## 실행

Python 3.13을 기준으로 `GPR` 폴더에서 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r Code/requirements.txt
python3 Code/process.py
python3 Code/verify_outputs.py
python3 Code/audit_coordinate_units.py
```

중간 결과의 기본 저장 위치는 `~/.cache/dam_damage_module/gpr/`입니다.
`GPR_WORK_DIR` 환경변수 또는 `--work-dir`로 변경할 수 있습니다.

```bash
python3 Code/process.py --work-dir /작업폴더/GPR_work
```

작업 폴더에 `result_image/`와 `anomaly_rb/anomalies_rulebased.csv`가 있으면
다시 검출하지 않고 Overlay를 생성하고 현재 변환·스캔 간격으로 CSV를 재산정할 수 있습니다.

```bash
python3 Code/process.py --export-only --work-dir /작업폴더/GPR_work
python3 Code/process.py --export-only --work-dir /작업폴더/GPR_work \
  --marker-half-size 14 --marker-width 4
python3 Code/verify_outputs.py --reference /기존결과폴더
```

마지막 검증 명령은 후보 수·픽셀 좌표·왕복시간, DZT 기준 거리와 보정 XYZ,
이미지 크기, 마커 주변 밖의 B-scan 배경 보존, 마커 확대 여부를 확인합니다.
마커 크기를 직접 변경한 경우 검증에도 같은 `--marker-half-size`,
`--marker-width` 값을 지정합니다.

현재 변환과 후보가 같으면 `Data/Calibration/grid_mapping.json`의 부재·격자 매핑을
내보내기에 재사용합니다. 시작·끝점이나 모델이 바뀌면 `review_positions.py --apply-grid-mapping`
으로 다시 계산합니다. 이 거리 검토에는 NumPy·Open3D가 필요합니다.

기존 3D 변환의 출처는 `Data/Calibration/README.md`, 원본 패키지 정리 이력은
`SOURCE_ARCHIVES.md`, 코드 이용 조건은 `LICENSE_NOTICE.md`와
`THIRD_PARTY_NOTICES.md`에 기록되어 있습니다.
