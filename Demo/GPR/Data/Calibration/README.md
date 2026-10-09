# GPR 모델 기준점 매핑

현재 `transform.json`은 사용자가 웹뷰어에서 지정한 두 모델 좌표를 적용한
표출 변환입니다. `calibration_status=selected_anchor_fit`입니다.

| 기준점 | mesh | 세계 XYZ(m) |
|---|---|---|
| 시작 | NOF_R_0157 | 243027.72157758917, 431263.943699504, 66.83210681095052 |
| 끝 | NOF_R_0162 | 243036.03283659572, 431238.8777409665, 66.83210681095052 |

기준 측선은 400MHz LINE_001입니다. 지정점 사이는 26.407940163m이고
DZT 기록 길이는 26.663902671m입니다. 두 점을 맞추기 위해 가로 진행 거리만
`26.407940163 / 26.663902671 = 0.990400410963`의 비율로 보간합니다.
이는 표시 위치에 대한 보간이며 DZT의 거리 단위를 바꾸지 않습니다.
CSV의 `x_m`은 실제 스캔 간격으로 계산한 원자료 거리입니다.
다른 주파수·가로 측선에도 같은 비율을 적용하므로 원래 촬영 길이의 차이는 유지됩니다.

세로 측선 위치 x=1,4,5.5,9,12,15,18,21,25m와 세로 진행 거리는 이 비율로
줄이지 않습니다. 가로 측선 사이 y=0,0.8,1.2m의 간격도 유지합니다.
회전 행렬은 직교하고 `scale=1`이며 가로축은 지정 두 점의 수평 방향입니다.
mesh에서 구한 평균 수평 방향과 약 0.006도 차이입니다.
`R[2][0]=0`이므로 LINE_001~003은 각각 시작부터 끝까지 Z가 같습니다.
세로 측선은 댐 경사면을 따라 높이가 변합니다.

| 파일 | 의미 |
|---|---|
| `selected_anchors.json` | 사용자가 전달한 시작·끝점, 원래 클릭 위치, 모델 SHA-256 |
| `transform.json` | 현재 원점·회전·가로 보간 비율·보존한 세로 간격 |
| `grid_mapping.json` | 후보 126개의 부재·격자, 표면 거리, 모델·변환 SHA-256 |
| `unit_review.json` | 48개 DZT/DZX 거리, 표시 길이, 고정 Z, 세로 간격, 지정점 일치 검증 |
| `transform_legacy.json` | 기존 결과에서 복원한 배율 0.082692956의 변환; 재현 기록용 |
| `legacy_positions.json` | 초기 후보 126개의 XYZ; 웹뷰어 비교용 |

후보를 최근접 격자 삼각형으로 매핑했으며 XYZ를 그 표면으로 이동하지 않습니다.
현재 126행 모두 `NOF_R_0157~0162` 및 상위 부재에 매핑됐고 표면과의 최대 거리는
0.015832m입니다. 재계산 시 허용 거리는 0.10m이며 이를 넘는 후보는 빈값으로 남깁니다.
매핑 캐시는 변환·후보 좌표·현재 모델이 같을 때만 내보내기에 재사용합니다.

현재 모델 기준점을 다시 적용할 때는 `GPR`에서 다음처럼 실행합니다.

```bash
python3 Code/fit_start_end.py --anchors Data/Calibration/selected_anchors.json \
  --fit-horizontal-endpoints --output Data/Calibration/transform.json
python3 Code/process.py --export-only --work-dir /path/to/existing_detection_results
python3 MappingReview/review_positions.py --apply-grid-mapping
python3 Code/audit_coordinate_units.py
python3 MappingReview/build_viewer.py
```

가로 보간 옵션 없이 실행하면 기록 길이와 0.10m 이상 다른 지정점은 거부합니다.
두 방법 모두 가로 Z 기울기를 허용하지 않으며 mesh 방향과 0.2도 이상 다른 점도 거부합니다.
`align_horizontal.py`는 지정점 적용 전의 방향 검토용입니다. 현재 변환에 실행하면
지정 끝점을 임의로 바꾸지 않도록 오류를 냅니다.

DZT의 실제 간격은 `dx=1/scans_per_metre`이고 길이는 `(N-1)*dx`, 후보 거리는
`x_px*dx`입니다. DZX의 `unitsPerScan`과 표시 단위(m 또는 cm)를 별도로 대조했습니다.
2.6GHz LINE_001의 DZX 끝점 distance는 표시 단위와 맞지 않아 거리 계산에 쓰지 않습니다.

이전 변환은 원본 line_endpoints.csv와 lines_3d.csv의 96개 시작·끝점에서 복원했습니다.
최대 복원 오차 3.5e-10m는 기존 결과의 재현 오차이며 현장 위치 정확도가 아닙니다.
현재 변환은 사용자가 모델에서 지정한 표출 기준점에 대한 매핑입니다.
현장 CRS·측량 정확도나 내부 손상 깊이를 검증한 좌표로 해석하지 않습니다.
원본 현장 4점 보정에는 실제 평면 상단 y를 명시하는 `--plane-y1`을 사용합니다.
