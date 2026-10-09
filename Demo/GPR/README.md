# GPR 이상 후보 표출 결과

대청댐 GSSI GPR 48개 측선에서 검출한 **이상 후보(ANM) 126개**의 표출 자료입니다.
측선별 Overlay 이미지와 후보의 3D 대표점·영상 위치·격자·부재 정보를 담은 CSV를 제공합니다.
`ANM`은 GPR 이상 후보이며 확정된 결함 종류를 뜻하지 않습니다.

## 폴더 구성

```text
GPR/
├── Data/
│   ├── processed data/{400MHz,900MHz,1.6GHz,2.6GHz}/
│   └── Calibration/
├── Code/
├── Output/
│   ├── Overlay/ANM/{400,900,1600,2600}/
│   └── Result/ANM_result.csv
└── README.md
```

| 경로 | 역할 |
|---|---|
| `Data/processed data/` | 네 주파수별 DZT/DZX 12쌍, 총 48쌍의 입력 자료. |
| `Data/Calibration/` | 지정 시작·끝점, 좌표 변환, 후보별 격자·부재 매핑 정보. |
| `Code/` | 전처리, 이상 후보 검출, 좌표·격자 매핑, CSV·Overlay 생성 코드. |
| `Output/Overlay/ANM/` | 측선별 Overlay PNG 48장. |
| `Output/Result/ANM_result.csv` | 이상 후보 126행의 표출 정보. UTF-8 BOM 인코딩. |

| 주파수 | 후보 수 | Overlay 수 |
|---:|---:|---:|
| 400MHz | 41 | 12 |
| 900MHz | 42 | 12 |
| 1600MHz | 26 | 12 |
| 2600MHz | 17 | 12 |
| 합계 | 126 | 48 |

## 결과 CSV 컬럼

| 컬럼 | 설명 |
|---|---|
| `image` | 후보가 검출된 원본 DZT 파일명. |
| `damage_id` | 후보 ID, `G000001`~`G000126`. |
| `damage_type` | 이상 후보 코드 `ANM`. |
| `damage_name_ko` | `GPR 이상 후보`. |
| `pixel_nodes_json` | B-scan의 후보 중심 좌표 `[[x,y]]`. 왼쪽 위가 원점이며 단위는 px. |
| `world_center_x_m`, `world_center_y_m`, `world_center_z_m` | 모델에서 표출할 3D 대표점 XYZ(m). |
| `member_name` | 후보와 가장 가까운 격자의 상위 부재 이름. |
| `grid_id` | 후보 XYZ에서 표면까지의 거리가 가장 가까운 격자 이름. |
| `freq_mhz` | 안테나 주파수: 400, 900, 1600, 2600MHz. |
| `line_no` | 측선 번호, 1~12. |
| `x_m` | 원자료의 측선 시작점으로부터 측정한 진행 거리(m). |
| `t_ns` | 왕복시간(ns). |
| `source_data_path` | 원본 DZT 경로. CSV 폴더 기준 상대경로. |
| `overlay_path` | 해당 측선 Overlay 경로. CSV 폴더 기준 상대경로. |
| `mapping_status` | `line_surface_projection_anchor_fit`: 지정 모델 기준점을 적용한 표면 투영. |

XYZ는 사용자가 모델에서 지정한 시작·끝점으로 계산한 표면 대표 좌표입니다.
가로 측선은 각각 일정한 Z를 유지하며 세로 측선 간 간격과 길이는 유지합니다.
왕복시간을 내부 깊이로 변환하지 않습니다.

후보 126개는 `NOF_R_0157~0162`와 상위 부재
`NOF_우안(청주측)_접속부(콘크리트_중력식)`에 연결됩니다.
CSV의 경로는 `Output/Result` 기준이며 `/`를 사용합니다.
예: `../Overlay/ANM/400/LINE_001_anomaly.png`.

## 실행

Python 3.13 환경에서 `GPR` 폴더로 이동한 뒤 실행합니다.

```bash
python3 -m pip install -r Code/requirements.txt
python3 Code/process.py
```

중간 결과는 `~/.cache/dam_damage_module/gpr/`에 저장합니다.
`GPR_WORK_DIR` 또는 `--work-dir`로 작업 폴더를 지정할 수 있습니다.

```bash
python3 Code/process.py --work-dir /작업폴더/GPR_work
```

기존 검출 결과에서 CSV와 Overlay를 다시 생성하는 경우:

```bash
python3 Code/process.py --export-only --work-dir /작업폴더/GPR_work
```

현재 좌표의 격자·부재 매핑은 `Data/Calibration/grid_mapping.json`에서 재사용합니다.
시작·끝점이나 모델을 바꾼 경우 NumPy·Open3D 환경에서 최근접 격자를 다시 계산합니다.

```bash
python3 Code/map_grids.py --apply-grid-mapping \
  --dam-model /모델/Daecheongdam/daecheongdam_regions_grid5m.gltf
```
