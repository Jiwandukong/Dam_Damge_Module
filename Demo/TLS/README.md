# TLS 박락 검출

대청댐 Part1·Part2 TLS 점군에서 국소 RANSAC 기준평면보다 안쪽으로 파인 영역을 박락 후보로 검출합니다. 원본 점군으로 면적·평균 깊이를 계산하고, 등록 점군으로 손상 중심 위치와 Grid·부재 정보를 연결합니다.

## 폴더 구성

`Data`에는 원본·등록 점군과 기준 모델, `Code`에는 실행 코드, `Output`에는 손상별 결과를 저장합니다.

```text
TLS/
├── Data/
│   ├── dacheung_part1.e57
│   ├── dachung_part2.e57
│   ├── Registered/
│   │   ├── regist_part1.e57
│   │   └── regist_part2.e57
│   └── Model/
│       ├── daecheongdam_regions_grid5m.gltf
│       ├── daecheongdam.bin
│       └── daecheongdam_regions_grid5m.bin
├── Code/
│   ├── process.py
│   ├── local_plane_spalling.py
│   ├── export_results.py
│   ├── visualize_results.py
│   ├── registered_source.py
│   ├── model_mapping.py
│   ├── config.json
│   ├── runtime.py
│   ├── requirements.txt
│   └── setup.sh
├── Output/
│   ├── Result/
│   │   ├── SPL/                # 후보별 LAS 42개
│   │   └── SPL_result.csv       # 후보 42행, 중심 위치·면적·깊이·대표 Grid
│   └── Visualize/
│       └── SPL/                # 후보별 PNG 42개
└── README.md
```

| 입력 | 점 수 | 검출 결과 | Grid 연결 | 부재 연결 |
|---|---:|---:|---:|---:|
| Part1 | 3,201,667 | 23건 | 23건 | 23건 |
| Part2 | 1,822,885 | 19건 | 0건 | 19건 |
| 합계 | 5,024,552 | 42건 | 23건 | 42건 |

손상 ID는 `TLS1-xxxx`, `TLS2-xxxx`이며, 각 손상은 CSV 1행·LAS 1개·PNG 1개에 대응합니다.

## 알고리즘과 설정

`Code/local_plane_spalling.py`는 3cm 셀별 관측점의 중앙값으로 표면을 구성합니다. 3m 구간에서 중심 1.5m를 제외한 주변점을 이용해 RANSAC 기준평면을 구하고, 0.75m 간격으로 구간을 겹쳐 평가합니다.

깊이·면적·점 수·주변 정상면 지지 조건을 만족하는 셀을 묶고, 길고 좁은 이음새와 경계 영역을 제외합니다. CSV의 면적·평균 깊이는 셀 기반 추정치이며, LAS의 깊이는 개별 관측점 값입니다.

| `Code/config.json` 설정 | 현재 값 | 역할 |
|---|---:|---|
| `grid_resolution_m` | 0.03m | 분석 셀 크기 |
| `reference_patch_size_m` | 3.0m | 주변 기준평면 추출 구간 |
| `evaluation_core_size_m` | 1.5m | 평가하는 중심 영역 |
| `patch_stride_m` | 0.75m | 구간 이동 간격 |
| `reference_ransac_threshold_m` | 0.01m | 기준평면 인라이어 허용 거리 |
| `minimum_depth_m` | 0.025m | 최소 안쪽 깊이 |
| `maximum_depth_m` | 0.20m | 후보 점의 최대 깊이 |
| `minimum_candidate_area_m2` | 0.01m² | 최소 후보 면적 |
| `joint_like_aspect_ratio` | 6 | 이 값 이상인 길쭉한 영역을 재검토로 분류 |

## 산출물

**LAS**는 손상 관측점의 등록 XYZ/RGB와 `depth_m`, `candidate_id`, `source_vertex_id`를 저장합니다. `source_vertex_id`는 해당 E57의 0부터 시작하는 점 인덱스입니다.

**PNG**는 ROV_LiDAR 형식의 **3840×2160 단일 3D overlay**입니다. 어두운 배경에 주변점을 회색으로, 손상점을 깊이 색상으로 표시합니다. 범례·컬러바·축·제목을 표시하지 않으며 형상과 깊이를 과장하지 않습니다.

**CSV**는 `SPL_result.csv` 하나이며 **42행·20개 컬럼**, UTF-8 BOM 형식입니다. 좌표와 깊이 단위는 m, 면적은 m², 점유율은 %입니다. 파일 경로는 `Output/Result` 기준입니다.

| 컬럼 | 의미 |
|---|---|
| `dataset` | 원본 E57 이름에서 확장자를 제외한 값 |
| `damage_id` | 손상 ID (`TLS1-xxxx`, `TLS2-xxxx`) |
| `source_damage_id` | 해당 Part 내부의 영역 번호 |
| `damage_type` | 손상 코드 `SPL` |
| `damage_name_ko` | 손상명 `박락` |
| `world_center_x_m` | 등록 관측점의 평균 X 좌표(m) |
| `world_center_y_m` | 등록 관측점의 평균 Y 좌표(m) |
| `world_center_z_m` | 등록 관측점의 평균 Z 좌표(m) |
| `grid_id` | 손상 중심에서 가장 가까운 모델 Grid. 1m 밖이면 빈값 |
| `member_name` | 가장 가까운 모델 부재. 1m 밖이면 빈값 |
| `area_m2` | 후보 셀의 추정 표면 면적(m²) |
| `mean_depth_m` | 후보 셀 깊이의 평균(m) |
| `point_count` | 후보 LAS의 관측점 수 |
| `grid_area_m2` | 대표 Grid의 점유율 분모, 현재 25m² |
| `grid_damage_area_m2` | 대표 Grid에 배정된 부분 손상 면적(m²) |
| `grid_damage_ratio_pct` | `grid_damage_area_m2 ÷ 25 × 100`으로 계산한 점유율(%) |
| `mapped_grid_count` | 손상이 연결된 Grid 수 |
| `registered_data_path` | 등록 E57의 상대경로 |
| `pointcloud_path` | 손상 LAS의 상대경로 |
| `visualization_path` | overlay PNG의 상대경로 |

## 실행

Python 3.13 환경에서 `TLS` 폴더로 이동한 뒤 실행합니다.

```bash
bash Code/setup.sh
python3 Code/process.py
```

설정은 `Code/config.json`에서 변경합니다. 기존 검출 결과로 CSV·LAS·PNG만 다시 생성하려면 다음 명령을 사용합니다.

```bash
python3 Code/process.py --export-only
```

최초 실행이나 검출 설정 변경 후에는 일반 실행을 사용합니다. 작업 캐시는 `~/.cache/dam_damage_module/tls_spalling/`에 저장되며, `--work-dir`로 위치를 변경할 수 있습니다.
