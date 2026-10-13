# TLS 박락 후보 검출 산출물

대청댐 Part1·Part2의 원본 TLS 점군에서 주변 기준평면보다 안쪽으로 파인 영역을 찾습니다. 겹치는 구간의 국소 평면을 RANSAC으로 추출하는 기하 방식이며, 학습 모델과 손상 라벨은 사용하지 않습니다. 결과는 **박락 후보 42건**으로, 현장 확인을 통해 박락 여부를 판정해야 합니다.

## 폴더 구성

GitHub `Demo/Multibeam`의 점군 센서 구성에 맞춰 원본, 실행 코드, 후보별 산출물을 구분합니다.

```text
TLS/
├── Data/
│   ├── dacheung_part1.e57
│   └── dachung_part2.e57
├── Code/
│   ├── process.py
│   ├── local_plane_spalling.py
│   ├── export_results.py
│   ├── config.json
│   ├── runtime.py
│   ├── requirements.txt
│   └── setup.sh
├── Output/
│   ├── Result/
│   │   ├── SPL/                # 후보별 LAS 42개
│   │   └── SPL_result.csv       # 후보 42행
│   └── Visualize/
│       └── SPL/                # 후보별 PNG 42개
└── README.md
```

| 입력 | 원본 점 수 | 박락 후보 | 작업 캐시의 재검토 영역 |
|---|---:|---:|---:|
| `dacheung_part1.e57` | 3,201,667 | 23 | 43 |
| `dachung_part2.e57` | 1,822,885 | 19 | 5 |
| 합계 | 5,024,552 | 42 | 48 |

각 후보는 기존 검출 ID(`TLS1-xxxx`, `TLS2-xxxx`)를 유지하며, LAS 1개·PNG 1개·CSV 1행에 대응합니다. 길고 좁은 이음새, 경계, 기준면 지지가 부족한 영역 등 `geometry_review` 상태인 48건은 최종 산출물에서 제외하고 작업 캐시에 유지합니다.

## 알고리즘과 설정

`Code/local_plane_spalling.py`는 원본 좌표를 표면 좌표계 U/V/W로 표현하고, 3cm 셀별 관측점의 중앙값으로 표면을 구성합니다. 3m 구간에서 중심 1.5m 영역을 제외한 주변점을 이용해 RANSAC 기준평면을 구합니다. 0.75m 간격으로 구간을 겹쳐 처리하며, 적어도 두 평면이 지지하고 기준면 간 차이가 12.5mm 이하인 영역을 평가합니다.

안쪽 깊이가 `max(최소 깊이, 4 × 주변 잡음 추정치, 2 × 기준면 불일치)` 이상인 셀을 묶습니다. 면적·점 수·주변 정상면 지지·길쭉함 조건으로 박락 후보와 재검토 영역을 구분하고, 원본 관측점도 깊이 조건을 만족할 때만 후보 LAS에 포함합니다. 면적과 CSV 깊이는 셀 기반 추정치이며 LAS 깊이는 개별 관측점 값입니다.

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

바깥쪽 방향은 현재 `outward_hint_xyz=[0,1,0]`으로 가정하며 현장 방향 검증은 완료되지 않았습니다. 깊이는 이 방향 가정과 주변 평면에 대한 상대값입니다.

## 산출물

LAS에는 실제 후보 관측점의 XYZ/RGB와 `depth_m`, `candidate_id`, `source_vertex_id`를 저장합니다. `source_vertex_id`는 해당 E57 스캔의 0부터 시작하는 원본 점 인덱스입니다. XYZ는 원본 E57 좌표계이며 LAS 저장 간격은 0.000001m입니다. 색상은 E57의 8비트 RGB를 LAS의 16비트 RGB로 변환합니다.

PNG는 3840×2160이며, 원본 RGB와 빨간 후보 점을 표시한 정면도, 깊이 색상과 초록색 기준면을 표시한 3D 시점을 함께 제공합니다. 시각화는 좌표의 회전·평행이동만 적용하고 깊이를 과장하지 않습니다.

CSV는 UTF-8 BOM 인코딩이며 경로는 `Output/Result` 기준입니다.

| 컬럼 | 의미 |
|---|---|
| `dataset` | 원본 E57 이름에서 확장자를 제외한 값 |
| `damage_id`, `source_damage_id` | 기존 후보 ID, 해당 Part 내부의 영역 번호 |
| `damage_type`, `damage_name_ko`, `detection_status` | `SPL`, 박락 후보, `spalling_candidate` |
| `world_center_x_m`, `world_center_y_m`, `world_center_z_m` | 개별 후보를 구성하는 원본 관측점 XYZ의 산술평균 중심점(m). 후보 LAS와 동일한 원본 E57 좌표계 |
| `coordinate_frame` | `source_E57_local` |
| `crs` | 확인된 EPSG 코드가 없어 현재 빈값 |
| `grid_id`, `grid_guid`, `member_name` | 모델 격자·부재 연결 정보. 현재 빈값 |
| `mapping_status` | 현재 `unregistered`: TLS와 모델의 좌표 정합 필요 |
| `area_m2` | 후보 셀의 추정 표면 면적(m²) |
| `mean_depth_m`, `p95_depth_m`, `max_depth_m` | 후보 셀 깊이의 평균, 95백분위, 최대값(m) |
| `point_count` | 후보 LAS의 관측점 수 |
| `source_data_path`, `pointcloud_path`, `visualization_path` | 원본 E57, 후보 LAS, 후보 PNG의 상대경로 |

CSV의 3차원 위치는 `world_center_x_m`, `world_center_y_m`, `world_center_z_m` 세 컬럼으로 제공합니다. 각 값은 해당 후보의 실제 관측점 중심이며, 좌표 기준은 `daechung_1.e57`에 스캔별 변환을 적용한 원본 점군의 공통 좌표계입니다. `world_center_*`라는 컬럼명은 센서별 산출물의 공통 형식이며, 현재 값에 EPSG:5186 변환이나 댐 모델 정합을 적용한 것은 아닙니다. 모델의 격자·부재 매핑은 공통 기준점 등으로 좌표를 정합한 후 수행해야 합니다.

## 실행

Python 3.13 환경에서 이 README가 있는 `TLS` 폴더로 이동한 뒤 실행합니다. CPU로 처리합니다.

```bash
bash Code/setup.sh
python3 Code/process.py
```

`Code/config.json`에서 설정을 바꾸고 같은 명령으로 다시 실행하면 산출물을 갱신합니다. 현재 설정·입력과 일치하는 작업 캐시에서 LAS·PNG·CSV만 다시 생성하려면 다음 명령을 사용합니다.

```bash
python3 Code/process.py --export-only
```

중간 배열·기준면·재검토 영역·실행 설정은 `~/.cache/dam_damage_module/tls_spalling/`에 저장합니다. `TLS_WORK_DIR` 또는 `--work-dir`로 작업 폴더를 바꿀 수 있습니다. 다른 작업 폴더에서 최초 실행할 때는 `--export-only` 없이 실행합니다. 입력·출력 위치는 `--data-dir`, `--output`으로 변경할 수 있습니다.

제공 범위는 Part1·Part2 원본, 실행 코드, 박락 후보별 최종 산출물입니다. 중간 처리 자료는 실행 시 외부 작업 캐시에 생성합니다.
