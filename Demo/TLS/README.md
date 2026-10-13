# TLS 박락 후보 검출 산출물

대청댐 Part1·Part2의 원본 TLS 점군에서 주변 기준평면보다 안쪽으로 파인 영역을 찾고, 등록된 E57의 실제 위치로 댐 모델의 Grid와 부재에 매핑합니다. 겹치는 구간의 국소 평면을 RANSAC으로 추출하는 기하 방식이며, 학습 모델과 손상 라벨은 사용하지 않습니다. 결과는 **박락 후보 42건**으로, 현장 확인을 통해 박락 여부를 판정해야 합니다. 지표는 면적, 깊이, 등록 좌표의 중심 위치, Grid 내 손상 면적 및 점유율입니다.

## 폴더 구성

GitHub `Demo/Multibeam`의 점군 센서 구성에 맞춰 원본, 실행 코드, 후보별 산출물을 구분합니다.

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

| 입력 | 원본 점 수 | 박락 후보 | Grid 매핑 | 부재 매핑 | 작업 캐시의 재검토 영역 |
|---|---:|---:|---:|---:|---:|
| `dacheung_part1.e57` | 3,201,667 | 23 | 23 | 23 | 43 |
| `dachung_part2.e57` | 1,822,885 | 19 | 0 | 19 | 5 |
| 합계 | 5,024,552 | 42 | 23 | 42 | 48 |

각 후보는 기존 검출 ID(`TLS1-xxxx`, `TLS2-xxxx`)를 유지하며, LAS 1개·PNG 1개·`SPL_result.csv` 1행에 대응합니다. CSV는 손상별 결과 파일 하나만 제공합니다. 여러 Grid에 걸친 부분 면적은 내부적으로 계산하고, 손상별 행에는 대표 Grid의 면적·점유율과 전체 배정 면적·미배정 면적·연결된 Grid 수를 기록합니다. 길고 좁은 이음새, 경계, 기준면 지지가 부족한 영역 등 `geometry_review` 상태인 48건은 최종 산출물에서 제외하고 작업 캐시에 유지합니다.

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

## 등록 좌표와 모델 매핑

등록 파일은 `Data/Registered/regist_part1.e57`, `regist_part2.e57`이며, 기준 모델은 `Data/Model/daecheongdam_regions_grid5m.gltf`입니다. 원본의 정밀한 형상, 등록 점군, 매핑용 모델을 모두 포함해 다운로드한 TLS 폴더에서 실행할 수 있습니다. 매핑용 모델은 `Dam_model/Daecheongdam` 기준 모델의 전체 형상·노드 변환·Grid·부재 GUID를 유지하고 텍스처와 재질을 제외한 복사본입니다. `Code/config.json`의 `mapping`에서 경로와 허용 거리·Grid 면적을 지정합니다. 경로는 `TLS` 폴더 기준입니다.

현재 등록 E57는 큰 XYZ 값이 단정밀도로 저장되어 X·Y 간격이 각각 약 0.015625m·0.03125m입니다. 0.025m 깊이 검출에 좌표 반올림이 섞이지 않도록 면적·깊이는 원본 E57 형상에서 다시 계산합니다. 두 등록 파일은 원본과 점 수·점 순서의 RGB·강도 값이 모두 일치하므로 같은 원본 레코드 인덱스로 등록 XYZ를 연결합니다. LAS와 손상 중심은 **등록 E57의 실제 좌표**를 사용하며, 추정한 강체변환으로 위치를 다시 생성하지 않습니다. 강체변환 잔차는 레코드 대응의 점검에만 사용합니다.

`Code/model_mapping.py`는 glTF의 전체 노드 변환을 적용하고 `(X,Y,Z) → (X,-Z,Y)`로 Z-up XYZ를 구성합니다. Grid 1,434개와 부재 13개의 삼각형 표면까지의 최단거리를 계산합니다. 삼각형의 내부·변·꼭짓점을 모두 평가하며, Grid 중심점까지의 거리로 대체하지 않습니다.

손상 중심에서 가장 가까운 Grid와 부재를 각각 구하고 **표면까지 1m 이하**일 때 연결합니다. Grid가 멀어도 부재가 1m 이내라면 부재만 연결하고 `mapping_status=member_only`로 기록합니다. 현재 Part1 중심의 Grid 이격은 0.0016~0.1254m입니다. Part2는 부재 이격이 0.4005~0.8095m여서 부재에 연결되지만, Grid 이격은 1.8036~6.8120m여서 Grid와 점유율이 빈값입니다. 거리만으로 Part2의 등록 정확성이나 Grid 누락 원인을 확정하지 않습니다.

면적 분할은 검출에 쓰인 **3cm 분석 셀 단위의 근사**입니다. 각 셀의 대응 등록점 평균 위치에서 가장 가까운 Grid 삼각형을 찾아, 1m 이내인 셀의 면적을 해당 Grid에 한 번 배정합니다. 셀 면적은 기존 기준면의 기울기를 반영하며, 배정 면적과 미배정 면적의 합이 원래 손상 면적과 일치하는지 확인합니다. 현재 Part1은 6.7596m² 모두 배정되고, Part2는 2.0030m²가 Grid 미배정 상태입니다.

```text
grid_damage_ratio_pct = grid_damage_area_m2 / 25.0 × 100
```

분모는 모델 속성의 실면적과 관계없이 요청한 고정값 **25m²**입니다. `SPL_result.csv`의 `grid_damage_area_m2`·`grid_damage_ratio_pct`는 손상 중심에 가장 가까운 대표 Grid에 배정된 부분의 면적·점유율입니다. 손상이 여러 Grid에 걸치면 `mapped_area_m2`에 모든 Grid에 배정된 면적 합계, `mapped_grid_count`에 연결된 Grid 수를 기록합니다. 이 파일은 Grid별 전체 집계가 아닌 손상별 결과입니다. 미매핑은 점유율 0%로 판정하지 않고 빈값으로 둡니다.

## 산출물

LAS에는 실제 후보 관측점의 등록 XYZ/RGB와 `depth_m`, `candidate_id`, `source_vertex_id`를 저장합니다. `source_vertex_id`는 원본 및 등록 E57 스캔의 공통된 0부터 시작하는 점 인덱스입니다. XYZ는 등록 파일과 같은 댐 모델 Z-up XYZ이며 LAS 저장 간격은 0.000001m입니다. 저장 간격이 등록 파일 자체의 좌표 정밀도를 높이지는 않습니다. 깊이는 원본의 정밀한 관측점에서 계산하고 색상은 E57의 8비트 RGB를 LAS의 16비트 RGB로 변환합니다.

PNG는 3840×2160이며, 원본 RGB와 빨간 후보 점을 표시한 정면도, 깊이 색상과 초록색 기준면을 표시한 3D 시점을 함께 제공합니다. 원본 형상의 표면 좌표계로 시각화하며 깊이를 과장하지 않습니다.

CSV는 UTF-8 BOM 인코딩이며 경로는 `Output/Result` 기준입니다.

| 컬럼 | 의미 |
|---|---|
| `dataset` | 원본 E57 이름에서 확장자를 제외한 값 |
| `damage_id`, `source_damage_id` | 기존 후보 ID, 해당 Part 내부의 영역 번호 |
| `damage_type`, `damage_name_ko`, `detection_status` | `SPL`, 박락 후보, `spalling_candidate` |
| `world_center_x_m`, `world_center_y_m`, `world_center_z_m` | 개별 후보의 대응 등록 관측점 XYZ의 산술평균 중심점(m). 후보 LAS와 같은 좌표계 |
| `coordinate_frame` | `Dam_model_Z_up_XYZ` |
| `crs` | 확인된 EPSG 코드가 없어 현재 빈값 |
| `grid_id`, `grid_guid` | 손상 중심에서 가장 가까운 모델 Grid. 1m 밖이면 빈값 |
| `member_name`, `member_guid` | 가장 가까운 모델 부재. 1m 밖이면 빈값 |
| `mapping_status` | 현재 `mapped` 23건, `member_only` 19건. 부분 셀 매핑과 전체 이격 초과도 별도 상태로 기록 |
| `area_m2` | 후보 셀의 추정 표면 면적(m²) |
| `mean_depth_m`, `p95_depth_m`, `max_depth_m` | 후보 셀 깊이의 평균, 95백분위, 최대값(m) |
| `point_count` | 후보 LAS의 관측점 수 |
| `nearest_grid_distance_m`, `nearest_member_distance_m` | 손상 중심에서 가장 가까운 각 표면까지의 거리(m). 미매핑도 거리 기록 |
| `mapping_max_distance_m` | 허용 이격, 현재 1m |
| `grid_area_m2` | 대표 Grid의 점유율 분모, 현재 25m² |
| `grid_damage_area_m2`, `grid_damage_ratio_pct` | 대표 Grid에 배정된 부분 손상 면적(m²), 해당 면적 ÷ 25 × 100(%) |
| `mapped_area_m2`, `unmapped_area_m2`, `mapped_grid_count` | 모든 Grid에 배정된 손상 면적, 미배정 면적, 연결된 Grid 수 |
| `source_data_path`, `registered_data_path`, `pointcloud_path`, `visualization_path` | 원본 E57, 등록 E57, 후보 LAS, 후보 PNG의 상대경로 |

CSV의 3차원 위치는 `world_center_x_m`, `world_center_y_m`, `world_center_z_m` 세 컬럼으로 제공합니다. 각 값은 해당 후보의 실제 등록 관측점 중심이며, 모델의 정합 기준 좌표계에 해당합니다. EPSG 코드는 확인되지 않아 `crs`는 빈값입니다.

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

등록 E57의 위치만 바꾼 경우에도 `--export-only`로 등록 좌표와 매핑을 다시 계산할 수 있습니다. 원본 검출 설정을 바꾸거나 최초 실행할 때는 일반 실행을 사용합니다. 등록 파일과 모델은 `mapping`에 지정한 위치에 있어야 하며, 매핑을 끄려면 `mapping.enabled=false`로 설정합니다. 이 경우 원본 좌표의 기존 기하 검출 산출물을 생성합니다.

중간 배열·기준면·재검토 영역·실행 설정은 `~/.cache/dam_damage_module/tls_spalling/`에 저장합니다. `TLS_WORK_DIR` 또는 `--work-dir`로 작업 폴더를 바꿀 수 있습니다. 다른 작업 폴더에서 최초 실행할 때는 `--export-only` 없이 실행합니다. 입력·출력 위치는 `--data-dir`, `--output`으로 변경할 수 있습니다.

검증·샘플·전처리 자료와 웹뷰어의 기존 작업본은 센서 폴더 밖의 `../unmodified_tls/TLS_work/`에 보존했습니다. 웹뷰어는 해당 폴더의 `viewer/serve.py --port 8894`로 실행하며, [기존 웹뷰어](https://tls-cloud-comparison-20261012.scsi17479.chatgpt.site)도 유지됩니다.
