# ROV LiDAR 손상 탐지 산출물

대청댐 ROV LiDAR의 세굴·슬래브 함몰 결과를 정리한 모듈입니다. **세굴(SC)은 최소 면적 5m²와 슬래브 오분류 4건 제외 기준으로 25개**를 제공하며, 세굴 결과는 유지했습니다. **함몰(DP)은 분할된 슬래브 입력 범위에서 재분석한 8개를 면적 제한 없이 모두 제공합니다.** 세굴의 면적·제외 규칙은 내보내기 단계에 적용합니다. 함몰의 기준 평면·MAD·격자 판정식은 원본대로 유지하고 분석 입력의 선정 방식만 바꿨습니다.

[Demo/Multibeam](https://github.com/Jiwandukong/Dam_Damge_Module/tree/main/Demo/Multibeam)의 손상별 LAS·PNG와 종류별 CSV 구성을 따릅니다. 최상위 폴더명은 요청한 소문자 `data`, `code`, `output`을 사용합니다.

## 폴더 구성

```text
ROV_LiDAR/
├── data/
│   ├── 01_EYAS_translated.e57
│   ├── 01_EYAS_translated_segmented.ply
│   └── 01_EYAS_translated_segmented_slab_zone.ply
├── code/
│   ├── process.py
│   ├── download_assets.py
│   ├── run.py
│   ├── export_results.py
│   ├── visualize_results.py
│   ├── map_model.py
│   ├── point_cloud_io.py
│   ├── prepare_slab_zone.py
│   ├── runtime.py
│   ├── algorithms/
│   ├── tools/whitebox/
│   ├── requirements.txt
│   ├── setup.sh
│   └── run.sh
├── output/
│   ├── Result/
│   │   ├── SC/                    # 세굴 관측 점군 LAS 25개
│   │   ├── DP/                    # DP-001.las ~ DP-008.las
│   │   ├── SC_result.csv          # 세굴 25행
│   │   └── DP_result.csv          # 함몰 8행
│   └── Visualize/
│       ├── SC/                    # 세굴 4K PNG 25개
│       └── DP/                    # DP-001.png ~ DP-008.png
└── README.md
```

최종 출력은 **LAS 33개, PNG 33개, CSV 2개**입니다. 검증 보고서, 실행 로그, 테스트 코드, 중간 분석 결과, 웹뷰어와 설치된 라이브러리는 포함하지 않습니다.

## 입력과 코드

대용량 입력 파일 3개는 [ROV LiDAR Release](https://github.com/Jiwandukong/Dam_Damge_Module/releases/tag/rov-lidar-demo-v1)의 [rov_lidar_data.zip](https://github.com/Jiwandukong/Dam_Damge_Module/releases/download/rov-lidar-demo-v1/rov_lidar_data.zip)으로 제공합니다. ZIP은 약 147MB이고, 압축 해제 후 약 881MB입니다. 저장소에는 코드와 최종 **CSV 2개·LAS 33개·PNG 33개**를 포함합니다.

`Demo/ROV_LiDAR` 폴더에서 다음 명령을 실행하면 `data/`에 입력을 복원합니다. ZIP과 각 파일의 SHA-256을 확인하며, 다운로드 캐시는 모듈 밖에 둡니다.

```bash
python3 -B code/download_assets.py
```

직접 다운로드할 때는 ZIP 내부의 `data/`를 `ROV_LiDAR` 폴더 아래에 압축 해제합니다. 입력 파일별 크기와 다운로드 안내는 [data/README.md](data/README.md)에 있습니다. 최종 산출물 확인에는 입력 다운로드가 필요하지 않습니다.

모델 격자·부재 연결에 사용하는 공통 3D 모델은 기존 [대청댐 모델 Release](https://github.com/Jiwandukong/Dam_Damge_Module/releases/tag/daecheong-dam-grid5m-v1)의 [daecheongdam_grid5m.zip](https://github.com/Jiwandukong/Dam_Damge_Module/releases/download/daecheong-dam-grid5m-v1/daecheongdam_grid5m.zip)에서 받습니다. 실행할 때 압축을 푼 glTF 경로를 `--dam-model`로 지정할 수 있습니다.

`01_EYAS_translated_segmented.ply`의 관측점 7,837,497개로 세굴을 분석합니다. `01_EYAS_translated_segmented_slab_zone.ply`는 슬래브 추출 입력입니다. `01_EYAS_translated.e57`는 원본 보관용이며 현재 분석은 준비된 PLY를 사용합니다.

`code/algorithms/`에는 원본 CUBE·세굴·자동 슬래브 추출·함몰 분석 코드를 보존했습니다. `process.py`가 분석 단계와 최종 출력 생성을 연결합니다. WhiteboxTools 2.4.0 실행 파일과 라이선스는 `code/tools/whitebox/`에 있습니다.

함몰 재분석의 기본 `input-zone` 방식은 `prepare_slab_zone.py`가 분할된 슬래브 PLY의 XY 범위를 그대로 사용합니다. 슬래브보다 높은 구조물은 이 측량 자료의 **상한 표고 Z=42.708m**로 제외하며, 낮은 점에는 하한 깊이·평면 거리·법선 조건을 적용하지 않습니다. 상한은 이전 자동추출 결과의 최고 표고 42.707382m를 mm 단위로 올림한 값입니다. 다른 측량 자료에서는 `--slab-upper-z`를 해당 표고 기준에 맞게 지정해야 합니다.

이 방식으로 원본 1,942,657점 중 1,029,303점을 함몰 분석에 전달하고, 원래 kNN 이상치 제거 후 1,008,291점으로 기준 평면과 격자 함몰을 계산했습니다. 이전 자동추출의 분석 입력은 826,370점이었습니다. 자동추출은 `--slab-mode auto`로 선택할 수 있습니다.

## 최종 결과

- SC: 원본 GPKG 경계 안의 실제 segmented PLY 관측점입니다. CUBE 격자점이나 보간점으로 대체하지 않았습니다. PLY에서 LAS로 변환할 때 원래 파이프라인과 같은 0.001m 저장 정밀도를 사용합니다. 원본 RGB와 intensity를 유지합니다.
- DP: 재분석한 함몰 패치 8개의 경계 안에 있는 점별 함몰 후보를 모두 저장합니다. 패치 면적·후보 군집 외곽 면적으로 제외하지 않으며, 미군집 후보점도 포함합니다. 격자 평균 임계값과 점별 임계값은 서로 달라 후보점이 없는 격자 패치는 실제 정제 관측점으로 내보냅니다. 현재 8개 패치는 모두 점별 후보를 포함합니다. `depression_depth_m`은 원래 기준 평면 잔차의 부호를 바꾼 점별 깊이입니다.
- PNG: 손상 점군과 주변 관측점의 3840×2160 3D 스냅샷입니다. XYZ의 비율을 유지하고 표본점으로 그립니다. SC 색은 표고, DP 색은 점별 함몰 깊이입니다. LAS에는 선택된 관측점 전체가 저장됩니다.
- CSV: 원본 면적·평균/최대 깊이와 중심 좌표, 모델 격자·부재명, LAS·PNG 경로를 기록합니다. 면적은 격자/경계 분석값으로 점별 후보의 외곽 면적과 다릅니다.

원본 78개 세굴 검출 중 25개를 내보내며 면적은 2,072.0m²입니다. `SC-073`(46.5m²), `SC-074`(26.5m²), `SC-077`(114.5m²), `SC-078`(99.0m²)은 슬래브 위치로 확인되었고 관측점 전체가 원본 슬래브 분석 입력과 1mm 이내로 일치해, 검토에 따른 세굴 오분류로 제외했습니다. 원본 탐지는 슬래브 제외 마스크 없이 전체 표면의 낮은 영역을 찾으므로 재질을 구분하지 않습니다. `code/export_results.py`의 제외 규칙은 해당 입력 이름과 원본 검출 ID에만 적용되며 면적 기준을 바꾸거나 재실행해도 네 건을 세굴로 내보내지 않습니다. SC ID 숫자는 원본 검출 번호를 유지하므로 파일 번호는 연속되지 않습니다.

| 함몰 ID | 원본 패치 면적(m²) | LAS 후보점 수 |
|---|---:|---:|
| DP-001 | 134.1561 | 16,112 |
| DP-002 | 43.4318 | 3,400 |
| DP-003 | 147.6682 | 12,192 |
| DP-004 | 0.9652 | 46 |
| DP-005 | 2.8955 | 109 |
| DP-006 | 24.1288 | 2,420 |
| DP-007 | 121.6091 | 10,202 |
| DP-008 | 2.8955 | 103 |

함몰 면적 합계는 477.7502m², 체적 합계는 218.2532m³입니다. 기존 자동추출 결과는 2개 패치·36.3272m²였으며, 입력 범위가 늘어나 기준 평면과 자동 격자도 다시 계산됐습니다. 현재 점별 후보 49,933개·DBSCAN 후보 군집 443개와 격자 평균 잔차로 구한 함몰 패치 8개는 서로 다른 탐지 단위입니다. `DP_result.csv`는 Multibeam 예시와 같이 함몰 패치 단위로 기록하고, 패치 안의 작은 군집과 미군집 후보점도 LAS에 유지합니다. 패치 밖 후보점은 외부 분석 캐시에 있으며 패치 손상 수에 합산하지 않습니다.

아래 대응은 예전 SC를 DP로 이름만 바꾼 것이 아니라, 재분석한 DP 경계와 제외된 SC 경계의 XY 겹침입니다.

| 제외된 SC | 재분석 DP | SC 경계 면적 중 DP에 포함된 비율 |
|---|---|---:|
| SC-073 | DP-001 | 100.0% |
| SC-074 | DP-002 | 98.6% |
| SC-077 | DP-003 | 99.1% |
| SC-078 | DP-007 | 97.2% |

DP ID는 이번 재분석의 검출 번호이므로 이전 자동추출 ID와 위치가 다릅니다. 이전 DP-001 위치는 새 DP-002에 포함되며, 이전 DP-002는 주로 새 DP-006과 일부 DP-008에 겹칩니다. [웹뷰어](https://daecheong-rov-lidar-damage-20261009.scsi17479.chatgpt.site)도 최종 세굴 25개·함몰 8개로 갱신했으며, 함몰은 면적과 관계없이 표시합니다.

## CSV 컬럼

두 CSV는 같은 컬럼과 UTF-8 BOM 인코딩을 사용합니다.

| 컬럼 | 내용 |
|---|---|
| `dataset`, `source_data_path` | 분석 입력 이름과 CSV 기준 입력 파일 상대경로 |
| `damage_id`, `source_damage_id` | LAS·PNG의 ID와 원본 검출 ID |
| `damage_type`, `damage_name_ko` | SC 세굴 / DP 슬래브 함몰 |
| `world_center_x_m`, `world_center_y_m`, `world_center_z_m` | 손상 경계 중심 XYZ, 단위 m |
| `crs` | 원본 파이프라인의 EPSG:5186 지정 |
| `grid_id`, `member_name` | 중심 XY가 실제 모델 격자 삼각형 안에 있을 때 연결한 격자·상위 부재 |
| `area_m2` | 원본 손상 면적, m² |
| `mean_depth_m`, `max_depth_m`, `median_depth_m` | 원본 깊이 통계, m. DP 중앙 깊이는 빈값 |
| `volume_loss_m3` | 원본 함몰 손실 체적, m³. SC는 빈값 |
| `point_count` | 해당 LAS에 저장한 관측점 수 |
| `pointcloud_path`, `visualization_path` | CSV 기준 LAS·PNG 상대경로 |
| `boundary_xy_json` | 내부 구멍을 포함한 원본 손상 경계 GeoJSON |

SC 중심 Z는 원본 CSV의 CUBE 표고이며, DP 중심 Z는 경계 중심에 가까운 실제 후보점의 표고입니다. 모델 연결은 관측 좌표를 이동시키지 않습니다. 모델 범위 밖의 결과는 `grid_id`와 `member_name`을 빈값으로 둡니다. 원본 PLY에는 CRS 메타데이터가 없고 EPSG:5186은 파이프라인의 기본 지정입니다. 모델에 겹치는 것만으로 독립 측량 기준점에 의한 정합이 검증된 것은 아닙니다.

## 실행

Python 3.13 환경을 사용합니다. 제공된 최종 출력은 별도 실행 없이 확인할 수 있습니다.

```bash
cd Demo/ROV_LiDAR
bash code/setup.sh
bash code/run.sh
```

기본 실행은 외부 작업 캐시에 보존된 분석 결과에서 최종 LAS·PNG·CSV만 다시 생성합니다. 캐시가 없는 새 환경에서는 입력과 공통 모델을 다운로드한 뒤 아래 `--task all` 명령으로 분석을 먼저 수행합니다. **Depression만 재분석하고 DP LAS·PNG·CSV만 갱신**하려면 다음 명령을 사용합니다. SC 분석과 SC 출력 파일에는 접근해 쓰지 않습니다.

```bash
bash code/run.sh --task slab --slab-mode input-zone --slab-upper-z 42.708
```

준비된 슬래브 LAS에서 함몰 계산만 다시 수행하려면 `--task slab-depression`, 준비된 분석 결과에서 DP 파일만 내보내려면 `--damage-type DP`를 사용합니다. `--task scour`는 SC만 갱신합니다.

다른 서버에서 처음 실행하거나 SC와 DP를 모두 처음부터 수행하려면 다음 명령을 사용합니다.

```bash
python3 -B code/download_assets.py
bash code/run.sh --task all --dam-model /압축해제경로/Daecheongdam/daecheongdam_regions_grid5m.gltf
```

분석 중간 파일과 Python 라이브러리는 기본적으로 `~/.cache/dam_damage_module/rov_lidar/`에 저장됩니다. `ROV_LIDAR_WORK_DIR` 환경변수 또는 `--work-dir`로 작업 위치를 변경할 수 있으며 모듈 폴더 밖이어야 합니다. 특정 Python 환경은 `ROV_LIDAR_PYTHON`으로 지정합니다. 모델 기본 경로는 `../Dam_model/Daecheongdam/daecheongdam_regions_grid5m.gltf`이고 `--dam-model`로 변경할 수 있습니다.

`--min-area`는 세굴에만 적용하며 값을 올려도 함몰 패치는 모두 유지합니다. 작은 세굴 원본 후보까지 포함한 출력은 별도 폴더에 생성할 수 있습니다. 이때도 검토에서 제외한 `SC-073`, `SC-074`, `SC-077`, `SC-078`은 포함하지 않습니다.

```bash
bash code/run.sh --min-area 0 --output-dir /원하는/별도/결과폴더
```

기존 자동추출 방식은 `--task slab --slab-mode auto`로 재실행할 수 있습니다. 원본 Open3D RANSAC에는 난수 시드가 지정되어 있지 않아 기준 평면과 패치 수가 달라질 수 있습니다. 이전 자동추출 분석과 DP 출력은 모듈 밖 `~/.cache/dam_damage_module/rov_lidar/history/slab_auto_before_input_zone_20261009/`에 보관했습니다.
