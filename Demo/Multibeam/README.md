# Multibeam 손상 탐지 산출물

대청댐 A 멀티빔 원본 점군에서 검출된 **세굴(SC) 2개, 슬래브 함몰(DP) 4개**를 손상별 LAS 점군·Visualiza 3D PNG·CSV로 정리했습니다. 기존 CUBE, 세굴 검출, 슬래브 추출, 함몰 검출 알고리즘과 원본 손상 수치를 유지했습니다. 현재 산출물은 기존 A 분석 결과에서 추출한 결과이며, 재계산 결과와 CSV의 `result_origin`으로 구분합니다.

## 폴더 구성

```text
Multibeam/
├── Data/
│   └── 0913_multibeam_A_align.las
├── Code/
│   ├── process.py
│   ├── export_results.py
│   ├── visualize_results.py
│   ├── download_assets.py
│   ├── algorithms/
│   ├── tools/
│   └── 실행·설치 보조 파일
├── Output/
│   ├── Visualiza/
│   │   ├── SC/SC-001.png, SC-002.png
│   │   └── DP/DP-001.png ~ DP-004.png
│   └── Result/
│       ├── SC/SC-001.las, SC-002.las
│       ├── DP/DP-001.las ~ DP-004.las
│       ├── SC_result.csv
│       └── DP_result.csv
└── README.md
```

| 경로 | 내용 |
|---|---|
| `Data/` | 원본 A LAS, 48,890,044개 관측점. |
| `Code/` | 원본 계산 코드, 실행 연결 코드, 손상별 점군 추출·이미지·CSV 생성 코드, 설치 의존성. |
| `Output/Result/SC/` | 기존 세굴 폴리곤 안의 원본 A LAS 관측점을 추출한 점군. 원본 XYZ와 점 속성을 유지합니다. |
| `Output/Result/DP/` | 기존 함몰 후보점 중 해당 패치에 속한 점을 원본 슬래브 LAS와 좌표로 매칭하여 추출한 점군. 원본 XYZ와 속성을 유지하고 `depression_depth_m` 속성을 추가합니다. |
| `Output/Visualiza/SC/` | 세굴 LAS의 실제 XYZ와 주변 원본 관측점으로 만든 3D 스냅샷. 점의 색은 실제 표고입니다. |
| `Output/Visualiza/DP/` | 함몰 LAS의 실제 XYZ와 주변 슬래브 관측점으로 만든 3D 스냅샷. 점의 색은 점별 함몰 깊이입니다. |
| `Output/Result/*.csv` | 손상별 치수, 중앙 표출 좌표, LAS·PNG 상대경로. UTF-8 BOM 인코딩. |

각 손상은 **LAS 1개와 PNG 1개**에 대응합니다. PNG는 1800×1050의 3D 스냅샷입니다. 왼쪽은 XYZ 실제 축척(1:1:1), 오른쪽은 높이 차를 읽기 위한 Z 확대 보기입니다. 두 보기는 같은 관측 점군이며, 오른쪽의 확대 배율을 이미지에 표시합니다. 축은 중앙점 기준 X/Y/Z 거리(m)이고 `+` 표시는 손상 중앙점입니다.

이미지에만 점 수 제한과 Z 표시 확대를 적용하며, LAS의 관측 XYZ·점 속성·전체 점 수는 유지합니다. 주변 관측점은 회색으로 표시합니다. 세굴의 색은 실제 표고, 함몰의 색은 해당 관측점의 `depression_depth_m`입니다. 점별 함몰 깊이와 CSV의 격자 패치 깊이는 원본 계산 대상이 다르므로 값이 다를 수 있습니다.

## 현재 산출물

| 손상 ID | 종류 | 면적(m²) | LAS 점 수 |
|---|---|---:|---:|
| SC-001 | 세굴 | 1.50 | 3,106 |
| SC-002 | 세굴 | 1,531.50 | 4,231,456 |
| DP-001 | 슬래브 함몰 | 194.59 | 174,150 |
| DP-002 | 슬래브 함몰 | 27.38 | 9,558 |
| DP-003 | 슬래브 함몰 | 38.14 | 26,607 |
| DP-004 | 슬래브 함몰 | 0.98 | 122 |

세굴은 기존 **posterior · 1.0m** 검출 결과를 사용했습니다. 함몰은 기존 `depression_patches.gpkg`의 4개 패치를 사용했습니다. 다른 방법·해상도의 결과는 같은 관측을 다르게 처리한 결과이므로 손상 건수에 합산하지 않습니다. B 관측과 제외된 E57는 처리 대상에서 제외합니다.

## 중앙 좌표와 깊이

- X/Y 중앙점은 원본 검출 폴리곤의 무게중심입니다. 단위는 m, 좌표계는 **EPSG:5186**입니다.
- 중앙점 Z는 그 X/Y에 가장 가까운 추출 관측점의 실제 표고입니다. 중앙점 X/Y와 표고를 표출 좌표로 사용합니다.
- LAS의 Z는 원래 관측 표고를 유지합니다.
- 세굴 CSV 깊이는 원본 계산과 동일하게 **Fill gap − 임계값 0.5m**입니다.
- 함몰 CSV 깊이·체적은 원본 기준 평면 대비 격자 분석 값입니다. LAS의 `depression_depth_m`은 해당 관측점의 원본 기준 평면 잔차에 음수를 취한 값입니다. CSV의 패치 통계와 점별 속성은 계산 대상이 다릅니다.
- 기존 세굴 SC-002 경계에는 꼭짓점의 자기 접촉이 있습니다. 원본 경계와 면적을 유지하여 그 경계 안의 점을 추출했습니다.

## CSV 컬럼

| 컬럼 | 의미 |
|---|---|
| `dataset` | 원본 관측 파일의 이름(확장자 제외). |
| `damage_id` | LAS와 PNG 파일명에 대응하는 손상 ID. |
| `source_damage_id` | 원본 검출 결과의 ID. |
| `damage_type`, `damage_name_ko` | SC 세굴 / DP 슬래브 함몰. |
| `world_center_x_m`, `world_center_y_m`, `world_center_z_m` | 손상 중앙 표출 좌표. |
| `crs` | EPSG:5186. |
| `area_m2` | 원본 손상 면적. |
| `mean_depth_m`, `max_depth_m` | 원본 패치 평균·최대 깊이. |
| `median_depth_m` | 원본 세굴 중앙 깊이. 함몰은 빈값. |
| `volume_loss_m3` | 원본 함몰 체적. 세굴은 빈값. |
| `point_count` | 해당 LAS에 저장된 점 수. |
| `pointcloud_selection` | `raw_points_in_polygon`: 세굴 영역 내 원본점. `damage_candidates`: 함몰 영역 내 원본 후보점. `cleaned_patch_points`: 점별 후보가 없는 검출 패치의 실제 정제 관측점. |
| `analysis_method`, `analysis_resolution_m` | 세굴 처리 방법·격자 해상도. 함몰 방법은 `reference_plane`. |
| `result_origin` | `existing_results`: 기존 검출 결과에서 추출. `computed`: 실행 코드로 새로 계산. |
| `source_data_path` | 원본 입력 경로. CSV 기준 상대경로. |
| `pointcloud_path`, `visualization_path` | 손상 LAS·PNG 경로. CSV 기준 상대경로. |
| `boundary_xy_json` | 원본 검출 경계와 내부 구멍을 포함한 GeoJSON 형식 좌표. EPSG:5186, m. |

## 다운로드

원본 LAS(약 1.27GB)와 SC-002 LAS(약 110MB)는 GitHub 저장소의 파일 크기 제한을 넘으므로 [Multibeam Release](https://github.com/Jiwandukong/Dam_Damge_Module/releases/tag/multibeam-demo-v2)로 제공합니다. 작은 LAS, Visualiza 3D PNG, CSV, 실행 코드는 저장소에 포함합니다.

| Release 파일 | 내용 |
|---|---|
| [multibeam_A_rawdata.zip](https://github.com/Jiwandukong/Dam_Damge_Module/releases/download/multibeam-demo-v2/multibeam_A_rawdata.zip) | `Data/0913_multibeam_A_align.las`. |
| [multibeam_A_results.zip](https://github.com/Jiwandukong/Dam_Damge_Module/releases/download/multibeam-demo-v2/multibeam_A_results.zip) | `Output/`의 LAS 6개, PNG 6개, CSV 2개 전체. |

`Demo/Multibeam` 폴더에서 다음 명령으로 다운로드·압축 해제합니다. ZIP과 내부 파일의 SHA256을 확인합니다.

```bash
python3 Code/download_assets.py
```

결과만 받는 경우:

```bash
python3 Code/download_assets.py --scope results
```

직접 다운로드할 때는 ZIP 안의 `Data/` 또는 `Output/`을 `Multibeam` 폴더 아래에 압축 해제합니다.

## 3D 스냅샷 다시 생성

기존 LAS와 CSV로 3D PNG만 다시 만들 수 있습니다. 손상 검출과 LAS 추출은 다시 수행하지 않습니다.

```bash
python3 Code/visualize_results.py
```

SC는 `Data`의 원본 LAS가 있으면 주변 관측점도 함께 표시합니다. DP의 주변 관측점을 함께 표시하려면 기존 분석 중간 결과의 슬래브 LAS를 지정합니다.

```bash
python3 Code/visualize_results.py --slab-source /path/to/slab.las
```

원본 분석을 실행하면 SC·DP의 주변 관측점까지 포함한 3D 스냅샷을 자동 생성합니다. 이미지 경로는 CSV의 `visualization_path`에 저장합니다.

## 원본에서 실행

Python 3.13 환경에서 실행합니다.

```bash
./Code/setup.sh
./Code/run.sh
```

원본이 없으면 A LAS를 Release에서 자동 다운로드합니다. `Data` 바로 아래의 LAS/LAZ/XYZ/E57를 입력별로 처리합니다. XYZ는 헤더 없는 X Y Z 3열이어야 합니다. 현재 제외 설정은 `Code/config.json`에 있습니다.

기존 처리 순서 CUBE → Fill gap → 세굴 → 슬래브 추출 → 함몰 검출을 유지합니다. 기본적으로 원래의 네 방법과 네 해상도를 계산하고, 최종 전달 산출물은 posterior 1.0m 세굴과 슬래브 함몰을 사용합니다.

```bash
# 동일 계산에서 완료 단계를 재사용
./Code/run.sh --resume

# posterior 1.0m만 계산
./Code/run.sh --methods posterior --resolutions 1.0

# 입력 지정
./Code/run.sh --input Data/0913_multibeam_A_align.las

# 환경과 원본 계산 코드 일치 확인
./Code/run.sh --check
```

계산 중간 파일·로그·검증 기록·전용 Python 환경은 기본적으로 `../unmodified_multibeam/Multibeam_work/`에 저장합니다. 실행 위치와 무관하게 `Data`와 `Output`은 이 프로젝트 폴더를 기준으로 합니다. 다른 작업 경로는 `MULTIBEAM_WORK_DIR` 또는 `--work-dir`로 지정합니다. 설치한 Python은 `MULTIBEAM_PYTHON`으로 지정할 수 있습니다.

완료된 새 계산에서 최종 산출물만 다시 추출하는 경우:

```bash
./Code/run.sh --export-only --input Data/0913_multibeam_A_align.las \
  --results-dir ../unmodified_multibeam/Multibeam_work/datasets/0913_multibeam_A_align
```

원본 predicted 방법에서 발생하는 division-by-zero 등 기존 계산의 실패는 작업 로그에 기록합니다. 알고리즘 SHA256을 항상 확인하며, 원본 노트북 보관본이 있는 환경에서는 계산 AST와 원본 13개 파일도 대조합니다.

함몰 격자 패치와 개별 점 후보는 원본 알고리즘에서 서로 다른 MAD 기준으로 검출합니다. 새로운 입력에서 검출 패치 안에 개별 후보점이 없으면 그 패치의 실제 정제 관측점을 추출하고 `pointcloud_selection`에 기록합니다. 현재 A 산출물 4개는 모두 기존 후보점을 사용했습니다.
