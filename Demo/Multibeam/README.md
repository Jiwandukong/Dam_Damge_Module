# Multibeam 손상 탐지 산출물

대청댐 A의 멀티빔 원본 점군과 **세굴(SC) 2개, 슬래브 함몰(DP) 3개**의 손상 결과입니다. 원본 데이터는 `0913_multibeam_A_align.las`이며, 48,890,044개의 관측점을 포함합니다. 손상 결과는 LAS 점군, 3D 스냅샷 PNG, 요약 CSV로 제공합니다.

## 폴더 구성

```text
Multibeam/
├── Data/
│   └── 0913_multibeam_A_align.las
├── Code/
├── Output/
│   ├── Visualize/
│   │   ├── SC/SC-001.png, SC-002.png
│   │   └── DP/DP-001.png ~ DP-003.png
│   └── Result/
│       ├── SC/SC-001.las, SC-002.las
│       ├── DP/DP-001.las ~ DP-003.las
│       ├── SC_result.csv
│       └── DP_result.csv
└── README.md
```

| 경로 | 내용 |
|---|---|
| `Data/` | 원본 A LAS, 48,890,044개 관측점. |
| `Code/` | 데이터 처리와 손상 결과 파일을 생성하는 코드. |
| `Output/Result/SC/` | 세굴 영역의 실제 관측 점군(LAS). |
| `Output/Result/DP/` | 슬래브 함몰의 실제 관측 점군(LAS). 점별 함몰 깊이 `depression_depth_m`(m)를 포함합니다. |
| `Output/Visualize/SC/` | 세굴 점군과 주변 관측점의 3D 스냅샷(PNG). 색은 표고를 나타냅니다. |
| `Output/Visualize/DP/` | 함몰 점군과 주변 관측점의 3D 스냅샷(PNG). 색은 함몰 깊이를 나타냅니다. |
| `Output/Result/*.csv` | 손상별 면적·깊이·좌표·부재명과 LAS·PNG 경로를 담은 요약표. UTF-8 BOM 인코딩. |

각 손상 ID는 **LAS 1개와 PNG 1개**, 해당 종류의 CSV 1개 행에 대응합니다. PNG 해상도는 3840×2160입니다. 전체 산출물은 **LAS 5개, PNG 5개, CSV 2개**입니다.

## 현재 산출물

| 손상 ID | 종류 | 면적(m²) | LAS 점 수 |
|---|---|---:|---:|
| SC-001 | 세굴 | 1.50 | 3,106 |
| SC-002 | 세굴 | 1,531.50 | 4,231,456 |
| DP-001 | 슬래브 함몰 | 194.59 | 174,150 |
| DP-002 | 슬래브 함몰 | 27.38 | 9,558 |
| DP-003 | 슬래브 함몰 | 38.14 | 26,607 |

CSV의 중심 좌표는 손상의 표출 위치입니다. X/Y 좌표계는 **EPSG:5186**, Z는 표고이며 단위는 m입니다. 면적은 m², 깊이는 m, 체적은 m³로 기록합니다.

## CSV 컬럼

| 컬럼 | 의미 |
|---|---|
| `dataset` | 원본 관측 파일의 이름(확장자 제외). |
| `damage_id` | LAS와 PNG 파일명에 대응하는 손상 ID. |
| `source_damage_id` | 원본 검출 결과의 ID. |
| `damage_type`, `damage_name_ko` | SC 세굴 / DP 슬래브 함몰. |
| `world_center_x_m`, `world_center_y_m`, `world_center_z_m` | 손상 중앙 표출 좌표. |
| `crs` | EPSG:5186. |
| `member_name` | 손상 중앙점을 모델 표면에 연결해 찾은 상위 부재·구획의 이름. 현재 SC·DP 5건은 `SPW_감세공_물받이공(수면_아래)`에 연결됩니다. |
| `area_m2` | 손상 면적(m²). |
| `mean_depth_m` | 손상 분석 결과의 평균 깊이(m). |
| `volume_loss_m3` | 함몰 손실 체적(m³). 세굴은 빈값. |
| `point_count` | 해당 LAS에 저장된 점 수. |
| `source_data_path` | 원본 입력 경로. CSV 기준 상대경로. |
| `pointcloud_path`, `visualization_path` | 손상 LAS·PNG 경로. CSV 기준 상대경로. |
| `boundary_xy_json` | 손상 영역의 경계 좌표(GeoJSON 형식, m). |

## 다운로드

원본 LAS와 전체 산출물 ZIP은 [Multibeam Release](https://github.com/Jiwandukong/Dam_Damge_Module/releases/tag/multibeam-demo-v7)에서 받을 수 있습니다. 저장소에는 코드, README, CSV, PNG와 작은 LAS가 포함되어 있습니다. 원본 LAS(약 1.27GB)와 SC-002 LAS(약 110MB)는 Release 다운로드에 포함됩니다.

| Release 파일 | 내용 |
|---|---|
| [multibeam_A_rawdata.zip](https://github.com/Jiwandukong/Dam_Damge_Module/releases/download/multibeam-demo-v7/multibeam_A_rawdata.zip) | `Data/0913_multibeam_A_align.las`. |
| [multibeam_A_results.zip](https://github.com/Jiwandukong/Dam_Damge_Module/releases/download/multibeam-demo-v7/multibeam_A_results.zip) | LAS 5개, 4K PNG 5개, CSV 2개. SC 2개·DP 3개이며 CSV의 부재 정보는 `member_name`만 포함합니다. |

`Demo/Multibeam` 폴더에서 다음 명령으로 원본과 산출물을 다운로드·압축 해제합니다.

```bash
python3 Code/download_assets.py
```

결과만 받는 경우:

```bash
python3 Code/download_assets.py --scope results
```

직접 다운로드할 때는 ZIP 안의 `Data/` 또는 `Output/`을 `Multibeam` 폴더 아래에 압축 해제합니다.
