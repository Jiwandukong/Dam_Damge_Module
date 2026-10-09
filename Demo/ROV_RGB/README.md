# ROV RGB 손상 탐지 결과

수중 ROV 영상에서 추출한 원본 RGB 프레임 35장을 Sunflicker 전처리와 DeepLabV3+/ResNet101 모델로 처리한 **균열(CRC)·박락(SPL) 탐지 결과**입니다. 예측한 손상 경계를 원본 영상의 픽셀 좌표로 기록하고, 손상별 Overlay 이미지와 클래스별 CSV를 제공합니다.

이 자료는 손상 탐지 결과를 댐 3D 모델과 함께 보여 주기 위한 시연용입니다. 이미지의 손상 영역은 모델의 예측 결과이며, **3D 좌표는 방류 여수로 격자에 배치한 합성 좌표**입니다. 실제 ROV 촬영 위치를 복원한 결과는 아닙니다. ROV 손상의 길이·폭·면적은 산출하지 않습니다.

## 폴더 구성

```text
ROV_RGB/
├── Data/                   # 원본 RGB 프레임 JPG 35장
├── Code/                   # 전처리·추론·CSV 및 Overlay 생성
├── Output/
│   ├── Overlay/
│   │   ├── CRC/            # 균열 Overlay PNG 35개
│   │   └── SPL/            # 박락 Overlay PNG 45개
│   └── Result/
│       ├── CRC_result.csv  # 균열 결과 35행
│       └── SPL_result.csv  # 박락 결과 45행
└── README.md
```

| 경로 | 역할 |
|---|---|
| `Data/` | 추론에 사용하는 원본 1920×1080 RGB 프레임. 파일명 `frame_XXXXXX.jpg`의 숫자는 원본 동영상의 프레임 번호입니다. |
| `Code/process.py` | 원본 입력을 전처리·추론하고 손상별 Overlay, CSV, 시연 좌표를 생성합니다. |
| `Code/preprocessing.py` | 원본 동영상의 이웃 프레임을 사용하는 Sunflicker 전처리와 검증된 처리 캐시를 관리합니다. |
| `Code/rov_model.py` | DeepLabV3+/ResNet101 모델을 불러오고 손상 클래스별 예측을 수행합니다. |
| `Code/demo_mapping.py` | 유사 프레임을 묶고 댐 모델의 허용 격자 표면에 시연 좌표를 배치합니다. |
| `Output/Overlay/CRC/`, `SPL/` | 손상별 정사각형 Overlay PNG. 파일명이 CSV의 `damage_id`와 같습니다. |
| `Output/Result/` | 균열·박락 클래스별 최종 CSV 두 파일. UTF-8 BOM 인코딩이며 같은 13개 컬럼을 사용합니다. |

## 결과 CSV 컬럼

두 CSV에는 아래 **13개 컬럼**을 같은 순서로 저장하며, 행은 `damage_id` 순서로 정렬합니다.

| 컬럼 | 설명 |
|---|---|
| `image` | 해당 손상이 검출된 원본 프레임 파일명. `Data/`의 JPG와 대응합니다. |
| `damage_id` | 예측 영역의 고유 ID. `R000001` 형식이며 해당 Overlay의 파일명과 대응합니다. |
| `damage_type` | 손상 코드. `CRC`는 균열, `SPL`은 박락입니다. |
| `damage_name_ko` | 손상 종류의 한글 이름. `균열` 또는 `박락`입니다. |
| `pixel_nodes_json` | 원본 프레임 기준 예측 손상 경계 표본 `[[x,y],…]`. 왼쪽 위가 원점이며 X는 오른쪽, Y는 아래쪽으로 증가합니다. 단위는 px입니다. |
| `world_center_x_m` | 시연용 3D 대표점의 X 좌표. 모델의 EPSG:5186 동쪽 방향 좌표(m)입니다. |
| `world_center_y_m` | 동일 대표점의 Y 좌표. 모델의 EPSG:5186 북쪽 방향 좌표(m)입니다. |
| `world_center_z_m` | 동일 대표점의 모델 표고(m). Z축은 위쪽 방향입니다. |
| `member_name` | 대표점이 놓인 격자의 실제 상위 부재 이름. 현재는 `SPW_여수로_교각_도수면`입니다. |
| `section_name` | 모델에서 해당 격자가 속한 배치 구획 이름. |
| `grid_id` | 대표점이 놓인 실제 모델 격자 이름. 예: `SPW_0220`. |
| `DRI` | 해당 예측 영역 픽셀의 손상 클래스 softmax 신뢰도 평균. 범위는 0~1이며 손상 크기·물리적 심각도를 뜻하지 않습니다. |
| `surface_local_x_m` | 댐 벽면에 평행한 가로 축을 기준으로 계산한 대표점 위치(m). 전역 X 좌표와 축·원점이 다른 벽면 기준 값입니다. |

## Release 다운로드 및 실행

큰 모델과 전처리 캐시는 Release에서 제공합니다.

| 파일 | 용도 |
|---|---|
| [rov_rgb_damage_demo.pt](https://github.com/Jiwandukong/Dam_Damge_Module/releases/download/rov-rgb-demo-v1/rov_rgb_damage_demo.pt) | DeepLabV3+/ResNet101 시연 모델, 약 175MiB. |
| [rov_rgb_sunflicker_cache.zip](https://github.com/Jiwandukong/Dam_Damge_Module/releases/download/rov-rgb-demo-v1/rov_rgb_sunflicker_cache.zip) | 제공 프레임 35장의 Sunflicker 처리 RGB와 검증 메타데이터, 약 99MiB. |
| [daecheongdam_grid5m.zip](https://github.com/Jiwandukong/Dam_Damge_Module/releases/download/daecheong-dam-grid5m-v1/daecheongdam_grid5m.zip) | 대청댐 격자화 glTF, BIN, 텍스처. 시연용 3D 좌표 배치에 사용합니다. |

Python 3.11~3.13 환경에서 이 README가 있는 `ROV_RGB` 폴더로 이동한 뒤 실행합니다. NVIDIA GPU를 사용할 수 있으며, CUDA를 사용할 수 없으면 CPU로 추론합니다.

```bash
bash Code/setup.sh
bash Code/run.sh
```

최초 실행 시 필요한 Release 파일을 자동 다운로드하고 SHA256을 확인합니다. 기본 저장 위치는 `~/.cache/dam_damage_module/rov_rgb/`이며 `ROV_RGB_WORK_DIR` 또는 `--work-dir`로 변경할 수 있습니다. 제공한 35장은 전처리 캐시로 처리하므로 원본 동영상이 없어도 재현할 수 있습니다.

다른 원본 프레임에 전처리를 적용하려면 원본 동영상을 함께 지정합니다. 이미 Sunflicker로 처리한 RGB 입력은 `--input-preprocessed`로 지정합니다.

```bash
bash Code/run.sh --input /원본프레임폴더 --video /원본동영상.mp4
bash Code/run.sh --input /전처리RGB폴더 --input-preprocessed
bash Code/run.sh --checkpoint /모델/rov_rgb_damage_demo.pt \
  --dam-model /모델/Daecheongdam/daecheongdam_regions_grid5m.gltf
```

기본 실행은 Overlay·CSV와 시연용 3D 좌표를 함께 생성합니다. `--no-demo-random-3d`를 추가하면 2D 손상 결과만 생성하며, 좌표·격자·부재·벽면 가로 위치는 빈값으로 저장합니다.

처리·좌표 검증용 상세 정보는 작업 폴더의 `result_details.json`, 실행 요약은 `inference_summary.json`에 별도로 저장합니다. 최종 CSV에는 위 표의 13개 컬럼만 기록합니다.
