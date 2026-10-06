# UAV RGB Damage Module

드론 RGB 원본 사진에서 콘크리트 손상을 찾고, 표출 시스템에서 사용할 2D·3D 위치와 정량값을 생성하는 프로그램입니다.

이 저장소의 모델은 학습이 완료된 상태입니다. 제3자는 모델을 다시 학습하지 않고 입력 사진을 넣어 결과를 생성할 수 있습니다.

## 가장 먼저 확인할 결과

실행이 끝나면 다음 세 종류를 우선 확인하면 됩니다.

| 확인 목적 | 파일 | 설명 |
|---|---|---|
| 시스템 연계 | `tables/damage_results.csv` 또는 `tables/damage_results.xlsx` | 손상 위치와 정량값을 담은 13개 열의 최종 표 |
| 육안 확인 | `overlays/<사진명>_overlay.png` | 원본 사진 위에 손상 영역을 반투명 색상으로 표시한 영상 |
| 손상 영역 사용 | `masks/<사진명>/` | 균열·박리·박락을 종류별로 분리한 영상 |

CSV와 Excel의 각 행은 **사진 한 장에서 탐지된 손상 한 건**을 의미합니다.

## 실행에 필요한 입력

새 사진으로 결과를 생성하려면 다음 파일이 필요합니다.

- 촬영 메타데이터(XMP)가 보존되고 파일명이 변경되지 않은 드론 RGB 원본 사진: JPG
- 해당 촬영 임무의 위치·자세 정보: `Timestamp.MRK` (JPG 촬영번호와 MRK 기록번호가 일치해야 함)
- 좌표가 부여된 대청댐 3D 모델: OBJ
- 학습이 완료된 SAM3 체크포인트: `checkpoint_final.pt`

저장소에는 바로 실행해 볼 수 있는 원본 사진 7장과 해당 MRK 파일이 포함되어 있습니다. 대용량 체크포인트와 OBJ는 GitHub Release에서 별도로 받습니다.

## 실행 환경

- Python 3.11 이상
- CUDA를 사용할 수 있는 NVIDIA GPU
- CUDA 지원 PyTorch
- 모델 준비 중 약 21GB 이상의 여유 공간

현재 공개 버전은 CPU 전용 서버에서 새 사진을 추론할 수 없습니다. 모델은 이미 학습되어 있으므로 재학습은 필요하지 않지만, 추론 실행에는 CUDA GPU가 필요합니다. 검증에 사용한 GPU는 RTX 4090 24GB이며 최소 VRAM은 별도로 검증하지 않았습니다.

이미 생성된 PNG·CSV·Excel을 확인하거나 표출 시스템에 연결하는 작업에는 CUDA GPU가 필요하지 않습니다.

## 폴더 구성

```text
UAV_RGB/
├── 01_RawData/       # 원본 사진, 비행정보, 3D 모델 배치 위치
├── 02_Model/         # 모델 다운로드·검증 정보와 체크포인트 배치 위치
├── 03_Processing/    # 실행 코드와 설정
├── 04_Output/        # 새 실행 결과와 공개 결과 예시
└── 05_Docs/          # 상세 기술 문서
```

주요 파일의 위치는 다음과 같습니다.

```text
01_RawData/
├── missions/DJI_202507021616_left03/
│   ├── images/       # 공개 샘플 원본 사진 7장
│   └── navigation/   # 샘플 사진의 MRK 및 비행 원자료
└── geometry/
    └── daecheong_dam_epsg5186_zup.obj

02_Model/sam3_aug512/checkpoints/
└── checkpoint_final.pt

04_Output/
├── runs/             # 새로 실행한 전체 결과
└── examples/         # GitHub에서 바로 확인할 수 있는 표 예시
```

## 처음 한 번 준비하기

### 1. 저장소와 실행 환경 준비

```bash
git clone https://github.com/Jiwandukong/UAV_RGB_Damage_Module.git UAV_RGB
cd UAV_RGB

python3.11 -m venv .venv
source .venv/bin/activate

git clone https://github.com/facebookresearch/sam3.git ../sam3
git -C ../sam3 checkout 46957e47805eaa273f4aa7bbbd25a88bca9108ce

python -m pip install --upgrade pip
python -m pip install -e ../sam3
python -m pip install -e ".[mesh-gpu]"
```

### 2. 학습된 모델 받기

모델은 [SAM3 AUG512 모델 Release](https://github.com/Jiwandukong/UAV_RGB_Damage_Module/releases/tag/sam3-aug512-v1)에서 제공합니다.

체크포인트가 약 10GB이므로 6개 조각으로 나누어 배포합니다. 다음 명령 하나가 조각을 내려받고, 각 파일을 검증하고, 하나의 `checkpoint_final.pt`로 복원합니다.

```bash
python 03_Processing/scripts/download_checkpoint.py
```

복원되는 위치와 파일 식별값은 다음과 같습니다.

```text
위치: 02_Model/sam3_aug512/checkpoints/checkpoint_final.pt
크기: 10,081,318,934 bytes
SHA-256: a2749dba62207575afac9ed42f923d7cdfd7e2d2f0ffb2210a103e91657d985d
```

다운로드한 조각과 복원된 파일이 함께 저장되므로 모델 준비 중 약 21GB의 공간을 사용합니다. Release 다운로드나 복원이 작동하지 않으면 저장소 관리자에게 약 10GB 원본 `checkpoint_final.pt`를 직접 전달받아 위 위치에 두십시오.

### 3. 3D OBJ 받기

3D 표출 좌표를 생성할 OBJ는 [대청댐 3D 모델 Release](https://github.com/Jiwandukong/UAV_RGB_Damage_Module/releases/tag/daecheong-dam-geometry-v1)에서 제공합니다.

```bash
curl --fail --location \
  --output 01_RawData/geometry/daecheong_dam_epsg5186_zup.obj \
  https://github.com/Jiwandukong/UAV_RGB_Damage_Module/releases/download/daecheong-dam-geometry-v1/daecheong_dam_epsg5186_zup.obj
```

```text
위치: 01_RawData/geometry/daecheong_dam_epsg5186_zup.obj
좌표 규약: X/Y = EPSG:5186, Z = 위쪽 방향, 단위 = m
크기: 184,020,785 bytes
SHA-256: a7566c8d8f70de91db18c70ae5f404b0f7885e679a1027fd3e1936afd2d1e470
```

브라우저로 직접 받는 경우에도 파일명과 저장 위치를 그대로 유지하십시오.

### 4. 입력 파일 검증

```bash
python 03_Processing/scripts/verify_checkpoint.py --strict-load
python 03_Processing/scripts/verify_assets.py
```

두 명령이 모두 성공해야 공개된 모델과 3D OBJ가 정확히 준비된 것입니다.

## 실행하기

### 포함된 샘플 사진 7장 실행

```bash
python 03_Processing/scripts/run_pipeline.py
```

### 기관이 보유한 사진 실행

사진 폴더와 동일한 촬영 임무의 MRK 파일을 지정합니다. 출력 위치는 존재하지 않거나 비어 있는 새 폴더여야 합니다.

```bash
python 03_Processing/scripts/verify_assets.py \
  --images /path/to/JPG_folder \
  --mrk /path/to/Timestamp.MRK

python 03_Processing/scripts/run_pipeline.py \
  --images /path/to/JPG_folder \
  --mrk /path/to/Timestamp.MRK \
  --output-dir 04_Output/runs/institution_run
```

사진 한 장만 처리하려면 `--images`에 JPG 파일 하나를 지정해도 됩니다. 전체 실행 옵션은 다음 명령으로 확인할 수 있습니다.

```bash
python 03_Processing/scripts/run_pipeline.py --help
```

## 결과 폴더 이해하기

```text
04_Output/runs/<실행 폴더>/
├── masks/<사진명>/
│   ├── CRC.png
│   ├── DLM.png
│   ├── SPL.png
│   ├── multilabel_rgb.png
│   └── preview.png
├── overlays/
│   └── <사진명>_overlay.png
├── tables/
│   ├── damage_results.csv
│   ├── damage_results.xlsx
│   └── per_image/<사진명>_damage_details.csv
└── metadata/
    ├── run_metadata.json
    └── per_image/<사진명>_metadata.json
```

| 결과 | 의미 |
|---|---|
| `CRC.png` | 균열 영역은 흰색, 나머지는 검정인 영상 |
| `DLM.png` | 박리 영역은 흰색, 나머지는 검정인 영상 |
| `SPL.png` | 박락 영역은 흰색, 나머지는 검정인 영상 |
| `multilabel_rgb.png` | R=균열, G=박리, B=박락인 다중 라벨 영상 |
| `preview.png` | 균열=빨강, 박리=노랑, 박락=파랑으로 표시한 확인용 영상 |
| `overlays/*.png` | 원본 사진 위에 같은 라벨 색상으로 손상을 반투명 표시한 영상 |
| `damage_results.csv` | 모든 사진의 손상 결과를 합친 최종 CSV |
| `damage_results.xlsx` | CSV와 같은 내용을 담은 최종 Excel |
| `per_image/*.csv` | 사진별로 나눈 손상 결과 CSV |
| `metadata/*.json` | 실행 환경과 입력 파일을 확인하기 위한 기록. 일반 표출 연계에는 필수가 아님 |

마스크, 미리보기, 오버레이는 원본 사진과 같은 가로·세로 pixel 크기로 저장됩니다. `multilabel_rgb.png`의 세 채널은 독립적이므로 한 pixel에 여러 손상이 겹칠 수 있습니다.

오버레이의 기본 색 불투명도는 80%입니다(`visualization.alpha: 0.80`). 배경은 원본 그대로 두고 손상 영역에만 색을 입힙니다. 겹친 손상은 해당 색의 평균으로 표시하므로 **초록은 박리+박락, 주황은 균열+박리, 보라는 균열+박락**을 뜻합니다. 세 종류가 모두 겹치면 갈색 계열이며, 원본 사진과 섞이면서 색조가 더 달라질 수 있습니다. 이는 추가 손상 종류나 심각도 표시가 아닙니다. 표시 농도를 바꿔도 손상 마스크·좌표·정량값·13개 컬럼은 바뀌지 않습니다.

## 손상 라벨

| 코드 | 영문 | 한글 | 제공되는 정량값 | 기본 표시 색상 |
|---|---|---|---|---|
| `CRC` | Crack | 균열 | 길이, 폭 | 빨강 |
| `DLM` | Delamination | 박리 | 면적 | 노랑 |
| `SPL` | Spalling | 박락 | 면적 | 파랑 |

## 최종 CSV·Excel 산출물

표출 시스템이 기본적으로 읽어야 할 파일은 `tables/damage_results.csv` 또는 `tables/damage_results.xlsx`입니다.

- 두 파일의 내용과 열 순서는 같습니다.
- Excel에는 `Damage_Details` 시트 하나만 있습니다.
- 최종 표에는 아래 13개 열만 생성하며 별도의 임의 열이나 요약 열을 추가하지 않습니다.

### 1. 손상 식별 정보

아래 네 열은 어느 사진에서 어떤 손상이 탐지됐는지 구분합니다. 2D·3D 표출 시에도 좌표 열과 함께 사용합니다.

| 열 이름 | 의미 |
|---|---|
| `image` | 손상이 탐지된 원본 사진 파일명. 현재 결과는 `.JPG` 확장자를 포함함 |
| `damage_id` | 한 번의 실행 안에서 손상 관측 건마다 순서대로 부여하는 ID. 예: `D000001` |
| `damage_type` | 손상 종류 코드: `CRC`, `DLM`, `SPL` |
| `damage_name_ko` | 손상 한글명: 균열, 박리, 박락 |

`damage_id`는 시설물의 영구 관리번호가 아닙니다. 같은 사진을 다시 실행하거나 입력 순서가 달라지면 ID가 달라질 수 있습니다.

### 2. 2D 표출에 필요한 정보

| 열 이름 | 의미 |
|---|---|
| `pixel_nodes_json` | 원본 사진 위에 손상 외곽선을 그리기 위한 대표점 배열. JSON 형식은 `[[x, y], ...]` |

좌표 원점 `(0, 0)`은 원본 사진의 왼쪽 위입니다. x는 오른쪽으로, y는 아래쪽으로 증가합니다. 이 값은 모든 손상 pixel의 목록이나 완전한 마스크가 아니라 **표출용 외곽선 대표점**입니다.

2D 표출에는 보통 `image`, `damage_id`, `damage_type`, `pixel_nodes_json`을 함께 사용합니다. `damage_type`에 따라 균열·박리·박락 색상을 적용하면 됩니다.

### 3. 3D 표출에 필요한 정보

| 열 이름 | 의미 |
|---|---|
| `world_center_x_m` | 3D 모델에서 손상 대표점의 EPSG:5186 X 좌표(m) |
| `world_center_y_m` | 3D 모델에서 손상 대표점의 EPSG:5186 Y 좌표(m) |
| `world_center_z_m` | 같은 대표점의 OBJ Z 좌표(m). Z축은 위쪽 방향 |

세 값을 `(X, Y, Z)` 한 점으로 묶어 3D 모델 위에 손상 마커를 표시합니다. 이 값은 손상 영역 전체의 3D 경계가 아니라 손상을 대표하는 중심 위치 한 점입니다.

### 4. 정량화에 필요한 정보

| 열 이름 | 적용 손상 | 의미 |
|---|---|---|
| `length_px` | CRC | 사진에서 탐지된 균열 영역의 긴 방향 크기(pixel) |
| `length_m` | CRC | `length_px`를 손상 위치의 pixel 크기로 환산한 근사 길이(m) |
| `width_px` | CRC | 사진에서 탐지된 균열 영역의 짧은 방향 크기(pixel) |
| `width_m` | CRC | `width_px`를 손상 위치의 pixel 크기로 환산한 근사 폭(m) |
| `area_m2` | DLM, SPL | 사진에서 탐지된 박리 또는 박락 영역을 환산한 근사 면적(m²) |

`length_px`와 `width_px`는 탐지된 균열 영역을 감싸는 방향별 크기입니다. 균열 중심선을 따라 측정한 길이나 정밀 개구폭 측정값은 아닙니다. m와 m² 값 역시 3D 모델을 이용한 근사값이므로 정밀측량값이나 안전판정값으로 사용하지 마십시오.

### 손상 종류별로 채워지는 정량 열

| 손상 | 길이·폭 4개 열 | `area_m2` |
|---|---|---|
| CRC 균열 | 값 사용 | 빈칸 |
| DLM 박리 | 빈칸 | 값 사용 |
| SPL 박락 | 빈칸 | 값 사용 |

빈칸은 숫자 0이 아닙니다. 해당 손상 종류에 적용되지 않거나 좌표·정량값을 생성하지 못했다는 뜻입니다. `world_center_x_m`, `world_center_y_m`, `world_center_z_m`이 비어 있으면 해당 행은 3D 마커를 만들 수 없습니다.

## 표출 시스템 연결 요약

| 화면 또는 기능 | 사용할 열 |
|---|---|
| 원본 사진 선택 | `image` |
| 손상 종류와 이름 표시 | `damage_id`, `damage_type`, `damage_name_ko` |
| 사진 위 2D 외곽선 표시 | `pixel_nodes_json` |
| 3D 모델 위 손상 마커 표시 | `world_center_x_m`, `world_center_y_m`, `world_center_z_m` |
| 균열 수치 표시 | `length_px`, `length_m`, `width_px`, `width_m` |
| 박리·박락 수치 표시 | `area_m2` |

## 산출물 해석 시 주의사항

- 한 행은 실제 시설물의 영구 손상 한 개가 아니라 **사진 한 장에서 관측된 손상 한 건**입니다.
- 같은 실제 손상이 여러 사진에 보이면 여러 행으로 기록될 수 있습니다. 현재 결과는 사진 간 중복을 제거하지 않습니다.
- CRC·DLM·SPL 영역은 서로 겹칠 수 있으므로 여러 종류의 면적을 단순 합산하면 중복될 수 있습니다.
- 빈칸을 0으로 바꾸거나 합계에 포함하지 마십시오.
- m 및 m² 정량값은 근사값입니다.

## 공개 산출물 예시

GitHub에는 샘플 원본 사진 7장으로 생성한 표 예시가 포함되어 있습니다. 총 718개의 손상 관측 행이며 CSV와 Excel 값이 동일합니다.

- [CSV 예시](04_Output/examples/left03_aug512_full_20260903/damage_results.csv)
- [Excel 예시](04_Output/examples/left03_aug512_full_20260903/damage_results.xlsx)
- [오버레이 예시 7장](04_Output/examples/left03_aug512_full_20260903/overlays)
- [예시 설명](04_Output/examples/left03_aug512_full_20260903/README.md)

## 라이선스

프로젝트 코드는 [MIT License](LICENSE)를 따릅니다. SAM3 모델은 Release에 포함된 [SAM License](02_Model/sam3_aug512/third_party/SAM_LICENSE)를 따릅니다.
