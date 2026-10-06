# 대청댐 손상 시연

**원본 사진 → SAM3 추론 → 손상 CSV·Excel과 512×512 오버레이**를 제공합니다.
이 `06.시연용` 폴더만 사용하며 상위 01~05 폴더는 필요하지 않습니다. 이미 생성된 결과를 표출할 때는 모델을 실행할 필요가 없습니다.

## 필요한 자료 위치

아래 경로와 명령은 모두 `06.시연용/` 기준입니다.

| 자료 | 위치 |
|---|---|
| 입력용 원본 사진 20장 | [01.시연데이터/원본사진/](01.시연데이터/원본사진/) |
| SAM3 모델·검증 정보 | [02.시연모델/SAM3/](02.시연모델/SAM3/) |
| 댐 OBJ·좌표계 정보 | [02.시연모델/댐3D모델/](02.시연모델/댐3D모델/) |
| 실행 코드 | [code/](code/) |
| 모델이 예측한 결과 | [04.시연산출물/모델추론결과/](04.시연산출물/모델추론결과/) |
| 라벨 기준 참고 결과 | [04.시연산출물/라벨참고결과/](04.시연산출물/라벨참고결과/) |

라벨 기준 결과는 비교·표출 선택을 위해 보존한 별도 자료이며, **모델 추론의 입력으로 사용하지 않습니다.**

손상별 시연 결과는 다음 두 경로에서 확인합니다.

- [모델 추론 전체 CRC / DLM / SPL 결과](04.시연산출물/모델추론결과/damage_type별%20결과/): 원래 inference 21,453건 전체와 class별 positive-only 타일.
- [사람 검수 후 KEEP된 CRC / DLM / SPL 라벨링 결과](04.시연산출물/라벨참고결과/damage_type별%20결과/): spatial valid 588건 검수 후 KEEP된 231건(CRC 78 / DLM 57 / SPL 96)과 KEEP-only 오버레이.

기존 mixed 모델·라벨 결과는 그대로 유지합니다. MODEL review/bulk EXCLUDE는 모델 게시본에 적용하지 않았으며, LABEL EXCLUDE는 원본 GT 삭제를 의미하지 않습니다.

## 모델 실행

### 1. 설치 — 처음 한 번

Python 3.11과 Git이 필요합니다. 아래 명령은 Linux·NVIDIA GPU 기준이며, 현재 시연 모델의 실데이터 추론을 GPU 환경에서 검증했습니다.
주요 실행 패키지는 제공된 결과를 만든 검증 버전으로 고정했습니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "git+https://github.com/facebookresearch/sam3.git@46957e47805eaa273f4aa7bbbd25a88bca9108ce"
python -m pip check
```

마지막 명령에서 `No broken requirements found.`가 나오면 패키지 의존성 검사를 통과한 것입니다.
기존 설치자는 가상환경 활성화 후 `python -m pip install -r requirements.txt`를 다시 실행하면 검증 버전으로 맞춰지고, 누락됐던 `pycocotools`도 설치됩니다.

### 2. 모델 다운로드 — 처음 한 번

```bash
python code/모델받기.py
python code/3D모델받기.py
```

- [SAM3 시연 모델 Release](https://github.com/Jiwandukong/UAV_RGB_Damage_Module/releases/tag/sam3-demo512-v1): 약 3.37GB. 두 조각을 받아 검증·복원합니다. 준비 중 약 6.8GB 여유 공간이 필요합니다.
- [댐 OBJ Release](https://github.com/Jiwandukong/UAV_RGB_Damage_Module/releases/tag/daecheong-dam-geometry-v1): 약 184MB. 기존 EPSG:5186/Z-up 모델입니다.

다운로드가 안 되면 관리자에게 파일을 직접 전달받아 아래 위치에 넣으십시오.

- `02.시연모델/SAM3/sam3_demo512_학습완료.pt`
- `02.시연모델/댐3D모델/daecheong_dam_epsg5186_zup.obj`

`학습기록.json`은 배포 모델을 확인하는 데 필요한 파일이므로 삭제하지 않습니다. 재학습이나 원본 10GB 학습 체크포인트는 필요하지 않습니다.

### 3. 원본 사진 한 장 추론

```bash
python code/시연도구.py infer \
  --images "01.시연데이터/원본사진/DJI_20250702150745_0300_V.JPG" \
  --model-record "02.시연모델/SAM3/학습기록.json" \
  --mesh "02.시연모델/댐3D모델/daecheong_dam_epsg5186_zup.obj" \
  --output "04.시연산출물/새_모델추론결과" \
  --device cuda
```

`--images`를 원하는 원본 사진 또는 사진 폴더로 바꾸면 됩니다. 폴더는 바로 아래 JPG·JPEG·PNG를 모두 처리합니다.
`--output`은 **아직 없는 새 폴더**로 지정합니다. 기존 결과를 덮어쓰지 않습니다.
다음 실행부터는 `source .venv/bin/activate` 후 추론 명령만 실행합니다.

## 결과 파일과 이미지 연결

실행 결과는 `--output`으로 지정한 폴더에 저장됩니다. 제공된 모델 결과와 라벨 참고 결과도 같은 방식으로 읽습니다.

| 결과 폴더 안의 파일·폴더 | 의미 |
|---|---|
| `damage_results.csv` | 손상별 위치·정량값·이미지 경로 |
| `damage_results.xlsx` | CSV와 같은 내용, `Damage_Details` 시트 |
| `원본사진/` | 자르기 전 원본 사진 |
| `512원본타일/` | 표출용 512×512 원본 이미지 |
| `512모델예측오버레이/` | 모델이 예측한 손상을 표시한 512×512 이미지 |
| `연결정보.json` | 손상 ID·타일·원본 내 타일 위치 |

라벨 참고 결과에서는 `512모델예측오버레이/` 대신 `512라벨오버레이/`를 사용합니다.
두 결과의 손상 ID는 각각 독립적입니다. 같은 ID라고 같은 손상을 의미하지 않습니다.

**CSV의 이미지 경로는 해당 CSV가 있는 폴더 기준 상대경로**입니다.
`tile_original_paths_json`과 `tile_overlay_paths_json`은 같은 순서로 대응하며, 같은 파일명은 같은 타일입니다.
손상이 여러 타일에 걸치면 배열에 모두 연결됩니다. 3D 표시점을 누를 때 해당 행의 이미지 경로를 사용하십시오.

### 표출용 컬럼

| 컬럼 | 의미 |
|---|---|
| `image` | 자르기 전 원본 사진 파일명 |
| `damage_id` | 결과 묶음 안에서 유일한 손상 ID |
| `damage_type` | CRC = Crack(균열), DLM = Delamination(박리), SPL = Spalling(박락) |
| `damage_name_ko` | 손상 종류의 한글 이름 |
| `pixel_nodes_json` | 원본 사진 기준 손상 외곽선 표본 좌표 |
| `world_center_x_m`, `world_center_y_m`, `world_center_z_m` | 3D 표시점(m). X/Y는 EPSG:5186, Z는 기존 OBJ 고도 |
| `source_image_path` | 자르기 전 원본 사진 경로 |
| `tile_original_paths_json` | 해당 손상의 원본 타일 경로 배열(JSON) |
| `tile_overlay_paths_json` | 같은 순서의 오버레이 경로 배열(JSON) |

## 이용 조건

자체 코드는 [MIT License](LICENSE), SAM3와 파생 모델은 [SAM License](code/third_party/SAM_LICENSE)를 확인하십시오.
사진·라벨 기반 결과·OBJ·모델 가중치는 코드의 MIT 라이선스로 자동 재허가되지 않습니다. 별도 재배포·사용 조건은 제공자에게 확인하십시오.
