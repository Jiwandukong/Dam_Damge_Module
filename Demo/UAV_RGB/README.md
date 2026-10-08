# UAV RGB 손상 탐지 결과

드론 원본 사진을 DINOv3 기반 파인튜닝 모델로 처리한 Prediction 결과입니다. 손상 영역을 격자화된 댐 3D 모델에 매핑하고, 손상별 Overlay와 CSV로 저장합니다. 3D 미연결(`unmapped`) 결과는 시연 산출물에서 제외합니다.

## 폴더 구성

| 경로 | 역할 |
|---|---|
| `Data/` | 드론 원본 사진 20장. 촬영 위치·방향 등 DJI 메타데이터를 포함합니다. |
| `Code/` | Processing 코드. `process.py`는 추론·매핑·산출물 생성, `dinov3_model.py`는 모델 구성, `dam_mapping.py`는 3D 매핑·치수 계산을 담당합니다. `requirements.txt`는 실행 의존성 목록입니다. |
| `Output/Overlay/CRC/` | 균열 Overlay 이미지. |
| `Output/Overlay/DLM/` | 박리 Overlay 이미지. |
| `Output/Overlay/SPL/` | 박락 Overlay 이미지. |
| `Output/Overlay/LKG/` | 누수·백태 Overlay 이미지. |
| `Output/Result/` | 손상 종류별 결과 CSV. |

Overlay는 해당 손상 영역을 빨간색으로 표시한 PNG이며, 손상 크기에 따라 128px 배수의 정사각형으로 자릅니다. 이미지 파일명은 CSV의 `damage_id`와 같습니다.

현재 결과는 총 **817건**입니다.

| CSV | 종류 | 건수 |
|---|---|---:|
| [CRC_result.csv](Output/Result/CRC_result.csv) | 균열 | 363 |
| [DLM_result.csv](Output/Result/DLM_result.csv) | 박리 | 87 |
| [SPL_result.csv](Output/Result/SPL_result.csv) | 박락 | 311 |
| [LKG_result.csv](Output/Result/LKG_result.csv) | 누수·백태 | 56 |

## 결과 CSV 컬럼

| 컬럼 | 설명 |
|---|---|
| `image` | 원본 사진 파일명. |
| `damage_id` | 예측 손상 ID. 해당 Overlay 파일명과 대응합니다. |
| `damage_type` | 손상 코드: `CRC` 균열, `DLM` 박리, `SPL` 박락, `LKG` 누수·백태. |
| `damage_name_ko` | 손상 종류의 한글 이름. |
| `pixel_nodes_json` | 원본 사진 기준 손상 경계 표본 좌표 `[[x,y],…]`. 왼쪽 위가 원점이며 단위는 px입니다. |
| `world_center_x_m` | 손상의 3D 대표점 X 좌표. EPSG:5186 동쪽 방향 좌표(m). |
| `world_center_y_m` | 손상의 3D 대표점 Y 좌표. EPSG:5186 북쪽 방향 좌표(m). |
| `world_center_z_m` | 손상의 3D 대표점 높이(m). Z축이 위쪽인 좌표계입니다. |
| `length_px` | 균열 예측 영역의 최소면적 외접사각형 장축 길이(px). |
| `length_m` | 국소 표면 축척으로 환산한 균열 추정 길이(m). |
| `width_px` | 균열 예측 영역의 최소면적 외접사각형 단축 폭(px). |
| `width_m` | 국소 표면 축척으로 환산한 균열 추정 폭(m). |
| `area_m2` | 박리·박락·누수·백태의 추정 면적(m²). 균열에는 사용하지 않습니다. |
| `area_px` | 해당 손상 영역을 구성하는 예측 픽셀 개수. |
| `member_name` | 연결된 3D 부재 또는 격자 메시의 이름. |
| `section_name` | 해당 부재가 속한 상위 구획 이름. |
| `grid_id` | 연결된 격자 번호·이름. 격자가 없는 부재에 연결되면 빈값입니다. |
| `grid_guid` | 해당 격자의 고유 식별자(GUID). |
| `DRI` | 해당 손상 예측 픽셀의 클래스 confidence 평균. 범위는 0~1입니다. |
| `source_image_path` | 원본 사진 경로. 해당 CSV 폴더 기준 상대경로입니다. |
| `overlay_path` | 손상 Overlay 경로. 해당 CSV 폴더 기준 상대경로입니다. |
| `mapping_status` | 3D 연결 상태. 현재 산출물에는 `mapped`만 저장합니다. |
| `measurement_status` | 물리치수 산출 상태. `local_surface_estimate`는 국소 표면 기반 추정 완료, 그 외 값은 산출 제외 사유입니다. |

## Release 다운로드 및 실행

필요한 파일은 아래 Release에서 제공합니다.

| Release | 파일 |
|---|---|
| [DINOv3 손상 탐지 모델](https://github.com/Jiwandukong/Dam_Damge_Module/releases/tag/dinov3-uav-demo-v1) | [dinov3_damage_demo.pt 다운로드](https://github.com/Jiwandukong/Dam_Damge_Module/releases/download/dinov3-uav-demo-v1/dinov3_damage_demo.pt) |
| [대청댐 격자화 3D 모델](https://github.com/Jiwandukong/Dam_Damge_Module/releases/tag/daecheong-dam-grid5m-v1) | [daecheongdam_grid5m.zip 다운로드](https://github.com/Jiwandukong/Dam_Damge_Module/releases/download/daecheong-dam-grid5m-v1/daecheongdam_grid5m.zip) — glTF, BIN 2개, 텍스처 18개 포함 |

Python 3.13 및 NVIDIA GPU 환경에서 `Demo/UAV_RGB/`로 이동한 뒤 실행합니다.

```bash
python -m pip install -r Code/requirements.txt
python Code/process.py
```

첫 실행 시 모델 파일이 없으면 Release에서 자동 다운로드하고 SHA256을 확인합니다. 저장 위치는 사용자 캐시 폴더 `~/.cache/dam_damage_module/`이며, 3D 모델 ZIP은 자동으로 압축 해제합니다. DINO 추론은 GPU 0번에서 실행하고 결과는 `Output/`에 저장합니다.

직접 다운로드할 경우 `daecheongdam_grid5m.zip`을 압축 해제한 뒤 파일 경로를 지정합니다.

```bash
python Code/process.py \
  --checkpoint /다운로드경로/dinov3_damage_demo.pt \
  --dam-model /압축해제경로/Daecheongdam/daecheongdam_regions_grid5m.gltf
```

DINOv3 모델과 파생 코드의 이용 조건은 [DINOV3_LICENSE.md](Code/DINOV3_LICENSE.md)를 확인합니다.
