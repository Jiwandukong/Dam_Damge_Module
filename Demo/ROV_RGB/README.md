# ROV RGB 손상 탐지

ROV 원본 프레임 **35장**에 Sunflicker 전처리와 DeepLabV3+/ResNet101을 적용한 시연 결과입니다. **균열(CRC) 35건·박락(SPL) 45건**, 총 80건을 제공합니다. 모델은 Train+Validation 115장으로 학습했으며, 제공한 35장은 학습 데이터에서 선정한 시연용 입력입니다.

| 경로 | 내용 |
|---|---|
| `Data/` | 원본 RGB 프레임 JPG 35장. |
| `Code/` | 전처리·모델 추론·결과 생성·3D 웹뷰어. |
| `Output/Overlay/CRC/`, `SPL/` | 손상별 정사각형 Overlay PNG 80개. |
| [Output/Result/result.csv](Output/Result/result.csv) | 통합 결과 80행. |
| `Output/Result/CRC_result.csv`, `SPL_result.csv` | 클래스별 결과 35행·45행. |

CSV에는 손상 ID, 픽셀 polygon, 신뢰도, 이미지 경로, XYZ, **`grid_id`·`grid_guid`·`member_id`·`member_name`**을 담습니다. 이미지 경로는 `Output/Result` 기준 상대경로이며, CSV 인코딩은 UTF-8 BOM입니다. 실측 치수는 없으며 면적·길이는 픽셀 단위입니다.

3D 위치는 대청댐 모델의 **방류 여수로 수면 인접 격자 50건과 바로 아래 행 30건**에 배치한 합성 시연 좌표입니다. 유사 프레임은 같은 행에서 벽면 가로 방향으로 0.25m씩 이어 배치합니다. XYZ는 EPSG:5186 X/Y와 모델 표고 Z이며, 실제 ROV 촬영 위치를 나타내지 않습니다.

## 실행

Python 3.11~3.13에서 `Demo/ROV_RGB` 폴더를 기준으로 실행합니다.

```bash
bash Code/setup.sh
bash Code/run.sh
```

필요한 파일은 최초 실행 시 SHA256을 확인해 자동으로 다운로드합니다. [ROV Release](https://github.com/Jiwandukong/Dam_Damge_Module/releases/tag/rov-rgb-demo-v1)는 **모델 약 175MiB**와 원본 영상에서 계산한 35장의 Sunflicker RGB 캐시를 제공합니다. [댐 모델 Release](https://github.com/Jiwandukong/Dam_Damge_Module/releases/tag/daecheong-dam-grid5m-v1)를 함께 사용합니다. 기본 캐시 위치는 `~/.cache/dam_damage_module/rov_rgb/`이며 `ROV_RGB_WORK_DIR`로 변경할 수 있습니다.

제공 프레임은 캐시로 재현합니다. 다른 프레임의 전처리에는 `bash Code/run.sh --input <프레임_폴더> --video <원본.mp4>`를 사용합니다. 모델·3D 모델을 직접 지정하려면 `--checkpoint`, `--dam-model`을 사용합니다.

## 웹뷰어

```bash
bash Code/tools/web_viewer/run.sh --bind 127.0.0.1 --port 8788
```

출력되는 접속 링크로 부재·격자·손상 위치와 원본·Overlay를 확인하고 CSV를 다운로드할 수 있습니다.
