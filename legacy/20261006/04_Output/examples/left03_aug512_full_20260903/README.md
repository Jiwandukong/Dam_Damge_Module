# Verified full sample result

Generated from all seven images in `DJI_202507021616_left03` with the verified AUG512 checkpoint and `daecheong_dam_epsg5186_zup.obj`.

- Images: 7
- SAM3 tiles: 616 (`88 × 7`)
- Damage rows: 718
- CRC: 110
- DLM: 338
- SPL: 270
- Valid OBJ center hits: 718/718
- Valid local GSD measurements: 718/718
- CRC total observed length: 18.01982464 m
- DLM total observed area: 1.2406479929 m²
- SPL total observed area: 1.0527978736 m²
- Mask output shape: 5280×3956 for every class/image
- Source resize: false
- Minimum retained component area: 8 pixels

The original full local run, including 35 masks and seven overlays at alpha 0.45, is preserved under `04_Output/runs/left03_aug512_simple_final` and is intentionally ignored by Git. This public example directory contains the unchanged 13-column CSV and its single-sheet Excel version, plus seven overlays re-rendered from that run's unchanged GPU masks at alpha 0.80. No CPU benchmark results are included here.

## 오버레이 사진 보기

아래 사진은 위 CSV·Excel과 같은 GPU 실행에서 생성한 결과입니다. 원본 사진 위에 균열(CRC)은 빨강, 박리(DLM)는 노랑, 박락(SPL)은 파랑으로 표시합니다.

사진을 선택하면 해당 PNG를 확인하거나 다운로드할 수 있습니다. 7장 모두 5280×3956 원본 크기이며, 이미지 크기와 손상 영역은 그대로 유지했습니다. 색 불투명도만 45%에서 80%로 높여 손상이 더 진하게 보이도록 다시 저장했습니다. 모델 추론이나 좌표·정량값 계산은 다시 수행하지 않았고 CSV·Excel은 변경하지 않았습니다.

### 손상은 3종인데 색이 더 많은 이유

같은 pixel에 여러 손상이 탐지되면 해당 색을 평균하여 표시합니다. 따라서 초록은 박리+박락, 주황은 균열+박리, 보라는 균열+박락, 갈색 계열은 세 종류가 모두 겹친 영역입니다. 원본 사진과 반투명하게 섞으므로 배경에 따라서도 색조가 달라집니다. 추가 손상 종류나 심각도를 뜻하는 색이 아닙니다.

겹친 영역의 손상 정보를 유지하기 위해 표시 우선순위로 한 종류를 숨기지 않았습니다. 원본의 손상 없는 pixel은 그대로 유지합니다.

### 같은 마스크로 표시 농도만 바꾸기

기존 손상 마스크가 있으면 모델·GPU 없이 오버레이만 다시 생성할 수 있습니다. `--output-dir`에는 아직 존재하지 않는 새 폴더를 지정합니다.

```bash
python 03_Processing/scripts/render_overlays.py \
  --images 01_RawData/missions/DJI_202507021616_left03/images \
  --masks-dir 04_Output/runs/left03_aug512_simple_final/masks \
  --output-dir 04_Output/runs/overlay_alpha080/overlays \
  --alpha 0.80
```

위 명령은 저장소 최상위 폴더에서 실행합니다. 참조한 `runs/`의 마스크는 로컬 실행 결과이므로, 새로 내려받은 저장소에서는 먼저 모델 실행으로 마스크를 생성해야 합니다.

### 사진 목록

- [DJI_20250702162019_0001_V](overlays/DJI_20250702162019_0001_V_overlay.png)
- [DJI_20250702162023_0005_V](overlays/DJI_20250702162023_0005_V_overlay.png)
- [DJI_20250702162024_0006_V](overlays/DJI_20250702162024_0006_V_overlay.png)
- [DJI_20250702162025_0007_V](overlays/DJI_20250702162025_0007_V_overlay.png)
- [DJI_20250702162031_0013_V](overlays/DJI_20250702162031_0013_V_overlay.png)
- [DJI_20250702162032_0014_V](overlays/DJI_20250702162032_0014_V_overlay.png)
- [DJI_20250702162033_0015_V](overlays/DJI_20250702162033_0015_V_overlay.png)

각 파일의 SHA-256은 `SHA256SUMS`에 기록되어 있습니다. 이미지와 표의 대응 기준은 원본 사진 파일명입니다.

CRC/DLM/SPL are independent multilabel masks; overlapping DLM/SPL pixels must not be summed as unique damaged surface area. Rows are per-image observations, not cross-image-deduplicated physical defects.
