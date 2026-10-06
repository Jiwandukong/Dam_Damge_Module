# 01_RawData

`missions/`에는 공개 실행용 UAV 샘플 임무가, `geometry/`에는 좌표 매핑용 OBJ의 로컬 배치 위치가 있습니다.

- 실제 계산 입력: JPG + `Timestamp.MRK` + 지리참조 OBJ
- 보존용 원자료: PPK NAV/OBS/RAW
- 영상은 탐지 전에 resize하지 않습니다.
- OBJ는 대용량이므로 일반 Git 파일에서 제외하고 [3D 모델 Release](https://github.com/Jiwandukong/UAV_RGB_Damage_Module/releases/tag/daecheong-dam-geometry-v1)에서 제공합니다. `manifests/assets.yaml`의 크기, SHA-256, vertex/face 수, bounds를 실행 전에 검증합니다.

`geometry/daecheong_dam_epsg5186_zup.obj`는 현재 개발 환경에서 원본 `dam - Cloud.obj`를 가리키는 로컬 심볼릭 링크입니다. 제3자 환경에서는 Release 파일을 같은 위치에 내려받아야 합니다.
