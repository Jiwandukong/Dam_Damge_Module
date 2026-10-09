# TIR 수동 손상 판독 자료

TIR/RGB 영상 80쌍을 수동 판독해 작성한 손상 polygon 64개입니다.

- `rawdata/`: 원본 TIR·RGB 영상. 시간대는 `daybreak`, `morning`, `afternoon`, `evening`입니다.
- `output/overlay/`: 손상 표시 PNG 80장.
- [output/tir_damage_results.csv](output/tir_damage_results.csv): 손상 64개의 시간대·XYZ·부재·grid·영상 경로를 담은 최종 결과입니다. 경로는 이 README가 있는 폴더 기준입니다.

XYZ는 촬영 위치 16곳의 대표점이며, 같은 위치의 네 시간대가 공유합니다.

수동 판독·라벨링·3D 위치 매핑 등 데이터의 manual process가 많아 전체 처리 과정의 자동화가 불가능하므로 Code를 포함하지 않습니다.
