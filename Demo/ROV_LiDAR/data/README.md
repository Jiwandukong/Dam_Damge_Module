# 입력 데이터 다운로드

입력 파일 3개는 [ROV LiDAR Release](https://github.com/Jiwandukong/Dam_Damge_Module/releases/tag/rov-lidar-demo-v1)의 [rov_lidar_data.zip](https://github.com/Jiwandukong/Dam_Damge_Module/releases/download/rov-lidar-demo-v1/rov_lidar_data.zip)에서 제공합니다. ZIP은 약 147MB이며, 압축 해제 후 약 881MB입니다.

| 파일 | 크기(bytes) | 용도 |
|---|---:|---|
| `01_EYAS_translated.e57` | 322,881,536 | 원본 보관 |
| `01_EYAS_translated_segmented.ply` | 447,191,845 | 세굴 분석 입력 |
| `01_EYAS_translated_segmented_slab_zone.ply` | 110,700,616 | 슬래브 함몰 분석 입력 |

`ROV_LiDAR` 폴더에서 실행하면 입력 파일을 이 `data/` 폴더에 복원합니다. Python 표준 라이브러리만 사용하며 ZIP과 각 입력 파일의 SHA-256을 확인합니다.

```bash
python3 -B code/download_assets.py
```

직접 다운로드할 때는 ZIP 내부의 `data/` 폴더를 `ROV_LiDAR` 아래에 압축 해제합니다. 저장소에서 제공하는 최종 CSV·LAS·PNG를 확인하는 데는 입력 다운로드가 필요하지 않습니다.

ZIP SHA-256: `f2ff79c5ff14a4ffac06ef75fa047dc92f4d9da2d4c5cb6b1da15ea4dab2eb7c`
