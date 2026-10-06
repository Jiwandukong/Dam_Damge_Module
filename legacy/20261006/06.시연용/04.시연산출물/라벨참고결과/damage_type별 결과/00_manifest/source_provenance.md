# 최종 MODEL / curated LABEL 출처

- MODEL source: `/data/disks/hdd03/gookhyun/Daechung_SAM3/github_output_staging/demo_split_v1/positive_only_v1/모델추론결과/damage_type별 결과`. 기존 positive-only SHA256 manifest와 전수 일치. CRC 8,559 / DLM 5,450 / SPL 7,444, 총 21,453건. MODEL review/bulk 상태는 읽거나 적용하지 않았습니다.
- LABEL 원본: `/data/disks/hdd03/gookhyun/Daechung_SAM3/demo20_stagec_dlm_spl_v1/06_reviewed_dlm_spl/final_v1`. 629개 shape를 기존 `source_annotation_id` / `shape_index` / `damage_id`에 629/629 대응 확인했습니다. ID 신규 생성 없음.
- LABEL mixed damage_results: `/data/disks/hdd03/gookhyun/Daechung_SAM3/github_work/UAV_RGB_Damage_Module_demo_split_v1/06.시연용/04.시연산출물/라벨참고결과`.
- LABEL positive-only source: `/data/disks/hdd03/gookhyun/Daechung_SAM3/github_output_staging/demo_split_v1/positive_only_v1/라벨참고결과/damage_type별 결과`. 기존 컬럼·정량값·ID·tile 연결을 유지하여 KEEP 행만 필터했습니다.
- 검수 동결: `/data/disks/hdd03/gookhyun/Daechung_SAM3/github_output_staging/demo_split_v1/label_review_freeze_v1`. DB SHA256: `38c04c88d0217ce620cf64cefb5a708309bf10cd51864cc6e3eb557468693fd8`.
- 원본 629건 / spatial valid 588건 / XYZ missing 41건. Spatial KEEP 231건, EXCLUDE 357건, UNREVIEWED 0건, MAYBE 0건. 최종 CRC 78 / DLM 57 / SPL 96.
- LABEL KEEP-only overlay: Pillow 12.2.0; full-image raster then tile crop; floor(v+0.5); CRC width=1; DLM/SPL polygon fill; alpha=0.5. 캐시 mask를 이용한 독립 기대 영상과 모든 픽셀 일치. KEEP과 제외 도형의 겹친 픽셀은 KEEP 근거로만 표시하고, 제외 도형만 있는 픽셀은 원본 RGB와 같습니다.
- 원본 LABEL/JSON/정량 결과 수정 및 자동 threshold 적용 없음. DB·이벤트·백업·임시 스크립트·검수 화면은 GitHub 게시 대상에서 제외했습니다.
- 두 게시 결과 폴더의 `.gitattributes`는 `*.csv -text`로 저장소 상위의 CSV 줄바꿈 변환을 재정의합니다. 감사한 원본 CSV 바이트를 Git blob에도 그대로 보존하기 위한 범위 한정 설정입니다.
