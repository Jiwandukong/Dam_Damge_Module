# 사람 검수 후 KEEP된 손상별 라벨링 결과

Authoritative human LABEL 629건(CRC 312 / DLM 86 / SPL 231) 중 XYZ valid 588건을 사람이 직접 공간분포 검수한 시연용 결과입니다. **status == KEEP AND XYZ valid**인 231건만 게시합니다.

EXCLUDE 357건은 환경부 시연용 표시에서 제외한 것이며 원본 GT 삭제가 아닙니다. 원본 LABEL 629건은 보존합니다. XYZ missing 41건은 이번 curated subset에서 제외했습니다. XYZ valid 미검수 및 MAYBE는 모두 0건입니다. LABEL에 자동 area/length threshold는 적용하지 않았습니다.

| 손상 | 최종 KEEP | positive tile |
|---|---:|---:|
| [CRC](CRC/) | 78 | 83 |
| [DLM](DLM/) | 57 | 77 |
| [SPL](SPL/) | 96 | 167 |

KEEP 손상이 실제로 존재하는 타일만 포함하며 빈 타일은 0개입니다. 각 class 오버레이는 KEEP geometry만 다시 그렸습니다. 좌표는 `floor(coordinate + 0.5)`, CRC는 width=1 line, DLM/SPL은 polygon fill, alpha는 0.5입니다. 모든 오버레이는 기존 검증된 개별 annotation mask와 픽셀 단위로 대조했습니다.

CSV/XLSX는 기존 16개 컬럼 순서와 `Damage_Details` 시트를 유지하며 경로는 각 class 폴더 기준 상대경로입니다. [게시 ID·제외 목록·검수 요약 및 출처](00_manifest/)의 제외 목록은 감사용이며 결과 손상에 포함되지 않습니다. MODEL과 LABEL damage_id는 독립된 ID 공간입니다.
