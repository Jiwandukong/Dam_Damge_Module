# 손상별 모델 추론 결과

원래 MODEL inference 전체 21,453건을 손상별로 분리한 결과입니다. MODEL Interactive Review Tool의 KEEP/EXCLUDE 및 bulk EXCLUDE는 적용하지 않았습니다. 기존 검증 완료 positive-only MODEL bundle을 그대로 사용했습니다.

| 손상 | 전체 instance | positive tile |
|---|---:|---:|
| [CRC](CRC/) | 8,559 | 1,732 |
| [DLM](DLM/) | 5,450 | 1,639 |
| [SPL](SPL/) | 7,444 | 1,697 |

해당 class 손상이 존재하는 타일만 포함하며 빈 타일은 0개입니다. 손상 instance는 모두 보존했습니다. CSV/XLSX의 이미지 경로는 각 class 폴더 기준 상대경로입니다. XLSX 시트는 `Damage_Details`입니다.
