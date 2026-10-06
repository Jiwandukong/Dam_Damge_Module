# Output specification

`tables/damage_results.xlsx`의 첫 sheet `Damage_Details`는 전달받은 산출물 예시의 핵심 열을 같은 순서로 제공합니다.

| 열 | 의미 |
|---|---|
| `image` | 원본 영상 파일명 |
| `damage_id` | run 전체에서 순차 부여한 손상 ID |
| `damage_type` | `CRC`, `DLM`, `SPL` |
| `damage_name_ko` | 균열, 박리, 박락 |
| `pixel_nodes_json` | 원본 영상 좌표의 손상 경계 표본 `[[x,y],...]` |
| `world_center_x_m` | EPSG:5186 X |
| `world_center_y_m` | EPSG:5186 Y |
| `world_center_z_m` | OBJ Z-up 교차점 |
| `length_px`, `length_m` | CRC 회전 사각형 장축과 환산값 |
| `width_px`, `width_m` | CRC 회전 사각형 단축과 환산값 |
| `area_m2` | DLM/SPL mask의 국부 표면 면적 |

사용자용 CSV와 Excel에는 위 13개 열만 기록합니다. Excel도 `Damage_Details` 단일 sheet만 생성하며, 별도 품질 열이나 요약·metadata sheet를 추가하지 않습니다. 빈 값은 해당 손상 유형에 적용되지 않는 항목이며 0으로 간주하면 안 됩니다.

PNG mask 좌표는 언제나 원본 `5280×3956`입니다. `multilabel_rgb.png`는 R=CRC, G=DLM, B=SPL이며 class 중첩을 보존합니다.

CRC/DLM/SPL은 서로 독립적인 multilabel mask입니다. 같은 픽셀이 두 class에 동시에 포함될 수 있으므로 DLM과 SPL 면적 합계를 고유 손상 총면적으로 해석하면 중복 계상될 수 있습니다.

각 행은 한 영상에서 검출된 **손상 관측 건**입니다. 서로 다른 영상에 같은 실제 손상이 보일 수 있으며 현재 버전은 영상 간 3D association/deduplication을 수행하지 않습니다. 따라서 여러 영상의 길이·면적 합계도 고유 손상 inventory로 사용하면 안 됩니다.
