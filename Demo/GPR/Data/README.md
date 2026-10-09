# Data

`processed data/`에는 네 주파수별 GSSI DZT/DZX 12쌍, 총 96개 파일이
있습니다. 폴더명과 파일명은 기존 파이프라인의 주파수 인식 규칙을
보존하기 위해 유지했습니다.

`Calibration/transform.json`은 사용자가 모델에서 지정한 시작·끝점에 맞춘
현재 표출 변환입니다. 가로 Z를 고정하고 세로 측선 간격·길이를 유지합니다.
`selected_anchors.json`에 지정점을, `grid_mapping.json`에 부재·격자 매핑을 보존합니다.
이전 변환은 `transform_legacy.json`에 별도로 보관합니다. 자세한 출처와 계산 규칙은
`Calibration/README.md`를 확인하십시오.

원자료 공개 권한과 좌표계는 제공 자료에 명시되지 않았습니다.
