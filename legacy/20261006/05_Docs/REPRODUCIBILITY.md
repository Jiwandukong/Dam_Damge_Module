# Reproducibility checklist

1. 저장소 루트에서 `(cd 01_RawData && sha256sum -c manifests/SHA256SUMS)`로 샘플 파일을 확인합니다.
2. OBJ size/SHA와 외부 EPSG:5186/Z-up 계약을 확인합니다.
3. 체크포인트 size/SHA를 확인한 뒤 pinned SAM3에서 strict load합니다.
4. `03_Processing/configs/daechung_aug512.yaml`을 실행 기록과 함께 보존합니다.
5. 원본 이미지를 resize하지 않았는지 metadata의 `source_resized=false`를 확인합니다.
6. 비배수 영상은 오른쪽/아래 padding과 원본 크기 crop을 사용했는지 확인합니다.
7. Excel의 근사/유효/실패사유 열을 보존합니다.

`edge_policy=pad`는 실제 DJI 원본을 다루기 위한 production extension입니다. AUG512의 1024 영상 4분할 진단 프로토콜과 동일하다고 주장하지 않습니다.
