# Publication checklist

이 폴더는 실행·검증까지 완료된 Git 후보이지만, 다음 항목이 결정되기 전에는 public push하지 않습니다.

1. `01_RawData/missions/`의 JPG/MRK/PPK 재배포와 대청댐 정밀 좌표 공개 승인을 확인합니다.
2. JPG XMP의 `CameraSerialNumber`, `DroneSerialNumber`, NTRIP host/mount point 공개 범위를 확인합니다. 삭제할 경우 원본 SHA manifest도 다시 작성하고 pose 필수 항목은 보존합니다.
3. 연구실 작성 코드는 원격 저장소에서 선택한 MIT License를 사용합니다. 별도로 UAV 데이터용 `DATA_LICENSE` 또는 재배포 허가를 확인합니다.
4. 파생 checkpoint 배포 권한을 확인하고 Release에 `SAM_LICENSE`를 함께 제공합니다.
5. OBJ 배포가 허용되면 `assets.yaml`과 동일한 파일을 별도 제공합니다. 허용되지 않으면 사용자 제공 필수 자산으로 유지합니다.
6. 공개 직전에 `python 03_Processing/scripts/verify_assets.py`, `python 03_Processing/scripts/verify_checkpoint.py --strict-load`, `pytest`를 다시 실행합니다.

현재 저장소는 commit·remote·push를 만들지 않은 상태이므로 관리자가 공개 범위를 먼저 결정할 수 있습니다.
