# 03_Processing

실행 코드의 책임은 명확히 분리되어 있습니다.

- `config.py`, `models.py`, `inference.py`: AUG512 계약 검증과 SAM3 추론
- `asset_contract.py`: `assets.yaml`에 고정된 OBJ size/SHA/topology/bounds 검증
- `camera_pose.py`, `mesh_ray.py`, `warp_ray.py`: XMP/MRK pose와 OBJ Ray 교차
- `surface_metrics.py`, `instances.py`: component geometry와 국부 GSD
- `reporting.py`: 마스크, 오버레이, CSV, JSON, Excel
- `pipeline.py`: 모델·메시를 한 번만 로드하는 배치 orchestration

기본 실행:

```bash
python 03_Processing/scripts/verify_assets.py
python 03_Processing/scripts/run_pipeline.py --profile
```

출력 폴더는 매 실행마다 새 폴더여야 합니다. 기존 결과와 새 결과가 섞이는 것을 코드가 거부합니다.
