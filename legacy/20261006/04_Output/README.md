# 04_Output

- `runs/`: 로컬 전체 실행 결과. Git에서 제외됩니다.
- `examples/`: 검증 후 선별한 소형 산출물 예시만 보관합니다.

각 run은 다음을 포함합니다.

```text
run/
├── masks/<image>/{CRC,DLM,SPL,multilabel_rgb,preview}.png
├── overlays/<image>_overlay.png
├── tables/
│   ├── damage_results.csv
│   ├── damage_results.xlsx
│   └── per_image/*.csv
└── metadata/{run_metadata.json,per_image/*.json}
```
