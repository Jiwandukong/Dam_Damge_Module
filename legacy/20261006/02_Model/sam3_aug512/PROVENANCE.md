# Public Provenance

## Experiment

- Experiment ID: `STAGE_B_REBUILD_512_QUERY_BALANCED_TRAIN1275_V1`
- Training date: 2026-08-17
- Final checkpoint SHA256: `a2749dba62207575afac9ed42f923d7cdfd7e2d2f0ffb2210a103e91657d985d`
- Final checkpoint model keys: 1,134
- Final optimizer updates: 1,275

## Lineage

```text
Official facebook/sam3
  -> AIHub + HRCDS + S2DS public-damage Stage A
  -> Daechung Train-85 512-tiled query-balanced Stage B
```

The source Stage-A experiment was `T1_PUBLIC1_UAV5_GUARDED_V4`. Its checkpoint SHA256 was `63fca826cb4a941271080f7dc475d5dd3195c0c539bf8c65989a250e7f0f93a0`. Handoff was model weights only with strict loading; optimizer, scheduler, scaler, RNG, epoch, and step state were reset.

## Stage-B contract

- 85 source images at 1024×1024
- Four non-overlapping 512×512 tiles per image
- 340 tiles: 174 non-empty and 166 fully empty
- 1,275 immutable queries and optimizer updates
- CRC/DLM/SPL: 425 queries each
- Positive/negative per class: 298/127
- Batch size and gradient accumulation: 1/1
- No skipped, failed, NaN/Inf, OOM, or retried query

## Diagnostic evaluation

The final evaluation used 22 source images at 1024×1024. Four 512×512 tile predictions per image were restored and class-wise OR-stitched into original coordinates before whole-image pixel metrics were calculated. This result is identified as `512_TILED_TEST22_DIAGNOSTIC_NOT_OFFICIAL_WHOLE_IMAGE` and does not replace the project's official whole-image evaluation.

The exact experiment-time SAM-3 commit was not preserved. Current deployment compatibility is a separate validation statement.
