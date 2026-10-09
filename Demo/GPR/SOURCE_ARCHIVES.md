# Source archive provenance and normalization record

## Source archives

| File | SHA-256 |
|---|---|
| `drive-download-20261008T090141Z-1-001.zip` | `a1f66fcbdca9535afc68f95229a677fe324663d6d4eb863d0cdc83d796e47d7a` |
| `Daechung_GPR.zip` | `8d1a6833d0622c9f8a7c1b14266c5c9dea063f3423c1c078b9dc24a46ca56fca` |

Both ZIP archives passed `unzip -t` integrity checks.

## Included

- The eight project-specific Python scripts originally under `new/`
- All 48 DZT/DZX scan pairs from `Daechung_GPR.zip`
- A display-only Output: 48 enlarged-marker Overlay PNGs and one 126-row CSV
- Original detection pixel positions, times, counts and pixel-aligned B-scan backgrounds
- GPRPy's MIT license and the DZT parsing behavior required by the pipeline

## Excluded

- Nested `.git` directories and histories
- `__pycache__`, build products, and package metadata
- RGPR, which is not imported by the Python pipeline
- GPR-Object-Detection, which is an unrelated external repository copy
- YOLO experiments, pretrained weights, incomplete training runs, and
  hard-coded local dataset paths; they do not produce the delivered
  rule-based results
- The broken exploratory `GPR_visualization.ipynb`

## Compatibility fixes that do not change detection numerics

1. `detect_anomalies_rulebased.process()` previously referenced the local
   `main()` variable `args` and could not run as a normal CLI. The same values
   are now explicit function arguments.
2. The GPRPy 1.0.14 DZT header/payload reader behavior was embedded in
   `gpr_gssi_io.py`. This removes unrelated Pmw/pyevtk GUI dependencies while
   producing the same 512×N arrays, `dt`, `dx`, and total distances.
3. The two duplicate frequency-preset tables were aligned to the values that
   produced the delivered rule-based outputs. The actual detector presets and
   thresholds were not changed.
4. `export_result_images.py` restores the missing producer for the delivered
   pixel-aligned grayscale images and mapping CSV.
5. `process.py` orchestrates the existing stages without changing their order.
6. Absolute Windows examples were replaced with repository-relative examples.

## Reference-output decisions

- The supplied source contained a special case that enlarged red crosses for
  1600MHz LINE_001–003, but the delivered overlays use the standard 5px
  half-width and 2px line width for every frequency. The delivered images were
  treated as the numerical baseline. In this display draft, the red crosses
  are enlarged to a 10px minimum half-size and 3px minimum line width, scaled
  proportionally for horizontal B-scans with width / height >= 4. The marker
  scale is max(1, width / 1,000). This includes LINE_001-003 at all four
  frequencies (12 images); markers on the other 36 images remain unchanged.
- The original four approximate 3D tie points and original `transform.json`
  were missing. The fixed transform was recovered from all delivered endpoint
  pairs, with a maximum absolute fit error of `3.5e-10m`. It is explicitly
  marked as recovered rather than original calibration.

## Initial original-package reproduction result (historical)

An independent full run over all 48 DZT/DZX pairs produced:

- Identical 48 pixel-aligned grayscale B-scan images
- Identical 48 rule-based anomaly Overlay images
- Identical candidate count and order: 126 total
- Matching CSV numeric values within `1e-8`; the 3D endpoint fit itself differs
  by no more than `3.5e-10m`

Matplotlib presentation figures are excluded from pixel equality because
fonts, layout, and rasterization can vary across Matplotlib versions. Their
underlying CSV values are included in the regression comparison.

## Display-draft changes (2026-10-09)

- Final Output contains only `Overlay/ANM/` and `Result/ANM_result.csv`.
- Intermediate images/CSVs are written to a separate processing work directory.
- Detection pixel coordinates, times, counts and IDs are preserved; standard display
  field names, portable relative paths and UTF-8 BOM encoding are added. Coordinates
  are subsequently revised by the unit review below.
- No member/grid association is invented; those fields remain empty.
- Markers are rendered from the unmarked pixel-aligned B-scans, so enlarging them
  does not change detection settings, candidate positions, or B-scan geometry.
- The separate LICENSES folder was removed from the draft. The GPRPy notice
  is retained in the DZT reader's source header.

## Coordinate unit rebuild (2026-10-09)

- The recovered legacy scale 0.082692956 shrank a 26.67m line to 2.21m.
  The display draft now uses scale 1 with the recovered rotation and plane-origin
  translation retained as assumptions. Absolute field placement remains unverified.
- DZT scans-per-metre determines metric trace spacing. N traces contain N-1
  intervals, correcting the previous endpoint inflation of one trace interval.
  Candidate `x_m` and world XYZ are recalculated from unchanged candidate pixel centers.
- `mapping_status=line_surface_projection_unit_review` distinguishes the review
  coordinates. Member/grid fields remain empty and time is not converted to depth.
- Tie-point calibration requires explicit `--plane-y1`, uses a rigid metre fit,
  and rejects incompatible dimensions instead of deriving a shrinkage scale.
- All 48 DZT spacings agree with independently normalized DZX `unitsPerScan`.
  The 2.6GHz LINE_001 DZX endpoint distance has inconsistent display units;
  that endpoint field is not used for the metric calculation.
- A fresh full pipeline over 48 scan pairs reproduced all 126 pixel centers and
  times and all 48 final Overlay images. The resulting CSV agrees with the export
  from existing detections, allowing only relative source-path differences.
- `Data/Calibration/` retains the legacy transform/positions and current unit audit.
  `MappingReview/` shows measured survey lines and a previous-position comparison.

## Start/end anchor review (2026-10-09)

- The requested starting region is present as `NOF_R_0157` in the model.
- `MappingReview/anchor_proposal.json` contains an unconfirmed guide ending at
  the first outdoor ancillary equipment boundary along the current line direction.
- The web viewer can pick exact triangle positions for start/end and copy their
  world XYZ. Picks remain in the browser and do not overwrite the result CSV.
- `Code/fit_start_end.py` prepares a unit-scale review transform from explicitly
  selected endpoints and rejects a distance inconsistent with the reference scan.

## Horizontal Z constraint (2026-10-09)

- The former along-line rotation component changed Z by about 0.338m over the
  400MHz LINE_001 scan. Its world Z component is now exactly zero.
- Rotation is derived from the area-weighted normals of NOF_R_0157~0163.
  The horizontal axis follows the dam face; scale remains 1 and the current
  origin remains provisional. Individual triangle tangent deviations are at
  most 0.075 degrees because the mesh face is not perfectly planar.
- All 12 horizontal scans (LINE_001~003 at four frequencies) maintain identical
  endpoint Z. Vertical scan height changes and survey spacing are preserved.
- Candidate XYZ, the unit audit, proximity report and viewer were regenerated.
  Candidate IDs, pixels, times, detection counts and all 48 Overlay PNGs are preserved.
- Start/end guide points have the same Z. User end picks are placed on the
  chosen mesh's intersection with the start-height plane; raw clicks are retained.
- Transform loading and endpoint/tie-point fitting enforce a horizontal axis,
  preventing a later rerun from restoring the old tilt.

## Applied user model anchors (2026-10-09)

- The user supplied a NOF_R_0157 start and NOF_R_0162 end, both at
  world Z=66.83210681095052m. The complete record is selected_anchors.json.
- Their distance is 26.407940163486508m versus the 400MHz LINE_001 measured
  length of 26.663902671258036m. A horizontal-only interpolation ratio of
  0.9904004109628168 fits both endpoints exactly. Physical DZT x_m is unchanged.
- The same ratio adjusts all horizontal along-scan positions. Frequency/line
  coverage differences remain; vertical station offsets, vertical scan lengths
  and cross-line spacing retain their metre values. No global shrink scale is used.
- All 48 survey lines and 126 candidate XYZ positions were regenerated.
  All horizontal endpoint Z differences are zero; the reference endpoints
  exactly reproduce the supplied points. The selected heading differs from
  the prior mesh-derived heading by approximately 0.00623 degrees.
- All candidates were assigned to nearest grid triangles NOF_R_0157~0162
  and their parent member. Maximum grid distance is 0.015832m; positions are
  not snapped to the mesh. Grid records are reused only for matching transforms,
  candidate coordinates and model geometry.
- The CSV status is line_surface_projection_anchor_fit. Input distances,
  detection IDs/pixels/times/counts and all 48 Overlay PNGs are preserved.
- Current model placement is based on user-selected model points; field CRS,
  survey accuracy and internal depth remain unverified.
