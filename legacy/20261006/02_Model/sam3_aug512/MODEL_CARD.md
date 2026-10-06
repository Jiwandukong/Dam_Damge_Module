# Model Card: Daechung Dam SAM-3 August 512 Query-Balanced

## Model details

- Model type: text-prompted concrete damage segmentation
- Base model: Official `facebook/sam3` image model
- Training date: 2026-08-17
- Classes/prompts: CRC/`crack`, DLM/`delamination`, SPL/`spalling`
- Training domain: public concrete-damage sources followed by Daechung Dam imagery
- Model weights: externally distributed; not included here
- Checkpoint SHA256: `a2749dba62207575afac9ed42f923d7cdfd7e2d2f0ffb2210a103e91657d985d`

The exact experiment-time SAM-3 git commit was not preserved. The release implementation was separately checked for compatibility with commit `46957e47805eaa273f4aa7bbbd25a88bca9108ce`.

## Training lineage

Official SAM-3 was adapted first on AIHub, HRCDS, and S2DS public-damage data. The resulting Stage-A model was transferred with `strict=True` model weights only; no optimizer, scheduler, scaler, RNG, epoch, or step state was inherited. Stage B used 85 Daechung images, not the July 1024 Stage-B checkpoint.

## Dataset construction and schedule

Each 1024×1024 image was split into four non-overlapping 512×512 tiles, for 340 tiles total. The fixed schedule had 1,275 queries and updates. Each class had 425 exposures: 298 positive and 127 trusted negative. The overall schedule had 894 positive and 381 negative queries.

The 512×512 value denotes the dataset tile. Training randomly resized tiles from 480 through 992 and padded them to the 1008×1008 SAM-3 model tensor.

## Evaluation protocol

Twenty-two 1024×1024 Test images were each split into four 512×512 non-overlapping tiles. Each tile/class pair was inferred independently. Candidate masks were filtered at score 0.5, thresholded at mask probability 0.5, OR-unioned within the tile, and restored to a 1024×1024 class canvas. Pixel TP/FP/FN/TN were calculated against whole-image GT only after stitching.

### 512-tiled Test-22 diagnostic

| Class | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| CRC | 0.2524589153 | 0.1372742971 | 0.1778453519 | 0.0976016783 |
| DLM | 0.0724842277 | 0.5437561168 | 0.1279167862 | 0.0683285792 |
| SPL | 0.2979792736 | 0.4339251266 | 0.3533267295 | 0.2145700279 |

- Macro F1: 0.2196962892
- Micro F1: 0.2583455161

This is a tiled diagnostic, not a replacement for the project's official whole-image Locked Test-22 evaluation.

## Intended use

- Research on concrete surface damage segmentation
- Reproduction of the validated 512 non-overlap tiling protocol
- Qualitative inspection and benchmark-oriented analysis with human review

## Out-of-scope use

- Safety certification or autonomous structural-condition decisions
- Claims of general performance across dams, materials, sensors, or environments
- Unvalidated overlap, arbitrary-edge padding, or silent resizing of the whole source image

## Limitations

CRC and DLM F1 are low in the diagnostic evaluation. The training and test domains are narrow, arbitrary image boundaries were not validated, probability calibration was not established, and the exact experiment-time upstream commit and package environment were not preserved.
