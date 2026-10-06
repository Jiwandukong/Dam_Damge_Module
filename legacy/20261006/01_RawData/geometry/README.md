# Geometry asset

Expected local file: `daecheong_dam_epsg5186_zup.obj`

- Original name: `dam - Cloud.obj`
- Size: `184,020,785` bytes
- SHA-256: `a7566c8d8f70de91db18c70ae5f404b0f7885e679a1027fd3e1936afd2d1e470`
- External coordinate contract: EPSG:5186 X/Y, metres, Z-up
- Bounds: `[242986.140625, 431066.09375, 32.6238327]` to `[243138.875, 431283.3125, 122.66383362]`

OBJ does not carry an embedded CRS. The contract above is project metadata and must remain with the file. The coordinate-less/Y-up GLB is intentionally unsupported.

Download the verified OBJ from the
[`daecheong-dam-geometry-v1` release](https://github.com/Jiwandukong/UAV_RGB_Damage_Module/releases/tag/daecheong-dam-geometry-v1):

```bash
curl --fail --location \
  --output 01_RawData/geometry/daecheong_dam_epsg5186_zup.obj \
  https://github.com/Jiwandukong/UAV_RGB_Damage_Module/releases/download/daecheong-dam-geometry-v1/daecheong_dam_epsg5186_zup.obj

python 03_Processing/scripts/verify_assets.py
```
