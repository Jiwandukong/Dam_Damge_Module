# Split checkpoint release

`checkpoint_final.pt` is about 10.08 GB and cannot be stored as one normal Git object or one GitHub Release asset. The release workflow therefore uses verified parts below 2 GB.

Maintainer packaging:

```bash
python 03_Processing/scripts/package_checkpoint.py \
  --download-base-url https://github.com/Jiwandukong/UAV_RGB_Damage_Module/releases/download/sam3-aug512-v1
```

Upload every file from `release/dist/`, `release_manifest.json`, and `../third_party/SAM_LICENSE` to the `sam3-aug512-v1` release. Commit only the manifest and license, never `dist/`. Redistribution of the derived checkpoint remains subject to that SAM License.

Consumer download:

```bash
python 03_Processing/scripts/download_checkpoint.py
```

Each part and the reconstructed file are SHA-256 checked before use.
If the GitHub repository is renamed, update `download_base_url` in the manifest or pass `--base-url`.

If the GitHub Release download or part reconstruction does not work, request the
original approximately 10 GB `checkpoint_final.pt` directly from the repository
maintainer and place it at
`02_Model/sam3_aug512/checkpoints/checkpoint_final.pt`.
