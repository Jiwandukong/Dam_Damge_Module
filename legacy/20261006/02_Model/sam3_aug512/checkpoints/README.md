# Local checkpoint location

Place the separately distributed `checkpoint_final.pt` here, then run:

```bash
python 03_Processing/scripts/verify_checkpoint.py --strict-load
```

The `.pt` file is ignored by Git. It contains trusted PyTorch pickle data and must only be loaded after size/SHA identity verification.
