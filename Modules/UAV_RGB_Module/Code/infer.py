"""Run the existing UAV RGB production pipeline from its sensor folder."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts/run_pipeline.py"), run_name="__main__")
