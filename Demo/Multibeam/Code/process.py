"""Entry point: Data raw point clouds to classified final Output artifacts."""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from run import main


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"오류: {error}", file=sys.stderr)
        sys.exit(1)
