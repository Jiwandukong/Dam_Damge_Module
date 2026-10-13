"""Load the TLS algorithm dependencies from an external workspace cache."""
import os
import sys
from pathlib import Path

DEFAULT_WORK = Path.home()/'.cache/dam_damage_module/tls_spalling'
WORK = Path(os.environ.get('TLS_WORK_DIR', DEFAULT_WORK)).expanduser().resolve()
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(DEFAULT_WORK/'python'))
sys.path.insert(0, str(WORK/'python'))
sys.dont_write_bytecode = True
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', str(WORK/'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
