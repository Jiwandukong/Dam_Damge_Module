"""Keep dependencies and intermediate analysis outside the distributed module."""
from pathlib import Path
import os
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('ROV_LIDAR_WORK_DIR', Path.home()/'.cache/dam_damage_module/rov_lidar')).expanduser().resolve()
if WORK == ROOT or ROOT in WORK.parents:
    raise ValueError('ROV_LIDAR_WORK_DIR must be outside ROV_LiDAR.')
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WORK/'python'))
os.environ['PYTHONPATH'] = str(WORK/'python') + os.pathsep + os.environ.get('PYTHONPATH', '')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('NUMBA_CACHE_DIR', str(WORK/'numba'))
os.environ.setdefault('MPLCONFIGDIR', str(WORK/'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '4')
