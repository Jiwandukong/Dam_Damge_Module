"""Retain the supplied slab ROI and its low observations for depression analysis."""
import sys
sys.dont_write_bytecode = True
from runtime import ROOT, WORK
from pathlib import Path
from copy import deepcopy
import argparse
import json
import os
import tempfile

import laspy
import numpy as np
from pyproj import CRS
from point_cloud_io import prepare_las


def prepare(source, output, upper_z, work):
    if not np.isfinite(upper_z):
        raise ValueError('The slab upper Z limit must be finite.')
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    count = total = 0
    with tempfile.TemporaryDirectory(prefix='slab-input-zone-', dir=work) as directory:
        staging = Path(directory)
        input_las = prepare_las(source, staging)
        destination = staging/'slab_extracted.las'
        with laspy.open(input_las) as reader:
            header = deepcopy(reader.header)
            header.add_crs(CRS.from_epsg(5186))
            with laspy.open(destination, mode='w', header=header) as writer:
                for points in reader.chunk_iterator(500_000):
                    total += len(points)
                    # Preserve the complete supplied XY region. Remove only
                    # upper structures; do not threshold low points or normals.
                    selected = points[np.asarray(points.z) <= upper_z]
                    if len(selected):
                        writer.write_points(selected)
                        count += len(selected)
        if not count:
            raise ValueError('No observations remain below the slab upper Z limit.')
        with laspy.open(destination) as reader:
            bounds = [reader.header.mins.tolist(), reader.header.maxs.tolist()]
        os.replace(destination, output/'slab_extracted.las')
    summary = {
        'selection_method': 'supplied_input_xy_region_with_upper_z_cap',
        'source': str(source.resolve()), 'full_points': total,
        'filled_zone_points': count, 'upper_z_m': upper_z,
        'lower_z_limit_m': None, 'bounds_xyz': bounds,
    }
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'COMPLETE: retained {count:,}/{total:,} original ROI observations; upper Z {upper_z:g} m; no lower depth limit', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/01_EYAS_translated_segmented_slab_zone.ply')
    parser.add_argument('--output-dir', type=Path, default=WORK/'analysis/slab_depression/slab_extraction')
    # Survey-specific cap: the retained original slab reached Z=42.707382 m.
    parser.add_argument('--upper-z', type=float, default=42.708)
    parser.add_argument('--work-dir', type=Path, default=WORK)
    args = parser.parse_args()
    work = args.work_dir.expanduser().resolve()
    if work == ROOT or ROOT in work.parents:
        parser.error('--work-dir must be outside ROV_LiDAR')
    prepare(args.input, args.output_dir, args.upper_z, work)


if __name__ == '__main__':
    main()
