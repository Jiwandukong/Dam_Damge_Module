"""Convert a CUBE surface TIFF to LAS, one point per finite cell; no Fill or interpolation."""
from pathlib import Path
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import CRS
import laspy

def export_las(input_tif, output_las):
    path=Path(input_tif)
    # BEGIN ORIGINAL ANALYSIS
    target = Path(output_las)
    temporary = target.with_suffix('.las.tmp')
    count = 0
    with rasterio.open(path) as src:
        assert src.transform.b == src.transform.d == 0
        header = laspy.LasHeader(point_format=6, version='1.4')
        header.scales = np.array([1e-6, 1e-6, 1e-6])
        header.offsets = np.array([src.transform.c, src.transform.f, 0.0])
        header.generating_software = 'CUBE TIFF to LAS'
        header.add_crs(CRS.from_wkt(src.crs.to_wkt()))
        header.add_extra_dim(laspy.ExtraBytesParams(name='grid_row', type=np.int32, description='Source CUBE raster row'))
        header.add_extra_dim(laspy.ExtraBytesParams(name='grid_col', type=np.int32, description='Source CUBE raster column'))
        header.add_extra_dim(laspy.ExtraBytesParams(name='cube_z', type=np.float32, description='Exact original float32 CUBE Z'))
        with laspy.open(temporary, mode='w', header=header) as writer:
            for row_start in range(0, src.height, 128):
                data = src.read(1, window=Window(0, row_start, src.width, min(128, src.height-row_start)), masked=True).filled(np.nan)
                row, col = np.where(np.isfinite(data))
                if not len(row):
                    continue
                points = laspy.ScaleAwarePointRecord.zeros(len(row), header=header)
                points.x = src.transform.c + (col + 0.5)*src.transform.a
                points.y = src.transform.f + (row + row_start + 0.5)*src.transform.e
                points.z = data[row, col]
                points.grid_row = (row + row_start).astype(np.int32)
                points.grid_col = col.astype(np.int32)
                points.cube_z = data[row, col]
                points.return_number[:] = 1
                points.number_of_returns[:] = 1
                points.synthetic[:] = 1
                writer.write_points(points)
                count += len(row)
        # Full readback verifies one point per finite grid cell and preserved float32 Z.
        source = src.read(1, masked=True).filled(np.nan)
        assert count == int(np.isfinite(source).sum())
        maxima = np.zeros(3)
        previous_id = -1
        read_count = 0
        with laspy.open(temporary) as reader:
            assert reader.header.point_count == count
            assert reader.header.parse_crs().equals(CRS.from_wkt(src.crs.to_wkt()))
            for points in reader.chunk_iterator(500000):
                rr = np.asarray(points.grid_row).astype(np.int64)
                cc = np.asarray(points.grid_col).astype(np.int64)
                ids = rr * src.width + cc
                assert ids[0] > previous_id and bool(np.all(np.diff(ids) > 0))
                previous_id = int(ids[-1])
                assert np.array_equal(np.asarray(points.cube_z), source[rr, cc])
                expected = [src.transform.c+(cc+.5)*src.transform.a, src.transform.f+(rr+.5)*src.transform.e, source[rr,cc]]
                for axis, actual in enumerate([points.x, points.y, points.z]):
                    maxima[axis] = max(maxima[axis], float(np.max(np.abs(np.asarray(actual)-expected[axis]))))
                read_count += len(points)
        assert read_count == count and bool(np.all(maxima <= 5.1e-7))
        os.replace(temporary, target)
        record = {'source_cube_tiff':str(path), 'output_las':str(target), 'method':path.stem.split('_order1a_')[0].rsplit('_',1)[1], 'resolution_m':float(path.stem.split('_res')[1]), 'point_count':count, 'source_rows':src.height, 'source_columns':src.width, 'coordinate_location':'pixel center', 'las_xyz_scale_m':1e-6, 'max_x_error_m':maxima[0], 'max_y_error_m':maxima[1], 'max_z_error_m':maxima[2], 'cube_z_float32_exact':True, 'source_crs_preserved':True, 'bytes':target.stat().st_size}
        print('LAS complete:',target.name,count,'points; full readback verified',flush=True)
    # END ORIGINAL ANALYSIS
    return record


