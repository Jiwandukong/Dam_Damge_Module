"""Original scour CSV: id, area, depths, centroid XYZ and polygon node XYZ."""
import argparse
from pathlib import Path

def export_csv(args):
    # BEGIN ORIGINAL ANALYSIS
    import geopandas as gpd
    import pandas as pd
    import rasterio
    import numpy as np
    import os
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='Geometry column does not contain geometry')
    input_gpkg = str(args.gpkg.resolve())
    tif_dir = str(args.tif.resolve().parent)
    stem = os.path.splitext(os.path.basename(input_gpkg))[0].replace('_fillgap', '')
    tif_path = str(args.tif.resolve())
    output_csv = str(args.output.resolve())

    def sample_z_at_points(dataset, xy_list):
        """(x, y) 리스트에 대해 z값 샘플링"""
        nodata = dataset.nodata
        zs = []
        for val in dataset.sample(xy_list):
            z = float(val[0])
            if nodata is not None and np.isfinite(nodata) and np.isclose(z, nodata):
                zs.append(np.nan)
            else:
                zs.append(z)
        return zs
    gdf = gpd.read_file(input_gpkg)
    if 'scour_id' not in gdf.columns:
        gdf['scour_id'] = [f'scour_{i + 1}' for i in range(len(gdf))]
    with rasterio.open(tif_path) as src:
        nodes_xyz_list = []
        centroid_xyz_list = []
        for geom in gdf.geometry:
            if geom.is_empty:
                nodes_xyz_list.append([])
                centroid_xyz_list.append((np.nan, np.nan, np.nan))
                continue
            exterior_coords = list(geom.exterior.coords)
            zs = sample_z_at_points(src, exterior_coords)
            nodes_xyz = [(x, y, z if np.isfinite(z) else np.nan) for (x, y), z in zip(exterior_coords, zs)]
            nodes_xyz_list.append(nodes_xyz)
            c = geom.centroid
            cz = sample_z_at_points(src, [(c.x, c.y)])[0]
            centroid_xyz_list.append((c.x, c.y, cz if np.isfinite(cz) else np.nan))
    if 'area_m2' not in gdf.columns:
        gdf['area_m2'] = gdf.geometry.area
    cols_base = ['scour_id', 'area_m2', 'mean_depth', 'max_depth', 'median_depth']
    cols_existing = [c for c in cols_base if c in gdf.columns]
    df_out = pd.DataFrame(gdf[cols_existing])
    df_out = df_out.rename(columns={'area_m2': 'area'})
    df_out['centroid'] = centroid_xyz_list
    df_out['nodes_xyz'] = nodes_xyz_list
    df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f'✅ CSV 저장 완료: {output_csv}')
    print(f'📊 총 폴리곤 개수: {len(df_out)}개')
    print(df_out.head(2))
    # END ORIGINAL ANALYSIS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['gpkg','tif','output']:parser.add_argument('--'+key,type=Path,required=True)
    export_csv(parser.parse_args())
if __name__=='__main__':main()
