"""Posterior/order1a/1.0 m CUBE -> Whitebox Fill/Subtract -> original scour contours."""
from pathlib import Path
import sys,os,argparse,tempfile,subprocess,shutil

os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MPLBACKEND','Agg')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from point_cloud_io import prepare_las
from scour_polygons import load_raster,polygonize_raster,filter_polygons,calculate_polygon_depths,save_to_gpkg
from export_csv import export_csv
from export_las import export_las
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(os.environ.get('ROV_LIDAR_WORK_DIR',str(Path.home()/'.cache/dam_damage_module/rov_lidar')))

def save_plot(cube_tif,gpkg,output_png,name):
    import numpy as np
    import geopandas as gpd
    import rasterio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gdf=gpd.read_file(gpkg)
    with rasterio.open(cube_tif) as src:
        bounds=src.bounds
        a=src.read(1,masked=True).filled(np.nan)
    fig,ax=plt.subplots(figsize=(14,10),constrained_layout=True)
    values=a[np.isfinite(a)]
    ax.imshow(a,extent=(bounds.left,bounds.right,bounds.bottom,bounds.top),cmap='gray',
        vmin=float(values.min()) if values.size else 0,vmax=float(values.max()) if values.size else 1)
    if len(gdf):
        gdf.plot(ax=ax,color='#ff3030',edgecolor='#d40000',alpha=.8,linewidth=1.3)
        for i in gdf.area_m2.nlargest(10).index:
            c=gdf.geometry.iloc[i].centroid
            ax.annotate(f'scour_{i+1}',(c.x,c.y),xytext=(7,7),textcoords='offset points',fontsize=9,color='darkred',
                bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'},arrowprops={'arrowstyle':'-','color':'darkred'})
    ax.set_title(f'{name} | posterior, 1 m | Scour locations: {len(gdf)} polygons\nRed = original scour polygons; labels = 10 largest areas')
    ax.set_xlabel('Input X (m)');ax.set_ylabel('Input Y (m)');ax.set_aspect('equal')
    ax.ticklabel_format(useOffset=False,style='plain')
    ax.set_xlim(bounds.left,bounds.right);ax.set_ylim(bounds.bottom,bounds.top)
    fig.savefig(output_png,dpi=160);plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/01_EYAS_translated_segmented.ply')
    parser.add_argument('--output-dir',type=Path,default=WORK/'analysis/scour')
    parser.add_argument('--cube-python',type=Path,default=Path(sys.executable))
    parser.add_argument('--whitebox-dir',type=Path)
    args=parser.parse_args()
    if not args.input.is_file():raise FileNotFoundError(args.input)
    if not args.cube_python.is_file():raise FileNotFoundError(f'CUBE Python: {args.cube_python}; use --cube-python')
    if args.whitebox_dir is None:
        args.whitebox_dir=ROOT/'code/tools/whitebox'
        if not (args.whitebox_dir/'whitebox_tools').is_file():
            raise FileNotFoundError('WhiteboxTools binary is missing: use --whitebox-dir')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    stem=args.input.stem+'_posterior_order1a_res1.0'
    with tempfile.TemporaryDirectory(prefix='lidar-scour-') as directory:
        work=Path(directory)
        input_las=prepare_las(args.input,work)
        cube=work/(stem+'.tif');filled=work/(stem+'_fill.tif');gap=work/(stem+'_fillgap.tif')
        subprocess.run([str(args.cube_python),str(Path(__file__).with_name('cube.py')),
            '--input',str(input_las),'--output',str(cube)],check=True)
        os.environ['WBT_PATH']=str(args.whitebox_dir.resolve())
        import whitebox
        wbt=whitebox.WhiteboxTools();wbt.set_whitebox_dir(str(args.whitebox_dir.resolve()));wbt.set_verbose_mode(True)
        rc=wbt.fill_depressions(dem=str(cube),output=str(filled),fix_flats=True)
        if rc!=0 or not filled.exists():raise RuntimeError(f'Whitebox FillDepressions failed: {rc}')
        rc=wbt.subtract(input1=str(filled),input2=str(cube),output=str(gap))
        if rc!=0 or not gap.exists():raise RuntimeError(f'Whitebox Subtract failed: {rc}')
        data,t,crs,dx,dy=load_raster(gap)
        polygons,areas=polygonize_raster(data,t,0.5)
        polygons,areas=filter_polygons(polygons,areas,0.5)
        depths=calculate_polygon_depths(data,polygons,t,0.5)
        gpkg=work/(stem+'.gpkg');csv=work/(stem+'.csv');png=work/(stem+'.png');las=work/(stem+'.las')
        save_to_gpkg(polygons,areas,depths,crs,str(gpkg))
        export_csv(argparse.Namespace(gpkg=gpkg,tif=cube,output=csv))
        save_plot(cube,gpkg,png,args.input.stem)
        export_las(cube,las)
        for path in [gpkg,csv,png,las]:
            shutil.move(str(path),str(args.output_dir/path.name))
    print(f'COMPLETE: {len(polygons)} scour candidates; CSV/PNG/GPKG/CUBE surface LAS in {args.output_dir}',flush=True)
if __name__=='__main__':main()
