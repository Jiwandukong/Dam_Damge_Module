"""Export accepted original polygons as measured LAS, PNG and summary CSV only."""
import sys
sys.dont_write_bytecode = True
from runtime import ROOT, WORK
from pathlib import Path
from contextlib import ExitStack
from copy import deepcopy
import argparse
import ast
import csv
import json
import os
import shutil
import sqlite3
import tempfile

import laspy
import numpy as np
import shapely
from shapely.geometry import mapping
from pyproj import CRS
from map_model import GridMapper
from point_cloud_io import prepare_las
from visualize_results import make_visualization

FIELDS=['dataset','damage_id','source_damage_id','damage_type','damage_name_ko',
  'world_center_x_m','world_center_y_m','world_center_z_m','crs','grid_id','member_name',
  'area_m2','mean_depth_m','max_depth_m','median_depth_m','volume_loss_m3','point_count',
  'source_data_path','pointcloud_path','visualization_path','boundary_xy_json']

# Reviewed false positives for this retained survey only: every measured point
# in these reviewed detections also belongs to the original slab-zone input. The
# source pipeline did not exclude concrete slabs from its scour search.
REVIEWED_SCOUR_EXCLUSIONS = {
    ('01_EYAS_translated_segmented', 'scour_73'),
    ('01_EYAS_translated_segmented', 'scour_74'),
    ('01_EYAS_translated_segmented', 'scour_77'),
    ('01_EYAS_translated_segmented', 'scour_78'),
}

def table(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def geopackage(path):
    with sqlite3.connect(path) as connection:
        name,column=connection.execute('select table_name,column_name from gpkg_geometry_columns').fetchone()
        connection.row_factory=sqlite3.Row
        features=[]
        for row in connection.execute(f'SELECT * FROM "{name}" ORDER BY fid'):
            row=dict(row);blob=row.pop(column);envelope=(blob[3]>>1)&7
            row['geometry']=shapely.from_wkb(blob[8+{0:0,1:32,2:48,3:48,4:64}[envelope]:])
            features.append(row)
        return features

def inside(geometry,x,y):
    west,south,east,north=geometry.bounds
    indices=np.flatnonzero((x>=west)&(x<=east)&(y>=south)&(y<=north))
    return indices[shapely.intersects_xy(geometry,x[indices],y[indices])] if len(indices) else indices

def features_from_analysis(analysis,min_area,source_stem='01_EYAS_translated_segmented',damage_types=('SC','DP')):
    sc=analysis/'scour';dp=analysis/'slab_depression/depression'
    paths={'SC':(sc/f'{source_stem}_posterior_order1a_res1.0.gpkg',sc/f'{source_stem}_posterior_order1a_res1.0.csv'),
      'DP':(dp/'depression_patches.gpkg',dp/'depression_patches.csv')}
    result=[]
    for kind in damage_types:
        geometry_path,csv_path=paths[kind]
        for path in [geometry_path,csv_path]:
            if not path.is_file():raise FileNotFoundError(f'{path}: run the corresponding analytical stage first, or supply a retained --analysis-dir')
        geos,attributes=geopackage(geometry_path),table(csv_path)
        if len(geos)!=len(attributes):raise ValueError('Original CSV/GPKG row counts disagree')
        for i,(geo,row) in enumerate(zip(geos,attributes)):
            area=float(row['area'])
            # The display area threshold applies to scour only. Preserve every
            # original depression patch, including a single-cell patch.
            if kind=='SC' and area<min_area:continue
            geometry=geo['geometry'];shapely.prepare(geometry)
            source_id=row['scour_id'] if kind=='SC' else f"depression_{row['patch_id']}"
            if kind=='SC' and (source_stem,source_id) in REVIEWED_SCOUR_EXCLUSIONS:
                continue
            suffix=int(source_id.split('_')[-1]);identifier=f'{kind}-{suffix:03d}'
            if kind=='SC':center=list(ast.literal_eval(row['centroid']))
            else:center=[float(geo['centroid_x']),float(geo['centroid_y']),0.0]
            west,south,east,north=geometry.bounds;span=max(east-west,north-south);margin=max(2.,span*.15)
            result.append({'kind':kind,'id':identifier,'source_id':source_id,'geometry':geometry,
              'area':area,'mean_depth':float(row['mean_depth']),'max_depth':float(row['max_depth']),
              'median_depth':float(row['median_depth']) if kind=='SC' else '',
              'volume':float(row['volume_loss']) if kind=='DP' else '',
              'center':center,'center_z':center[2],'point_count':0,
              'context_bounds':[west-margin,south-margin,east+margin,north+margin],
              'context':np.empty((0,3)),'context_seen':0})
    return result

def add_context(feature,xyz,limit=24000):
    if not len(xyz):return
    # Bounded deterministic context sample; the exported LAS retain every selected observation.
    step=max(1,len(xyz)//limit);sample=xyz[::step]
    existing=feature['context'];combined=np.concatenate((existing,sample))
    if len(combined)>limit:
        combined=combined[np.linspace(0,len(combined)-1,limit,dtype=np.int64)]
    feature['context']=combined

def extract_scour(source,features,staging):
    if not features:return
    with laspy.open(source) as reader,ExitStack() as stack:
        header=deepcopy(reader.header);header.add_crs(CRS.from_epsg(5186))
        writers={f['id']:stack.enter_context(laspy.open(staging/f"Result/SC/{f['id']}.las",mode='w',header=header)) for f in features}
        for points in reader.chunk_iterator(500000):
            x,y,z=np.asarray(points.x),np.asarray(points.y),np.asarray(points.z)
            for f in features:
                indices=inside(f['geometry'],x,y)
                if len(indices):writers[f['id']].write_points(points[indices]);f['point_count']+=len(indices)
                west,south,east,north=f['context_bounds']
                nearby=np.flatnonzero((x>=west)&(x<=east)&(y>=south)&(y<=north))
                if len(nearby):
                    nearby=nearby[~shapely.intersects_xy(f['geometry'],x[nearby],y[nearby])]
                    add_context(f,np.column_stack((x[nearby],y[nearby],z[nearby])))

def extract_depression(analysis,features,staging):
    if not features:return
    directory=analysis/'slab_depression/depression'
    records=np.loadtxt(directory/'damage_candidate_points.csv',delimiter=',',skiprows=1)
    if records.ndim==1:records=records.reshape(1,-1)
    records=records[:,[0,1,2,6]]
    residual_points=None
    with laspy.open(analysis/'slab_depression/slab_extraction/slab_extracted.las') as reader:
        header=deepcopy(reader.header)
    header.add_crs(CRS.from_epsg(5186))
    header.add_extra_dim(laspy.ExtraBytesParams(name='depression_depth_m',type=np.float64,description='Minus original plane residual'))
    for f in features:
        # Keep all original candidates in the patch, including small clusters
        # and unclustered points. Candidate clusters are not patch IDs.
        idx=inside(f['geometry'],records[:,0],records[:,1])
        selected=records[idx]
        f['point_selection']='point_candidates_in_patch'
        if not len(selected):
            # Grid-average MAD and pointwise MAD are distinct original tests.
            # A detected grid patch may contain no pointwise candidates. Keep
            # the patch using its actual cleaned observations, as in Multibeam.
            if residual_points is None:
                residual_points=np.loadtxt(directory/'residual_points.csv',delimiter=',',skiprows=1,usecols=(0,1,2,6))
                residual_points=np.atleast_2d(residual_points)
            idx=inside(f['geometry'],residual_points[:,0],residual_points[:,1])
            selected=residual_points[idx]
            f['point_selection']='cleaned_observations_in_grid_patch'
        if not len(selected):raise ValueError(f"{f['id']}: no original observed points lie within the depression boundary")
        selected_header=deepcopy(header)
        points=laspy.ScaleAwarePointRecord.zeros(len(selected),header=selected_header)
        las=laspy.LasData(selected_header,points);las.x,las.y,las.z=selected[:,0],selected[:,1],selected[:,2]
        las.depression_depth_m=-selected[:,3]
        las.write(staging/f"Result/DP/{f['id']}.las")
        f['point_count']=len(selected)
        xy=np.array(f['center'][:2]);nearest=np.argmin(np.sum((selected[:,:2]-xy)**2,axis=1))
        # This is a measured representative height; X/Y and the original patch metrics are retained.
        f['center_z']=float(selected[nearest,2]);f['center'][2]=f['center_z']
    source=analysis/'slab_depression/slab_extraction/slab_extracted.las'
    with laspy.open(source) as reader:
        for points in reader.chunk_iterator(500000):
            x,y,z=np.asarray(points.x),np.asarray(points.y),np.asarray(points.z)
            for f in features:
                west,south,east,north=f['context_bounds']
                idx=np.flatnonzero((x>=west)&(x<=east)&(y>=south)&(y<=north))
                idx=idx[~shapely.intersects_xy(f['geometry'],x[idx],y[idx])]
                add_context(f,np.column_stack((x[idx],y[idx],z[idx])))

def export(args):
    work=args.work_dir.expanduser().resolve();output=args.output_dir.resolve()
    if work==output or output in work.parents or work in output.parents:raise ValueError('Work and final output directories must be separate')
    work.mkdir(parents=True,exist_ok=True)
    kinds=('SC','DP') if args.damage_type=='all' else (args.damage_type,)
    features=features_from_analysis(args.analysis_dir,args.min_area,args.source_ply.stem,kinds)
    mapper=GridMapper(args.dam_model) if args.dam_model.is_file() else None
    with tempfile.TemporaryDirectory(prefix='final-export-',dir=work) as directory:
        staging=Path(directory)
        for kind in kinds:
            (staging/f'Result/{kind}').mkdir(parents=True)
            (staging/f'Visualize/{kind}').mkdir(parents=True)
        sc=[f for f in features if f['kind']=='SC'];dp=[f for f in features if f['kind']=='DP']
        extract_depression(args.analysis_dir,dp,staging)
        if sc:
            source=prepare_las(args.source_ply,staging)
            extract_scour(source,sc,staging)
            # Temporary converted input is outside the final artifact directories.
        for f in features:
            if f['point_count']==0:raise ValueError(f"{f['id']}: empty measured point cloud")
            print(f"{f['id']}: {f['point_count']:,} measured points; rendering 4K PNG",flush=True)
            make_visualization(f,staging)
        for kind in kinds:
            rows=[]
            for f in features:
                if f['kind']!=kind:continue
                mapped=mapper.associate(f['center'])[0] if mapper else {'grid_id':'','member_name':''}
                source=args.source_ply if kind=='SC' else args.slab_zone
                rows.append({'dataset':source.stem,'damage_id':f['id'],'source_damage_id':f['source_id'],
                  'damage_type':kind,'damage_name_ko':'세굴' if kind=='SC' else '슬래브 함몰',
                  **{f'world_center_{axis}_m':float(v) for axis,v in zip('xyz',f['center'])},'crs':'EPSG:5186',**mapped,
                  'area_m2':f['area'],'mean_depth_m':f['mean_depth'],'max_depth_m':f['max_depth'],
                  'median_depth_m':f['median_depth'],'volume_loss_m3':f['volume'],'point_count':f['point_count'],
                  'source_data_path':os.path.relpath(source,output/'Result'),
                  'pointcloud_path':f"{kind}/{f['id']}.las",'visualization_path':f"../Visualize/{kind}/{f['id']}.png",
                  'boundary_xy_json':json.dumps(mapping(f['geometry']),ensure_ascii=False,separators=(',',':'))})
            with (staging/f'Result/{kind}_result.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
        # All expensive work completes before any final output is replaced.
        previous=[]
        for kind in kinds:
            path=output/f'Result/{kind}_result.csv'
            if path.is_file():previous.extend(table(path))
        produced={str(p.relative_to(staging)) for top in ['Result','Visualize'] for p in (staging/top).rglob('*') if p.is_file()}
        for top in ['Result','Visualize']:
            for path in (staging/top).rglob('*'):
                if path.is_file():
                    destination=output/path.relative_to(staging);destination.parent.mkdir(parents=True,exist_ok=True)
                    os.replace(path,destination)
        for row in previous:
            for sub in [f"Result/{row['damage_type']}/{row['damage_id']}.las",f"Visualize/{row['damage_type']}/{row['damage_id']}.png"]:
                if sub not in produced:(output/sub).unlink(missing_ok=True)
    counts=', '.join(f"{kind} {sum(f['kind']==kind for f in features)}" for kind in kinds)
    print(f'COMPLETE: {counts}; {len(features)} LAS, {len(features)} PNG, {len(kinds)} CSV in {output}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir',type=Path,default=WORK/'analysis')
    p.add_argument('--source-ply',type=Path,default=ROOT/'data/01_EYAS_translated_segmented.ply')
    p.add_argument('--slab-zone',type=Path,default=ROOT/'data/01_EYAS_translated_segmented_slab_zone.ply')
    p.add_argument('--output-dir',type=Path,default=ROOT/'output')
    p.add_argument('--damage-type',choices=['all','SC','DP'],default='all',help='Export and replace only the selected damage type')
    p.add_argument('--min-area',type=float,default=5.,help='Minimum scour area in m²; depression patches and candidates are always retained without an area filter')
    p.add_argument('--dam-model',type=Path,default=ROOT.parent/'Dam_model/Daecheongdam/daecheongdam_regions_grid5m.gltf')
    p.add_argument('--work-dir',type=Path,default=WORK)
    args=p.parse_args()
    if not np.isfinite(args.min_area) or args.min_area<0:p.error('--min-area must be finite and nonnegative')
    export(args)

if __name__=='__main__':main()
