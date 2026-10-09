#!/usr/bin/env python3
"""ROV raw frames -> Sunflicker -> model prediction -> damage CSV and square Overlay."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
import json
import math
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from skimage.measure import find_contours
from paths import ROOT,WORK,VIDEO,sha,read_json,write_json
from demo_mapping import SpillwayWaterlineGridSampler
from result_export import RESULT_COLUMNS as COLUMNS, write_results

KINDS={1:'CRC',2:'SPL'}
NAMES={'CRC':'균열','SPL':'박락'}
COLORS={'CRC':(255,0,0),'SPL':(255,255,0)}
def crop_box(bbox,size):
    left,top,right,bottom=bbox;w,h=size
    extent=max(right-left,bottom-top);margin=max(128,math.ceil(extent*.2))
    desired=math.ceil((extent+2*margin)/128)*128
    available=min(size)//128*128
    side=min(desired,available) if available>=extent else math.ceil(extent/128)*128
    def origin(a,b,total):
        center=math.floor((a+b-side)/2)
        return min(max(0,center),total-side) if side<=total else center
    x,y=origin(left,right,w),origin(top,bottom,h)
    if not (x<=left<right<=x+side and y<=top<bottom<=y+side):raise ValueError('Crop clips prediction')
    return x,y,side

def contour_rings(mask,x0,y0):
    rings=[]
    for ring in find_contours(np.pad(mask,1),.5,fully_connected='high'):
        if len(ring)>256:ring=ring[np.linspace(0,len(ring)-1,256).round().astype(int)]
        points=[[float(x-1+x0),float(y-1+y0)] for y,x in ring]
        if points and points[0]!=points[-1]:points.append(points[0])
        rings.append(points)
    return rings

def export_image(image_path,mask,probability,output,rows,counter,min_area):
    raw=np.asarray(Image.open(image_path).convert('RGB'));h,w=raw.shape[:2]
    if mask.shape!=(h,w) or probability.shape!=(3,h,w):raise ValueError('Prediction/source dimensions differ')
    frame_stats={}
    for class_id,kind in KINDS.items():
        count,labels,stats,_=cv2.connectedComponentsWithStats((mask==class_id).astype(np.uint8),connectivity=8)
        represented=0;filtered=0;exported=0
        for component in range(1,count):
            x0,y0,bw,bh,area=map(int,stats[component]);represented+=area
            if area<min_area:filtered+=area;continue
            local=labels[y0:y0+bh,x0:x0+bw]==component
            yy,xx=np.nonzero(local);closest=int(np.argmin((xx-xx.mean())**2+(yy-yy.mean())**2))
            center=[float(xx[closest]+x0),float(yy[closest]+y0)]
            bbox=[x0,y0,x0+bw,y0+bh];rings=contour_rings(local,x0,y0)
            x,y,side=crop_box(bbox,(w,h))
            crop=Image.fromarray(raw).crop((x,y,x+side,y+side));selected=np.zeros((side,side),np.uint8)
            selected[y0-y:y0-y+bh,x0-x:x0-x+bw]=local.astype(np.uint8)*255
            painted=crop.copy();painted.paste(COLORS[kind],(0,0),Image.fromarray(selected));overlay=Image.blend(crop,painted,.3)
            counter[0]+=1;damage_id=f'R{counter[0]:06d}'
            destination=output/'Overlay'/kind/(damage_id+'.png');destination.parent.mkdir(parents=True,exist_ok=True)
            temp=destination.with_suffix('.png.new');overlay.save(temp,format='PNG');temp.replace(destination)
            row=dict.fromkeys(COLUMNS,'')
            row.update(image=image_path.name,damage_id=damage_id,damage_type=kind,damage_name_ko=NAMES[kind],
                       pixel_nodes_json=json.dumps(rings[0] if rings else [],separators=(',',':')),
                       DRI=float(probability[class_id,y0:y0+bh,x0:x0+bw][local].mean(dtype=np.float64)),
                       source_image_path=os.path.relpath(image_path,output/'Result').replace(os.sep,'/'),
                       overlay_path=f'../Overlay/{kind}/{damage_id}.png',mapping_status='pending',measurement_status='pixel_only',
                       pixel_center_x_px=center[0],pixel_center_y_px=center[1],bbox_px_json=json.dumps(bbox,separators=(',',':')),
                       crop_origin_x_px=x,crop_origin_y_px=y,crop_size_px=side,
                       overlay_padding_px_json=json.dumps([max(0,-x),max(0,-y),max(0,x+side-w),max(0,y+side-h)]),
                       result_origin='model_prediction_demo_in_sample')
            rows.append(row);exported+=1
        if represented!=int((mask==class_id).sum()):raise ValueError('Components lost prediction pixels')
        frame_stats[kind]=dict(components=count-1,exported=exported,filtered_pixels=filtered,mask_pixels=represented)
    return frame_stats

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'Data');parser.add_argument('--output',type=Path,default=ROOT/'Output')
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--video',type=Path,default=VIDEO)
    parser.add_argument('--work-dir',type=Path,default=WORK);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--input-preprocessed',action='store_true');parser.add_argument('--min-area-px',type=int,default=1)
    parser.add_argument('--demo-random-3d',action=argparse.BooleanOptionalAction,default=True,
                        help='Group similar frames along downstream waterline spillway grids (default on)')
    parser.add_argument('--dam-model',type=Path,help='Optional gridded glTF; default downloads the shared Release model')
    parser.add_argument('--mapping-seed',type=int,default=42)
    args=parser.parse_args()
    if args.min_area_px<1:parser.error('--min-area-px must be >=1')
    args.work_dir=args.work_dir.resolve();args.work_dir.mkdir(parents=True,exist_ok=True)
    args.output=args.output.resolve();args.input=args.input.resolve()
    from download_assets import prepare_release_assets
    prepare_release_assets(args)
    assets=read_json(ROOT/'Code/assets.json');checkpoint=args.checkpoint or args.work_dir/'models'/assets['checkpoint_filename']
    if not checkpoint.exists():raise FileNotFoundError('Use --checkpoint for the ROV demo model: '+str(checkpoint))
    model_sha=sha(checkpoint)
    if checkpoint==args.work_dir/'models'/assets['checkpoint_filename'] and model_sha!=assets['checkpoint_sha256']:raise ValueError('Default model SHA256 mismatch')
    paths=[args.input] if args.input.is_file() else sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if not paths:parser.error('No input frames')
    if len({p.stem for p in paths})!=len(paths):parser.error('Duplicate image stems')
    sampler=SpillwayWaterlineGridSampler(args.dam_model,seed=args.mapping_seed) if args.demo_random_3d else None
    if sampler is not None:sampler.prepare_frames(paths)
    import torch
    from rov_model import load_model,predict
    from preprocessing import Preprocessor
    if args.device.startswith('cuda') and not torch.cuda.is_available():args.device='cpu'
    torch.set_num_threads(4);cv2.setNumThreads(1)
    model,size,cfg=load_model(checkpoint,args.device)
    processor=None if args.input_preprocessed else Preprocessor(args.video,args.work_dir)
    prediction_dir=args.work_dir/'prediction_cache';prediction_dir.mkdir(exist_ok=True)
    rows=[];counter=[0];observations=[]
    try:
        for i,path in enumerate(paths,1):
            if args.input_preprocessed:rgb=np.asarray(Image.open(path).convert('RGB'));origin='supplied_preprocessed_rgb'
            else:rgb,origin=processor.get(path)
            mask,probability=predict(model,rgb,size,args.device)
            mask_file=prediction_dir/(path.stem+'.png');Image.fromarray(mask).save(mask_file)
            stats=export_image(path,mask,probability,args.output,rows,counter,args.min_area_px)
            observations.append(dict(image=path.name,image_sha256=sha(path),mask_sha256=sha(mask_file),preprocessing_origin=origin,classes=stats))
            print(f'ROV Prediction {i}/{len(paths)}: {path.name} ({len(rows)} observations)',flush=True)
    finally:
        if processor is not None:processor.close()
    if sampler is not None:
        for row in rows:sampler.sample(row)
    final_result=write_results(args.output,rows)
    for kind in KINDS.values():
        # Remove stale files generated by an earlier invocation of this exporter.
        active={r['damage_id']+'.png' for r in rows if r['damage_type']==kind}
        folder=args.output/'Overlay'/kind;folder.mkdir(parents=True,exist_ok=True)
        for p in folder.glob('R*.png'):
            if p.name not in active:p.unlink()
    summary=dict(images=len(paths),counts={kind:sum(r['damage_type']==kind for r in rows) for kind in KINDS.values()},total_observations=len(rows),
                 model_sha256=model_sha,checkpoint=str(checkpoint),input_size=list(size),
                 mapping_status='demo_random' if sampler is not None else 'pending',measurement_status='pixel_only',
                 min_area_px=args.min_area_px,purpose=cfg.get('purpose','demo'),evaluation_scope='in_sample_for_supplied_demo_frames',
                 observations=observations)
    if sampler is not None:summary['demo_mapping']=sampler.summary()
    summary['final_result']=final_result
    write_json(args.work_dir/'inference_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='observations'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
