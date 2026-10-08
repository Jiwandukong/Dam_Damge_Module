#!/usr/bin/env python3
"""Drone JPEG -> DINOv3 Prediction -> native dam mapping -> class CSV/Overlay.

Run from the UAV_RGB folder:
  python Code/process.py

The complete trained checkpoint and native gridded glTF are downloaded from
GitHub Releases into the user cache when the default files are absent.
Override --checkpoint and --dam-model to use manually downloaded files.
Only physical GPU 0 is used for inference.
No annotation JSON, GT mask, comparison report, or training manifest is read.
Use --prediction-cache to export the already generated real predictions.
CSV paths are relative to Output/Result; XYZ is EPSG:5186, Z-up.
Only predictions mapped onto the dam surface are exported for the demo.
Dimensions reuse the existing approximate local directional GSD method.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from urllib.request import Request, urlopen
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import find_contours

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('CRC', 'DLM', 'SPL', 'LEAK_EFF')
OUTPUT_CLASSES = ('CRC', 'DLM', 'SPL', 'LKG')
NAMES = {'CRC': '균열', 'DLM': '박리', 'SPL': '박락', 'LKG': '누수·백태'}
COLORS = {'CRC': (255,0,0), 'DLM': (0,255,0), 'SPL': (255,255,0), 'LKG': (0,0,255)}
RELEASE_BASE = 'https://github.com/Jiwandukong/Dam_Damge_Module/releases/download'
CACHE = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache'))) / 'dam_damage_module'
MODEL_ASSET = 'dinov3_damage_demo.pt'
DAM_ASSET = 'daecheongdam_grid5m.zip'
MODEL_TAG = 'dinov3-uav-demo-v1'
DAM_TAG = 'daecheong-dam-grid5m-v1'
MODEL_SHA256 = '05d636ae06d2b58327526f64c0b0599e31c4484abd188f82f83ec7d6a5573acc'
DAM_SHA256 = '795eed4f684ddb7ada2669b2555d627d2b337782587763fc0ffb332acd9820cf'
DAM_MODEL_SHA256 = '4096486bd10baa62e51988e0424eb66b1191b58dcc58a410b97a6e5d03edb4f7'
LOCAL_CHECKPOINT = ROOT.parent / 'DINOv3_Damage_Demo/runs/20261006_train_reconstruction/demo_model.pt'
LOCAL_MODEL = ROOT.parent / 'Dam_model/Daecheongdam/daecheongdam_regions_grid5m.gltf'
DEFAULT_CHECKPOINT = LOCAL_CHECKPOINT if LOCAL_CHECKPOINT.is_file() else CACHE/MODEL_ASSET
DEFAULT_MODEL = LOCAL_MODEL if LOCAL_MODEL.is_file() else CACHE/'Daecheongdam/daecheongdam_regions_grid5m.gltf'
COLUMNS = [
    'image', 'damage_id', 'damage_type', 'damage_name_ko', 'pixel_nodes_json',
    'world_center_x_m', 'world_center_y_m', 'world_center_z_m',
    'length_px', 'length_m', 'width_px', 'width_m', 'area_m2', 'area_px',
    'member_name', 'section_name', 'grid_id', 'grid_guid', 'DRI',
    'source_image_path', 'overlay_path',
    'mapping_status', 'measurement_status',
]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download_verified(url, destination, expected_sha256):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_file() and sha(destination)==expected_sha256:
        return
    temporary = destination.with_name(destination.name+'.partial')
    print('Downloading Release asset: '+destination.name,flush=True)
    try:
        request = Request(url,headers={'User-Agent':'Dam-Damage-Module-UAV'})
        with urlopen(request,timeout=60) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response,stream,length=8*1024*1024)
        if sha(temporary)!=expected_sha256:
            raise ValueError('Release asset SHA256 mismatch: '+destination.name)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_release_assets(args):
    if not args.prediction_cache and not args.checkpoint.is_file():
        if args.checkpoint!=CACHE/MODEL_ASSET:
            raise FileNotFoundError('Checkpoint not found: '+str(args.checkpoint))
        download_verified(f'{RELEASE_BASE}/{MODEL_TAG}/{MODEL_ASSET}',args.checkpoint,MODEL_SHA256)
    archive = CACHE/DAM_ASSET
    cached_model = CACHE/'Daecheongdam/daecheongdam_regions_grid5m.gltf'
    needs_dam_model = not args.dam_model.is_file()
    if args.dam_model==cached_model and not needs_dam_model:
        needs_dam_model = (sha(args.dam_model)!=DAM_MODEL_SHA256 or
                           not archive.is_file() or sha(archive)!=DAM_SHA256)
    if needs_dam_model:
        if args.dam_model!=cached_model:
            raise FileNotFoundError('Native dam model not found: '+str(args.dam_model))
        download_verified(f'{RELEASE_BASE}/{DAM_TAG}/{DAM_ASSET}?sha256={DAM_SHA256}',archive,DAM_SHA256)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (CACHE/member.filename).resolve()
                if not target.is_relative_to(CACHE.resolve()):
                    raise ValueError('Invalid Release ZIP path')
            bundle.extractall(CACHE)
        if not args.dam_model.is_file():
            raise FileNotFoundError('Expected gridded glTF is absent from the Release package')
        if sha(args.dam_model)!=DAM_MODEL_SHA256:
            raise ValueError('Gridded glTF SHA256 mismatch after Release extraction')


def crop_box(bbox, size):
    left, top, right, bottom = bbox
    width, height = size
    margin = max(128, math.ceil(max(right-left, bottom-top)*.2))
    side = math.ceil((max(right-left, bottom-top)+2*margin)/128)*128
    side = min(side, min(size)//128*128)
    if side < max(right-left, bottom-top):
        raise ValueError('The full prediction cannot fit a source-only square crop: '+str(bbox))
    x = min(max(0, math.floor((left+right-side)/2)), width-side)
    y = min(max(0, math.floor((top+bottom-side)/2)), height-side)
    if not (x <= left < right <= x+side and y <= top < bottom <= y+side):
        raise ValueError('Crop would clip the prediction')
    return x, y, side, side


def positions(length, tile, stride):
    if length <= tile:
        return [0]
    result = list(range(0, length-tile+1, stride))
    if result[-1] != length-tile:
        result.append(length-tile)
    return result


def tiled_prediction(model, rgb, settings):
    import torch
    tile, stride, batch_size = (settings[k] for k in ('tile', 'stride', 'batch_size'))
    height, width = rgb.shape[:2]
    coords = [(x,y) for y in positions(height,tile,stride) for x in positions(width,tile,stride)]
    blends = settings['blend']
    if isinstance(blends,str):
        blends = dict.fromkeys(CLASSES,blends)
    windows = []
    for kind in CLASSES:
        axis = .05+.95*np.hanning(tile) if blends[kind]=='hann' else np.ones(tile)
        windows.append(np.outer(axis,axis).astype(np.float32))
    window = np.stack(windows)
    total, cover = np.zeros((4,height,width),np.float32), np.zeros((4,height,width),np.float32)
    mean = np.array([.485,.456,.406],np.float32)
    std = np.array([.229,.224,.225],np.float32)
    with torch.inference_mode():
        for offset in range(0,len(coords),batch_size):
            batch = coords[offset:offset+batch_size]
            inputs = []
            for x,y in batch:
                crop = rgb[y:y+tile,x:x+tile]
                if crop.shape[:2] != (tile,tile):
                    padded = np.zeros((tile,tile,3),np.uint8)
                    padded[:crop.shape[0],:crop.shape[1]] = crop
                    crop = padded
                normalized = ((crop.astype(np.float32)/255-mean)/std).transpose(2,0,1).copy()
                inputs.append(torch.from_numpy(normalized))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                probabilities = model(torch.stack(inputs).to('cuda:0')).float().sigmoid().cpu().numpy()
            for (x,y),p in zip(batch,probabilities):
                h,w = min(tile,height-y),min(tile,width-x)
                total[:,y:y+h,x:x+w] += p[:,:h,:w]*window[:,:h,:w]
                cover[:,y:y+h,x:x+w] += window[:,:h,:w]
    if not np.all(cover>0):
        raise ValueError('Inference tiles did not cover the complete image')
    return total/cover


def contour_rings(mask, x0, y0):
    rings = []
    for ring in find_contours(np.pad(mask,1),.5,fully_connected='high'):
        if len(ring)>256:
            ring = ring[np.linspace(0,len(ring)-1,256).round().astype(int)]
        rings.append([[float(x-1+x0),float(y-1+y0)] for y,x in ring])
    return rings


def map_component(surface, context, pixel_center, rings):
    flat = [xy for ring in rings for xy in ring]
    hits = surface.cast(context,[pixel_center]+flat)
    center_hit = hits[0] or next((h for h in hits[1:] if h is not None),None)
    return center_hit


def association(surface, hit):
    fields = dict.fromkeys(('member_name','section_name','grid_id','grid_guid'),'')
    if hit is None:
        return fields
    model,node_id = surface.g,hit['node_id']
    node = model['nodes'][node_id]
    fields.update(member_name=node['name'])
    parents = {child:i for i,n in enumerate(model['nodes']) for child in n.get('children',[])}
    roots = set(model['scenes'][model.get('scene',0)]['nodes'])
    section_id = node_id
    while section_id in parents and parents[section_id] not in roots:
        section_id = parents[section_id]
    section = model['nodes'][section_id]
    fields.update(section_name=section['name'])
    if all(k in node['extras'] for k in ('face','row','col')):
        fields.update(grid_id=node['name'],grid_guid=node['extras']['guid'])
    return fields


def dimensions(surface, context, mask, bbox, class_index, pixel_center):
    from dam_mapping import (measure_oriented_pixel_geometry, build_surface_measurement_stencil,
                             surface_measurement_from_hits, review_measurement)
    geometry = measure_oriented_pixel_geometry(mask,offset_x=bbox[0],offset_y=bbox[1])
    stencil = build_surface_measurement_stencil(geometry,center_xy=np.asarray(pixel_center),
                 image_width=context.intrinsics.width,image_height=context.intrinsics.height)
    rays = [pixel_center] + (stencil.xy.tolist() if stencil is not None else [])
    hits = surface.cast(context,rays)
    spatial_hits = []
    for hit in hits[1:]:
        record = {'mesh_ray_hit':hit is not None}
        if hit is not None:
            record.update({f'world_{axis}_m':hit['world_obj_xyz'][i] for i,axis in enumerate('xyz')})
        spatial_hits.append(record)
    spatial = surface_measurement_from_hits({'class_id':class_index+1,'area_px':int(mask.sum())},geometry,stencil,spatial_hits)
    spatial.update(xyz_valid=hits[0] is not None,mesh_ray_t_m=hits[0]['ray_distance_m'] if hits[0] is not None else None)
    review = review_measurement(spatial,context.intrinsics.focal_length_px)
    result = {k:spatial.get(k) for k in ('length_px','width_px')}
    result.update({k:spatial.get(k) if review['physical_values_exported'] else None for k in ('length_m','width_m','area_m2')})
    result['measurement_status'] = 'local_surface_estimate' if review['physical_values_exported'] else ';'.join(review['reasons'])
    return result


def export_image(image_path, probability, mask_for_class, surface, output, all_rows, component_counter, cached_mapping=None):
    from dam_mapping import read_dji_xmp, intrinsics_from_xmp, pose_from_xmp
    image = Image.open(image_path).convert('RGB')
    xmp = read_dji_xmp(image_path)
    context = SimpleNamespace(intrinsics=intrinsics_from_xmp(image_path,xmp),pose=pose_from_xmp(image_path,xmp))
    if probability.shape != (4,image.height,image.width):
        raise ValueError('Unexpected probability dimensions: '+image_path.name)
    for channel, (kind, published_kind) in enumerate(zip(CLASSES,OUTPUT_CLASSES)):
        mask = mask_for_class(channel,kind)
        labels,count = ndimage.label(mask,structure=np.ones((3,3),np.uint8))
        covered = 0
        for component_id,box in enumerate(ndimage.find_objects(labels),1):
            ys,xs = box
            local = labels[box]==component_id
            area = int(local.sum());covered += area
            yy,xx = np.nonzero(local)
            closest = int(np.argmin((xx-xx.mean())**2+(yy-yy.mean())**2))
            pixel_center = [float(xx[closest]+xs.start),float(yy[closest]+ys.start)]
            bbox = [xs.start,ys.start,xs.stop,ys.stop]
            rings = contour_rings(local,xs.start,ys.start)
            component_counter[0] += 1
            damage_id = f'P{component_counter[0]:06d}'
            dri = float(np.asarray(probability[channel][box])[local].mean(dtype=np.float64))
            cached = cached_mapping.get((image_path.name,kind,component_id)) if cached_mapping is not None else None
            if cached_mapping is not None and cached is None:
                raise ValueError('Cached native mapping has no prediction '+damage_id)
            if cached is not None:
                if cached['damage_id']!=damage_id or cached['area_px']!=area or cached['bbox_px']!=bbox or cached['prediction_pixel_center']!=pixel_center:
                    raise ValueError('Cached prediction component changed: '+damage_id)
                if not math.isclose(cached['DRI'],dri,abs_tol=1e-12):
                    raise ValueError('Cached prediction confidence changed: '+damage_id)
                world = cached['world_center_xyz']
                fields = dict(member_name=cached['native_mesh_name'],
                    **{k:cached[k] for k in ('section_name','grid_id','grid_guid')})
            else:
                hit = map_component(surface,context,pixel_center,rings)
                world = hit['world_obj_xyz'] if hit is not None else None
                fields = association(surface,hit)
            if world is None:
                # Keep observation IDs stable even when a demo-only row is omitted.
                stale = output/'Overlay'/published_kind/f'{damage_id}.png'
                if stale.is_file():
                    stale.unlink()
                continue
            x,y,w,h = crop_box(bbox,image.size)
            raw = image.crop((x,y,x+w,y+h))
            selected = np.zeros((h,w),np.uint8)
            selected[ys.start-y:ys.stop-y,xs.start-x:xs.stop-x] = local.astype(np.uint8)*255
            painted = raw.copy();painted.paste(COLORS[published_kind],(0,0),Image.fromarray(selected))
            overlay = Image.blend(raw,painted,.3)
            destination = output/'Overlay'/published_kind/f'{damage_id}.png'
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary = destination.with_name(destination.name+'.new')
            overlay.save(temporary,format='PNG');temporary.replace(destination)
            row = dict.fromkeys(COLUMNS)
            row.update(image=image_path.name,damage_id=damage_id,damage_type=published_kind,damage_name_ko=NAMES[published_kind],
                pixel_nodes_json=json.dumps(rings[0] if rings else [],separators=(',',':')),
                area_px=area,DRI=dri,source_image_path=os.path.relpath(image_path,output/'Result').replace(os.sep,'/'),
                overlay_path=f'../Overlay/{published_kind}/{damage_id}.png',
                mapping_status='mapped',**fields)
            row.update({f'world_center_{axis}_m':world[i] if world is not None else None for i,axis in enumerate('xyz')})
            row.update(dimensions(surface,context,local,bbox,channel,pixel_center))
            all_rows.append(row)
        if covered != int(mask.sum()):
            raise ValueError('The exported components did not retain all prediction pixels')
    image.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input',type=Path,default=ROOT/'Data')
    parser.add_argument('--output',type=Path,default=ROOT/'Output')
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--dam-model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--prediction-cache',type=Path,help='Existing real probability NPYs and final prediction PNGs')
    parser.add_argument('--mapping-cache',type=Path,help='Optional existing native prediction intersections; no GT fields are used')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.prediction_cache else '0'
    from dam_mapping import DisplaySurface
    paths = [args.input] if args.input.is_file() else sorted(p for p in args.input.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if not paths:
        parser.error('No input photos found')
    prepare_release_assets(args)
    cached_mapping = None
    if args.mapping_cache:
        if not args.prediction_cache:
            parser.error('--mapping-cache requires --prediction-cache')
        source = json.loads(args.mapping_cache.read_text())
        if source['model_sha256'] != sha(args.dam_model):
            raise ValueError('Native model differs from the cached prediction mapping')
        for spec in source['processing']:
            name = f'{Path(spec["image"]).stem}_{spec["damage_type"]}.png'
            if sha(args.prediction_cache/name)!=spec['mask_sha256']:
                raise ValueError('Cached final prediction mask changed: '+name)
        cached_mapping = {(p['image'],p['damage_type'],p['prediction_component_id']):p for p in source['predictions']}
    surface = DisplaySurface(args.dam_model)
    if args.prediction_cache:
        settings = json.loads((args.prediction_cache/'inference_summary.json').read_text())
        thresholds = settings['threshold']
    else:
        import torch
        from dinov3_model import load_model
        if not torch.cuda.is_available():
            raise RuntimeError('Physical GPU 0 is required for DINO inference')
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        model,settings = load_model(args.checkpoint)
        thresholds = settings['thresholds']
    rows,component_counter = [],[0]
    for number,path in enumerate(paths,1):
        if args.prediction_cache:
            probability = np.load(args.prediction_cache/f'{path.stem}_probabilities.npy',mmap_mode='r',allow_pickle=False)
        else:
            rgb = np.asarray(Image.open(path).convert('RGB'))
            probability = tiled_prediction(model,rgb,settings)
        def masks(channel,kind):
            threshold = thresholds[kind] if isinstance(thresholds,dict) else thresholds
            predicted = probability[channel]>=threshold
            if args.prediction_cache:
                with Image.open(args.prediction_cache/f'{path.stem}_{kind}.png') as image:
                    saved = np.asarray(image)>0
                if not np.array_equal(predicted,saved):
                    raise ValueError('Probability and final mask differ: '+path.name+' '+kind)
            return predicted
        export_image(path,probability,masks,surface,args.output,rows,component_counter,cached_mapping)
        del probability
        print(f'Prediction export {number}/{len(paths)}: {path.name} ({len(rows)} damages)',flush=True)
    (args.output/'Result').mkdir(parents=True,exist_ok=True)
    for kind in OUTPUT_CLASSES:
        destination = args.output/'Result'/f'{kind}_result.csv'
        temporary = destination.with_name(destination.name+'.new')
        with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=COLUMNS);writer.writeheader()
            writer.writerows(r for r in rows if r['damage_type']==kind)
        temporary.replace(destination)
    print(json.dumps(dict(total=len(rows),unmapped_excluded=component_counter[0]-len(rows),class_counts=dict(Counter(r['damage_type'] for r in rows)),
        mapped=sum(r['mapping_status']=='mapped' for r in rows),grid_matched=sum(bool(r['grid_id']) for r in rows),
        physical_measurements=sum(r['measurement_status']=='local_surface_estimate' for r in rows)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
