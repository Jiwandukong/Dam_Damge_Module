#!/usr/bin/env python3
"""Build local-only viewer assets from the current ROV CSVs and dam model."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import csv
import hashlib
import json
import shutil
import numpy as np
from PIL import Image
from paths import ROOT, WORK, read_json, write_json, sha
from demo_mapping import SpillwayWaterlineGridSampler
from download_assets import ensure_dam_model
from result_export import read_class_results

HERE = Path(__file__).resolve().parent
DIST = WORK/'web_viewer/dist'
BASE = HERE/'renderer_base.js'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Renderer patch target changed: '+old[:80])
    return source.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dam-model', type=Path)
    args = parser.parse_args()
    result_dir = ROOT/'Output/Result'
    rows = read_class_results(result_dir)
    if not rows or any(row['mapping_status'] != 'demo_random' for row in rows):
        raise ValueError('The current ROV outputs have no demonstration 3D positions')
    seeds = {int(row['mapping_seed']) for row in rows}
    if len(seeds) != 1:
        raise ValueError('Expected one mapping seed in the final CSV')
    model_path = args.dam_model or ensure_dam_model()
    sampler = SpillwayWaterlineGridSampler(model_path, seed=seeds.pop())
    sampler.prepare_frames({(result_dir/row['source_image_path']).resolve() for row in rows})
    mapping = sampler.summary()
    geometry = {(g['node_id'],g['mesh_id'],g['primitive_id']):g for g in sampler.geometries if not g['water']}
    DIST.mkdir(parents=True, exist_ok=True)
    (DIST/'result.csv').unlink(missing_ok=True)
    assets = DIST/'model'; assets.mkdir(exist_ok=True)
    model = read_json(sampler.path)
    for buffer in model['buffers']:
        shutil.copy2(sampler.path.parent/buffer['uri'], assets/buffer['uri'])
    textures = []
    for i, record in enumerate(model['images']):
        destination = assets/f'texture_{i}.jpg'
        with Image.open(sampler.path.parent/record['uri']) as image:
            image = image.convert('RGB'); image.thumbnail((512,512),Image.Resampling.LANCZOS)
            image.save(destination,quality=88)
        textures.append('model/'+destination.name)
    Image.new('RGB',(1,1),'white').save(assets/'white.png')
    write_json(DIST/'model_payload.json',dict(gltf=model,
               buffer_uris=['model/'+b['uri'] for b in model['buffers']],images=textures+['model/white.png']))
    images = DIST/'images'; images.mkdir(exist_ok=True)
    records = []; copied = set(); source_hashes = {}
    for name in ['CRC_result.csv','SPL_result.csv']:
        csv_path = ROOT/'Output/Result'/name
        source_hashes[str(csv_path.relative_to(ROOT))] = sha(csv_path)
        shutil.copy2(csv_path,DIST/csv_path.name)
    for row in rows:
        kind=row['damage_type']
        if kind not in {'CRC','SPL'} or row['mapping_status']!='demo_random':
            raise ValueError('Unexpected final result class or position status')
        source = (result_dir/row['source_image_path']).resolve()
        overlay = (result_dir/row['overlay_path']).resolve()
        if ROOT not in source.parents or ROOT not in overlay.parents:
            raise ValueError('Image path escapes the ROV module')
        raw_preview = 'images/'+source.stem+'.jpg'
        with Image.open(source) as image:
            width,height = image.size
            if source not in copied:
                preview = image.convert('RGB');preview.thumbnail((1280,1280),Image.Resampling.LANCZOS)
                preview.save(DIST/raw_preview,quality=90);copied.add(source)
                source_hashes[str(source.relative_to(ROOT))] = sha(source)
        overlay_preview = 'images/'+row['damage_id']+'.png'
        with Image.open(overlay) as image:
            preview = image.convert('RGB');preview.thumbnail((1280,1280),Image.Resampling.LANCZOS)
            preview.save(DIST/overlay_preview)
        source_hashes[str(overlay.relative_to(ROOT))] = sha(overlay)
        key = tuple(int(row[k]) for k in ['surface_node_id','surface_mesh_id','surface_primitive_id'])
        g = geometry[key]
        expected = dict(grid_id=g['name'], grid_guid=g['guid'], member_id=g['member_id'],
                        member_name=g['member_name'], member_node_id=str(g['member_node_id']))
        if any(row[field] != value for field, value in expected.items()):
            raise ValueError('CSV grid or member differs from the dam model: '+row['damage_id'])
        triangle = g['triangles'][int(row['surface_triangle_id'])]
        normal = np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);normal/=np.linalg.norm(normal)
        point = [float(row[k]) for k in ['world_center_x_m','world_center_y_m','world_center_z_m']]
        barycentric = np.asarray(json.loads(row['surface_barycentric_json']))
        if not np.allclose(barycentric @ triangle, point, rtol=0, atol=1e-7):
            raise ValueError('CSV position differs from the dam surface: '+row['damage_id'])
        records.append(dict(id=row['damage_id'],kind=kind,name=row['damage_name_ko'],image=row['image'],
            image_width=width,image_height=height,nodes=json.loads(row['pixel_nodes_json']),
            world_xyz=point,gltf_xyz=[point[0],point[2],-point[1]],
            normal_gltf=[float(normal[0]),float(normal[2]),float(-normal[1])],
            node_id=int(row['surface_node_id']),member=row['member_name'],section=row['section_name'],grid=row['grid_id'],
            member_id=row['member_id'],member_node_id=int(row['member_node_id']),grid_guid=row['grid_guid'],
            grid_band=mapping['eligible_grid_metadata'][row['grid_id']]['band'],
            confidence=float(row['DRI']),area_px=int(row['area_px']),water_level=float(row['water_level_m']),
            submergence=float(row['submergence_m']),mapping_status=row['mapping_status'],
            frame_group=row['frame_group_id'],frame_group_index=int(row['frame_group_index']),
            frame_group_size=int(row['frame_group_size']),frame_offset_x_m=float(row['frame_offset_x_m']),
            crop_size=int(row['crop_size_px']),raw_preview=raw_preview,overlay_preview=overlay_preview))
    records.sort(key=lambda r:r['id'])
    metadata=dict(count=len(records),frames=len(copied),counts={k:sum(r['kind']==k for r in records) for k in ['CRC','SPL']},
                  purpose='synthetic demo locations; neural-network damage predictions',seed=mapping['seed'],
                  model_sha256=sampler.model_sha,buffer_sha256=sampler.buffer_hashes,source_sha256=source_hashes,
                  placement_scope=mapping['scope'],eligible_grid_ids=mapping['eligible_grid_ids'],
                  frame_group_count=mapping['frame_group_count'],frame_step_m=mapping['frame_step_m'],
                  grid_band_counts={b:sum(r['grid_band']==b for r in records) for b in ['waterline','lower']},
                  result_csvs=['CRC_result.csv','SPL_result.csv'],
                  grid_and_member_ids_present=all(r['grid'] and r['member_id'] for r in records),
                  external_asset_requests=False)
    write_json(DIST/'rov_results.json',dict(summary=metadata,records=records))
    for name in ['index.html','style.css']:
        shutil.copy2(HERE/name,DIST/name)
    js = BASE.read_text()
    js = replace_once(js,"const payload=JSON.parse(document.getElementById('model-data').textContent), model=payload.gltf;",
        "const response=await fetch('model_payload.json');if(!response.ok)throw Error('모델 자료를 불러올 수 없습니다.');const payload=await response.json(),model=payload.gltf;")
    js = replace_once(js,"const buffers=payload.buffers.map(s=>decode(s).buffer);",
        "const buffers=await Promise.all(payload.buffer_uris.map(async uri=>{const r=await fetch(uri);if(!r.ok)throw Error('모델 형상 파일을 불러올 수 없습니다.');return r.arrayBuffer();}));")
    js = replace_once(js,"if(d.terrain&&!document.getElementById('land').checked)continue;",
        "if(d.name==='수면'&&!document.getElementById('water').checked)continue;if(d.terrain&&d.name!=='수면'&&!document.getElementById('land').checked)continue;const layer=document.getElementById('layer').value;if(layer==='grid'&&!d.region||layer==='member'&&d.region)continue;")
    js = replace_once(js,"textures[model.textures[t.index].source]","textures[t?model.textures[t.index].source:textures.length-1]")
    js = replace_once(js,"gl.uniform1i(loc.mode,picking?2:colorMode?1:0);","gl.uniform1i(loc.mode,picking?2:colorMode||!t?1:0);")
    js = replace_once(js,"}gl.bindFramebuffer(gl.FRAMEBUFFER,null);}\n function updateStatus",
        "}drawMarkers(vp,picking);gl.bindFramebuffer(gl.FRAMEBUFFER,null);}\n function updateStatus")
    js = replace_once(js," function resize(){",'\n'+(HERE/'controls.js').read_text()+'\n function resize(){')
    start=js.index("canvas.addEventListener('pointerup'");end=js.index("canvas.addEventListener('pointercancel'",start)
    js=js[:start]+"""canvas.addEventListener('pointerup',e=>{if(!drag)return;const click=drag.move<5&&!drag.pan;drag=null;if(click){draw(true);gl.bindFramebuffer(gl.FRAMEBUFFER,pickFramebuffer);const r=canvas.getBoundingClientRect(),p=new Uint8Array(4),x=Math.floor((e.clientX-r.left)*canvas.width/r.width),y=canvas.height-1-Math.floor((e.clientY-r.top)*canvas.height/r.height);gl.readPixels(x,y,1,1,gl.RGBA,gl.UNSIGNED_BYTE,p);const index=p[0]+p[1]*256+p[2]*65536-1;gl.bindFramebuffer(gl.FRAMEBUFFER,null);if(index>=draws.length&&index<draws.length+records.length){selectDamage(index-draws.length,false);}else{selected=index;const d=draws[index];document.getElementById('selection').textContent=d?`${d.name} · 모델 부재`:'선택 없음';draw();}}});"""+js[end:]
    js=replace_once(js,"draw();updateStatus();window.MODEL_PREVIEW_READY=true;",
        "draw();updateStatus();const requested=new URLSearchParams(location.search).get('damage');const first=records.findIndex(r=>r.id===requested);selectDamage(first>=0?first:0,false);fitDamage();window.MODEL_PREVIEW_READY=true;")
    js=js.replace("document.getElementById('error').style.display='block';","document.getElementById('error').hidden=false;")
    js=js.replace('表示できませんでした。上のエラーを確認してください。','표시할 수 없습니다. 오류 내용을 확인해 주세요.')
    # Replace the old model-only status with the ROV observation count.
    start=js.index(' function updateStatus(){');end=js.index('\n',start)
    js=js[:start]+" function updateStatus(){status.textContent=`대청댐 ${draws.length.toLocaleString()} 부재 · 손상 ${records.length}건 · 표시 ${visibleIndices().length}건`; }"+js[end:]
    (DIST/'viewer.js').write_text(js)
    write_json(WORK/'web_viewer/build_manifest.json',metadata)
    print(json.dumps(dict(viewer=str(DIST),count=len(records),frames=len(copied),local_assets_only=True),ensure_ascii=False))


if __name__=='__main__':main()
