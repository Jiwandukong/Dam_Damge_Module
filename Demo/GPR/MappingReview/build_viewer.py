"""Prepare local GPR inspection data and preview textures; keep source files unchanged."""

import csv
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

HERE = Path(__file__).resolve().parent
GPR = HERE.parent
ROOT = GPR.parent
MODEL = ROOT / "Dam_model/Daecheongdam/daecheongdam_regions_grid5m.gltf"
CSV = GPR / "Output/Result/ANM_result.csv"
REVIEW = HERE / "proximity_review.json"
sys.path.insert(0, str(GPR / "Code"))
from coordinate_mapping import load_transform, candidate_world


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Renderer patch target differs: " + old[:100])
    return source.replace(old, new, 1)


def main():
    model_sha, csv_sha = sha(MODEL), sha(CSV)
    review = json.loads(REVIEW.read_text())
    if review["model_sha256"] != model_sha or review["candidate_csv_sha256"] != csv_sha:
        raise ValueError("Model/CSV changed since the proximity diagnostic; refresh the diagnostic first")
    model = json.loads(MODEL.read_text())
    for uri, expected in review["buffer_sha256"].items():
        if sha(MODEL.parent/uri) != expected:
            raise ValueError("Model geometry changed since the proximity diagnostic")
    calibration = load_transform()
    proposal = json.loads((HERE / 'anchor_proposal.json').read_text())
    if proposal['model_sha256'] != model_sha:
        raise ValueError('Model differs from the start/end guide')
    if not proposal.get('horizontal_z_locked') or proposal['start']['world_xyz'][2] != proposal['end']['world_xyz'][2]:
        raise ValueError('The start/end guide must maintain constant world Z')
    if calibration.get('mesh_alignment', {}).get('model_sha256') != model_sha:
        raise ValueError('Recompute horizontal mesh alignment for the current model')
    unit_review = json.loads((GPR / "Data/Calibration/unit_review.json").read_text())
    if unit_review["csv_sha256"] != csv_sha:
        raise ValueError("Run Code/audit_coordinate_units.py for the current CSV first")
    legacy = json.loads((GPR / "Data/Calibration/legacy_positions.json").read_text())
    legacy_points = {r["damage_id"]:r["world_xyz"] for r in legacy["candidates"]}
    textures = HERE / "textures"
    textures.mkdir(exist_ok=True)
    image_uris = []
    for index, record in enumerate(model["images"]):
        target = textures / f"texture_{index}.jpg"
        if not target.exists():
            with Image.open(MODEL.parent / record["uri"]) as image:
                image = image.convert("RGB")
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                image.save(target, quality=90)
        image_uris.append(f"/textures/{target.name}")
    Image.new("RGB", (1, 1), "white").save(textures / "white.png")
    payload = {"gltf": model, "buffer_uris": ["/model/" + b["uri"] for b in model["buffers"]],
               "images": image_uris + ["/textures/white.png"]}
    (HERE / "model_payload.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    details = {d["damage_id"]: d for d in review["results"]["grid"]["details"]}
    with CSV.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    candidates = []
    for row in rows:
        point = [float(row["world_center_" + a + "_m"]) for a in "xyz"]
        detail = details[row["damage_id"]]
        if point != detail["candidate_xyz"]:
            raise ValueError("Candidate XYZ differs from the diagnostic")
        near = detail["nearest_point_xyz"]
        freq, line = int(row["freq_mhz"]), int(row["line_no"])
        candidates.append({**row, "freq_mhz": freq, "line_no": line, "x_m": float(row["x_m"]),
            "t_ns": float(row["t_ns"]), "world_xyz": point, "gltf_xyz": [point[0], point[2], -point[1]],
            "pixel_center": json.loads(row["pixel_nodes_json"])[0],
            "nearest_grid_id": detail["nearest_node"], "nearest_grid_member": detail["parent_node"],
            "grid_distance_m": detail["distance_m"], "nearest_grid_xyz": near,
            "legacy_world_xyz": legacy_points[row["damage_id"]],
            "legacy_gltf_xyz": [legacy_points[row["damage_id"]][0],legacy_points[row["damage_id"]][2],-legacy_points[row["damage_id"]][1]],
            "nearest_grid_gltf": [near[0], near[2], -near[1]],
            "overlay_url": f"/overlays/{freq}/LINE_{line:03d}_anomaly.png"})
    lines = []
    for scan in unit_review["scans"]:
        start = candidate_world(scan['line_no'], 0, calibration)
        end = candidate_world(scan['line_no'], scan['total_distance_m'], calibration)
        lines.append(dict(freq_mhz=scan["freq_mhz"],line_no=scan["line_no"],length_m=scan["total_distance_m"],
            display_length_m=scan['model_line_length_m'],
            start_gltf=[start[0],start[2],-start[1]],end_gltf=[end[0],end[2],-end[1]]))
    (HERE / "anomalies.json").write_text(json.dumps({"model_sha256": model_sha, "csv_sha256": csv_sha,
        "coordinate_assumption": review["axis_convention"], "calibration":calibration,
        "grid_summary":{k:v for k,v in review["results"]["grid"].items() if k!="details"},
        "anchor_proposal":proposal, "lines":lines, "candidates": candidates}, ensure_ascii=False, separators=(",", ":")))
    js = (HERE / "renderer_base.js").read_text()
    js = replace_once(js, "const payload=JSON.parse(document.getElementById('model-data').textContent), model=payload.gltf;", "const response=await fetch('model_payload.json');if(!response.ok)throw Error('댐 모델을 불러올 수 없습니다.');const payload=await response.json(),model=payload.gltf;")
    js = replace_once(js, "const buffers=payload.buffers.map(s=>decode(s).buffer);", "const buffers=await Promise.all(payload.buffer_uris.map(async uri=>{const r=await fetch(uri);if(!r.ok)throw Error('모델 형상 파일을 불러올 수 없습니다.');return r.arrayBuffer();}));")
    js = replace_once(js, "if(d.terrain&&!document.getElementById('land').checked)continue;", "if(!isDrawVisible(d))continue;")
    js = replace_once(js, "textures[model.textures[t.index].source]", "textures[t?model.textures[t.index].source:textures.length-1]")
    js = replace_once(js, "gl.uniform1i(loc.mode,picking?2:colorMode?1:0);", "gl.uniform1i(loc.mode,picking?2:colorMode||!t?1:0);")
    js = replace_once(js, "gl.uniform1i(loc.selected,i===selected?1:0);", "gl.uniform1i(loc.selected,i===selected||isAnchorHighlight(d.name)?1:0);")
    js = replace_once(js, "}gl.bindFramebuffer(gl.FRAMEBUFFER,null);}\n function updateStatus", "}drawMarkers(vp,picking);gl.bindFramebuffer(gl.FRAMEBUFFER,null);}\n function updateStatus")
    js = replace_once(js, " function resize(){", "\n" + (HERE / "controls.js").read_text() + "\n" + (HERE / "anchor_controls.js").read_text() + "\n function resize(){")
    start = js.index("canvas.addEventListener('pointerup'")
    end = js.index("canvas.addEventListener('pointercancel'", start)
    js = js[:start] + """canvas.addEventListener('pointerup',e=>{if(!drag)return;const click=drag.move<5&&!drag.pan;drag=null;if(click){if(anchorMode){pickAnchor(e.clientX,e.clientY);return;}draw(true);gl.bindFramebuffer(gl.FRAMEBUFFER,pickFramebuffer);const r=canvas.getBoundingClientRect(),p=new Uint8Array(4),x=Math.floor((e.clientX-r.left)*canvas.width/r.width),y=canvas.height-1-Math.floor((e.clientY-r.top)*canvas.height/r.height);gl.readPixels(x,y,1,1,gl.RGBA,gl.UNSIGNED_BYTE,p);const index=p[0]+p[1]*256+p[2]*65536-1;gl.bindFramebuffer(gl.FRAMEBUFFER,null);if(index>=draws.length&&index<draws.length+candidates.length){selectCandidate(index-draws.length,false);}else{selected=index;const d=draws[index];document.getElementById('selection').textContent=d?d.name+' · 모델 표면 선택':'선택 없음';draw();}}});""" + js[end:]
    js = js.replace("Math.max(radius*0.03,", "Math.max(0.25,")
    js = replace_once(js, "document.getElementById('front').onclick=()=>{target=[0,0,0];distance=radius*2.2;azimuth=Math.PI*0.43;elevation=Math.PI*0.14;draw();};", "document.getElementById('front').onclick=()=>{azimuth=-Math.PI*.18;elevation=.12;draw();};")
    js = replace_once(js, "document.getElementById('top').onclick=()=>{target=[0,0,0];distance=radius*2.4;elevation=Math.PI/2-0.01;draw();};", "document.getElementById('top').onclick=()=>{elevation=Math.PI/2-.01;draw();};")
    js = replace_once(js, "document.getElementById('fit').onclick=()=>{target=[0,0,0];distance=radius*2.5;azimuth=-Math.PI*0.38;elevation=Math.PI*0.22;draw();};", "document.getElementById('fit').onclick=fitDam;")
    js = replace_once(js, "draw();updateStatus();window.MODEL_PREVIEW_READY=true;", "const request=new URLSearchParams(location.search);const initial=candidates.findIndex(r=>r.damage_id===request.get('candidate'));selectCandidate(initial>=0?initial:0,false);if(initial>=0)focusCandidate(initial,6);else fitDam();if(request.get('anchors')==='1')document.getElementById('anchor-fit').click();updateStatus();window.MODEL_PREVIEW_READY=true;")
    js = js.replace("document.getElementById('error').style.display='block';", "document.getElementById('error').hidden=false;")
    js = js.replace("status.textContent='表示できませんでした。上のエラーを確認してください。';", "status.textContent='表示に失敗しました。';").replace("表示に失敗しました。", "표시에 실패했습니다. 오류 내용을 확인해 주세요.")
    old_status = js[js.index(" function updateStatus(){"):js.index(" canvas.addEventListener('contextmenu'")]
    js = js.replace(old_status, " function updateStatus(){status.textContent=`현재 Dam_model · GPR ${visibleCandidates.length} / ${candidates.length}개 · 가로 Z 일정 · 세로 간격 유지 · 지정 기준점 적용`; }\n", 1)
    (HERE / "viewer.js").write_text(js)
    if sha(MODEL) != model_sha or sha(CSV) != csv_sha:
        raise AssertionError("Source model/CSV changed during viewer build")
    print(f"Built local viewer: {len(candidates)} candidates; source model and CSV preserved")


if __name__ == "__main__":
    main()
