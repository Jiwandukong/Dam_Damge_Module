// Exact triangle picks for reviewing field anchors. Picks stay in this browser;
// copying the JSON does not publish or overwrite the result CSV.
let anchorMode=null;
const anchorGuide=review.anchor_proposal;
const anchorApplied=review.calibration.model_anchors_applied?review.calibration.selected_anchors:null;
const anchorStorageKey='gpr-anchor-review:horizontal-v1:'+review.model_sha256;
let anchorPoints=anchorApplied?JSON.parse(JSON.stringify({start:anchorApplied.start,end:anchorApplied.end})):{start:null,end:null};
try{const saved=JSON.parse(localStorage.getItem(anchorStorageKey)||'null');if(saved?.model_sha256===review.model_sha256){for(const role of ['start','end']){const p=saved[role];if(p?.world_xyz?.length===3&&p.world_xyz.every(Number.isFinite)&&typeof p.mesh==='string')anchorPoints[role]=p;}}}catch(_e){}
if(anchorPoints.end&&(!anchorPoints.start||anchorPoints.end.world_xyz[2]!==anchorPoints.start.world_xyz[2]))anchorPoints.end=null;
const anchorPointBuffer=makeMarkerBuffer(),anchorLineBuffer=makeMarkerBuffer();
function worldToGltf(p){return [p[0],p[2],-p[1]];}
function gltfToWorld(p){return [p[0],-p[2],p[1]];}
function anchorRecord(){return {model_sha256:review.model_sha256,reference_frequency_mhz:400,reference_line_no:1,measured_length_m:anchorGuide.measured_length_m,horizontal_z_locked:true,start:anchorPoints.start,end:anchorPoints.end};}
function rebuildAnchors(){
  const points=[],lines=[],colors=[[.35,.95,.92],[.98,.51,.95]];
  if(document.getElementById('anchor-guide').checked){
    for(const [i,role]of ['start','end'].entries()){const p=anchorGuide[role].gltf_xyz;points.push(...markerValue(p,colors[i],-200,0));lines.push(...markerValue(p,colors[i],-200,0));}
  }
  for(const [i,role]of ['start','end'].entries()){const point=anchorPoints[role];if(point)points.push(...markerValue(worldToGltf(point.world_xyz),i?[1,.3,.25]:[.4,1,.35],-200,0));}
  if(anchorPoints.start&&anchorPoints.end){for(const [i,role]of ['start','end'].entries())lines.push(...markerValue(worldToGltf(anchorPoints[role].world_xyz),i?[1,.3,.25]:[.4,1,.35],-200,0));}
  upload(anchorPointBuffer,points);upload(anchorLineBuffer,lines);
}
function drawAnchors(vp){
  gl.useProgram(markerProgram);gl.uniformMatrix4fv(markerLoc.vp,false,new Float32Array(vp));gl.uniform1i(markerLoc.picking,0);gl.uniform1i(markerLoc.ghost,0);gl.uniform1f(markerLoc.pointScale,1.5*Math.min(devicePixelRatio||1,2));
  gl.disable(gl.DEPTH_TEST);gl.uniform1i(markerLoc.lineMode,1);gl.bindVertexArray(anchorLineBuffer.vao);gl.drawArrays(gl.LINES,0,anchorLineBuffer.count);
  gl.uniform1i(markerLoc.lineMode,0);gl.bindVertexArray(anchorPointBuffer.vao);gl.drawArrays(gl.POINTS,0,anchorPointBuffer.count);gl.depthFunc(gl.LESS);
}
function isDrawVisible(d){
  if(d.terrain&&!document.getElementById('land').checked)return false;
  if(d.name==='옥외_부대설비'&&!document.getElementById('anchor-equipment').checked)return false;
  if(document.getElementById('anchor-context').checked)return d.region||d.name==='옥외_부대설비';
  const layer=document.getElementById('layer').value;
  return !(layer==='grid'&&!d.region||layer==='member'&&d.region);
}
function isAnchorHighlight(name){return document.getElementById('anchor-context').checked&&name==='NOF_R_0157';}
function anchorMessage(message){document.getElementById('anchor-status').textContent=message;}
function renderAnchorValues(){
  const values=document.getElementById('anchor-values');values.replaceChildren();
  for(const role of ['start','end']){const point=anchorPoints[role];values.append(dom('dt',role==='start'?'시작':'끝'),dom('dd',point?point.mesh+' · '+point.world_xyz.map(x=>x.toFixed(6)).join(', ')+' m':'미지정'));}
  if(anchorPoints.start&&anchorPoints.end){const measured=anchorGuide.measured_length_m,length=Math.hypot(...minus(anchorPoints.end.world_xyz,anchorPoints.start.world_xyz));values.append(dom('dt','시작·끝 높이 차'),dom('dd',(anchorPoints.end.world_xyz[2]-anchorPoints.start.world_xyz[2]).toFixed(6)+' m'),dom('dt','수평 거리'),dom('dd',length.toFixed(3)+' m'),dom('dt','측정 길이'),dom('dd',measured.toFixed(3)+' m'),dom('dt','길이 차이'),dom('dd',(length-measured).toFixed(3)+' m'));}
  document.getElementById('anchor-json').value=anchorPoints.start||anchorPoints.end?JSON.stringify(anchorRecord(),null,2):'';
  for(const [id,role]of [['anchor-start','start'],['anchor-end','end']])document.getElementById(id).setAttribute('aria-pressed',String(anchorMode===role));
  rebuildAnchors();
}
function focusAnchor(role){
  const active=anchorPoints[role]||anchorApplied?.[role];
  const point=active?worldToGltf(active.world_xyz):role==='start'?anchorGuide.grid_surface_centroid.gltf_xyz:anchorGuide.end.gltf_xyz;
  // The end is a mesh boundary guide. Focus a metre before it on the dam side.
  const direction=anchorApplied?minus(worldToGltf(anchorApplied.end.world_xyz),worldToGltf(anchorApplied.start.world_xyz)):minus(anchorGuide.end.gltf_xyz,anchorGuide.start.gltf_xyz),norm=Math.hypot(...direction);
  target=point.map((x,k)=>x-center[k]-(role==='end'?direction[k]/norm:0));distance=12;cameraDirection(candidates[0]);draw();
}
function beginAnchor(role){
  if(role==='end'&&!anchorPoints.start){anchorMessage('시작점을 먼저 지정하세요. 끝점은 시작점과 같은 Z의 댐 표면에 맞춥니다.');return;}
  anchorMode=role;document.getElementById('anchor-context').checked=true;document.getElementById('anchor-guide').checked=!anchorApplied;
  document.getElementById('anchor-equipment').checked=role==='start';
  anchorMessage(role==='start'?'NOF_R_0157 안의 실제 측정 시작점을 클릭하세요.':'끝점의 Z를 시작점과 같은 높이로 고정합니다. 설비 직전의 댐 표면을 클릭하세요.');renderAnchorValues();focusAnchor(role);
}
const anchorBounds=draws.map(d=>{const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];for(let i=0;i<d.positions.length;i++){const k=i%3;low[k]=Math.min(low[k],d.positions[i]);high[k]=Math.max(high[k],d.positions[i]);}return {low,high};});
function rayBox(eye,direction,bounds){
  let low=0,high=Infinity;
  for(let i=0;i<3;i++){if(Math.abs(direction[i])<1e-12){if(eye[i]<bounds.low[i]||eye[i]>bounds.high[i])return Infinity;continue;}const a=(bounds.low[i]-eye[i])/direction[i],b=(bounds.high[i]-eye[i])/direction[i];low=Math.max(low,Math.min(a,b));high=Math.min(high,Math.max(a,b));if(low>high)return Infinity;}
  return low;
}
function surfaceAt(clientX,clientY){
  const rect=canvas.getBoundingClientRect(),nx=2*(clientX-rect.left)/rect.width-1,ny=1-2*(clientY-rect.top)/rect.height;
  const eye=[target[0]+distance*Math.cos(elevation)*Math.sin(azimuth),target[1]+distance*Math.sin(elevation),target[2]+distance*Math.cos(elevation)*Math.cos(azimuth)];
  const forward=unit(minus(target,eye)),right=unit(cross(forward,[0,1,0])),up=cross(right,forward),tan=Math.tan(Math.PI/8);
  const direction=unit(forward.map((x,i)=>x+right[i]*nx*tan*rect.width/rect.height+up[i]*ny*tan));
  let nearest=Infinity,result=null;
  for(let index=0;index<draws.length;index++){
    const d=draws[index];if(!isDrawVisible(d)||rayBox(eye,direction,anchorBounds[index])>nearest)continue;
    const p=d.positions,ix=d.indices;
    for(let offset=0;offset<ix.length;offset+=3){
      const a=Array.from(p.subarray(ix[offset]*3,ix[offset]*3+3)),b=Array.from(p.subarray(ix[offset+1]*3,ix[offset+1]*3+3)),c=Array.from(p.subarray(ix[offset+2]*3,ix[offset+2]*3+3));
      const e1=minus(b,a),e2=minus(c,a),h=cross(direction,e2),det=dot(e1,h);if(Math.abs(det)<1e-12)continue;
      const inverse=1/det,s=minus(eye,a),u=inverse*dot(s,h);if(u<0||u>1)continue;
      const q=cross(s,e1),v=inverse*dot(direction,q);if(v<0||u+v>1)continue;
      const length=inverse*dot(e2,q);if(length<0||length>=nearest)continue;
      nearest=length;const position=eye.map((x,k)=>x+length*direction[k]+center[k]);result={mesh:d.name,world_xyz:gltfToWorld(position),normal_world:gltfToWorld(unit(cross(e1,e2))),triangle_index:offset/3};
    }
  }
  return result;
}
function horizontalSurfacePoint(hit,z){
  let nearest=Infinity,result=null;
  for(const d of draws){
    if(d.name!==hit.mesh)continue;
    const p=d.positions,ix=d.indices;
    for(let offset=0;offset<ix.length;offset+=3){
      const vertices=[0,1,2].map(i=>gltfToWorld(Array.from(p.subarray(ix[offset+i]*3,ix[offset+i]*3+3), (v,k)=>v+center[k]))),section=[];
      for(let i=0;i<3;i++){
        const a=vertices[i],b=vertices[(i+1)%3],za=a[2]-z,zb=b[2]-z;
        if(Math.abs(za)<1e-9)section.push([a[0],a[1],z]);
        if(za*zb<0){const t=za/(za-zb);section.push([a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]),z]);}
      }
      if(section.length<2)continue;
      for(let i=0;i<section.length;i++)for(let j=i+1;j<section.length;j++){
        const a=section[i],b=section[j],delta=minus(b,a),length2=dot(delta,delta);
        if(length2<1e-18)continue;
        const t=Math.max(0,Math.min(1,dot(minus(hit.world_xyz,a),delta)/length2)),point=a.map((v,k)=>v+t*delta[k]);point[2]=z;
        const distance=Math.hypot(point[0]-hit.world_xyz[0],point[1]-hit.world_xyz[1]);
        if(distance<nearest){nearest=distance;result={...hit,world_xyz:point,raw_world_xyz:hit.world_xyz,normal_world:unit(cross(minus(vertices[1],vertices[0]),minus(vertices[2],vertices[0]))),triangle_index:offset/3,horizontal_z_locked:true};}
      }
    }
  }
  return result;
}
function pickAnchor(clientX,clientY){
  let hit=surfaceAt(clientX,clientY);if(!hit){anchorMessage('모델 표면을 클릭해 주세요.');return;}
  if(anchorMode==='start'&&hit.mesh!=='NOF_R_0157'){anchorMessage('현재 클릭한 면은 '+hit.mesh+'입니다. 시작 영역 NOF_R_0157을 클릭해 주세요.');return;}
  if(anchorMode==='end'&&!/^NOF_R_\d+$/.test(hit.mesh)){anchorMessage('옥외 부대설비 직전의 댐 격자 표면을 클릭해 주세요. 현재 면: '+hit.mesh);return;}
  if(anchorMode==='end'){
    hit=horizontalSurfacePoint(hit,anchorPoints.start.world_xyz[2]);
    if(!hit){anchorMessage('이 격자에는 시작점과 같은 Z의 표면이 없습니다. 같은 높이의 댐 격자를 클릭하세요.');return;}
  }else anchorPoints.end=null;
  anchorPoints[anchorMode]=hit;anchorMode=null;
  try{localStorage.setItem(anchorStorageKey,JSON.stringify(anchorRecord()));}catch(_e){}
  renderAnchorValues();draw();
  if(anchorPoints.start&&anchorPoints.end){const length=Math.hypot(...minus(anchorPoints.end.world_xyz,anchorPoints.start.world_xyz)),delta=length-anchorGuide.measured_length_m;anchorMessage(Math.abs(delta)>.10?'두 점 사이 길이가 측정 길이와 '+delta.toFixed(3)+'m 다릅니다. 실제 시작·끝 위치와 기준 측선을 확인해 주세요.':'시작·끝이 지정됐습니다. 좌표 복사로 이 채팅에 전달해 주세요.');}
  else anchorMessage('시작점이 지정됐습니다. 끝점 찍기로 나머지 기준점을 지정하세요.');
}
document.getElementById('anchor-start').onclick=()=>beginAnchor('start');
document.getElementById('anchor-end').onclick=()=>beginAnchor('end');
document.getElementById('anchor-cancel').onclick=()=>{anchorMode=null;renderAnchorValues();anchorMessage('기준점 선택을 종료했습니다.');draw();};
document.getElementById('anchor-fit').onclick=()=>{
  anchorMode=null;document.getElementById('anchor-context').checked=true;document.getElementById('anchor-guide').checked=!anchorApplied;renderAnchorValues();
  document.getElementById('anchor-equipment').checked=true;
  const anchors=anchorApplied?[worldToGltf(anchorApplied.start.world_xyz),worldToGltf(anchorApplied.end.world_xyz)]:[anchorGuide.start.gltf_xyz,anchorGuide.end.gltf_xyz];
  const positions=[...anchors,...review.lines.filter(r=>r.line_no===1).flatMap(r=>[r.start_gltf,r.end_gltf])];
  const low=[0,1,2].map(k=>Math.min(...positions.map(p=>p[k]))),high=[0,1,2].map(k=>Math.max(...positions.map(p=>p[k])));target=low.map((v,k)=>(v+high[k])/2-center[k]);distance=Math.hypot(...minus(high,low))*1.8;cameraDirection(candidates[0]);anchorMessage(anchorApplied?'전달한 시작·끝점을 CSV에 적용했습니다. 세로 측선 간격과 길이를 유지하고 가로 위치를 두 기준점에 맞췄습니다.':'청록·보라 점과 선은 자동 제안입니다. 시작·끝점 찍기로 실제 기준점을 지정하세요. 현재 CSV는 유지됩니다.');draw();
};
for(const id of ['anchor-context','anchor-guide','anchor-equipment'])document.getElementById(id).onchange=()=>{rebuildAnchors();draw();};
document.getElementById('anchor-clear').onclick=()=>{anchorPoints={start:null,end:null};anchorMode=null;try{localStorage.removeItem(anchorStorageKey);}catch(_e){}renderAnchorValues();anchorMessage('지정점을 지웠습니다.');draw();};
document.getElementById('anchor-copy').onclick=()=>{const input=document.getElementById('anchor-json');if(!input.value){anchorMessage('먼저 기준점을 지정해 주세요.');return;}input.focus();input.select();let copied=false;try{copied=document.execCommand('copy');}catch(_e){}anchorMessage(copied?'좌표를 복사했습니다. 이 채팅에 붙여 넣어 주세요.':'좌표 입력란을 선택했습니다. Ctrl+C로 복사해 주세요.');};
window.GPR_ANCHOR_API={begin:beginAnchor,record:anchorRecord,surfaceAt,horizontalSurfacePoint,mode:()=>anchorMode,guide:anchorGuide};
renderAnchorValues();
if(anchorApplied)anchorMessage('전달한 기준점이 적용됐습니다. 기준 길이는 26.408m이며 원자료의 26.664m는 보존합니다.');
