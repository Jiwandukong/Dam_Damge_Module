const reviewResponse = await fetch('anomalies.json');
if (!reviewResponse.ok) throw Error('GPR 후보 자료를 불러올 수 없습니다.');
const review = await reviewResponse.json(), candidates = review.candidates;
const frequencyColors = {400:[.37,.89,.76],900:[.38,.73,1],1600:[.91,.61,1],2600:[1,.72,.38]};
let currentCandidate = -1, visibleCandidates = candidates.map((_,i)=>i),visibleLines=review.lines;
document.getElementById('grid-range').textContent=`${review.grid_summary.min_distance_m.toFixed(2)}–${review.grid_summary.max_distance_m.toFixed(2)} m`;
window.GPR_REVIEW_STATS = {total:candidates.length,visible:candidates.length,selected:null,model_sha256:review.model_sha256,scale:review.calibration.scale,calibration_status:review.calibration.calibration_status,horizontal_distance_scale:review.calibration.horizontal_distance_scale||1,mapped_grid_count:candidates.filter(r=>r.grid_id).length};
const markerVertex = `#version 300 es
layout(location=0)in vec3 position;layout(location=1)in vec3 color;layout(location=2)in vec3 pickColor;layout(location=3)in float recordId;
uniform mat4 vp;uniform float selectedId;uniform float pointScale;uniform bool ghost;
out vec3 rgb;out vec3 pick;
void main(){bool chosen=abs(recordId-selectedId)<0.1;gl_Position=vp*vec4(position,1.0);gl_PointSize=(ghost?12.0:chosen?26.0:12.0)*pointScale;rgb=chosen&&!ghost?vec3(1.0):color;pick=pickColor;}`;
const markerFragment = `#version 300 es
precision highp float;in vec3 rgb;in vec3 pick;uniform bool picking;uniform bool lineMode;uniform bool ghost;out vec4 pixel;
void main(){if(lineMode){pixel=vec4(rgb,1.0);return;}float d=length(gl_PointCoord-vec2(.5));if(d>.5)discard;vec3 c=d>.35?(ghost?vec3(1.0):vec3(.025,.055,.08)):rgb;pixel=vec4(picking?pick:c,1.0);}`;
const markerProgram=gl.createProgram();
gl.attachShader(markerProgram,shader(gl.VERTEX_SHADER,markerVertex));gl.attachShader(markerProgram,shader(gl.FRAGMENT_SHADER,markerFragment));gl.linkProgram(markerProgram);
if(!gl.getProgramParameter(markerProgram,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(markerProgram));
const markerLoc=Object.fromEntries(['vp','selectedId','pointScale','picking','lineMode','ghost'].map(k=>[k,gl.getUniformLocation(markerProgram,k)]));
function makeMarkerBuffer(){
  const vao=gl.createVertexArray(),buffer=gl.createBuffer();gl.bindVertexArray(vao);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
  for(const [i,size,offset]of [[0,3,0],[1,3,12],[2,3,24],[3,1,36]]){gl.enableVertexAttribArray(i);gl.vertexAttribPointer(i,size,gl.FLOAT,false,40,offset);}
  return {vao,buffer,count:0};
}
const pointBuffer=makeMarkerBuffer(),offsetBuffer=makeMarkerBuffer(),surveyBuffer=makeMarkerBuffer(),legacyBuffer=makeMarkerBuffer();
function markerValue(position,color,index,pickId){return [...position.map((x,k)=>x-center[k]),...color,(pickId&255)/255,((pickId>>8)&255)/255,((pickId>>16)&255)/255,index];}
function upload(buffer,values){gl.bindVertexArray(buffer.vao);gl.bindBuffer(gl.ARRAY_BUFFER,buffer.buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(values),gl.DYNAMIC_DRAW);buffer.count=values.length/10;}
function rebuildMarkers(){
  // Draw the selected point last, including in the picking buffer, so nearby
  // coincident candidates cannot obscure the currently selected ID.
  const order=visibleCandidates.filter(i=>i!==currentCandidate);if(visibleCandidates.includes(currentCandidate))order.push(currentCandidate);
  const values=[];for(const i of order){const r=candidates[i];values.push(...markerValue(r.gltf_xyz,frequencyColors[r.freq_mhz],i,draws.length+i+1));}upload(pointBuffer,values);
  const r=candidates[currentCandidate];
  upload(offsetBuffer,r?[...markerValue(r.gltf_xyz,[1,.73,.38],-2,0),...markerValue(r.nearest_grid_gltf,[.12,.25,.32],-3,0)]:[]);
  const frequency=document.getElementById('frequency').value,line=document.getElementById('line-filter').value;
  visibleLines=review.lines.filter(r=>(frequency==='all'||String(r.freq_mhz)===frequency)&&(line==='all'||String(r.line_no)===line));
  upload(surveyBuffer,visibleLines.flatMap(r=>[...markerValue(r.start_gltf,frequencyColors[r.freq_mhz],-100,0),...markerValue(r.end_gltf,frequencyColors[r.freq_mhz],-100,0)]));
  upload(legacyBuffer,visibleCandidates.flatMap(i=>markerValue(candidates[i].legacy_gltf_xyz,[.55,.62,.67],-100,0)));
  window.GPR_REVIEW_STATS.line_count=visibleLines.length;
}
function drawMarkers(vp,picking){
  gl.useProgram(markerProgram);gl.uniformMatrix4fv(markerLoc.vp,false,new Float32Array(vp));gl.uniform1f(markerLoc.selectedId,currentCandidate);
  gl.uniform1f(markerLoc.pointScale,Math.min(devicePixelRatio||1,2));gl.uniform1i(markerLoc.picking,picking?1:0);
  if(document.getElementById('xray').checked)gl.disable(gl.DEPTH_TEST);else{gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);}
  if(!picking&&document.getElementById('survey-lines').checked){gl.bindVertexArray(surveyBuffer.vao);gl.uniform1i(markerLoc.ghost,0);gl.uniform1i(markerLoc.lineMode,1);gl.drawArrays(gl.LINES,0,surveyBuffer.count);}
  if(!picking&&document.getElementById('legacy').checked){gl.bindVertexArray(legacyBuffer.vao);gl.uniform1i(markerLoc.ghost,0);gl.uniform1i(markerLoc.lineMode,0);gl.drawArrays(gl.POINTS,0,legacyBuffer.count);}
  if(!picking&&document.getElementById('offset').checked&&offsetBuffer.count){
    gl.bindVertexArray(offsetBuffer.vao);gl.uniform1i(markerLoc.ghost,0);gl.uniform1i(markerLoc.lineMode,1);gl.drawArrays(gl.LINES,0,2);
    gl.uniform1i(markerLoc.lineMode,0);gl.uniform1i(markerLoc.ghost,1);gl.drawArrays(gl.POINTS,1,1);
  }
  gl.uniform1i(markerLoc.lineMode,0);gl.uniform1i(markerLoc.ghost,0);gl.bindVertexArray(pointBuffer.vao);gl.drawArrays(gl.POINTS,0,pointBuffer.count);gl.depthFunc(gl.LESS);
  if(!picking)drawAnchors(vp);
}
const dom=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
function renderList(){
  const list=document.getElementById('candidate-list');list.replaceChildren();
  document.getElementById('candidate-count').textContent=`${visibleCandidates.length} / ${candidates.length}개`;
  for(const i of visibleCandidates){const r=candidates[i],b=dom('button',undefined,'candidate-row');b.type='button';b.dataset.candidate=r.damage_id;b.setAttribute('aria-pressed',String(i===currentCandidate));
    b.style.setProperty('--candidate-color',`rgb(${frequencyColors[r.freq_mhz].map(x=>Math.round(x*255)).join(',')})`);
    b.append(dom('strong',r.damage_id),dom('span',`${r.freq_mhz} MHz`),dom('small',`LINE_${String(r.line_no).padStart(3,'0')} · ${r.nearest_grid_id} · ${r.grid_distance_m.toFixed(3)} m`));
    b.onclick=()=>selectCandidate(i,true);list.append(b);
  }
  if(!visibleCandidates.length)list.append(dom('p','조건에 맞는 후보가 없습니다.','empty'));
}
function cameraDirection(r){
  const normal=review.calibration.selected_anchors?.start?.normal_world;
  const n=normal?unit([normal[0],normal[2],-normal[1]]):unit(minus(r.gltf_xyz,r.nearest_grid_gltf));azimuth=Math.atan2(n[0],n[2])+.7;elevation=Math.max(-1,Math.min(1,Math.asin(n[1])-.18));
}
function focusCandidate(i,d=4){const r=candidates[i];if(!r)return;target=r.gltf_xyz.map((x,k)=>x-center[k]);distance=d;cameraDirection(r);draw();}
function selectCandidate(i,focus=true){
  const r=candidates[i];if(!r)return;currentCandidate=i;
  selected=draws.findIndex(d=>d.name===r.nearest_grid_id);
  const detail=document.getElementById('candidate-detail');detail.replaceChildren(dom('h2',r.damage_id),dom('div',`${r.freq_mhz} MHz · LINE_${String(r.line_no).padStart(3,'0')}`,'tag'));
  const fields=dom('dl',undefined,'fields'),add=(name,value,cls)=>fields.append(dom('dt',name),dom('dd',value,cls));
  for(const [axis,value]of r.world_xyz.entries())add(`XYZ${axis===0?' · X':axis===1?' · Y':' · Z'}`,value.toFixed(6)+' m','coords');
  add('최근접 격자',r.nearest_grid_id);add('상위 부재',r.nearest_grid_member);add('격자 거리',r.grid_distance_m.toFixed(4)+' m');
  add('보정 상태',review.calibration.model_anchors_applied?'지정 기준점 적용 · 세로 간격 유지':'거리 단위 보정 · 위치 검토 중');add('CSV 매핑',r.grid_id||'부재·격자 미기입');add('원자료 거리',r.x_m.toFixed(3)+' m');
  if(r.line_no<=3&&review.calibration.horizontal_distance_scale)add('모델 표시 거리',(r.x_m*review.calibration.horizontal_distance_scale).toFixed(3)+' m');
  add('왕복시간',r.t_ns.toFixed(3)+' ns');
  add('영상 중심',r.pixel_center.map(v=>v.toFixed(2)).join(' / ')+' px');detail.append(fields);
  const actions=dom('div',undefined,'detail-actions');const zoom=dom('button','선택점 확대');zoom.onclick=()=>focusCandidate(i);const range=dom('button','GPR 범위');range.onclick=fitCandidates;actions.append(zoom,range);detail.append(actions);
  const heading=dom('div',undefined,'image-heading');heading.append(dom('span','해당 측선 Overlay'));const a=dom('a','원본 크기로 보기');a.href=r.overlay_url;a.target='_blank';a.rel='noopener';heading.append(a);detail.append(heading);
  const image=dom('img',undefined,'overlay-image'+(r.line_no<=3?' long':''));image.src=r.overlay_url;image.alt=`${r.freq_mhz} MHz LINE_${String(r.line_no).padStart(3,'0')} GPR 이상 후보 Overlay`;detail.append(image);
  detail.append(dom('p','Overlay의 빨간 십자는 측선 전체의 후보입니다. 현재 선택 후보의 영상 중심은 위 좌표를 확인하세요.'));
  document.getElementById('selection').textContent=`${r.damage_id} · ${r.freq_mhz} MHz · 최근접 ${r.nearest_grid_id} · 거리 ${r.grid_distance_m.toFixed(3)} m`;
  window.GPR_REVIEW_STATS.selected=r.damage_id;
  rebuildMarkers();renderList();if(focus)focusCandidate(i,6);else draw();
  history.replaceState(null,'',`?candidate=${encodeURIComponent(r.damage_id)}`);
}
function fitCandidates(){
  const records=visibleCandidates.map(i=>candidates[i]);if(!records.length)return;
  const positions=[...records.map(r=>r.gltf_xyz),...visibleLines.flatMap(r=>[r.start_gltf,r.end_gltf])];
  const low=[0,1,2].map(k=>Math.min(...positions.map(p=>p[k]))),high=[0,1,2].map(k=>Math.max(...positions.map(p=>p[k])));
  target=low.map((v,k)=>(v+high[k])/2-center[k]);distance=Math.max(12,Math.hypot(...minus(high,low))*2.5);cameraDirection(records[0]);draw();
}
function fitDam(){
  const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
  for(const d of draws){if(d.terrain)continue;for(let i=0;i<d.positions.length;i++){const k=i%3;low[k]=Math.min(low[k],d.positions[i]);high[k]=Math.max(high[k],d.positions[i]);}}
  target=low.map((v,k)=>(v+high[k])/2);
  distance=Math.hypot(...minus(high,low))*1.1;azimuth=-Math.PI*.38;elevation=Math.PI*.22;draw();
}
function applyFilters(){
  const freq=document.getElementById('frequency').value,line=document.getElementById('line-filter').value,search=document.getElementById('search').value.trim().toUpperCase();
  visibleCandidates=candidates.flatMap((r,i)=>(freq==='all'||String(r.freq_mhz)===freq)&&(line==='all'||String(r.line_no)===line)&&(!search||r.damage_id.includes(search))?[i]:[]);
  window.GPR_REVIEW_STATS.visible=visibleCandidates.length;
  if(!visibleCandidates.includes(currentCandidate)){
    if(visibleCandidates.length){selectCandidate(visibleCandidates[0],false);return;}
    currentCandidate=-1;selected=-1;window.GPR_REVIEW_STATS.selected=null;document.getElementById('candidate-detail').replaceChildren(dom('p','조건에 맞는 후보가 없습니다.'));document.getElementById('selection').textContent='선택 없음';
  }
  rebuildMarkers();renderList();draw();updateStatus();
}
for(let n=1;n<=12;n++){const option=dom('option',`LINE_${String(n).padStart(3,'0')}`);option.value=String(n);document.getElementById('line-filter').append(option);}
document.getElementById('frequency').onchange=applyFilters;document.getElementById('line-filter').onchange=applyFilters;document.getElementById('search').oninput=applyFilters;
document.getElementById('candidate-fit').onclick=fitCandidates;document.getElementById('candidate-close').onclick=()=>focusCandidate(currentCandidate);
for(const id of ['xray','offset','layer','survey-lines','legacy'])document.getElementById(id).onchange=()=>draw();
window.GPR_REVIEW_API={
  select:(id,focus=true)=>{const i=candidates.findIndex(r=>r.damage_id===id);if(i<0)throw Error('알 수 없는 후보');selectCandidate(i,focus);},
  fit:fitCandidates,
  filter:(freq='all',line='all')=>{document.getElementById('frequency').value=String(freq);document.getElementById('line-filter').value=String(line);document.getElementById('search').value='';applyFilters();},
  selected:()=>candidates[currentCandidate]||null,
  camera:()=>({target:[...target],distance,azimuth,elevation}),
};
rebuildMarkers();renderList();
