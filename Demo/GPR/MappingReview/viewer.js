'use strict';
(async function(){
 const status=document.getElementById('status'), canvas=document.getElementById('canvas');
 try {
 const response=await fetch('model_payload.json');if(!response.ok)throw Error('댐 모델을 불러올 수 없습니다.');const payload=await response.json(),model=payload.gltf;
 const gl=canvas.getContext('webgl2',{antialias:true,preserveDrawingBuffer:true});
 if(!gl)throw Error('WebGL2를 사용할 수 없습니다. WebGL2를 지원하는 브라우저에서 이 파일을 열어 주세요.');
 const decode=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
 const buffers=await Promise.all(payload.buffer_uris.map(async uri=>{const r=await fetch(uri);if(!r.ok)throw Error('모델 형상 파일을 불러올 수 없습니다.');return r.arrayBuffer();}));
 const widths={SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT4:16};
 const sizes={5120:1,5121:1,5122:2,5123:2,5125:4,5126:4};
 function acc(id){const a=model.accessors[id],v=model.bufferViews[a.bufferView],n=widths[a.type],size=sizes[a.componentType],stride=v.byteStride||n*size,offset=(v.byteOffset||0)+(a.byteOffset||0),dv=new DataView(buffers[v.buffer]);const out=new Float64Array(a.count*n);for(let i=0;i<a.count;i++)for(let j=0;j<n;j++){const q=offset+i*stride+j*size;let x;switch(a.componentType){case 5126:x=dv.getFloat32(q,true);break;case 5125:x=dv.getUint32(q,true);break;case 5123:x=dv.getUint16(q,true);break;case 5122:x=dv.getInt16(q,true);break;case 5121:x=dv.getUint8(q);break;default:x=dv.getInt8(q);}if(a.normalized&&a.componentType!==5126)x=a.componentType===5121?x/255:a.componentType===5123?x/65535:a.componentType===5120?Math.max(-1,x/127):Math.max(-1,x/32767);out[i*n+j]=x;}return out;}
 const identity=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
 function multiply(a,b){const c=new Array(16).fill(0);for(let j=0;j<4;j++)for(let i=0;i<4;i++)for(let k=0;k<4;k++)c[j*4+i]+=a[k*4+i]*b[j*4+k];return c;}
 function local(n){if(n.matrix)return n.matrix;const t=n.translation||[0,0,0],s=n.scale||[1,1,1],q=n.rotation||[0,0,0,1],[x,y,z,w]=q;return[(1-2*y*y-2*z*z)*s[0],(2*x*y+2*w*z)*s[0],(2*x*z-2*w*y)*s[0],0,(2*x*y-2*w*z)*s[1],(1-2*x*x-2*z*z)*s[1],(2*y*z+2*w*x)*s[1],0,(2*x*z+2*w*y)*s[2],(2*y*z-2*w*x)*s[2],(1-2*x*x-2*y*y)*s[2],0,t[0],t[1],t[2],1];}
 const draws=[], min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
 function walk(id,parent){const n=model.nodes[id],world=multiply(parent,local(n));if(n.mesh!==undefined){const mesh=model.meshes[n.mesh];for(const primitive of mesh.primitives){const p=acc(primitive.attributes.POSITION),normal=acc(primitive.attributes.NORMAL),uv=primitive.attributes.TEXCOORD_0===undefined?new Float64Array(p.length/3*2):acc(primitive.attributes.TEXCOORD_0),positions=new Float64Array(p.length),normals=new Float32Array(normal.length);for(let i=0;i<p.length;i+=3){for(let k=0;k<3;k++){positions[i+k]=world[k]*p[i]+world[4+k]*p[i+1]+world[8+k]*p[i+2]+world[12+k];min[k]=Math.min(min[k],positions[i+k]);max[k]=Math.max(max[k],positions[i+k]);normals[i+k]=world[k]*normal[i]+world[4+k]*normal[i+1]+world[8+k]*normal[i+2];}}const ix=primitive.indices===undefined?Float64Array.from({length:p.length/3},(_,i)=>i):acc(primitive.indices),indices=Uint32Array.from(ix),name=n.name||mesh.name||`시설 ${id}`;draws.push({name,node:n,nodeId:id,meshId:n.mesh,material:primitive.material,positions,normals,uv:new Float32Array(uv),indices,region:/^(SPW|NOF_R|NOF_L)_\d+/.test(name),terrain:/지형|수면/.test(name)||(/^SPW_\d+/.test(name)&&n.extras?.face===1)});}}for(const child of n.children||[])walk(child,world);}
 for(const id of model.scenes[model.scene||0].nodes)walk(id,identity());
 const center=min.map((x,k)=>(x+max[k])/2),extent=min.map((x,k)=>max[k]-x),radius=Math.hypot(...extent)/2;
 let triangles=0;for(const d of draws){for(let i=0;i<d.positions.length;i++)d.positions[i]-=center[i%3];triangles+=d.indices.length/3;}
 function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
 const vertex=`#version 300 es
 layout(location=0)in vec3 p;layout(location=1)in vec3 n;layout(location=2)in vec2 uv;uniform mat4 vp;out vec2 texcoord;out vec3 normal;void main(){gl_Position=vp*vec4(p,1.0);texcoord=uv;normal=normalize(n);}`;
 const fragment=`#version 300 es
 precision highp float;in vec2 texcoord;in vec3 normal;uniform sampler2D image;uniform vec4 color;uniform int mode;uniform vec3 pick;uniform bool selected;out vec4 pixel;void main(){if(mode==2){pixel=vec4(pick,1.0);return;}vec3 rgb=mode==0?texture(image,texcoord).rgb*color.rgb:color.rgb*(0.70+0.30*abs(dot(normalize(normal),normalize(vec3(0.4,0.8,0.5)))));if(selected)rgb=mix(rgb,vec3(1.0,0.88,0.25),0.55);pixel=vec4(rgb,1.0);}`;
 const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
 const loc=Object.fromEntries(['vp','image','color','mode','pick','selected'].map(k=>[k,gl.getUniformLocation(program,k)]));
 function attr(data,index,size){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);gl.enableVertexAttribArray(index);gl.vertexAttribPointer(index,size,gl.FLOAT,false,0,0);}
 for(const d of draws){d.vao=gl.createVertexArray();gl.bindVertexArray(d.vao);attr(new Float32Array(d.positions),0,3);attr(d.normals,1,3);attr(d.uv,2,2);const ib=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,d.indices,gl.STATIC_DRAW);}
 const textures=await Promise.all(payload.images.map(uri=>new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>{const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.REPEAT);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.REPEAT);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR);gl.generateMipmap(gl.TEXTURE_2D);resolve(t);};image.onerror=reject;image.src=uri;})));
 const minus=(a,b)=>a.map((x,k)=>x-b[k]),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],unit=a=>{const n=Math.hypot(...a);return a.map(x=>x/n);},dot=(a,b)=>a.reduce((s,x,k)=>s+x*b[k],0);
 function view(eye,target){const z=unit(minus(eye,target)),x=unit(cross([0,1,0],z)),y=cross(z,x);return[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1];}
 function perspective(aspect){const f=1/Math.tan(Math.PI/8),near=Math.max(radius/2000,0.005),far=radius*150;return[f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0];}
 function regionColor(d){let base=d.name.startsWith('SPW')?(d.node.extras?.face===1?[0.20,0.42,0.88]:[0.24,0.70,0.96]):d.name.startsWith('NOF_R')?[0.90,0.45,0.76]:[1.0,0.73,0.30];const n=Number(d.name.split('_').at(-1)),v=0.64+0.36*((n*0.61803398875)%1);return[...base.map(x=>x*v),1];}
 let azimuth=-Math.PI*0.38,elevation=Math.PI*0.22,distance=radius*2.5,target=[0,0,0],selected=-1,drag=null,pickFramebuffer=null,pickColor=null,pickDepth=null,lastSize='';

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

 function resize(){const ratio=Math.min(window.devicePixelRatio||1,2),w=Math.round(canvas.clientWidth*ratio),h=Math.round(canvas.clientHeight*ratio);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}const key=w+':'+h;if(lastSize===key)return;lastSize=key;if(pickFramebuffer){gl.deleteFramebuffer(pickFramebuffer);gl.deleteTexture(pickColor);gl.deleteRenderbuffer(pickDepth);}pickFramebuffer=gl.createFramebuffer();pickColor=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,pickColor);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,null);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);pickDepth=gl.createRenderbuffer();gl.bindRenderbuffer(gl.RENDERBUFFER,pickDepth);gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT24,w,h);gl.bindFramebuffer(gl.FRAMEBUFFER,pickFramebuffer);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,pickColor,0);gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,pickDepth);gl.bindFramebuffer(gl.FRAMEBUFFER,null);}
 function draw(picking=false){resize();gl.bindFramebuffer(gl.FRAMEBUFFER,picking?pickFramebuffer:null);gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(...(picking?[0,0,0,1]:[0.10,0.16,0.23,1]));gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.disable(gl.BLEND);gl.disable(gl.CULL_FACE);if(picking)gl.disable(gl.DITHER);else gl.enable(gl.DITHER);const eye=[target[0]+distance*Math.cos(elevation)*Math.sin(azimuth),target[1]+distance*Math.sin(elevation),target[2]+distance*Math.cos(elevation)*Math.cos(azimuth)],vp=multiply(perspective(canvas.width/canvas.height),view(eye,target));gl.useProgram(program);gl.uniformMatrix4fv(loc.vp,false,new Float32Array(vp));gl.uniform1i(loc.image,0);const colorMode=document.getElementById('mode').value==='regions';for(let i=0;i<draws.length;i++){const d=draws[i];if(!isDrawVisible(d))continue;const mat=model.materials[d.material],pbr=mat.pbrMetallicRoughness||{},t=pbr.baseColorTexture;gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,textures[t?model.textures[t.index].source:textures.length-1]);gl.uniform1i(loc.mode,picking?2:colorMode||!t?1:0);gl.uniform4fv(loc.color,colorMode?(d.region?regionColor(d):[0.44,0.50,0.57,1]):pbr.baseColorFactor||[1,1,1,1]);const id=i+1;gl.uniform3f(loc.pick,(id&255)/255,((id>>8)&255)/255,((id>>16)&255)/255);gl.uniform1i(loc.selected,i===selected||isAnchorHighlight(d.name)?1:0);gl.bindVertexArray(d.vao);gl.drawElements(gl.TRIANGLES,d.indices.length,gl.UNSIGNED_INT,0);}drawMarkers(vp,picking);gl.bindFramebuffer(gl.FRAMEBUFFER,null);}
 function updateStatus(){status.textContent=`현재 Dam_model · GPR ${visibleCandidates.length} / ${candidates.length}개 · 가로 Z 일정 · 세로 간격 유지 · 지정 기준점 적용`; }
 canvas.addEventListener('contextmenu',e=>e.preventDefault());canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,move:0,pan:e.button===2||e.shiftKey};});canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.move+=Math.abs(dx)+Math.abs(dy);drag.x=e.clientX;drag.y=e.clientY;if(drag.pan){const s=distance/canvas.clientHeight*0.8;target[0]-=dx*s*Math.cos(azimuth);target[2]+=dx*s*Math.sin(azimuth);target[1]+=dy*s;}else{azimuth-=dx*0.006;elevation=Math.max(-Math.PI/2+0.01,Math.min(Math.PI/2-0.01,elevation+dy*0.006));}draw();});canvas.addEventListener('pointerup',e=>{if(!drag)return;const click=drag.move<5&&!drag.pan;drag=null;if(click){if(anchorMode){pickAnchor(e.clientX,e.clientY);return;}draw(true);gl.bindFramebuffer(gl.FRAMEBUFFER,pickFramebuffer);const r=canvas.getBoundingClientRect(),p=new Uint8Array(4),x=Math.floor((e.clientX-r.left)*canvas.width/r.width),y=canvas.height-1-Math.floor((e.clientY-r.top)*canvas.height/r.height);gl.readPixels(x,y,1,1,gl.RGBA,gl.UNSIGNED_BYTE,p);const index=p[0]+p[1]*256+p[2]*65536-1;gl.bindFramebuffer(gl.FRAMEBUFFER,null);if(index>=draws.length&&index<draws.length+candidates.length){selectCandidate(index-draws.length,false);}else{selected=index;const d=draws[index];document.getElementById('selection').textContent=d?d.name+' · 모델 표면 선택':'선택 없음';draw();}}});canvas.addEventListener('pointercancel',()=>drag=null);canvas.addEventListener('wheel',e=>{e.preventDefault();distance=Math.max(0.25,Math.min(radius*25,distance*Math.exp(e.deltaY*0.001)));draw();},{passive:false});
 document.getElementById('fit').onclick=fitDam;document.getElementById('front').onclick=()=>{azimuth=-Math.PI*.18;elevation=.12;draw();};document.getElementById('top').onclick=()=>{elevation=Math.PI/2-.01;draw();};document.getElementById('save').onclick=()=>{draw();const a=document.createElement('a');a.download='daecheongdam_preview.png';a.href=canvas.toDataURL('image/png');a.click();};document.getElementById('mode').onchange=()=>draw();document.getElementById('land').onchange=()=>draw();window.addEventListener('resize',()=>draw());const request=new URLSearchParams(location.search);const initial=candidates.findIndex(r=>r.damage_id===request.get('candidate'));selectCandidate(initial>=0?initial:0,false);if(initial>=0)focusCandidate(initial,6);else fitDam();if(request.get('anchors')==='1')document.getElementById('anchor-fit').click();updateStatus();window.MODEL_PREVIEW_READY=true;window.MODEL_PREVIEW_STATS={meshCount:draws.length,triangleCount:triangles,center,extent};
 }catch(e){document.getElementById('error').hidden=false;document.getElementById('error').textContent=String(e)+'\n'+String(e.stack||'');status.textContent='표시에 실패했습니다. 오류 내용을 확인해 주세요.';window.MODEL_PREVIEW_ERROR=String(e);}
})();
