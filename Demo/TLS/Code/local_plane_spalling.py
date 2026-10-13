#!/usr/bin/env python3
"""Find geometric depression candidates using overlapping local reference planes.

Input: the two original E57 partial clouds. Output classes are algorithm
candidates, never human ground truth. Normal polarity is an explicit assumption.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import runtime
import numpy as np
from scipy import ndimage
import pye57

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_source(part, data_dir=None):
    path = (Path(data_dir) if data_dir is not None else ROOT/'Data')/(part+'.e57')
    cache = runtime.WORK/(part+'_source.npz')
    manifest = cache.with_suffix('.json')
    fingerprint = sha256(path)
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text()).get('source_sha256') == fingerprint:
        with np.load(cache) as saved:
            return saved['xyz'], saved['rgb'], saved['intensity'], fingerprint
    with pye57.E57(str(path)) as source:
        if source.scan_count != 1:
            raise ValueError('Expected one scan in each original partial E57')
        header = source.get_header(0)
        data = source.read_scan_raw(0, ignore_unsupported_fields=True)
        xyz = np.column_stack([data['cartesian'+key] for key in 'XYZ'])
        if header.has_pose():
            xyz = source.to_global(xyz, header.rotation, header.translation)
        rgb = np.column_stack([data['color'+key] for key in ('Red','Green','Blue')])
        intensity = data.get('intensity', np.zeros(len(xyz), dtype=np.float32))
    # Keep the original record order: source_vertex_id indexes the E57 scan.
    np.savez(cache, xyz=xyz, rgb=rgb, intensity=intensity)
    manifest.write_text(json.dumps({'source_sha256':fingerprint, 'source':str(path)})+'\n')
    return xyz, rgb, intensity, fingerprint


def surface_frame(xyz, outward):
    valid = np.isfinite(xyz).all(axis=1)
    points = xyz[valid]
    rng = np.random.default_rng(12026)
    sample = points if len(points) < 100000 else points[rng.choice(len(points), 100000, replace=False)]
    center = sample.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((sample-center).T))
    basis = vectors[:, [2,1,0]].T
    if np.dot(basis[2], outward) < 0:
        basis[2] *= -1
    if basis[0,0] < 0:
        basis[0] *= -1
    if np.dot(np.cross(basis[0],basis[1]), basis[2]) < 0:
        basis[1] *= -1
    # Refit only the central surface band to keep distant stray returns from
    # tilting the coordinate frame. This frame is NOT the damage reference plane.
    projected = (sample-center)@basis.T
    keep = np.abs(projected[:,2]-np.median(projected[:,2])) < .25
    values, vectors = np.linalg.eigh(np.cov((sample[keep]-sample[keep].mean(axis=0)).T))
    normal = vectors[:,0]
    if normal@outward < 0:
        normal *= -1
    along = basis[0]-normal*(basis[0]@normal)
    along /= np.linalg.norm(along)
    across = np.cross(normal,along)
    basis = np.array([along, across, normal])
    center = sample[keep].mean(axis=0)
    return center, basis, (xyz-center)@basis.T, valid


def aggregate_grid(uvw, valid, cfg):
    res = cfg['grid_resolution_m']
    median = np.median(uvw[valid,2])
    kept = valid & (np.abs(uvw[:,2]-median) <= cfg['global_normal_distance_limit_m'])
    uv_origin = np.floor(uvw[kept,:2].min(axis=0)/res)*res
    ij = np.zeros((len(uvw),2),dtype=np.int32)
    ij[kept] = np.floor((uvw[kept,:2]-uv_origin)/res).astype(np.int32)
    nx,ny = ij[kept].max(axis=0)+1
    ids = np.full(len(uvw),-1,dtype=np.int32)
    ids[kept] = ij[kept,1]*nx+ij[kept,0]
    source = np.flatnonzero(kept)
    # Sort by cell then surface depth, so the per-cell median is exact and does
    # not require a Python loop over hundreds of thousands of cells.
    order = np.lexsort((uvw[source,2],ids[source]))
    source = source[order]
    occupied,starts,counts = np.unique(ids[source],return_index=True,return_counts=True)
    low = starts+(counts-1)//2
    high = starts+counts//2
    z = np.full(nx*ny,np.nan,dtype=np.float64)
    z[occupied] = (uvw[source[low],2]+uvw[source[high],2])/2
    count = np.zeros(nx*ny,dtype=np.int32)
    count[occupied] = counts
    return {'z':z.reshape(ny,nx), 'counts':count.reshape(ny,nx), 'point_cell':ids,
            'kept':kept, 'uv_origin':uv_origin, 'nx':int(nx), 'ny':int(ny), 'resolution_m':res}


def fit_reference_plane(uv, w, cfg, rng):
    """RANSAC on equally weighted cell medians, followed by inlier refits."""
    if len(w) < cfg['minimum_reference_cells']:
        return None
    choose = rng.choice(len(w),min(len(w),cfg['reference_max_sample_cells']),replace=False)
    q, z = uv[choose], w[choose]
    best = None
    best_count = 0
    trials = cfg['reference_ransac_trials']
    threshold = cfg['reference_ransac_threshold_m']
    max_slope = np.tan(np.radians(cfg['maximum_local_tilt_deg']))
    for _ in range(trials):
        triplet = rng.choice(len(z),3,replace=False)
        q0,q1,q2 = q[triplet]
        du1,dv1 = q1-q0;du2,dv2 = q2-q0
        det = du1*dv2-du2*dv1
        if abs(det) < cfg.get('reference_min_triangle_area2_m2', .03):
            continue
        dz1,dz2 = z[triplet[1:]]-z[triplet[0]]
        a = (dz1*dv2-dz2*dv1)/det
        b = (du1*dz2-du2*dz1)/det
        if np.hypot(a,b) > max_slope:
            continue
        c = z[triplet[0]]-a*q0[0]-b*q0[1]
        residual = z-(a*q[:,0]+b*q[:,1]+c)
        n = int(np.count_nonzero(np.abs(residual)/np.sqrt(1+a*a+b*b) <= threshold))
        if n > best_count:
            best_count = n;best = np.array([a,b,c])
    if best is None:
        return None
    design = np.column_stack([uv,np.ones(len(uv))])
    for _ in range(4):
        factor = np.sqrt(1+best[0]**2+best[1]**2)
        residual = (w-design@best)/factor
        inliers = np.abs(residual) <= threshold
        if inliers.sum() < cfg['minimum_reference_cells']:
            return None
        best = np.linalg.lstsq(design[inliers],w[inliers],rcond=None)[0]
    if np.hypot(*best[:2]) > max_slope:
        return None
    residual = (w-design@best)/np.sqrt(1+best[0]**2+best[1]**2)
    inliers = np.abs(residual) <= threshold
    ratio = float(inliers.mean())
    quadrant = (uv[:,0]>=0).astype(int)+2*(uv[:,1]>=0).astype(int)
    quadrant_counts = np.bincount(quadrant[inliers],minlength=4)
    if ratio < cfg['minimum_reference_inlier_ratio'] or np.sum(quadrant_counts>=10) < cfg['minimum_reference_quadrants']:
        return None
    r = residual[inliers]
    sigma = max(.001, float(1.4826*np.median(np.abs(r-np.median(r)))))
    return {'coefficients':best,'sigma':sigma,'inlier_ratio':ratio,'reference_cells':len(w),
            'inlier_cells':int(inliers.sum()),'inlier_rmse_m':float(np.sqrt(np.mean(r*r)))}


def local_reference_grid(grid, cfg, progress=False):
    z,count = grid['z'],grid['counts']
    ny,nx = z.shape
    res = cfg['grid_resolution_m']
    valid = np.isfinite(z)&(count>=cfg['minimum_points_per_cell'])
    patch = cfg['reference_patch_size_m']/2
    core = cfg['evaluation_core_size_m']/2
    stride = cfg['patch_stride_m']
    if core>=patch or stride>2*core:
        raise ValueError('Reference patch must surround the evaluation core; stride cannot leave core gaps')
    fields = {name:np.zeros((ny,nx),dtype=np.float64) for name in ('weight','prediction','prediction_square','slope_u','slope_v','sigma')}
    support = np.zeros((ny,nx),dtype=np.uint16)
    origin = grid['uv_origin']
    x = origin[0]+(np.arange(nx)+.5)*res
    y = origin[1]+(np.arange(ny)+.5)*res
    centers_x = np.arange(np.floor(x[0]/stride)*stride,x[-1]+stride,stride)
    centers_y = np.arange(np.floor(y[0]/stride)*stride,y[-1]+stride,stride)
    rng = np.random.default_rng(cfg['random_seed'])
    planes = []
    attempts = 0
    for cy in centers_y:
        sy = slice(max(0,int(np.searchsorted(y,cy-patch))),min(ny,int(np.searchsorted(y,cy+patch,side='right'))))
        for cx in centers_x:
            sx = slice(max(0,int(np.searchsorted(x,cx-patch))),min(nx,int(np.searchsorted(x,cx+patch,side='right'))))
            if not valid[sy,sx].any():
                continue
            xx,yy = np.meshgrid(x[sx]-cx,y[sy]-cy)
            central = (np.abs(xx)<=core)&(np.abs(yy)<=core)
            if np.count_nonzero(valid[sy,sx]&central)<15:
                continue
            ring = valid[sy,sx]&~central
            attempts += 1
            fitted = fit_reference_plane(np.column_stack([xx[ring],yy[ring]]),z[sy,sx][ring],cfg,rng)
            if fitted is None:
                continue
            a,b,c = fitted['coefficients']
            prediction = a*xx+b*yy+c
            evaluate = valid[sy,sx]&central
            # Overlapping cores are blended by distance and fit quality. Only
            # independently supported surrounding-plane fits contribute.
            # The weight and its first derivative vanish at core boundaries;
            # adding/removing an overlapping fit must not create a depth step.
            taper = np.maximum(1-(xx/core)**2,0)**2*np.maximum(1-(yy/core)**2,0)**2
            weight = taper*fitted['inlier_ratio']/max(fitted['sigma'],.002)**2
            weight = np.where(evaluate,weight,0)
            for name,value in [('weight',1),('prediction',prediction),('prediction_square',prediction**2),('slope_u',a),('slope_v',b),('sigma',fitted['sigma'])]:
                fields[name][sy,sx] += weight*value
            support[sy,sx] += evaluate
            planes.append({**{key:value for key,value in fitted.items() if key!='coefficients'},
                           'plane_id':len(planes),'center_u_m':float(cx),'center_v_m':float(cy),
                           'slope_u':float(a),'slope_v':float(b),'center_w_m':float(c)})
            if progress and len(planes)%100==0:
                print(f'  planes={len(planes)} attempted={attempts}',flush=True)
    good = valid&(support>=cfg['minimum_plane_supports'])&(fields['weight']>0)
    for name in ('prediction','prediction_square','slope_u','slope_v','sigma'):
        np.divide(fields[name],fields['weight'],out=fields[name],where=fields['weight']>0)
        fields[name][~good] = np.nan
    factor = np.sqrt(1+fields['slope_u']**2+fields['slope_v']**2)
    disagreement = np.sqrt(np.maximum(fields['prediction_square']-fields['prediction']**2,0))/factor
    good &= disagreement<=cfg['maximum_reference_disagreement_m']
    disagreement[~valid] = np.nan
    depth = (fields['prediction']-z)/factor
    depth[~good] = np.nan
    threshold = np.maximum(np.maximum(cfg['minimum_depth_m'],cfg['noise_sigma_multiplier']*fields['sigma']),2*disagreement)
    return {**fields,'good':good,'support':support,'depth':depth,'threshold':threshold,
            'factor':factor,'disagreement':disagreement,'planes':planes,'attempted_patches':attempts,
            'resolution_m':res,'maximum_depth_m':cfg['maximum_depth_m']}


def candidate_regions(grid, reference, cfg):
    depth,threshold = reference['depth'],reference['threshold']
    mask = reference['good']&(depth>=threshold)&(depth<=cfg['maximum_depth_m'])
    radius = cfg['region_link_radius_cells']
    linked = ndimage.binary_dilation(mask,iterations=radius) if radius else mask
    labels,n = ndimage.label(linked,structure=np.ones((3,3)))
    labels = np.where(mask,labels,0)
    ids = np.unique(labels[mask])
    output = np.zeros_like(labels,dtype=np.int32)
    regions = []
    res = cfg['grid_resolution_m']
    boxes = ndimage.find_objects(labels)
    for label_id in ids:
        box = boxes[label_id-1]
        if box is None:
            continue
        ysl,xsl = box
        pad = cfg['region_surrounding_width_cells']+radius+1
        sy = slice(max(0,ysl.start-pad),min(labels.shape[0],ysl.stop+pad))
        sx = slice(max(0,xsl.start-pad),min(labels.shape[1],xsl.stop+pad))
        region = labels[sy,sx]==label_id
        area = float(np.nansum(reference['factor'][sy,sx][region])*res*res)
        points = int(grid['counts'][sy,sx][region].sum())
        if area<cfg['minimum_candidate_area_m2'] or points<cfg['minimum_candidate_points']:
            continue
        around = ndimage.binary_dilation(region,iterations=cfg['region_surrounding_width_cells'])&~region
        observed_around = around&reference['good'][sy,sx]
        surrounding_coverage = float(observed_around.sum()/max(1,around.sum()))
        shallow = observed_around&(depth[sy,sx]<threshold[sy,sx])
        shallow_fraction = float(shallow.sum()/max(1,observed_around.sum()))
        yi,xi = np.nonzero(region)
        uv = np.column_stack([xi,yi])*res
        if len(uv)>2:
            eig = np.linalg.eigvalsh(np.cov(uv.T))
            aspect = float(np.sqrt(max(eig[1],res*res/12)/max(eig[0],res*res/12)))
        else:
            aspect = 1.
        flags = []
        if surrounding_coverage<cfg['minimum_surrounding_coverage']:
            flags.append('coverage_or_boundary_review')
        if shallow_fraction<cfg['minimum_shallow_surrounding_fraction']:
            flags.append('open_depression_or_step_review')
        if aspect>=cfg['joint_like_aspect_ratio']:
            flags.append('elongated_joint_like_review')
        if area>=cfg['broad_region_area_m2']:
            flags.append('broad_depression_review')
        status = 'geometry_review' if flags else 'spalling_candidate'
        candidate_id = len(regions)+1
        output[sy,sx][region] = candidate_id
        d = depth[sy,sx][region]
        regions.append({'candidate_id':candidate_id,'status':status,'review_flags':';'.join(flags),
                        'covered_area_m2':area,'grid_cells':int(region.sum()),'cell_point_count':points,
                        'mean_depth_m':float(d.mean()),'median_depth_m':float(np.median(d)),
                        'p95_depth_m':float(np.quantile(d,.95)),'max_depth_m':float(d.max()),
                        'covered_volume_proxy_m3':float(np.sum(d*reference['factor'][sy,sx][region])*res*res),
                        'surrounding_coverage':surrounding_coverage,'shallow_surrounding_fraction':shallow_fraction,
                        'aspect_ratio':aspect,'center_u_m':float(grid['uv_origin'][0]+(xi.mean()+sx.start+.5)*res),
                        'center_v_m':float(grid['uv_origin'][1]+(yi.mean()+sy.start+.5)*res),
                        'mean_reference_w_m':float(reference['prediction'][sy,sx][region].mean()),
                        'mean_surface_w_m':float(grid['z'][sy,sx][region].mean())})
    return output,regions


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else ['candidate_id','status','review_flags','covered_area_m2','mean_depth_m']
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def point_results(uvw,grid,reference,region_map):
    ids = grid['point_cell']
    depth = np.full(len(uvw),np.nan,dtype=np.float32)
    candidate = np.full(len(uvw),-1,dtype=np.int32)
    valid = ids>=0
    candidate[~valid] = -2
    source = np.flatnonzero(valid)
    cell = ids[source]
    good = reference['good'].ravel()[cell]
    source = source[good];cell = cell[good]
    nx = grid['nx'];res = grid['resolution_m']
    u = grid['uv_origin'][0]+(cell%nx+.5)*res
    v = grid['uv_origin'][1]+(cell//nx+.5)*res
    a = reference['slope_u'].ravel()[cell]
    b = reference['slope_v'].ravel()[cell]
    predicted = reference['prediction'].ravel()[cell]+a*(uvw[source,0]-u)+b*(uvw[source,1]-v)
    d = (predicted-uvw[source,2])/reference['factor'].ravel()[cell]
    depth[source] = d
    candidate[source] = 0
    region = region_map.ravel()[cell]
    eligible = (region>0)&(d>=reference['threshold'].ravel()[cell])&(d<=reference['maximum_depth_m'])
    candidate[source[eligible]] = region[eligible]
    return depth,candidate


def write_candidate_ply(path,xyz,rgb,depth,candidate):
    selected = np.flatnonzero(candidate>0)
    records = np.empty(len(selected),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),
                                            ('depth_m','<f4'),('candidate_id','<i4'),('source_vertex_id','<u4')])
    for i,key in enumerate('xyz'):
        records[key] = xyz[selected,i]
    for i,key in enumerate(('red','green','blue')):
        records[key] = rgb[selected,i]
    records['depth_m'] = depth[selected];records['candidate_id'] = candidate[selected];records['source_vertex_id'] = selected
    header = ('ply\nformat binary_little_endian 1.0\ncomment Geometric candidates; not human ground truth\n'
              f'element vertex {len(records)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty float depth_m\n'
              'property int candidate_id\nproperty uint source_vertex_id\nend_header\n')
    with path.open('wb') as stream:
        stream.write(header.encode());records.tofile(stream)


def run_part(part,cfg,out,data_dir=None):
    start = time.perf_counter()
    print('PART '+part,flush=True)
    xyz,rgb,intensity,source_hash = load_source(part,data_dir)
    center,basis,uvw,valid = surface_frame(xyz,np.asarray(cfg['outward_hint_xyz']))
    grid = aggregate_grid(uvw,valid,cfg)
    print(f'  source={len(xyz):,} grid={grid["nx"]}x{grid["ny"]}',flush=True)
    reference = local_reference_grid(grid,cfg,progress=True)
    reference['resolution_m'] = cfg['grid_resolution_m']
    reference['maximum_depth_m'] = cfg['maximum_depth_m']
    region_map,regions = candidate_regions(grid,reference,cfg)
    depth,candidate = point_results(uvw,grid,reference,region_map)
    folder = out/part;folder.mkdir(parents=True,exist_ok=True)
    for row in regions:
        row['damage_id'] = f'{"TLS1" if part=="dacheung_part1" else "TLS2"}-{row["candidate_id"]:04d}'
        for position,level in [('reference',row['mean_reference_w_m']),('surface',row['mean_surface_w_m'])]:
            point = center+np.array([row['center_u_m'],row['center_v_m'],level])@basis
            for value,axis in zip(point,'xyz'):
                row[f'center_{position}_{axis}_m'] = float(value)
        row['point_count'] = int(np.count_nonzero(candidate==row['candidate_id']))
    plane_rows = []
    for plane in reference['planes']:
        p = dict(plane)
        n = basis[2]-p['slope_u']*basis[0]-p['slope_v']*basis[1];n/=np.linalg.norm(n)
        anchor = center+np.array([p['center_u_m'],p['center_v_m'],p['center_w_m']])@basis
        for axis,value in zip('xyz',n):p['normal_'+axis] = float(value)
        p['plane_d'] = float(-n@anchor)
        plane_rows.append(p)
    write_csv(folder/'candidates.csv',regions)
    write_csv(folder/'reference_planes.csv',plane_rows,fields=list(plane_rows[0]) if plane_rows else ['plane_id'])
    np.savez_compressed(folder/'point_candidates.npz',source_vertex_id=np.arange(len(xyz),dtype=np.uint32),
                        depth_m=depth,candidate_id=candidate,frame_center_xyz=center,frame_basis=basis,
                        labels_are_ground_truth=np.array(False))
    np.savez_compressed(folder/'surface_grid.npz',measured_w=grid['z'].astype(np.float32),point_counts=grid['counts'],
                        reference_w=reference['prediction'].astype(np.float32),depth_m=reference['depth'].astype(np.float32),
                        threshold_m=reference['threshold'].astype(np.float32),supports=reference['support'],
                        region_id=region_map,valid_reference=reference['good'],reference_disagreement_m=reference['disagreement'].astype(np.float32),uv_origin=grid['uv_origin'],
                        resolution_m=cfg['grid_resolution_m'],frame_center_xyz=center,frame_basis=basis)
    write_candidate_ply(folder/'candidate_points.ply',xyz,rgb,depth,candidate)
    sensitivity = []
    for threshold in [.015,.02,.025,.03,.04,.05]:
        alternate = dict(cfg,minimum_depth_m=threshold)
        ref = dict(reference,threshold=np.maximum(np.maximum(threshold,cfg['noise_sigma_multiplier']*reference['sigma']),2*reference['disagreement']))
        _,rows = candidate_regions(grid,ref,alternate)
        sensitivity.append({'minimum_depth_m':threshold,'all_regions':len(rows),
                            'spalling_candidate_regions':sum(r['status']=='spalling_candidate' for r in rows),
                            'covered_candidate_area_m2':sum(r['covered_area_m2'] for r in rows if r['status']=='spalling_candidate')})
    write_csv(folder/'depth_sensitivity.csv',sensitivity)
    summary = {'part':part,'source_file':part+'.e57','source_sha256':source_hash,'source_points':len(xyz),
               'grid_resolution_m':cfg['grid_resolution_m'],'occupied_cells':int((grid['counts']>0).sum()),
               'reliable_reference_cells':int(reference['good'].sum()),'reference_planes':len(plane_rows),
               'attempted_patches':reference['attempted_patches'],'outward_normal_xyz':basis[2].tolist(),
               'maximum_reference_disagreement_m':cfg['maximum_reference_disagreement_m'],
               'outward_direction_verified':cfg['outward_direction_verified'],'minimum_depth_m':cfg['minimum_depth_m'],
               'regions':len(regions),'spalling_candidates':sum(r['status']=='spalling_candidate' for r in regions),
               'geometry_review_regions':sum(r['status']=='geometry_review' for r in regions),
               'candidate_points':int(np.count_nonzero(candidate>0)),
               'unassessed_points':int(np.count_nonzero(candidate==-1)),
               'excluded_points':int(np.count_nonzero(candidate==-2)),
               'elapsed_seconds':round(time.perf_counter()-start,2),'labels_are_ground_truth':False,
               'point_label_semantics':{'-2':'excluded_nonfinite_or_distant','-1':'no_reliable_reference',
                                        '0':'assessed_not_candidate','positive':'candidate region ID'},
               'depth_sensitivity':sensitivity}
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'Code/config.json')
    parser.add_argument('--output',type=Path,default=runtime.WORK/'local_plane_spalling')
    parser.add_argument('--part',choices=['dacheung_part1','dachung_part2'],action='append')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.part:cfg['parts'] = args.part
    outward = np.asarray(cfg['outward_hint_xyz'],dtype=float)
    if outward.shape!=(3,) or not np.isfinite(outward).all() or np.linalg.norm(outward)==0:
        raise ValueError('A nonzero finite outward direction is required')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'run_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    summaries = [run_part(part,cfg,args.output) for part in cfg['parts']]
    (args.output/'run_summary.json').write_text(json.dumps({'method':'overlapping_ring_RANSAC_local_planes',
        'source':'original_E57','human_labels_used':False,'outward_direction_verified':cfg['outward_direction_verified'],
        'parts':summaries},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
