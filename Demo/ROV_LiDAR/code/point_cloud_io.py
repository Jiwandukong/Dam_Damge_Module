"""Read the CloudCompare PLY inputs without filtering or moving points."""
from pathlib import Path
import numpy as np
import laspy

def prepare_las(source, work_directory):
    source=Path(source).resolve()
    if source.suffix.lower() in {'.las','.laz'}:
        return source
    if source.suffix.lower()!='.ply':
        raise ValueError('Input must be the segmented PLY, LAS or LAZ; E57 is retained as raw data.')
    with source.open('rb') as f:
        header=[]
        while True:
            line=f.readline().decode('ascii').strip()
            header.append(line)
            if line=='end_header':break
            if not line or len(header)>100:raise ValueError('Invalid PLY header')
    if 'format ascii 1.0' not in header:
        raise ValueError('The existing pipeline inputs use ASCII PLY. Export ASCII PLY from CloudCompare.')
    count=int(next(s for s in header if s.startswith('element vertex ')).split()[-1])
    properties=[s.split()[-1] for s in header if s.startswith('property ')]
    if properties!=['x','y','z','red','green','blue','scalar_Intensity']:
        raise ValueError(f'Unexpected PLY properties: {properties}')
    data=np.loadtxt(source,skiprows=len(header),dtype=np.float64)
    if data.shape!=(count,7) or not np.isfinite(data).all():
        raise ValueError('Point count mismatch or nonfinite input. No points are silently removed.')
    xyz=data[:,:3]
    header_las=laspy.LasHeader(point_format=3,version='1.2')
    header_las.scales=np.array([0.001,0.001,0.001])
    header_las.offsets=xyz.min(axis=0)
    if np.any(np.ptp(xyz,axis=0)/header_las.scales>np.iinfo(np.int32).max):
        raise ValueError('Coordinate range exceeds LAS integer storage.')
    header_las.add_extra_dim(laspy.ExtraBytesParams(name='scalar_Intensity',type=np.float32))
    las=laspy.LasData(header_las)
    las.x,las.y,las.z=xyz[:,0],xyz[:,1],xyz[:,2]
    for i,name in enumerate(['red','green','blue'],3):
        if not np.all((data[:,i]>=0)&(data[:,i]<=255)&(data[:,i]==np.rint(data[:,i]))):
            raise ValueError('Invalid uint8 RGB values.')
        setattr(las,name,data[:,i].astype(np.uint16)*257)
    las.scalar_Intensity=data[:,6].astype(np.float32)
    las.intensity=np.clip(np.rint(data[:,6]),0,65535).astype(np.uint16)
    destination=Path(work_directory)/(source.stem+'.las')
    las.write(destination)
    # Full XYZ readback preserves the former input's millimetre storage precision.
    seen=0
    with laspy.open(destination) as reader:
        assert reader.header.point_count==count
        for points in reader.chunk_iterator(500000):
            n=len(points)
            for axis,values in enumerate([points.x,points.y,points.z]):
                assert np.max(np.abs(np.asarray(values)-xyz[seen:seen+n,axis]))<=0.00050001
            seen+=n
    assert seen==count
    print(f'PLY -> temporary LAS: {count:,} points; no filtering or coordinate transform.',flush=True)
    return destination
