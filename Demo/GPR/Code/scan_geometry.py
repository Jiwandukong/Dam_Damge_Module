"""Read metric trace spacing from DZT headers, independently of display units."""

import math
from pathlib import Path
import struct


def dzt_scan_geometry(path):
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(54)
    if len(header) != 54:
        raise ValueError(f"Truncated DZT header: {path}")
    header_count = struct.unpack_from("<h", header, 2)[0]
    samples = struct.unpack_from("<h", header, 4)[0]
    bits = struct.unpack_from("<h", header, 6)[0]
    spm = struct.unpack_from("<f", header, 14)[0]
    position = struct.unpack_from("<f", header, 22)[0]
    channels = struct.unpack_from("<h", header, 52)[0]
    if samples <= 0 or bits not in (8, 16, 32) or not math.isfinite(spm) or spm <= 0:
        raise ValueError(f"DZT lacks valid samples or scans-per-metre spacing: {path}")
    offset = 1024 * (header_count if header_count < 1024 else channels)
    trace_bytes = samples * (bits // 8)
    payload = path.stat().st_size - offset
    if offset < 0 or payload <= 0 or payload % trace_bytes:
        raise ValueError(f"Invalid DZT trace payload: {path}")
    traces = payload // trace_bytes
    return dict(n_traces=traces, n_samples=samples, scans_per_metre=spm,
                dx_m=1.0/spm, total_distance_m=(traces-1)/spm, start_m=position)
