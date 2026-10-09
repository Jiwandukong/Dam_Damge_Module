# GPRPy DZT reader notice
# MIT License
#
# Copyright (c) 2018 Near Surface Geophysics
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""GSSI DZT/DZX input helpers used by the Daecheong GPR demo.

The DZT reader follows the same header parsing, sample conversion, profile
sample and time-axis calculation used by GPRPy 1.0.14. It is
kept locally so the demo does not need GPRPy's GUI/VTK-only dependencies just
to read a DZT file. Physical trace spacing is 1/scans_per_metre, with N-1
intervals for N traces; this corrects the legacy distance endpoint inflation.
"""

from __future__ import annotations

import os
import struct
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _read_dzt_like_gprpy(dzt_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a GSSI DZT using the algorithm from GPRPy's gprIO_DZT.readdzt."""
    with open(dzt_path, "rb") as fid:
        fid.read(2)  # rh_tag
        rh_data = struct.unpack("<h", fid.read(2))[0]
        rh_nsamp = struct.unpack("<h", fid.read(2))[0]
        rh_bits = struct.unpack("<h", fid.read(2))[0]
        fid.read(2)  # rh_zero
        rhf_sps = struct.unpack("<f", fid.read(4))[0]
        rhf_spm = struct.unpack("<f", fid.read(4))[0]
        fid.read(4)  # rhf_mpm
        rhf_position = struct.unpack("<f", fid.read(4))[0]
        rhf_range = struct.unpack("<f", fid.read(4))[0]
        rh_npass = struct.unpack("<h", fid.read(2))[0]
        rhb_cdt = struct.unpack("<f", fid.read(4))[0]
        fid.read(4)  # rhb_mdt
        fid.read(2)  # rh_mapOffset
        fid.read(2)  # rh_mapSize
        fid.read(2)  # rh_text
        fid.read(2)  # rh_ntext
        fid.read(2)  # rh_proc
        fid.read(2)  # rh_nproc
        rh_nchan = struct.unpack("<h", fid.read(2))[0]

    if rh_nsamp <= 0:
        raise ValueError(f"Invalid samples-per-trace in DZT header: {rh_nsamp}")
    if rh_bits not in (8, 16, 32):
        raise ValueError(f"Unsupported DZT sample width: {rh_bits} bits")

    min_header_size = 1024
    offset_bytes = (
        min_header_size * rh_data
        if rh_data < min_header_size
        else min_header_size * rh_nchan
    )
    dtype = {8: np.dtype("u1"), 16: np.dtype("<u2"), 32: np.dtype("<i4")}[rh_bits]
    vec = np.fromfile(dzt_path, dtype=dtype)
    header_values = int(offset_bytes / (rh_bits / 8))
    data_values = vec[header_values:]

    if data_values.size % rh_nsamp:
        raise ValueError(
            f"DZT payload size is not divisible by {rh_nsamp} samples per trace: "
            f"{dzt_path}"
        )
    if rh_bits in (8, 16):
        data_values = data_values - (2**rh_bits) / 2.0

    # GPRPy returns samples/time on axis 0 and traces/distance on axis 1.
    data = np.reshape(data_values, (-1, rh_nsamp)).T.copy()
    info = {
        "rh_nsamp": rh_nsamp,
        "rhf_sps": rhf_sps,
        "rhf_spm": rhf_spm,
        "rhf_position": rhf_position,
        "rhf_range": rhf_range,
        "rh_npass": rh_npass,
        "rhb_cdt": rhb_cdt,
    }
    return data, info


def _read_dzx_ascii_fallback(dzx_path: Optional[str]) -> Optional[np.ndarray]:
    """Retain the legacy numeric-text fallback for nonstandard companions.

    Normal DZX companions are XML.  The supplied data get distance and time
    metadata from the DZT header, so this fallback is not used for the demo.
    """
    if dzx_path is None or not os.path.exists(dzx_path):
        return None
    values = []
    with open(dzx_path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("<"):
                continue
            parts = line.replace("\t", " ").replace(",", " ").split()
            for part in parts:
                try:
                    values.append(float(part))
                    break
                except ValueError:
                    continue
    if not values:
        return None
    x = np.asarray(values, dtype=float)
    if np.any(np.diff(x) < 0):
        x = np.cumsum(np.abs(np.diff(np.r_[0.0, x])))
    return x


def load_gssi_dzt_dxt(
    dzt_path: str,
    dxt_path: Optional[str] = None,
    default_dx: float = 0.02,
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """Return ``data, dt, dx, meta`` in the established pipeline convention.

    ``data`` is shaped ``(n_time_samples, n_traces)``; ``dt`` is seconds and
    ``dx`` is metres per trace.
    """
    data, info = _read_dzt_like_gprpy(dzt_path)
    twtt = np.linspace(0.0, float(info["rhf_range"]), int(info["rh_nsamp"]))
    if twtt.size < 2:
        raise ValueError(f"DZT must contain at least two time samples: {dzt_path}")
    dt = float(twtt[1] - twtt[0]) * 1e-9

    spacing_source = float(info["rhf_spm"])
    if not np.isfinite(spacing_source) or spacing_source <= 0:
        raise ValueError("DZT lacks valid scans-per-metre metadata; provide measured spacing instead of guessing units")
    meta: Dict[str, Any] = {
        "freq_mhz": None,
        "gprpy_time_range": (float(twtt[0]), float(twtt[-1])),
        "gprpy_total_time_ns": float(twtt[-1] - twtt[0]),
    }

    if spacing_source:
        start = float(info["rhf_position"])
        # N traces contain N-1 physical intervals. Display settings in the
        # DZX may use cm, while this header field is scans per metre.
        profile_pos = start + np.linspace(
            0.0, (data.shape[1]-1) / spacing_source, data.shape[1]
        )
        end = float(profile_pos[-1])
        dx = float((end - start) / (data.shape[1] - 1)) if data.shape[1] > 1 else default_dx
        meta.update(
            {
                "gprpy_distance_range": (start, end),
                "gprpy_total_distance": end - start,
                "total_distance_m": end - start,
                "distance_range": (start, end),
            }
        )
    else:
        positions = _read_dzx_ascii_fallback(dxt_path)
        if positions is not None and len(positions) == data.shape[1] and len(positions) > 2:
            dx = float(np.median(np.diff(positions)))
            start, end = float(positions[0]), float(positions[-1])
        else:
            dx = float(default_dx)
            start, end = 0.0, float(data.shape[1] * dx)
        meta.update(
            {
                "total_distance_m": end - start,
                "distance_range": (start, end),
            }
        )

    return data, dt, dx, meta
