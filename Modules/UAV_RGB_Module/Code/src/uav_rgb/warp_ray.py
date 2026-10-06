from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp


@wp.kernel
def _mesh_raycast_kernel(
    mesh_id: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    hit_flags: wp.array(dtype=wp.int32),
    hit_t: wp.array(dtype=wp.float32),
    hit_face: wp.array(dtype=wp.int32),
    hit_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    query = wp.mesh_query_ray(mesh_id, origins[tid], directions[tid], 1.0e6)
    if query.result:
        hit_flags[tid] = 1
        hit_t[tid] = query.t
        hit_face[tid] = query.face
        hit_points[tid] = origins[tid] + directions[tid] * query.t
    else:
        hit_flags[tid] = 0
        hit_t[tid] = -1.0
        hit_face[tid] = -1
        hit_points[tid] = wp.vec3(0.0, 0.0, 0.0)


class WarpMeshRaycaster:
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        device: str = "cuda:0",
        origin_offset: np.ndarray | None = None,
    ):
        wp.init()
        if str(device).startswith("cuda") and not wp.is_cuda_available():
            raise RuntimeError("Warp CUDA device was requested, but CUDA is not available.")

        self.device = str(device)
        vertices_f64 = np.asarray(vertices, dtype=np.float64)
        if origin_offset is None:
            origin_offset = 0.5 * (vertices_f64.min(axis=0) + vertices_f64.max(axis=0))
        self.origin_offset = np.asarray(origin_offset, dtype=np.float64)

        vertices_f32 = (vertices_f64 - self.origin_offset).astype(np.float32)
        faces_i32 = np.asarray(faces, dtype=np.int32).reshape(-1)
        self.points = wp.array(vertices_f32, dtype=wp.vec3, device=self.device)
        self.indices = wp.array(faces_i32, dtype=wp.int32, device=self.device)
        constructor = "lbvh" if self.device.startswith("cuda") else "sah"
        self.mesh = wp.Mesh(
            points=self.points,
            velocities=None,
            indices=self.indices,
            bvh_constructor=constructor,
        )

    def intersect_rays(self, origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        origins_f32 = (np.asarray(origins, dtype=np.float64) - self.origin_offset).astype(np.float32)
        directions_f32 = np.asarray(directions, dtype=np.float32)
        n = int(len(origins_f32))
        if n == 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0, 3), dtype=np.float32),
            )

        origins_wp = wp.array(origins_f32, dtype=wp.vec3, device=self.device)
        directions_wp = wp.array(directions_f32, dtype=wp.vec3, device=self.device)
        hit_flags = wp.empty(n, dtype=wp.int32, device=self.device)
        hit_t = wp.empty(n, dtype=wp.float32, device=self.device)
        hit_face = wp.empty(n, dtype=wp.int32, device=self.device)
        hit_points = wp.empty(n, dtype=wp.vec3, device=self.device)

        wp.launch(
            kernel=_mesh_raycast_kernel,
            dim=n,
            inputs=[self.mesh.id, origins_wp, directions_wp, hit_flags, hit_t, hit_face, hit_points],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        points = hit_points.numpy().astype(np.float64) + self.origin_offset
        return hit_flags.numpy(), hit_t.numpy(), hit_face.numpy(), points

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "warp",
            "device": self.device,
            "origin_offset": tuple(float(v) for v in self.origin_offset),
        }
