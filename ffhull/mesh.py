"""Fixed-capacity structure-of-arrays topology for the ffHull polyhedron.

A star-shaped polyhedron built by inserting points into a tetrahedron grows by
+2 triangles per inserted vertex, so at most ``2n - 4`` triangles ever exist.
The subsequent Flip-Flop phase never increases the triangle count.  We
therefore preallocate everything up front; append-style allocation of the two
new triangles per split uses a single atomic counter (``tri_count``).

Adjacency convention
--------------------
``tri_v[t] = (v0, v1, v2)`` are oriented vertex (point) indices.
``tri_adj[t][i]`` is the triangle across the edge *opposite* vertex ``i``
(i.e. edge ``(v[(i+1)%3], v[(i+2)%3])``).  ``tri_adj_slot[t][i]`` is the local
slot ``j`` in that neighbour whose ``tri_adj[neighbour][j]`` points back to
``t``.  Storing the reciprocal slot makes every split/flip update
constant-time with no neighbour search.
"""

import numpy as np
import warp as wp

INT_MAX = np.iinfo(np.int32).max


class Mesh:
    def __init__(self, points_np: np.ndarray, device: str):
        assert points_np.ndim == 2 and points_np.shape[1] == 3
        self.n = int(points_np.shape[0])
        self.device = device
        cap = 2 * self.n + 8  # triangle capacity, >= 2n-4
        self.cap = cap

        self.points = wp.array(points_np.astype(np.float64), dtype=wp.vec3d, device=device)

        # Topology (triangle-indexed)
        self.tri_v = wp.zeros(cap, dtype=wp.vec3i, device=device)
        self.tri_adj = wp.zeros(cap, dtype=wp.vec3i, device=device)
        self.tri_adj_slot = wp.zeros(cap, dtype=wp.vec3i, device=device)
        self.tri_active = wp.zeros(cap, dtype=wp.int32, device=device)

        # Point-indexed association
        self.point_owner = wp.full(self.n, -1, dtype=wp.int32, device=device)

        # Growth scratch (triangle-indexed)
        self.face_score = wp.zeros(cap, dtype=wp.float64, device=device)
        self.face_pivot = wp.full(cap, INT_MAX, dtype=wp.int32, device=device)
        self.face_children = wp.full(cap, -1, dtype=wp.vec3i, device=device)

        # Counters / flags (single element device arrays)
        self.tri_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.scratch_i = wp.zeros(1, dtype=wp.int32, device=device)
        self.changed = wp.zeros(1, dtype=wp.int32, device=device)

        # Flip scratch (triangle-indexed)
        self.tri_claim = wp.full(cap, INT_MAX, dtype=wp.int32, device=device)
        self.prop_slot = wp.full(cap, -1, dtype=wp.int32, device=device)
        self.prop_type = wp.zeros(cap, dtype=wp.int32, device=device)  # 0 none,2 22,3 31
        # Vertex labels: 0 unknown/extreme, 1 non-extreme
        self.vertex_label = wp.zeros(self.n, dtype=wp.int32, device=device)

    def set_tri_count(self, k: int):
        self.tri_count.assign(np.array([k], dtype=np.int32))

    def get_tri_count(self) -> int:
        return int(self.tri_count.numpy()[0])
