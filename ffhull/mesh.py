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
    def __init__(self, points_wp: "wp.array", device: str):
        """``points_wp`` is a device ``wp.array(dtype=wp.vec3d)`` of ``n`` points.
        Its contents are copied into the internal n+1-slot buffer on the device
        (slot ``n`` is reserved for the kernel point ``s``), so construction never
        touches the host."""
        assert points_wp.ndim == 1, "expected a wp.array(dtype=wp.vec3d) of points"
        self.n = int(points_wp.shape[0])
        self.device = device
        cap = 2 * self.n + 8  # triangle capacity, >= 2n-4
        self.cap = cap

        # Point array has n+1 slots: index n holds the kernel point s, written by
        # init_tetra_gpu, so no kernel needs a host-side s (keeps everything
        # on-device and graph-capturable).  Copied on-device from the caller's
        # array -- no host staging.
        self.points = wp.empty(self.n + 1, dtype=wp.vec3d, device=device)
        wp.copy(self.points[0:self.n], points_wp)
        self.s_idx = self.n

        # Topology (triangle-indexed).  These large arrays are fully written
        # before they are read (init_tetra_gpu/split for topology; per-round reset
        # kernels for the growth/flip scratch), so allocate WITHOUT zeroing --
        # the memset of ~0.3 GB dominated per-call time.
        self.tri_v = wp.empty(cap, dtype=wp.vec3i, device=device)
        self.tri_adj = wp.empty(cap, dtype=wp.vec3i, device=device)
        self.tri_adj_slot = wp.empty(cap, dtype=wp.vec3i, device=device)
        self.tri_active = wp.empty(cap, dtype=wp.int32, device=device)

        # Point-indexed association (written for all n by init_associate)
        self.point_owner = wp.empty(self.n, dtype=wp.int32, device=device)

        # Growth scratch (reset each round by reset_growth before use)
        self.face_score = wp.empty(cap, dtype=wp.float64, device=device)
        self.face_pivot = wp.empty(cap, dtype=wp.int32, device=device)
        self.face_children = wp.empty(cap, dtype=wp.vec3i, device=device)

        # Counters / flags (single element device arrays)
        self.tri_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.old_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.scratch_i = wp.zeros(1, dtype=wp.int32, device=device)
        self.changed = wp.zeros(1, dtype=wp.int32, device=device)
        self.cond = wp.zeros(1, dtype=wp.int32, device=device)
        self.iter_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.convex_flag = wp.zeros(1, dtype=wp.int32, device=device)

        # Flip scratch (reset each round by reset_flip before use)
        self.tri_claim = wp.empty(cap, dtype=wp.int32, device=device)
        self.prop_slot = wp.empty(cap, dtype=wp.int32, device=device)
        self.prop_type = wp.empty(cap, dtype=wp.int32, device=device)  # 0 none,2 22,3 31
        # Vertex labels: 0 unknown/extreme, 1 non-extreme
        self.vertex_label = wp.zeros(self.n, dtype=wp.int32, device=device)

    def set_tri_count(self, k: int):
        self.tri_count.assign(np.array([k], dtype=np.int32))

    def get_tri_count(self) -> int:
        return int(self.tri_count.numpy()[0])

    def rebind(self, points_wp: "wp.array"):
        """Reuse this mesh for a new point set of the SAME size: re-copy the
        points (device ``wp.array(dtype=wp.vec3d)``) and reset the state that
        isn't reinitialised by the kernels (counters + vertex labels).  Avoids
        re-allocating the 2n arrays."""
        assert int(points_wp.shape[0]) == self.n
        # write into the first n slots (slot n = s, rewritten by init_tetra_gpu)
        wp.copy(self.points[0:self.n], points_wp)
        self.vertex_label.zero_()
        for a in (self.tri_count, self.old_count, self.scratch_i,
                  self.changed, self.cond, self.iter_count, self.convex_flag):
            a.zero_()
