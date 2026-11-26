from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .geom import apply, inv, magnitude, voxel


class Index:
    def __init__(self, points, cfg):
        self.points = voxel(points, cfg["voxel"])
        self.tree = cKDTree(self.points)
        # PCA in batches keeps memory bounded for a global map.
        self.normals = np.zeros_like(self.points, dtype=float)
        self.planar = np.zeros(len(self.points), bool)
        for start in range(0, len(self.points), 8192):
            s = slice(start, start + 8192)
            d, i = self.tree.query(self.points[s], k=min(10, len(self.points)))
            if d.ndim != 2 or d.shape[1] < 5:
                continue
            nb = self.points[i]
            nb -= nb.mean(axis=1, keepdims=True)
            val, vec = np.linalg.eigh(nb.transpose(0, 2, 1) @ nb)
            self.normals[s] = vec[:, :, 0]
            self.planar[s] = (
                (val[:, 1] - val[:, 0]) / np.maximum(val[:, 2], 1e-10) > 0.16
            ) & (d[:, -1] < 1.4)


@dataclass
class Fit:
    T: np.ndarray
    ok: bool
    overlap: float = 0.0
    residual: float = 0.0
    weak: int = 0


def register(points, index, initial, cfg, radius=None):
    if len(points) < cfg["min_points"]:
        return Fit(initial, False)
    T = initial.copy()
    radius = radius or cfg["max_corr"]
    weak = 0
    for _ in range(13):
        q = apply(T, points)
        d, ii = index.tree.query(q, distance_upper_bound=radius)
        good = np.isfinite(d)
        if good.sum() < cfg["min_points"] or good.mean() < cfg["min_overlap"] * 0.75:
            return Fit(initial, False, float(good.mean()))
        target = index.points[ii[good]]
        residual = q[good] - target
        # Jacobian around CURRENT robot center, invariant to the map origin.
        arm = q[good] - T[:3, 3]
        normal = index.normals[ii[good]]
        plane = index.planar[ii[good]]
        a, n = arm[plane], normal[plane]
        r = np.sum(n * residual[plane], axis=1)
        J = np.c_[np.cross(a, n), n]
        w = 1 / (1 + (r / 0.15) ** 2) ** 2
        H = J.T @ (w[:, None] * J)
        g = J.T @ (w * r)
        # Isotropic point matches at edges or where too few neighbors define a plane.
        a, e = arm[~plane], residual[~plane]
        w = 0.25 / (1 + np.sum(e * e, axis=1) / 0.15**2) ** 2
        if len(a):
            cov = a.T @ (w[:, None] * a)
            x, y, z = np.sum(w[:, None] * a, axis=0)
            cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            H[:3, :3] += np.eye(3) * cov.trace() - cov
            H[:3, 3:] += cross
            H[3:, :3] += cross.T
            H[3:, 3:] += np.eye(3) * w.sum()
            g += np.r_[np.sum(w[:, None] * np.cross(a, e), axis=0), w @ e]
        # Normalize rotation columns by RMS range. Project weak modes out of the update.
        scale = np.r_[np.full(3, 1 / max(np.sqrt(np.mean(arm**2) * 3), 1)), np.ones(3)]
        val, vec = np.linalg.eigh(H * scale[:, None] * scale[None, :])
        keep = val > max(cfg["weak_information"] * good.sum(), 1e-8)
        weak = int((~keep).sum())
        if weak == 6:
            return Fit(initial, False, float(good.mean()), weak=weak)
        step = -scale * (vec[:, keep] @ ((vec[:, keep].T @ (g * scale)) / val[keep]))
        T[:3, :3] = Rotation.from_rotvec(step[:3]).as_matrix() @ T[:3, :3]
        T[:3, 3] += step[3:]
        if np.linalg.norm(step) < 0.001:
            break
    d, _ = index.tree.query(apply(T, points), distance_upper_bound=cfg["max_corr"])
    mask = np.isfinite(d)
    overlap = float(mask.mean())
    residual = float(np.median(d[mask])) if mask.any() else radius
    dist, angle = magnitude(inv(initial) @ T)
    ok = (
        overlap >= cfg["min_overlap"]
        and residual <= cfg["max_residual"]
        and dist <= cfg["max_correction_m"]
        and angle <= cfg["max_correction_deg"]
    )
    return Fit(T if ok else initial, bool(ok), overlap, residual, weak)
