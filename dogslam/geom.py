import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def pose(rot=None, xyz=None):
    T = np.eye(4)
    if rot is not None:
        T[:3, :3] = rot.as_matrix()
    if xyz is not None:
        T[:3, 3] = xyz
    return T


def inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def apply(T, points):
    return points @ T[:3, :3].T + T[:3, 3]


def rotation(T):
    return Rotation.from_matrix(T[:3, :3])


def magnitude(T):
    return np.linalg.norm(T[:3, 3]), np.degrees(rotation(T).magnitude())


def blend(A, B, u):
    D = inv(A) @ B
    return A @ pose(Rotation.from_rotvec(rotation(D).as_rotvec() * u), D[:3, 3] * u)


def voxel(points, size, cap=None):
    if not len(points):
        return points
    _, idx = np.unique(
        np.floor(points / size).astype(np.int64), axis=0, return_index=True
    )
    if cap and len(idx) > cap:
        idx = idx[np.linspace(0, len(idx) - 1, cap, dtype=int)]
    return points[idx]


def write_tum(path, times, poses):
    a = np.asarray(poses)
    np.savetxt(
        path,
        np.c_[times, a[:, :3, 3], Rotation.from_matrix(a[:, :3, :3]).as_quat()],
        fmt="%.9f",
        header="t x y z qx qy qz qw",
    )


class Series:
    def __init__(self, samples, gap):
        self.gap = gap
        self.t = np.array([t for t, _ in samples])
        if len(self.t) and (
            not np.isfinite(self.t).all() or np.any(np.diff(self.t) <= 0)
        ):
            raise ValueError(
                "Nonmonotonic auxiliary stamps. Inspect clocks, splits and offsets."
            )
        if len(samples):
            self.poses = np.stack([T for _, T in samples])
            if len(samples) > 1:
                self.slerp = Slerp(self.t, Rotation.from_matrix(self.poses[:, :3, :3]))

    def at(self, t):
        i = np.searchsorted(self.t, t)
        if i < len(self.t) and self.t[i] == t:
            return self.poses[i]
        if i == 0 or i == len(self.t) or self.t[i] - self.t[i - 1] > self.gap:
            return None
        u = (t - self.t[i - 1]) / (self.t[i] - self.t[i - 1])
        return pose(
            self.slerp(t), (1 - u) * self.poses[i - 1, :3, 3] + u * self.poses[i, :3, 3]
        )

    def delta(self, a, b):
        A, B = self.at(a), self.at(b)
        if A is None or B is None or b - a > self.gap:
            return None
        # Both endpoints have interpolation support. An interval no longer
        # than gap cannot span an interior dropout longer than gap.
        return inv(A) @ B
