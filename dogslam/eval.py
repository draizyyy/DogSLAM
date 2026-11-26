import numpy as np
from scipy.spatial.transform import Rotation
from .geom import inv, pose, rotation, Series


def read(path):
    a = np.loadtxt(path, ndmin=2)
    if (
        a.shape[1] != 8
        or len(a) < 2
        or not np.isfinite(a).all()
        or np.any(np.diff(a[:, 0]) <= 0)
    ):
        raise ValueError("Expected monotonic finite TUM: t x y z qx qy qz qw")
    return a[:, 0], np.array([pose(Rotation.from_quat(row[4:]), row[1:4]) for row in a])


def rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def evaluate(est, gt, align, gap, delta, fixed=None):
    te, E = read(est)
    tg, G = read(gt)
    truth = Series(list(zip(tg, G)), gap)
    times, obs, ref = [], [], []
    for t, T in zip(te, E):
        g = truth.at(t)
        if g is not None:
            times.append(t)
            obs.append(T)
            ref.append(g)
    if len(obs) < 3:
        raise ValueError(
            "Fewer than three time-associated poses. Check epochs, units and --max-gap"
        )
    obs, ref = np.array(obs), np.array(ref)
    A = np.eye(4)
    if fixed is not None:
        if align != "none":
            raise ValueError("Use an existing map transform with --align none")
        A = np.asarray(fixed, dtype=float)
        if (
            A.shape != (4, 4)
            or not np.isfinite(A).all()
            or not np.allclose(A[3], [0, 0, 0, 1])
            or not np.allclose(A[:3, :3].T @ A[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(A[:3, :3]), 1)
        ):
            raise ValueError("Invalid rigid map alignment")
        align = "fixed-map"
    elif align == "first":
        A = ref[0] @ inv(obs[0])
    elif align == "se3":
        x, y = obs[:, :3, 3], ref[:, :3, 3]
        u, s, vt = np.linalg.svd((x - x.mean(0)).T @ (y - y.mean(0)))
        if s[1] < 1e-9:
            raise ValueError(
                "SE3 alignment underdetermined on line; choose first or none"
            )
        R = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
        A = pose(Rotation.from_matrix(R), y.mean(0) - R @ x.mean(0))
    P = A @ obs
    dist = np.linalg.norm(P[:, :3, 3] - ref[:, :3, 3], axis=1)
    rot = [np.degrees(rotation(inv(g) @ p).magnitude()) for p, g in zip(P, ref)]
    rel, angles = [], []
    est_series = Series(list(zip(te, E)), gap)
    for t, e, g in zip(times, obs, ref):
        e2, g2 = est_series.at(t + delta), truth.at(t + delta)
        if e2 is not None and g2 is not None:
            D = inv(inv(g) @ g2) @ (inv(e) @ e2)
            rel.append(np.linalg.norm(D[:3, 3]))
            angles.append(np.degrees(rotation(D).magnitude()))
    return dict(
        alignment=align,
        gt_T_map=A.tolist(),
        max_interpolation_gap_s=gap,
        associated=len(times),
        estimate_count=len(te),
        coverage=len(times) / len(te),
        ate_rmse_m=rms(dist),
        position_p95_m=float(np.percentile(dist, 95)),
        orientation_rmse_deg=rms(rot),
        rpe_interval_s=delta,
        rpe_pairs=len(rel),
        rpe_translation_rmse_m=rms(rel),
        rpe_rotation_rmse_deg=rms(angles),
    )
