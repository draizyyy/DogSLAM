"""A single verified revisit constraint for the announced single-loop mapping route."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from .geom import inv, magnitude, blend, pose
from .registration import Index, register


def xytheta(T):
    return np.array([T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])])


def planar(x):
    return pose(Rotation.from_euler("z", x[2]), [*x[:2], 0])


def relative(a, b):
    d = b - a
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack(
        (
            c * d[..., 0] + s * d[..., 1],
            -s * d[..., 0] + c * d[..., 1],
            np.arctan2(np.sin(d[..., 2]), np.cos(d[..., 2])),
        ),
        axis=-1,
    )


def close_loop(keys, poses, cfg):
    if not cfg["loop"] or len(keys) < 5:
        return {"applied": False, "reason": "disabled or too few keyframes"}
    raw = np.stack([xytheta(k["T"]) for k in keys])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(raw[:, :2], axis=0), axis=1))]
    proposals = []
    for j in range(1, len(keys)):
        d = np.linalg.norm(raw[:j, :2] - raw[j, :2], axis=1)
        valid = (arc[j] - arc[:j] > cfg["loop_min_path"]) & (d < cfg["loop_radius"])
        for i in np.flatnonzero(valid):
            proposals.append((d[i], i, j))
    for _, i, j in sorted(proposals)[:12]:
        A, B = keys[i], keys[j]
        guess = inv(A["T"]) @ B["T"]
        strict = dict(
            cfg,
            min_overlap=max(0.70, cfg["min_overlap"]),
            max_residual=min(0.13, cfg["max_residual"]),
        )
        forward = register(B["points"], Index(A["points"], cfg), guess, strict)
        if not forward.ok or forward.weak:
            continue
        reverse = register(A["points"], Index(B["points"], cfg), inv(guess), strict)
        d, y = magnitude(forward.T @ reverse.T)
        if not reverse.ok or reverse.weak or d > 0.13 or y > 3:
            continue
        # The optimized graph is planar even when the LiDAR poses tilt. Project
        # the measured relative pose AFTER composing in the map frame.
        edges = np.array([(k, k + 1) for k in range(len(keys) - 1)] + [(i, j)])
        obs = np.vstack(
            [relative(raw[:-1], raw[1:]), relative(raw[i], xytheta(A["T"] @ forward.T))]
        )

        def error(v):
            X = np.vstack([raw[0], v.reshape(-1, 3)])
            delta = relative(X[edges[:, 0]], X[edges[:, 1]]) - obs
            delta[:, 2] = np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))
            return (delta / np.array([0.07, 0.07, 0.035])).ravel()

        pattern = lil_matrix((3 * len(edges), 3 * (len(keys) - 1)), dtype=int)
        for e, pair in enumerate(edges):
            for k in pair:
                if k:
                    pattern[3 * e : 3 * e + 3, 3 * (k - 1) : 3 * k] = 1
        sol = least_squares(
            error,
            raw[1:].ravel(),
            jac_sparsity=pattern.tocsr(),
            loss="huber",
            max_nfev=60,
        )
        if not sol.success:
            continue
        opt = np.vstack([raw[0], sol.x.reshape(-1, 3)])
        corrections = [planar(b) @ inv(planar(a)) for a, b in zip(raw, opt)]
        ids = [k["idx"] for k in keys]
        for n, T in enumerate(poses):
            l = max(0, np.searchsorted(ids, n, side="right") - 1)
            r = min(l + 1, len(ids) - 1)
            u = np.clip((n - ids[l]) / max(ids[r] - ids[l], 1), 0, 1)
            poses[n] = blend(corrections[l], corrections[r], u) @ T
        for k in keys:
            k["T"] = poses[k["idx"]]
        return dict(
            applied=True,
            first_key=int(i),
            last_key=int(j),
            overlap=forward.overlap,
            reverse_check_m=float(d),
            reverse_check_deg=float(y),
        )
    return dict(
        applied=False, reason="no verified nonadjacent match", candidates=len(proposals)
    )
