import csv
import json
from pathlib import Path
from time import monotonic
import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation
from .backend import close_loop
from .geom import apply, blend, inv, magnitude, pose, voxel, write_tum
from .grid import occupancy, overlay, render, seeds
from .registration import Fit, Index, register


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def initialize(points, bag, index, grid, origin, res, cfg, hint, initial_z):
    if hint:
        guesses = [
            pose(Rotation.from_euler("z", hint[2], degrees=True), [*hint[:2], initial_z])
            @ bag.level
        ]
    else:
        guesses = seeds(points, bag.level, grid, origin, res, cfg)
    fits = []
    for g in guesses:
        f = register(
            points,
            index,
            g,
            dict(cfg, max_correction_m=1.4, max_correction_deg=30),
            radius=max(cfg["max_corr"], 1.0),
        )
        if f.ok:
            fits.append((f.overlap - f.residual, f))
    if not fits:
        raise ValueError(
            "No supported initial pose on this map; verify frames/map or supply --initial X Y YAW_DEG"
        )
    fits.sort(key=lambda x: -x[0])
    best, f = fits[0]
    for score, other in fits[1:]:
        d, angle = magnitude(inv(f.T) @ other.T)
        if (d > 1.0 or angle > 25) and best - score < 0.04:
            raise ValueError(
                "Ambiguous global start. Supply --initial X Y YAW_DEG from the floor plan"
            )
    return f


def process(
    bag, cfg, out, mapdir=None, initial=None, limit=None, baseline=False, initial_z=0.0
):
    started = monotonic()
    loc = mapdir is not None
    keys, poses, times, diag = [], [], [], []
    kin_times, kin_poses = [], []
    kin_anchor = None
    anchor_time = None
    lastkin = None
    index = None
    metadata = {}
    if loc:
        with np.load(Path(mapdir) / "landmarks.npz", allow_pickle=False) as f:
            index = Index(f["points"], cfg)
        meta = yaml.safe_load((Path(mapdir) / "map.yaml").read_text())
        grid = np.asarray(Image.open(Path(mapdir) / meta["image"]))[::-1]
        origin = np.array(meta["origin"][:2])
        res = meta["resolution"]
    motion = np.eye(4)
    last_dt = None
    for t, p, sensor, deskew in bag.scans():
        if not times:
            if len(p) < cfg["min_points"] or (baseline and bag.kin.at(t) is None):
                continue
            fit = (
                initialize(p, bag, index, grid, origin, res, cfg, initial, initial_z)
                if loc
                else Fit(bag.level.copy(), True)
            )
            source = "global" if loc and not initial else "initial"
            status = "init"
        else:
            dt = t - times[-1]
            smooth = np.eye(4)
            if last_dt and dt <= cfg["max_gap_s"]:
                smooth = blend(np.eye(4), motion, min(dt / last_dt, 2))
            delta, source = bag.delta(times[-1], t, smooth)
            guess = poses[-1] @ delta
            if baseline:
                if source != "kin" or dt > cfg["max_gap_s"]:
                    raise ValueError(
                        "Kinematic baseline requires synchronized odometry covering all used scans"
                    )
                fit = Fit(guess, True)
                status = "kinematic"
            else:
                fit = register(p, index, guess, cfg)
                status = "matched" if fit.ok else "prior_only"
            motion = inv(poses[-1]) @ fit.T if fit.ok else np.eye(4)
            last_dt = dt
        poses.append(fit.T)
        times.append(t)
        kin = bag.kin.at(t)
        if kin is not None:
            if kin_anchor is None:
                kin_anchor = poses[-1] @ inv(kin)
                anchor_time = t
            kin_times.append(t)
            kin_poses.append(kin_anchor @ kin)
            lastkin = t
        diag.append(
            dict(
                t=t,
                status=status,
                prior=source,
                points=len(p),
                overlap=fit.overlap,
                median_nn_m=fit.residual,
                weak_modes=fit.weak,
                deskew=deskew,
            )
        )
        if not loc and fit.ok and len(p) >= cfg["min_points"]:
            dist, angle = (
                magnitude(inv(keys[-1]["T"]) @ fit.T) if keys else (np.inf, np.inf)
            )
            if dist >= cfg["keyframe_m"] or angle >= cfg["keyframe_deg"]:
                keys.append(
                    dict(
                        T=fit.T,
                        points=p.astype("float32"),
                        idx=len(times) - 1,
                        origin=sensor,
                    )
                )
                if not baseline:
                    index = Index(
                        np.vstack(
                            [
                                apply(k["T"], k["points"])
                                for k in keys[-cfg["local_keyframes"] :]
                            ]
                        ),
                        cfg,
                    )
        if limit and len(times) >= limit:
            break
    if len(times) < 2:
        raise ValueError(
            "Fewer than two usable scans; check topics, filters and duration"
        )
    if not loc:
        write_tum(out / "frontend.tum", times, poses)
        loop = (
            close_loop(keys, poses, cfg)
            if not baseline
            else {"applied": False, "reason": "kinematic baseline"}
        )
        grid, origin, res = occupancy(keys, cfg)
        render(out, grid, origin, res, poses)
        (out / "map.yaml").write_text(
            yaml.safe_dump(
                dict(
                    image="map.pgm",
                    resolution=float(res),
                    origin=[*map(float, origin), 0.0],
                    negate=0,
                    occupied_thresh=0.65,
                    free_thresh=0.196,
                    mode="trinary",
                )
            )
        )
        global_points = voxel(
            np.vstack([apply(k["T"], k["points"]) for k in keys]), cfg["voxel"]
        )
        np.savez_compressed(
            out / "landmarks.npz", points=global_points.astype("float32")
        )
        metadata.update(
            loop=loop, keyframes=len(keys), occupied_cells=int((grid == 0).sum())
        )
    else:
        metadata["initial_hint"] = (
            dict(x_m=initial[0], y_m=initial[1], yaw_deg=initial[2], z_m=initial_z)
            if initial is not None
            else None
        )
        overlay(
            Image.fromarray(grid[::-1]),
            origin,
            res,
            poses,
            out / "inspection_route.png",
        )
    write_tum(out / "trajectory.tum", times, poses)
    if kin_times:
        write_tum(out / "kinematic.tum", kin_times, kin_poses)
    with (out / "tracking.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=diag[0])
        writer.writeheader()
        writer.writerows(diag)
    good = [d for d in diag if d["status"] == "matched"]
    failures = sum(d["status"] == "prior_only" for d in diag)
    streak = longest = 0
    for d in diag:
        streak = streak + 1 if d["status"] == "prior_only" else 0
        longest = max(streak, longest)
    metadata.update(
        mode="localize" if loc else ("kinematic" if baseline else "mapping"),
        bag_start_s=times[0],
        bag_end_s=times[-1],
        base=cfg["base"],
        lidar=bag.lidar,
        imu=bag.imu,
        odom=bag.odom,
        scans=len(times),
        matched=len(good),
        prior_only=failures,
        longest_prior_only_streak=longest,
        tracking="degraded"
        if failures > (len(times) - 1) * 0.2 or longest >= 5
        else "ok",
        kin_samples=len(kin_times),
        kin_anchor_s=anchor_time,
        kin_end_s=lastkin,
        median_match_residual_m=float(np.median([d["median_nn_m"] for d in good]))
        if good
        else None,
        path_m=float(
            np.linalg.norm(np.diff(np.asarray(poses)[:, :3, 3], axis=0), axis=1).sum()
        ),
        gyro_bias_rad_s=bag.bias.tolist() if bag.bias is not None else None,
        elapsed_s=round(monotonic() - started, 3),
    )
    save_json(out / "report.json", metadata)
    (out / "used.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return metadata
