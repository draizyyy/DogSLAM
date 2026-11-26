"""Export final map and timestamped base poses for ROS 2 playback."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
import yaml


def export_bag(run, mapdir, out):
    run, mapdir = Path(run), Path(mapdir)
    meta = yaml.safe_load((mapdir / "map.yaml").read_text(encoding="utf-8"))
    trajectory = np.loadtxt(run / "trajectory.tum", ndmin=2)
    if not len(trajectory) or trajectory.shape[1] != 8 or not np.isfinite(trajectory).all():
        raise ValueError("Expected finite TUM base trajectory")
    if np.any(np.diff(trajectory[:, 0]) <= 0):
        raise ValueError("Trajectory timestamps must increase")
    store = get_typestore(Stores.ROS2_HUMBLE)
    msg = store.types
    point = msg["geometry_msgs/msg/Point"]
    quat = msg["geometry_msgs/msg/Quaternion"]
    pose = msg["geometry_msgs/msg/Pose"]
    stamp = msg["builtin_interfaces/msg/Time"]
    header = msg["std_msgs/msg/Header"]
    with Image.open(mapdir / meta["image"]) as im:
        pixels = np.asarray(im.convert("L"))[::-1]
    data = np.full(pixels.shape, -1, dtype=np.int8)
    data[pixels == 0] = 100
    data[pixels == 254] = 0
    x, y, yaw = meta["origin"]
    origin = pose(point(x, y, 0.), quat(0., 0., np.sin(yaw / 2), np.cos(yaw / 2)))
    first_ns = round(trajectory[0, 0] * 1e9)
    info = msg["nav_msgs/msg/MapMetaData"](
        stamp(*divmod(first_ns, 10**9)), meta["resolution"], pixels.shape[1], pixels.shape[0], origin
    )
    with Writer(out, version=8) as writer:
        grid = writer.add_connection("/map", "nav_msgs/msg/OccupancyGrid", typestore=store)
        base = writer.add_connection("/dogslam/base_pose", "geometry_msgs/msg/PoseStamped", typestore=store)
        next_map = first_ns
        for t, x, y, z, qx, qy, qz, qw in trajectory:
            ns = round(t * 1e9)
            h = header(stamp(*divmod(ns, 10**9)), "map")
            if ns >= next_map:
                m = msg[grid.msgtype](h, info, data.ravel())
                writer.write(grid, ns, store.serialize_cdr(m, grid.msgtype))
                next_map = ns + 10**9
            m = msg[base.msgtype](h, pose(point(x, y, z), quat(qx, qy, qz, qw)))
            writer.write(base, ns, store.serialize_cdr(m, base.msgtype))
    return {"poses": len(trajectory), "base": json.loads((run / "report.json").read_text())["base"]}
