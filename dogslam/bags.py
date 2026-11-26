from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from scipy.spatial.transform import Rotation

from .geom import apply, inv, pose, Series, voxel

CLOUD = {
    "sensor_msgs/msg/PointCloud2",
    "sensor_msgs/msg/LaserScan",
    "livox_ros_driver/msg/CustomMsg",
    "livox_ros_driver2/msg/CustomMsg",
}
IMU = {"sensor_msgs/msg/Imu"}
ODOM = {"nav_msgs/msg/Odometry"}
TF = "tf2_msgs/msg/TFMessage"


def load_config(path):
    defaults = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config/default.yaml").read_text(encoding="utf-8-sig")
    )
    if path:
        updates = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
        extra = updates.keys() - defaults.keys()
        if extra:
            raise ValueError(f"Unknown settings: {extra}")
        defaults.update(updates)
    if defaults["clock"] not in ("header", "bag"):
        raise ValueError("clock must be header or bag")
    if defaults["height_min"] >= defaults["height_max"] or defaults["resolution"] <= 0:
        raise ValueError("Check height_min, height_max and resolution")
    min_free = defaults["map_min_free_observations"]
    if type(min_free) is not int or min_free < 1:
        raise ValueError("map_min_free_observations must be a positive integer")
    return defaults


@contextmanager
def open_bag(path, msg_dir=None):
    path = Path(path)
    if path.suffix in {".db3", ".mcap"}:
        if not (path.parent / "metadata.yaml").exists():
            raise ValueError(
                "ROS2: supply the bag directory with metadata.yaml and all split files. For standalone files use ros2 bag reindex on a copy."
            )
        path = path.parent
    store = get_typestore(Stores.ROS2_HUMBLE if path.is_dir() else Stores.ROS1_NOETIC)
    if msg_dir:
        defs = {}
        for p in Path(msg_dir).glob("*/msg/*.msg"):
            defs.update(
                get_types_from_msg(
                    p.read_text(encoding="utf-8-sig"), f"{p.parent.parent.name}/msg/{p.stem}"
                )
            )
        if not defs:
            raise ValueError("Expected MSG_DIR/package/msg/Type.msg")
        store.register(defs)
    with AnyReader([path], default_typestore=store) as reader:
        if msg_dir:
            reader.typestore.register(defs)
        yield reader


def resolve(reader, value, types, required=False):
    options = sorted({c.topic for c in reader.connections if c.msgtype in types})
    if value == "auto":
        if len(options) > 1:
            raise ValueError(f"Select one topic explicitly: {options}")
        value = options[0] if options else None
    if value is None and not required:
        return None
    if value not in options:
        raise ValueError(
            f"Topic {value!r} unavailable for {types}; candidates: {options}"
        )
    return value


def stamp(msg, ns, clock):
    if clock == "bag":
        return ns / 1e9
    t = msg.header.stamp
    return t.sec + t.nanosec * 1e-9


def xyz(v):
    return np.array([v.x, v.y, v.z], dtype=float)


def quat(q):
    return Rotation.from_quat([q.x, q.y, q.z, q.w])


def inspect_bag(reader, samples):
    out = {}
    for c in reader.connections:
        entry = out.setdefault(c.topic, dict(type=c.msgtype, messages=0))
        entry["messages"] += c.msgcount
    supported = CLOUD | IMU | ODOM
    for topic, record in out.items():
        if record["type"] not in supported:
            continue
        times, latencies, frames = [], [], set()
        for c, ns, raw in reader.messages(
            connections=[c for c in reader.connections if c.topic == topic]
        ):
            m = reader.deserialize(raw, c.msgtype)
            t = stamp(m, ns, "header")
            times.append(t)
            latencies.append(ns / 1e9 - t)
            frames.add(m.header.frame_id)
            if hasattr(m, "fields"):
                record["fields"] = [f.name for f in m.fields]
                record["sample_size"] = int(m.width * m.height)
                if len(times) <= 3:
                    candidates = {"t", "dt", "time", "timestamp", "offset_time", "time_offset"}
                    for f in m.fields:
                        if f.name.lower() not in candidates:
                            continue
                        stats = dict(header_s=t, datatype=int(f.datatype))
                        try:
                            _, raw_times = clouds(m, f.name)
                            finite = raw_times[np.isfinite(raw_times)]
                            if len(finite):
                                stats.update(min_raw=float(finite.min()), max_raw=float(finite.max()))
                        except (ValueError, TypeError, KeyError, BufferError) as exc:
                            stats["decode_error"] = str(exc)
                        record.setdefault("point_time_samples", {}).setdefault(f.name, []).append(stats)
            if hasattr(m, "child_frame_id"):
                record["child_frame"] = m.child_frame_id
            if len(times) >= samples:
                break
        if times:
            d = np.diff(times)
            record.update(
                frames=sorted(frames),
                sampled=len(times),
                first_header_s=times[0],
                header_nonincreasing=int(np.sum(d <= 0)),
                header_hz=float(1 / np.median(d))
                if len(d) and np.median(d) > 0
                else None,
                max_sample_gap_s=float(d.max()) if len(d) else None,
                bag_minus_header_median_s=float(np.median(latencies)),
            )
    edges = set()
    tf_connections = [
        c
        for c in reader.connections
        if c.topic.strip("/") == "tf_static" and c.msgtype == TF
    ]
    if tf_connections:
        for c, _, raw in reader.messages(connections=tf_connections):
            for tr in reader.deserialize(raw, c.msgtype).transforms:
                edges.add((tr.header.frame_id, tr.child_frame_id))
    return dict(
        duration_s=reader.duration * 1e-9, topics=out, static_tf_edges=sorted(edges)
    )


def lookup(edges, base, sensor, override):
    base, sensor = base.strip("/"), sensor.strip("/")
    if override is not None:
        if len(override) != 6:
            raise ValueError("Extrinsics require x y z roll pitch yaw (m, deg)")
        return pose(
            Rotation.from_euler("xyz", override[3:], degrees=True), override[:3]
        )
    seen = {base}
    queue = deque([(base, np.eye(4))])
    while queue:
        node, acc = queue.popleft()
        if node == sensor:
            return acc
        for name, T in edges[node]:
            if name not in seen:
                seen.add(name)
                queue.append((name, acc @ T))
    raise ValueError(
        f"Missing static transform {base} <- {sensor}; supply calibrated extrinsics in YAML"
    )


def clouds(msg, field):
    if hasattr(msg, "ranges"):
        r = np.asarray(msg.ranges)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        return np.c_[r * np.cos(a), r * np.sin(a), np.zeros(len(a))], np.arange(
            len(r)
        ) * msg.time_increment if field == "beam_time" else None
    if hasattr(msg, "points") and not hasattr(msg, "fields"):
        return np.array([(p.x, p.y, p.z) for p in msg.points]).reshape(-1, 3), (
            np.array([getattr(p, field) for p in msg.points]) if field else None
        )
    types = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    fields = {f.name: f for f in msg.fields}
    names = ["x", "y", "z"] + ([field] if field else [])
    for key in names:
        if key not in fields or fields[key].count != 1:
            raise ValueError(
                f"Missing scalar PointCloud2 field {key}; present: {list(fields)}"
            )
    dtype = np.dtype(
        dict(
            names=names,
            formats=[
                (">" if msg.is_bigendian else "<") + types[fields[k].datatype]
                for k in names
            ],
            offsets=[fields[k].offset for k in names],
            itemsize=msg.point_step,
        )
    )
    arr = np.ndarray(
        (msg.height, msg.width),
        dtype=dtype,
        buffer=msg.data,
        strides=(msg.row_step, msg.point_step),
    )
    return np.column_stack([arr[k].ravel() for k in names[:3]]), arr[
        field
    ].ravel().astype(float) if field else None


class Bag:
    def __init__(self, reader, cfg):
        self.reader, self.cfg = reader, cfg
        self.lidar = resolve(reader, cfg["lidar"], CLOUD, True)
        self.imu = resolve(reader, cfg["imu"], IMU)
        self.odom = resolve(reader, cfg["odom"], ODOM)
        edges = defaultdict(list)
        odom, inertial = [], []
        imu_frame, parent_frame = None, None
        self.bias = None
        self.level = np.eye(4)
        names = {self.imu, self.odom, "/tf_static", "tf_static"}
        selected = [c for c in reader.connections if c.topic in names]
        messages = reader.messages(connections=selected) if selected else ()
        for c, ns, raw in messages:
            m = reader.deserialize(raw, c.msgtype)
            if c.msgtype == TF:
                for tr in m.transforms:
                    a, b = tr.header.frame_id.strip("/"), tr.child_frame_id.strip("/")
                    T = pose(quat(tr.transform.rotation), xyz(tr.transform.translation))
                    edges[a].append((b, T))
                    edges[b].append((a, inv(T)))
                continue
            t = stamp(m, ns, cfg["clock"])
            if c.topic == self.odom:
                if parent_frame is not None and parent_frame != m.header.frame_id:
                    raise ValueError("Odometry parent frame changed inside bag")
                parent_frame = m.header.frame_id
                if m.child_frame_id.strip("/") != cfg["base"].strip("/"):
                    raise ValueError(
                        f"Odometry child {m.child_frame_id!r} differs from base {cfg['base']!r}"
                    )
                odom.append(
                    (
                        t + cfg["odom_offset_s"],
                        pose(quat(m.pose.pose.orientation), xyz(m.pose.pose.position)),
                    )
                )
            elif c.topic == self.imu:
                if imu_frame is not None and imu_frame != m.header.frame_id:
                    raise ValueError("IMU frame changed")
                imu_frame = m.header.frame_id
                inertial.append(
                    (
                        t + cfg["imu_offset_s"],
                        xyz(m.angular_velocity),
                        xyz(m.linear_acceleration),
                    )
                )
        self.edges = edges
        self.kin = Series(odom, cfg["max_gap_s"])
        self.gyro = Series([], cfg["max_gap_s"])
        if inertial:
            self._imu(
                inertial, lookup(edges, cfg["base"], imu_frame, cfg["base_T_imu"])
            )

    def _imu(self, rows, extr):
        cfg = self.cfg
        ts = np.array([r[0] for r in rows])
        w = np.stack([r[1] for r in rows]) @ extr[:3, :3].T
        acc = np.stack([r[2] for r in rows]) @ extr[:3, :3].T
        if not np.isfinite(w).all() or np.any(np.diff(ts) <= 0):
            raise ValueError("Invalid gyro data or repeated/nonmonotonic time")
        mask = ts < ts[0] + cfg["static_seconds"]
        if mask.sum() >= 5 and np.ptp(ts[mask]) >= cfg["static_seconds"] * 0.75:
            g = np.median(acc[mask], axis=0)
            if (
                np.max(np.linalg.norm(w[mask], axis=1)) < 0.12
                and 8.8 < np.linalg.norm(g) < 10.8
                and np.max(np.std(acc[mask], axis=0)) < 0.45
            ):
                self.bias = np.median(w[mask], axis=0)
                w = w - self.bias
                roll = np.arctan2(g[1], g[2])
                pitch = np.arctan2(-g[0], np.hypot(g[1], g[2]))
                self.level = pose(Rotation.from_euler("xyz", [roll, pitch, 0]))
        R = Rotation.identity()
        result = [(ts[0], pose(R))]
        for i, dt in enumerate(np.diff(ts)):
            if dt <= cfg["max_gap_s"]:
                R = R * Rotation.from_rotvec(0.5 * (w[i] + w[i + 1]) * dt)
            result.append((ts[i + 1], pose(R)))
        self.gyro = Series(result, cfg["max_gap_s"])

    def delta(self, a, b, fallback):
        k = self.kin.delta(a, b)
        g = self.gyro.delta(a, b)
        d = (k if k is not None else fallback).copy()
        if g is not None:
            d[:3, :3] = g[:3, :3]
        return d, ("kin" if k is not None else "prediction") + (
            "+gyro" if g is not None else ""
        )

    def scans(self):
        cfg = self.cfg
        con = [c for c in self.reader.connections if c.topic == self.lidar]
        prev = frame = extr = None
        for c, ns, raw in self.reader.messages(connections=con):
            msg = self.reader.deserialize(raw, c.msgtype)
            t = stamp(msg, ns, cfg["clock"])
            if prev is not None and t <= prev:
                raise ValueError(f"Nonmonotonic lidar stamps at {t}; inspect clock")
            if frame is None:
                frame = msg.header.frame_id
                extr = lookup(self.edges, cfg["base"], frame, cfg["base_T_lidar"])
            elif frame != msg.header.frame_id:
                raise ValueError("LiDAR frame changed")
            points, offsets = clouds(msg, cfg["point_time_field"])
            if cfg["point_time_field"] and offsets is None:
                raise ValueError("Invalid point time field for this message type")
            ranges = np.linalg.norm(points, axis=1)
            mask = (
                np.isfinite(ranges)
                & (ranges >= cfg["min_range"])
                & (ranges <= cfg["max_range"])
            )
            points = apply(extr, points[mask])
            deskew = "off"
            if offsets is not None and len(points):
                dt = offsets[mask] * cfg["point_time_scale"] + cfg["point_time_shift_s"]
                if (
                    not np.isfinite(dt).all()
                    or np.max(np.abs(dt)) > cfg["max_scan_span_s"]
                ):
                    raise ValueError(
                        "Check point time units and reference: offsets exceed max_scan_span_s"
                    )
                bins = np.rint(dt / 0.005).astype(int)
                corrected = 0
                for b in np.unique(bins):
                    sel = bins == b
                    tb = t + float(np.median(dt[sel]))
                    d, source = self.delta(min(t, tb), max(t, tb), np.eye(4))
                    if source != "prediction":
                        points[sel] = apply(d if tb >= t else inv(d), points[sel])
                        corrected += int(sel.sum())
                deskew = f"{corrected}/{len(points)}"
            box = cfg["body_box"]
            if box is not None:
                points = points[
                    ~np.all((points >= box[:3]) & (points <= box[3:]), axis=1)
                ]
            prev = t
            yield t, voxel(points, cfg["voxel"], cfg["max_points"]), extr[:3, 3], deskew
