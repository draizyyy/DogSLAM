"""Seeded two-session fixture. GT is written separately and never read by the estimator."""

import numpy as np
from scipy.spatial.transform import Rotation
from rosbags.rosbag2 import Writer, StoragePlugin
from rosbags.typesys import Stores, get_typestore
from .geom import inv, pose, rotation, write_tum


def path(t, inspection):
    u = np.clip((t - 1.3) / (9 if inspection else 18), 0, 1)
    a = u * 2 * np.pi
    if inspection:
        xy = [1.35 + 2.0 * u, 2.0 - 0.95 * u + 0.18 * np.sin(a)]
        yaw = 0.55 - 0.92 * u
    else:
        xy = [3 * np.sin(a), 1.8 * (1 - np.cos(a))]
        yaw = a
    sway = 0.03 * np.sin(12 * t) * np.sin(np.pi * u)
    return pose(
        Rotation.from_euler("xyz", [sway, sway * 0.7, yaw]),
        [*xy, 0.025 * np.sin(16 * t) * np.sin(np.pi * u)],
    )


def ranges(o, d, t):
    # Room interior (exit intersection) with asymmetric office fixtures (entry).
    boxes = [
        ([-6, -3, -0.5], [7, 7, 2.45]),
        ([-4.8, 0.5, -0.5], [-4, 1.5, 1.8]),
        ([4.6, -1.8, -0.5], [5.1, 0.5, 1.3]),
        ([-0.8, 4.8, -0.5], [1.4, 5.2, 1.1]),
        ([-2.2, -2.6, -0.5], [-1.3, -1.9, 2.1]),
        ([4.4, 3.5, -0.5], [5.7, 4.2, 1.45]),
        ([-3.0, 5.6, -0.5], [-2.5, 6.1, 2.45]),
    ]
    if 4 < t < 8:
        boxes.append(([0.2 * t, -1.2, -0.5], [0.2 * t + 0.3, -0.85, 1.1]))
    depth = np.full(len(d), np.inf)
    dr = np.where(np.abs(d) < 1e-10, 1e-10, d)
    for i, (lo, hi) in enumerate(boxes):
        a, b = (lo - o) / dr, (hi - o) / dr
        entry = np.minimum(a, b).max(axis=1)
        exit = np.maximum(a, b).min(axis=1)
        dist = exit if i == 0 else entry
        depth = np.minimum(depth, np.where((exit >= entry) & (dist > 0), dist, np.inf))
    return depth


def generate(out, inspection):
    name = "inspection" if inspection else "survey"
    store = get_typestore(Stores.ROS2_HUMBLE)
    C = lambda typ: store.types[typ]
    vec = C("geometry_msgs/msg/Vector3")
    point = C("geometry_msgs/msg/Point")
    quat = C("geometry_msgs/msg/Quaternion")

    def header(ns, frame):
        return C("std_msgs/msg/Header")(
            C("builtin_interfaces/msg/Time")(ns // 10**9, ns % 10**9), frame
        )

    extr = pose(Rotation.from_euler("z", 7, degrees=True), [0.17, 0.02, 0.19])
    rng = np.random.default_rng(82 if inspection else 43)
    az, el = np.meshgrid(
        np.linspace(-np.pi, np.pi, 160, endpoint=False), np.radians([-15, -7, 0, 7, 15])
    )
    az, el = az.ravel(), el.ravel()
    direction = np.c_[np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]
    fields = [
        C("sensor_msgs/msg/PointField")(k, 4 * i, 7, 1)
        for i, k in enumerate(["x", "y", "z", "dt"])
    ]
    times, poses = [], []
    base_ns = (1700001000 if inspection else 1700000000) * 10**9
    with Writer(
        out / name,
        version=9,
        storage_plugin=StoragePlugin.MCAP if inspection else StoragePlugin.SQLITE3,
    ) as writer:

        def channel(n, typ):
            return writer.add_connection(n, typ, typestore=store)

        def put(c, ns, m):
            writer.write(c, ns + 2000000, store.serialize_cdr(m, c.msgtype))

        lidar = channel("/lidar", "sensor_msgs/msg/PointCloud2")
        imu = channel("/imu", "sensor_msgs/msg/Imu")
        odom = channel("/leg_odom", "nav_msgs/msg/Odometry")
        tf = channel("/tf_static", "tf2_msgs/msg/TFMessage")
        tr = C("geometry_msgs/msg/Transform")(
            vec(*extr[:3, 3]), quat(*rotation(extr).as_quat())
        )
        put(
            tf,
            base_ns,
            C(tf.msgtype)(
                [
                    C("geometry_msgs/msg/TransformStamped")(
                        header(base_ns, "base_link"), "laser", tr
                    )
                ]
            ),
        )
        for i in range((11 if inspection else 20) * 50 + 1):
            t = i * 0.02
            ns = base_ns + i * 20000000
            T = path(t, inspection)
            rate = rotation(inv(T) @ path(t + 0.001, inspection)).as_rotvec() / 0.001
            a = T[:3, :3].T @ np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.02, 3)
            put(
                imu,
                ns,
                C(imu.msgtype)(
                    header(ns, "base_link"),
                    quat(0.0, 0.0, 0.0, 1.0),
                    np.zeros(9),
                    vec(*(rate + [0.001, -0.002, 0.005])),
                    np.zeros(9),
                    vec(*a),
                    np.zeros(9),
                ),
            )
            if i % 2 == 0:
                od = inv(path(0, inspection)) @ T
                od[:3, 3] *= 1.06
                od[:3, :3] = (
                    Rotation.from_euler("z", 0.007 * max(0, t - 1.3)).as_matrix()
                    @ od[:3, :3]
                )
                p = C("geometry_msgs/msg/Pose")(
                    point(*od[:3, 3]), quat(*rotation(od).as_quat())
                )
                tw = C("geometry_msgs/msg/Twist")(
                    vec(0.0, 0.0, 0.0), vec(0.0, 0.0, 0.0)
                )
                put(
                    odom,
                    ns,
                    C(odom.msgtype)(
                        header(ns, "odom"),
                        "base_link",
                        C("geometry_msgs/msg/PoseWithCovariance")(p, np.zeros(36)),
                        C("geometry_msgs/msg/TwistWithCovariance")(tw, np.zeros(36)),
                    ),
                )
            if i % 5 == 0:
                offsets = np.floor((az + np.pi) / (2 * np.pi) * 8) * 0.01 - 0.04
                points = np.empty_like(direction)
                for dt in np.unique(offsets):
                    mask = offsets == dt
                    S = path(t + dt, inspection) @ extr
                    dist = ranges(S[:3, 3], direction[mask] @ S[:3, :3].T, t + dt)
                    points[mask] = (
                        direction[mask]
                        * (dist + rng.normal(0, 0.015, len(dist)))[:, None]
                    )
                a = np.c_[points, offsets].astype("<f4")
                put(
                    lidar,
                    ns,
                    C(lidar.msgtype)(
                        header(ns, "laser"),
                        1,
                        len(a),
                        fields,
                        False,
                        16,
                        16 * len(a),
                        a.view(np.uint8).ravel(),
                        True,
                    ),
                )
                times.append(ns / 1e9)
                poses.append(T)
    write_tum(out / (name + "_gt.tum"), times, poses)
