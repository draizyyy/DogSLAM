"""Extract measured Go2 motor angles on the cloud header clock for RViz replay."""
import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    logs, headers, joint_times, angles = [], [], [], []
    with AnyReader([args.bag]) as reader:
        connections = [c for c in reader.connections if c.topic in ("/lowstate", "/utlidar/cloud_deskewed")]
        for c, t, raw in reader.messages(connections=connections):
            if c.topic == "/lowstate" and joint_times and t - joint_times[-1] < 15_000_000:
                continue
            msg = reader.deserialize(raw, c.msgtype)
            if c.topic == "/lowstate":
                joint_times.append(t)
                angles.append([motor.q for motor in msg.motor_state[:12]])
            else:
                logs.append(t)
                stamp = msg.header.stamp
                headers.append(stamp.sec + stamp.nanosec * 1e-9)
    offset = np.median(np.asarray(headers) - np.asarray(logs) * 1e-9)
    times = np.asarray(joint_times) * 1e-9 + offset
    np.savez_compressed(args.output, times=times, angles=np.asarray(angles, dtype=np.float32))
    print(len(times), times[0], times[-1], "header minus log", offset)
