#!/usr/bin/env python3
# Record canonical hand keypoints from ZMQ broadcast into a .npy file.
# Run AFTER starting manus_ros2_bridge.py (or manus_mocap_core.py).
#
# Usage:
#   python record_manus.py --name manus_data --duration 60
#   Saves: data/manus_data.npy  shape=(N_frames, 21, 3)

import argparse
import os
import time
import numpy as np
from geort.mocap.manus_mocap import ManusMocap
from geort.utils.path import get_human_data_output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',     type=str,   default='manus_data', help='Output file name (no extension)')
    parser.add_argument('--duration', type=float,  default=60.0,         help='Recording duration in seconds')
    parser.add_argument('--fps',      type=float,  default=30.0,         help='Target sampling rate (frames/s)')
    parser.add_argument('--port',     type=int,    default=8765,         help='ZMQ port')
    args = parser.parse_args()

    mocap = ManusMocap(port=args.port)

    print(f"Waiting for first frame on ZMQ port {args.port} ...")
    while True:
        result = mocap.get()
        if result['result'] is not None:
            break
        time.sleep(0.05)

    print(f"Recording for {args.duration}s at ~{args.fps} fps. Move your hand!")
    frames = []
    dt = 1.0 / args.fps
    t_end = time.time() + args.duration

    while time.time() < t_end:
        t0 = time.time()
        result = mocap.get()
        if result['result'] is not None:
            frames.append(result['result'].copy())   # (21, 3)
        elapsed = time.time() - t0
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    mocap.close()

    data = np.array(frames, dtype=np.float32)   # (N, 21, 3)
    print(f"Captured {len(data)} frames. Shape: {data.shape}")

    os.makedirs("data", exist_ok=True)
    save_path = get_human_data_output_path(args.name)
    np.save(save_path, data)
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
