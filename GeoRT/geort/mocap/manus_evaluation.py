# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from geort.mocap.manus_mocap import ManusMocap
from geort.env.hand import HandKinematicModel
from geort import load_model, get_config
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hand', type=str, default='allegro')
    parser.add_argument('-ckpt_tag', type=str, default='alex')  # Your CKPT Tag.

    args = parser.parse_args()

    # GeoRT Model.
    model = load_model(args.ckpt_tag)
    
    # Motion Capture.
    mocap = ManusMocap()
    
    # Robot Simulation.
    config = get_config(args.hand)
    hand = HandKinematicModel.build_from_config(config, render=True)
    try:
        viewer_env = hand.get_viewer_env()
    except Exception as exc:
        print(f"Failed to initialize SAPIEN viewer: {exc}", flush=True)
        mocap.close()
        os._exit(1)
    
    # Run!
    had_live_data = False
    try:
        while True:
            viewer_env.update()

            result = mocap.get()

            if result['status'] == 'recording' and result["result"] is not None:
                qpos = model.forward(result["result"])
                if qpos is not None and hand.set_qpos_target(qpos):
                    had_live_data = True
                continue

            if result['status'] in {'stale', 'no data'} and had_live_data:
                hand.reset_qpos_target()
                had_live_data = False

            if result['status'] == 'quit':
                break
    finally:
        mocap.close()

if __name__ == '__main__':
    main()
