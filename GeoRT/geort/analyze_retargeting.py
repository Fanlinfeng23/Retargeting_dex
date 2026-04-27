import argparse
import json
from pathlib import Path

import numpy as np

from geort import get_config, load_model
from geort.env.hand import HandKinematicModel
from geort.export import GeoRTRetargetingModel
from geort.utils.config_utils import parse_config_keypoint_info


def compute_metrics(diff):
    abs_diff = np.abs(diff)
    l2 = np.linalg.norm(diff, axis=-1)
    return {
        "mae": float(abs_diff.mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "mean_l2": float(l2.mean()),
        "p95_l2": float(np.percentile(l2, 95)),
        "max_l2": float(l2.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-hand", type=str, required=True)
    parser.add_argument("-ckpt_tag", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("-data", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    config = get_config(args.hand)
    info = parse_config_keypoint_info(config)
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        model = GeoRTRetargetingModel(
            model_path=checkpoint_dir / "last.pth",
            config_path=checkpoint_dir / "config.json",
        )
        ckpt_label = checkpoint_dir.name
    else:
        model = load_model(args.ckpt_tag)
        ckpt_label = args.ckpt_tag

    hand = HandKinematicModel.build_from_config(config, render=False, visualization_mode=True)
    hand.initialize_keypoint(info["link"], info["offset"])

    data_path = Path("data") / f"{args.data}.npy"
    human_all = np.load(data_path).astype(np.float32)
    if args.max_frames > 0:
        human_all = human_all[:args.max_frames]

    human = human_all[:, info["human_id"], :3]

    robot = []
    qpos_list = []
    for frame in human_all:
        qpos = model.forward(frame)
        if qpos is None:
            continue
        qpos_list.append(qpos)
        robot.append(hand.keypoint_from_qpos(qpos, ret_vec=True))

    robot = np.asarray(robot, dtype=np.float32)
    human = human[: len(robot)]
    qpos_list = np.asarray(qpos_list, dtype=np.float32)

    diff = robot - human
    report = {
        "hand": args.hand,
        "ckpt_tag": ckpt_label,
        "data": args.data,
        "n_frames": int(len(robot)),
        "n_keypoints": int(robot.shape[1]),
        "overall": compute_metrics(diff),
        "per_group": {},
        "per_tip": {},
    }

    start = 0
    for group_name, group_size, tip_idx in zip(info["group_name"], info["group_size"], info["tip_indices"]):
        group_slice = slice(start, start + group_size)
        report["per_group"][group_name] = compute_metrics(diff[:, group_slice, :])
        report["per_tip"][group_name] = compute_metrics(diff[:, tip_idx:tip_idx + 1, :])
        start += group_size

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("analysis") / f"{args.hand}_{ckpt_label}_{args.data}_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
