#!/usr/bin/env /usr/bin/python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from manus_l20_dex_retarget import (
    DEFAULT_INPUT_PATH,
    _is_valid_keypoints,
    build_retargeting,
    extract_semantic_keypoints,
    hand_to_canonical,
)


THIS_DIR = Path(__file__).resolve().parent
GEORT_ROOT = THIS_DIR.parent
DEFAULT_CONFIG_PATH = THIS_DIR / "linkerhand_g20_right_vector.yml"
DEFAULT_ASSET_DIR = GEORT_ROOT / "assets"
DEFAULT_OUTPUT_PATH = GEORT_ROOT / "analysis" / "manus_g20_dex_retargeting.npz"

TIP_HUMAN_INDICES = np.array([4, 8, 12, 16, 20], dtype=int)
TIP_LINK_NAMES = [
    "thumb_distal",
    "index_distal",
    "middle_distal",
    "ring_distal",
    "pinky_distal",
]

G20_COMMAND_NAMES = [
    "thumb_base",
    "index_base",
    "middle_base",
    "ring_base",
    "little_base",
    "thumb_abduction",
    "index_abduction",
    "middle_abduction",
    "ring_abduction",
    "little_abduction",
    "thumb_roll",
    "reserved_11",
    "reserved_12",
    "reserved_13",
    "reserved_14",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "little_tip",
]

G20_ARC_SPECS = [
    ("thumb_base", "thumb_cmc_pitch", 0.0, 0.91, -1),
    ("index_base", "index_mcp_pitch", 0.0, 1.30, -1),
    ("middle_base", "middle_mcp_pithch", 0.0, 1.30, -1),
    ("ring_base", "ring_mcp_pitch", 0.0, 1.30, -1),
    ("little_base", "pinky_mcp_pitch", 0.0, 1.30, -1),
    ("thumb_abduction", "thumb_cmc_roll", 0.0, 1.54, -1),
    ("index_abduction", "index_mcp_roll", -0.26, 0.26, 0),
    ("middle_abduction", "middle_mcp_roll", -0.26, 0.26, 0),
    ("ring_abduction", "ring_mcp_roll", -0.26, 0.26, 0),
    ("little_abduction", "pinky_mcp_roll", -0.26, 0.26, 0),
    ("thumb_roll", "thumb_cmc_yaw", 0.0, 1.63, -1),
    ("reserved_11", None, 0.0, 0.0, 0),
    ("reserved_12", None, 0.0, 0.0, 0),
    ("reserved_13", None, 0.0, 0.0, 0),
    ("reserved_14", None, 0.0, 0.0, 0),
    ("thumb_tip", "thumb_mcp", 0.0, 1.29, -1),
    ("index_tip", "index_pip", 0.0, 1.42, -1),
    ("middle_tip", "middle_pip", 0.0, 1.42, -1),
    ("ring_tip", "ring_pip", 0.0, 1.42, -1),
    ("little_tip", "pinky_pip", 0.0, 1.42, -1),
]


def summarize_vector(values: Sequence[float], names: Sequence[str], count: int = 6) -> str:
    return ", ".join(
        f"{name}={float(value):+.3f}" for name, value in zip(names[:count], values[:count])
    )


def compute_reference(keypoints: np.ndarray, human_indices: np.ndarray) -> np.ndarray:
    return keypoints[human_indices[1], :] - keypoints[human_indices[0], :]


def retarget_frame_to_qpos(retargeting, frame_left: np.ndarray) -> np.ndarray:
    ref_value = compute_reference(frame_left, retargeting.optimizer.target_link_human_indices)
    return retargeting.retarget(ref_value).astype(np.float32)


def compute_tip_scale_g20(retargeting, human_keypoints: np.ndarray) -> Optional[float]:
    if not _is_valid_keypoints(human_keypoints):
        return None

    robot = retargeting.optimizer.robot
    robot.compute_forward_kinematics(np.zeros(robot.dof))
    base_link_index = robot.get_link_index("wrist_base_link")
    base_pos = robot.get_link_pose(base_link_index)[:3, 3]

    robot_tip_lengths = []
    for link_name in TIP_LINK_NAMES:
        link_index = robot.get_link_index(link_name)
        link_pos = robot.get_link_pose(link_index)[:3, 3]
        robot_tip_lengths.append(np.linalg.norm(link_pos - base_pos))
    robot_tip_lengths = np.asarray(robot_tip_lengths, dtype=np.float64)

    human_tip_lengths = np.linalg.norm(
        human_keypoints[TIP_HUMAN_INDICES] - human_keypoints[0], axis=1
    )
    valid = human_tip_lengths > 0.06
    if valid.sum() < 3:
        return None

    ratios = robot_tip_lengths[valid] / human_tip_lengths[valid]
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) < 3:
        return None
    return float(np.clip(np.median(ratios), 0.5, 4.0))


def qpos_to_arc(qpos: np.ndarray, joint_names: Sequence[str], reserved_value: float) -> np.ndarray:
    qpos_dict = {name: float(value) for name, value in zip(joint_names, qpos)}
    out = []
    for _, joint_name, lower, upper, _ in G20_ARC_SPECS:
        if joint_name is None:
            out.append(float(reserved_value))
            continue
        out.append(float(np.clip(qpos_dict[joint_name], lower, upper)))
    return np.asarray(out, dtype=np.float32)


def arc_to_raw(g20_arc: np.ndarray, reserved_raw_value: float) -> np.ndarray:
    out = []
    for value, (_, joint_name, lower, upper, direction) in zip(g20_arc, G20_ARC_SPECS):
        if joint_name is None:
            out.append(float(np.clip(reserved_raw_value, 0.0, 255.0)))
            continue

        if upper <= lower:
            out.append(0.0)
            continue

        norm = (float(value) - lower) / (upper - lower)
        if direction == -1:
            raw = 255.0 * (1.0 - norm)
        else:
            raw = 255.0 * norm
        out.append(float(np.clip(np.round(raw), 0.0, 255.0)))
    return np.asarray(out, dtype=np.float32)


def default_publish_topic(output_mode: str) -> str:
    if output_mode == "arc":
        return "/g20_dex_retarget/joint_states_arc"
    return "/cb_right_hand_control_cmd"


def run_offline(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    asset_dir = Path(args.asset_dir).resolve()
    input_path = Path(args.npy_path).resolve()
    output_path = Path(args.output_path).resolve() if args.output_path else None

    data = np.load(input_path)
    if data.ndim != 3 or data.shape[1:] != (21, 3):
        raise ValueError(f"Expected (N, 21, 3) data, got {data.shape}")

    retargeting = build_retargeting(config_path, asset_dir, args.scaling)
    if args.auto_scale:
        for frame in data:
            if not _is_valid_keypoints(frame):
                continue
            canonical = hand_to_canonical(frame).astype(np.float32)
            scale = compute_tip_scale_g20(retargeting, canonical)
            if scale is not None:
                retargeting.optimizer.scaling = scale
                print(f"[offline] auto scale = {scale:.4f}")
                break

    qpos_frames = []
    g20_arc_frames = []
    g20_raw_frames = []
    processed = 0
    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else len(data)

    t0 = time.perf_counter()
    for frame_idx, frame in enumerate(data[:max_frames]):
        if not _is_valid_keypoints(frame):
            continue
        canonical = hand_to_canonical(frame).astype(np.float32)
        qpos = retarget_frame_to_qpos(retargeting, canonical)
        g20_arc = qpos_to_arc(qpos, retargeting.joint_names, args.reserved_arc_value)
        g20_raw = arc_to_raw(g20_arc, args.reserved_raw_value)

        qpos_frames.append(qpos)
        g20_arc_frames.append(g20_arc)
        g20_raw_frames.append(g20_raw)
        processed += 1

        if args.print_every > 0 and processed % args.print_every == 0:
            print(
                f"[offline] frame={frame_idx + 1} processed={processed} "
                f"g20_arc: {summarize_vector(g20_arc, G20_COMMAND_NAMES)} "
                f"| g20_raw: {summarize_vector(g20_raw, G20_COMMAND_NAMES)}"
            )

    elapsed = time.perf_counter() - t0
    if not qpos_frames:
        raise RuntimeError("No valid frames were retargeted.")

    qpos_frames = np.asarray(qpos_frames, dtype=np.float32)
    g20_arc_frames = np.asarray(g20_arc_frames, dtype=np.float32)
    g20_raw_frames = np.asarray(g20_raw_frames, dtype=np.float32)

    solve_ms = 1000.0 * retargeting.accumulated_time / max(1, retargeting.num_retargeting)
    print(
        f"[offline] valid_frames={len(qpos_frames)} total_frames={min(len(data), max_frames)} "
        f"avg_solve_ms={solve_ms:.3f} wall_time_s={elapsed:.3f}"
    )
    print(
        f"[offline] qpos range = [{qpos_frames.min():+.4f}, {qpos_frames.max():+.4f}] "
        f"g20_arc range = [{g20_arc_frames.min():+.4f}, {g20_arc_frames.max():+.4f}] "
        f"g20_raw range = [{g20_raw_frames.min():+.1f}, {g20_raw_frames.max():+.1f}] "
        f"scale={retargeting.optimizer.scaling:.4f}"
    )
    print(f"[offline] first g20_arc: {summarize_vector(g20_arc_frames[0], G20_COMMAND_NAMES, count=8)}")
    print(f"[offline] first g20_raw: {summarize_vector(g20_raw_frames[0], G20_COMMAND_NAMES, count=8)}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            qpos_g20=qpos_frames,
            qpos_joint_names=np.asarray(retargeting.joint_names),
            g20_arc=g20_arc_frames,
            g20_raw=g20_raw_frames,
            g20_command_names=np.asarray(G20_COMMAND_NAMES),
            scaling=np.asarray([retargeting.optimizer.scaling], dtype=np.float32),
            input_path=np.asarray([str(input_path)]),
            config_path=np.asarray([str(config_path)]),
        )
        print(f"[offline] saved to {output_path}")

    return 0


def run_ros(args: argparse.Namespace) -> int:
    import rclpy
    from manus_ros2_msgs.msg import ManusGlove
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    class ManusG20DexRetargetNode(Node):
        def __init__(self):
            super().__init__("manus_g20_dex_retarget")
            self.retargeting = build_retargeting(
                Path(args.config).resolve(),
                Path(args.asset_dir).resolve(),
                args.scaling,
            )
            self.print_every = max(1, args.print_every)
            self.auto_scale = args.auto_scale
            self.auto_scale_done = not args.auto_scale
            self.processed = 0
            self.skipped = 0
            publish_topic = args.publish_topic or default_publish_topic(args.output_mode)

            self.publisher = self.create_publisher(JointState, publish_topic, 10)
            self.subscription = self.create_subscription(
                ManusGlove,
                f"/manus_glove_{args.glove_id}",
                self.glove_callback,
                10,
            )
            self.get_logger().info(
                f"Listening on /manus_glove_{args.glove_id}, publishing {publish_topic}, "
                f"output_mode={args.output_mode}, scale={self.retargeting.optimizer.scaling:.4f}"
            )

        def glove_callback(self, msg: ManusGlove):
            keypoints = extract_semantic_keypoints(msg.raw_nodes)
            if keypoints is None:
                self.skipped += 1
                return
            if not _is_valid_keypoints(keypoints):
                self.skipped += 1
                return

            try:
                canonical = hand_to_canonical(keypoints).astype(np.float32)
            except ValueError:
                self.skipped += 1
                return
            if self.auto_scale and not self.auto_scale_done:
                scale = compute_tip_scale_g20(self.retargeting, canonical)
                if scale is not None:
                    self.retargeting.optimizer.scaling = scale
                    self.auto_scale_done = True
                    self.get_logger().info(f"Auto scale = {scale:.4f}")

            qpos = retarget_frame_to_qpos(self.retargeting, canonical)
            g20_arc = qpos_to_arc(qpos, self.retargeting.joint_names, args.reserved_arc_value)
            g20_raw = arc_to_raw(g20_arc, args.reserved_raw_value)
            position = g20_arc if args.output_mode == "arc" else g20_raw

            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.header.frame_id = args.frame_id
            joint_msg.name = list(G20_COMMAND_NAMES)
            joint_msg.position = position.astype(np.float64).tolist()
            self.publisher.publish(joint_msg)

            self.processed += 1
            if self.processed % self.print_every == 0:
                solve_ms = 1000.0 * self.retargeting.accumulated_time / max(
                    1, self.retargeting.num_retargeting
                )
                self.get_logger().info(
                    f"processed={self.processed} skipped={self.skipped} "
                    f"avg_solve_ms={solve_ms:.3f} "
                    f"g20_{args.output_mode}: {summarize_vector(position, G20_COMMAND_NAMES)}"
                )

    rclpy.init()
    node = ManusG20DexRetargetNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        if "context is not valid" not in str(exc):
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retarget Manus right-hand keypoints to dedicated G20 parameters using the official LHG20 URDF."
    )
    parser.add_argument("--input", choices=["npy", "ros"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--scaling", type=float, default=None)
    parser.add_argument("--auto-scale", action="store_true")
    parser.add_argument("--print-every", type=int, default=120)

    parser.add_argument("--npy-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--max-frames", type=int, default=0)

    parser.add_argument("--glove-id", type=int, default=0)
    parser.add_argument("--output-mode", choices=["raw", "arc"], default="raw")
    parser.add_argument("--publish-topic", default="")
    parser.add_argument("--frame-id", default="wrist_base_link")
    parser.add_argument("--reserved-raw-value", type=float, default=255.0)
    parser.add_argument("--reserved-arc-value", type=float, default=0.0)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input == "npy":
        return run_offline(args)
    return run_ros(args)


if __name__ == "__main__":
    raise SystemExit(main())
