#!/usr/bin/env /usr/bin/python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GEORT_ROOT = THIS_DIR.parent
WS_ROOT = GEORT_ROOT.parent
DEX_RETARGETING_SRC = WS_ROOT / "dex-retargeting" / "src"
if str(DEX_RETARGETING_SRC) not in sys.path:
    sys.path.insert(0, str(DEX_RETARGETING_SRC))

from dex_retargeting.retargeting_config import RetargetingConfig


DEFAULT_CONFIG_PATH = THIS_DIR / "linkerhand_l20_right_vector.yml"
DEFAULT_ASSET_DIR = GEORT_ROOT / "assets"
DEFAULT_INPUT_PATH = GEORT_ROOT / "data" / "manus_data.npy"
DEFAULT_OUTPUT_PATH = GEORT_ROOT / "analysis" / "manus_l20_dex_retargeting.npz"

PIN_JOINT_NAMES = [
    "index_joint0",
    "index_joint1",
    "index_joint2",
    "index_joint3",
    "little_joint0",
    "little_joint1",
    "little_joint2",
    "little_joint3",
    "middle_joint0",
    "middle_joint1",
    "middle_joint2",
    "middle_joint3",
    "ring_joint0",
    "ring_joint1",
    "ring_joint2",
    "ring_joint3",
    "thumb_joint0",
    "thumb_joint1",
    "thumb_joint2",
    "thumb_joint3",
    "thumb_joint4",
]

GEORT_JOINT_NAMES = [
    "thumb_joint0",
    "thumb_joint1",
    "thumb_joint2",
    "thumb_joint3",
    "thumb_joint4",
    "index_joint0",
    "index_joint1",
    "index_joint2",
    "index_joint3",
    "middle_joint0",
    "middle_joint1",
    "middle_joint2",
    "middle_joint3",
    "ring_joint0",
    "ring_joint1",
    "ring_joint2",
    "ring_joint3",
    "little_joint0",
    "little_joint1",
    "little_joint2",
    "little_joint3",
]

TIP_LINK_NAMES = [
    "thumb_link5",
    "index_link4",
    "middle_link4",
    "ring_link4",
    "little_link4",
]
TIP_HUMAN_INDICES = np.array([4, 8, 12, 16, 20], dtype=int)


def _normalize(vec: np.ndarray, name: str, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError(f"Degenerate axis {name}: norm={norm}")
    return vec / norm


def hand_to_canonical(hand_point: np.ndarray) -> np.ndarray:
    z_axis = hand_point[9] - hand_point[0]
    z_axis = _normalize(z_axis, "wrist->middle_mcp")
    y_axis_aux = hand_point[5] - hand_point[13]
    y_axis_aux = _normalize(y_axis_aux, "index_mcp->ring_mcp")

    x_axis = np.cross(y_axis_aux, z_axis)
    x_axis = _normalize(x_axis, "palm_normal")
    y_axis = np.cross(z_axis, x_axis)
    y_axis = _normalize(y_axis, "palm_y")

    rotation_base = np.array([x_axis, y_axis, z_axis]).transpose()
    transform = np.eye(4)
    transform[:3, :3] = rotation_base
    transform[:3, 3] = hand_point[0]

    transform_inv = np.linalg.inv(transform)
    pts = np.concatenate([np.array(hand_point), np.ones((21, 1))], axis=-1)
    pts = pts @ transform_inv.transpose()
    return pts[:, :3]


def _is_valid_keypoints(keypoints: np.ndarray, eps: float = 1e-6) -> bool:
    if keypoints.shape != (21, 3):
        return False
    if not np.isfinite(keypoints).all():
        return False
    wrist = keypoints[0]
    max_radius = np.max(np.linalg.norm(keypoints - wrist, axis=1))
    return np.isfinite(max_radius) and max_radius > eps


def _node_position(node) -> np.ndarray:
    pos = node.pose.position
    return np.array([pos.x, pos.y, pos.z], dtype=np.float64)


def _order_chain_nodes(nodes) -> list:
    if len(nodes) == 0:
        return []

    node_by_id = {node.node_id: node for node in nodes}
    child_map = {}
    root = None

    for node in nodes:
        if node.parent_node_id in node_by_id:
            child_map.setdefault(node.parent_node_id, []).append(node)
        else:
            root = node

    if root is None:
        root = min(nodes, key=lambda node: node.node_id)

    ordered = []
    current = root
    visited = set()
    while current is not None and current.node_id not in visited:
        ordered.append(current)
        visited.add(current.node_id)
        children = sorted(
            child_map.get(current.node_id, []), key=lambda node: node.node_id
        )
        current = children[0] if children else None
    return ordered


def extract_semantic_keypoints(raw_nodes) -> Optional[np.ndarray]:
    chain_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    hand_nodes = [node for node in raw_nodes if node.chain_type == "Hand"]
    if len(hand_nodes) == 0:
        return None

    wrist = min(hand_nodes, key=lambda node: node.node_id)
    keypoints = [_node_position(wrist)]

    for chain_name in chain_names:
        chain_nodes = [node for node in raw_nodes if node.chain_type == chain_name]
        ordered = _order_chain_nodes(chain_nodes)
        if len(ordered) < 4:
            return None
        keypoints.extend([_node_position(node) for node in ordered[-4:]])

    keypoints = np.array(keypoints, dtype=np.float64)
    return keypoints if keypoints.shape == (21, 3) else None


def compute_reference(keypoints: np.ndarray, human_indices: np.ndarray) -> np.ndarray:
    return keypoints[human_indices[1], :] - keypoints[human_indices[0], :]


def resolve_joint_order(joint_order: str) -> Sequence[str]:
    if joint_order == "pin":
        return PIN_JOINT_NAMES
    if joint_order == "geort":
        return GEORT_JOINT_NAMES
    raise ValueError(f"Unsupported joint order: {joint_order}")


def build_retargeting(
    config_path: Path,
    asset_dir: Path,
    scaling: Optional[float],
):
    RetargetingConfig.set_default_urdf_dir(str(asset_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    if scaling is not None:
        retargeting.optimizer.scaling = float(scaling)
    return retargeting


def compute_tip_scale(retargeting, human_keypoints: np.ndarray) -> Optional[float]:
    if not _is_valid_keypoints(human_keypoints):
        return None

    robot = retargeting.optimizer.robot
    robot.compute_forward_kinematics(np.zeros(robot.dof))
    base_link_index = robot.get_link_index("base_link")
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


def reorder_qpos(qpos_pin: np.ndarray, source_names: Sequence[str], target_names: Sequence[str]) -> np.ndarray:
    index_map = np.array([source_names.index(name) for name in target_names], dtype=int)
    return qpos_pin[index_map]


def summarize_qpos(qpos: np.ndarray, joint_names: Sequence[str], count: int = 5) -> str:
    parts = []
    for name, value in zip(joint_names[:count], qpos[:count]):
        parts.append(f"{name}={value:+.3f}")
    return ", ".join(parts)


def run_offline(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    asset_dir = Path(args.asset_dir).resolve()
    input_path = Path(args.npy_path).resolve()
    output_path = Path(args.output_path).resolve() if args.output_path else None

    data = np.load(input_path)
    if data.ndim != 3 or data.shape[1:] != (21, 3):
        raise ValueError(f"Expected (N, 21, 3) data, got {data.shape}")

    retargeting = build_retargeting(config_path, asset_dir, args.scaling)
    desired_joint_names = list(resolve_joint_order(args.joint_order))
    source_joint_names = list(retargeting.joint_names)

    if args.auto_scale:
        for frame in data:
            scale = compute_tip_scale(retargeting, frame)
            if scale is not None:
                retargeting.optimizer.scaling = scale
                print(f"[offline] auto scale = {scale:.4f}")
                break

    qpos_frames = []
    processed = 0
    human_indices = retargeting.optimizer.target_link_human_indices
    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else len(data)

    t0 = time.perf_counter()
    for frame_idx, frame in enumerate(data[:max_frames]):
        if not _is_valid_keypoints(frame):
            continue
        ref_value = compute_reference(frame, human_indices)
        qpos_pin = retargeting.retarget(ref_value)
        qpos_ordered = reorder_qpos(qpos_pin, source_joint_names, desired_joint_names)
        qpos_frames.append(qpos_ordered.astype(np.float32))
        processed += 1

        if args.print_every > 0 and processed % args.print_every == 0:
            print(
                f"[offline] frame={frame_idx + 1} processed={processed} "
                f"{summarize_qpos(qpos_ordered, desired_joint_names)}"
            )

    elapsed = time.perf_counter() - t0
    qpos_frames = np.asarray(qpos_frames, dtype=np.float32)
    if qpos_frames.size == 0:
        raise RuntimeError("No valid frames were retargeted.")

    solve_ms = 1000.0 * retargeting.accumulated_time / max(1, retargeting.num_retargeting)
    print(
        f"[offline] valid_frames={len(qpos_frames)} total_frames={min(len(data), max_frames)} "
        f"avg_solve_ms={solve_ms:.3f} wall_time_s={elapsed:.3f}"
    )
    print(
        f"[offline] qpos range = [{qpos_frames.min():+.4f}, {qpos_frames.max():+.4f}] "
        f"scale={retargeting.optimizer.scaling:.4f}"
    )
    print(f"[offline] first frame: {summarize_qpos(qpos_frames[0], desired_joint_names, count=8)}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            qpos=qpos_frames,
            joint_names=np.asarray(desired_joint_names),
            source_joint_names=np.asarray(source_joint_names),
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

    class ManusL20DexRetargetNode(Node):
        def __init__(self):
            super().__init__("manus_l20_dex_retarget")
            self.retargeting = build_retargeting(
                Path(args.config).resolve(),
                Path(args.asset_dir).resolve(),
                args.scaling,
            )
            self.desired_joint_names = list(resolve_joint_order(args.joint_order))
            self.source_joint_names = list(self.retargeting.joint_names)
            self.human_indices = self.retargeting.optimizer.target_link_human_indices
            self.print_every = max(1, args.print_every)
            self.auto_scale = args.auto_scale
            self.auto_scale_done = not args.auto_scale
            self.processed = 0
            self.skipped = 0

            self.publisher = self.create_publisher(JointState, args.publish_topic, 10)
            self.subscription = self.create_subscription(
                ManusGlove,
                f"/manus_glove_{args.glove_id}",
                self.glove_callback,
                10,
            )

            self.get_logger().info(
                f"Listening on /manus_glove_{args.glove_id}, publishing {args.publish_topic}, "
                f"joint_order={args.joint_order}, scale={self.retargeting.optimizer.scaling:.4f}"
            )

        def glove_callback(self, msg: ManusGlove):
            keypoints = extract_semantic_keypoints(msg.raw_nodes)
            if keypoints is None:
                positions = {}
                for node in msg.raw_nodes:
                    positions[node.node_id] = _node_position(node)
                if len(positions) < 21:
                    self.skipped += 1
                    return
                try:
                    keypoints = np.array([positions[i] for i in range(21)], dtype=np.float64)
                except KeyError:
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
                scale = compute_tip_scale(self.retargeting, canonical)
                if scale is not None:
                    self.retargeting.optimizer.scaling = scale
                    self.auto_scale_done = True
                    self.get_logger().info(f"Auto scale = {scale:.4f}")

            ref_value = compute_reference(canonical, self.human_indices)
            qpos_pin = self.retargeting.retarget(ref_value)
            qpos_ordered = reorder_qpos(
                qpos_pin,
                self.source_joint_names,
                self.desired_joint_names,
            )

            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.header.frame_id = args.frame_id
            joint_msg.name = self.desired_joint_names
            joint_msg.position = qpos_ordered.tolist()
            self.publisher.publish(joint_msg)

            self.processed += 1
            if self.processed % self.print_every == 0:
                solve_ms = 1000.0 * self.retargeting.accumulated_time / max(
                    1, self.retargeting.num_retargeting
                )
                self.get_logger().info(
                    f"processed={self.processed} skipped={self.skipped} "
                    f"avg_solve_ms={solve_ms:.3f} "
                    f"{summarize_qpos(qpos_ordered, self.desired_joint_names)}"
                )

    rclpy.init()
    node = ManusL20DexRetargetNode()
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
        description="Retarget Manus hand keypoints to LinkerHand L20 using dex-retargeting."
    )
    parser.add_argument("--input", choices=["npy", "ros"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--joint-order", choices=["pin", "geort"], default="geort")
    parser.add_argument("--scaling", type=float, default=None)
    parser.add_argument("--auto-scale", action="store_true")
    parser.add_argument("--print-every", type=int, default=120)

    parser.add_argument("--npy-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--max-frames", type=int, default=0)

    parser.add_argument("--glove-id", type=int, default=0)
    parser.add_argument("--publish-topic", default="/l20_dex_retarget/joint_states")
    parser.add_argument("--frame-id", default="base_link")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input == "npy":
        return run_offline(args)
    return run_ros(args)


if __name__ == "__main__":
    raise SystemExit(main())
