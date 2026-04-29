#!/usr/bin/env /usr/bin/python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GEORT_ROOT = THIS_DIR.parent
WS_ROOT = GEORT_ROOT.parent
DEX_RETARGETING_SRC = WS_ROOT / "dex-retargeting" / "src"
if str(DEX_RETARGETING_SRC) not in sys.path:
    sys.path.insert(0, str(DEX_RETARGETING_SRC))

from dex_retargeting.retargeting_config import RetargetingConfig


DEFAULT_CONFIG_PATH = THIS_DIR / "linkerhand_g20_right_vector.yml"
DEFAULT_ASSET_DIR = GEORT_ROOT / "assets"
DEFAULT_INPUT_PATH = GEORT_ROOT / "data" / "manus_data.npy"
DEFAULT_OUTPUT_PATH = GEORT_ROOT / "analysis" / "manus_g20_dex_retargeting.npz"

OPERATOR2MANO_RIGHT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# The official dex-retargeting examples assume robot URDFs are authored in the
# same right-hand MANO frame as SingleHandDetector. The LinkerHand LHG20 URDF is
# authored in its product frame: fingers extend along +Z, but finger flexion
# moves distal links toward +X. The MANO-frame Manus points flex toward -X, so
# reflect X once before building G20 reference vectors. This is a fixed frame
# alignment for the G20 URDF, not a replacement for dex-retargeting.
G20_DEX_TO_URDF_FRAME = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

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

# LinkerHand SDK G20 uses the same 20-value order as L20. These constants are
# copied from LinkerHand/utils/mapping.py: l20_r_min, l20_r_max, l20_r_derict.
G20_ARC_MIN = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.297,
        -0.26,
        -0.26,
        -0.26,
        -0.26,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)
G20_ARC_MAX = np.array(
    [
        0.87,
        1.4,
        1.4,
        1.4,
        1.4,
        0.683,
        0.26,
        0.26,
        0.26,
        0.26,
        1.78,
        0.0,
        0.0,
        0.0,
        0.0,
        1.29,
        1.08,
        1.08,
        1.08,
        1.08,
    ],
    dtype=np.float32,
)
G20_RAW_DIRECTION = np.array(
    [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1],
    dtype=np.int8,
)
G20_RESERVED_INDICES = {11, 12, 13, 14}

G20_COMMAND_JOINT_NAMES = [
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pithch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
    "thumb_cmc_roll",
    "index_mcp_roll",
    "middle_mcp_roll",
    "ring_mcp_roll",
    "pinky_mcp_roll",
    "thumb_cmc_yaw",
    None,
    None,
    None,
    None,
    "thumb_mcp",
    "index_pip",
    "middle_pip",
    "ring_pip",
    "pinky_pip",
]

DISTAL_LINK_NAMES = [
    "thumb_distal",
    "index_distal",
    "middle_distal",
    "ring_distal",
    "pinky_distal",
]
TIP_HUMAN_INDICES = np.array([4, 8, 12, 16, 20], dtype=int)


def _normalize(vec: np.ndarray, name: str, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError(f"Degenerate axis {name}: norm={norm}")
    return vec / norm


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Official dex-retargeting wrist frame estimator for MediaPipe-ordered points."""
    keypoint_3d_array = np.asarray(keypoint_3d_array, dtype=np.float64)
    if keypoint_3d_array.shape != (21, 3):
        raise ValueError(f"Expected (21, 3) keypoints, got {keypoint_3d_array.shape}")

    points = keypoint_3d_array[[0, 5, 9], :]
    x_vector = points[0] - points[2]

    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2, :]

    x_axis = x_vector - np.sum(x_vector * normal) * normal
    x_axis = _normalize(x_axis, "official wrist-to-middle projection")
    z_axis = np.cross(x_axis, normal)
    z_axis = _normalize(z_axis, "official palm lateral")

    if np.sum(z_axis * (centered[1] - centered[2])) < 0:
        normal *= -1
        z_axis *= -1

    return np.stack([x_axis, normal, z_axis], axis=1)


def hand_to_dex_frame(hand_point: np.ndarray) -> np.ndarray:
    """Convert world/Manus 21 points to dex-retargeting right-hand MANO frame."""
    keypoint_3d_array = np.asarray(hand_point, dtype=np.float64)
    if keypoint_3d_array.shape != (21, 3):
        raise ValueError(f"Expected (21, 3) keypoints, got {keypoint_3d_array.shape}")

    keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
    wrist_rot = estimate_frame_from_hand_points(keypoint_3d_array)
    return keypoint_3d_array @ wrist_rot @ OPERATOR2MANO_RIGHT


def _is_valid_keypoints(keypoints: np.ndarray, eps: float = 1e-6) -> bool:
    keypoints = np.asarray(keypoints)
    if keypoints.shape != (21, 3):
        return False
    if not np.isfinite(keypoints).all():
        return False
    wrist = keypoints[0]
    max_radius = np.max(np.linalg.norm(keypoints - wrist, axis=1))
    return np.isfinite(max_radius) and max_radius > eps


def prepare_npy_frame(frame: np.ndarray, npy_frame: str) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    if npy_frame == "dex":
        return frame
    if npy_frame in {"world", "geort"}:
        return hand_to_dex_frame(frame).astype(np.float32)
    raise ValueError(f"Unsupported --npy-frame value: {npy_frame}")


def apply_frame_alignment(keypoints: np.ndarray, frame_alignment: str) -> np.ndarray:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if frame_alignment == "none":
        return keypoints
    if frame_alignment == "g20":
        return keypoints @ G20_DEX_TO_URDF_FRAME.T
    raise ValueError(f"Unsupported --frame-alignment value: {frame_alignment}")


def _node_position(node) -> np.ndarray:
    pos = node.pose.position
    return np.array([pos.x, pos.y, pos.z], dtype=np.float64)


def _order_chain_nodes_by_joint_type(nodes) -> list:
    by_type = {}
    for node in nodes:
        by_type.setdefault(node.joint_type, []).append(node)

    def pick(*joint_types):
        for joint_type in joint_types:
            candidates = by_type.get(joint_type, [])
            if candidates:
                return min(candidates, key=lambda n: n.node_id)
        return None

    ordered = [pick("MCP"), pick("PIP"), pick("DIP", "IP"), pick("TIP")]
    if all(node is not None for node in ordered):
        return ordered

    priority = {"MCP": 0, "PIP": 1, "IP": 2, "DIP": 2, "TIP": 3}
    typed = [node for node in nodes if node.joint_type in priority]
    if len(typed) >= 4:
        return sorted(typed, key=lambda n: (priority[n.joint_type], n.node_id))[-4:]
    return []


def _order_chain_nodes(nodes) -> list:
    if len(nodes) == 0:
        return []

    ordered_by_joint_type = _order_chain_nodes_by_joint_type(nodes)
    if len(ordered_by_joint_type) >= 4:
        return ordered_by_joint_type

    node_by_id = {node.node_id: node for node in nodes}
    child_map = {}
    root = None
    for node in nodes:
        if node.parent_node_id in node_by_id:
            child_map.setdefault(node.parent_node_id, []).append(node)
        else:
            root = node

    if root is None:
        root = min(nodes, key=lambda n: n.node_id)

    ordered = []
    current = root
    visited = set()
    while current is not None and current.node_id not in visited:
        ordered.append(current)
        visited.add(current.node_id)
        children = sorted(child_map.get(current.node_id, []), key=lambda n: n.node_id)
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

    keypoints = np.asarray(keypoints, dtype=np.float64)
    return keypoints if keypoints.shape == (21, 3) else None


def compute_retargeting_reference(retargeting, keypoints: np.ndarray) -> np.ndarray:
    retargeting_type = retargeting.optimizer.retargeting_type
    indices = retargeting.optimizer.target_link_human_indices
    if retargeting_type == "POSITION":
        return keypoints[indices, :]

    origin_indices = indices[0, :]
    task_indices = indices[1, :]
    return keypoints[task_indices, :] - keypoints[origin_indices, :]


def build_retargeting(config_path: Path, asset_dir: Path, scaling: Optional[float]):
    RetargetingConfig.set_default_urdf_dir(str(asset_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    if scaling is not None:
        retargeting.optimizer.scaling = float(scaling)
    return retargeting


def retarget_frame_to_qpos(retargeting, keypoints: np.ndarray) -> np.ndarray:
    ref_value = compute_retargeting_reference(retargeting, keypoints)
    return retargeting.retarget(ref_value).astype(np.float32)


def compute_tip_scale_g20(retargeting, human_keypoints: np.ndarray) -> Optional[float]:
    if not _is_valid_keypoints(human_keypoints):
        return None

    robot = retargeting.optimizer.robot
    robot.compute_forward_kinematics(np.zeros(robot.dof))
    base_link_index = robot.get_link_index("wrist_base_link")
    base_pos = robot.get_link_pose(base_link_index)[:3, 3]

    robot_lengths = []
    for link_name in DISTAL_LINK_NAMES:
        link_index = robot.get_link_index(link_name)
        link_pos = robot.get_link_pose(link_index)[:3, 3]
        robot_lengths.append(np.linalg.norm(link_pos - base_pos))
    robot_lengths = np.asarray(robot_lengths, dtype=np.float64)

    human_lengths = np.linalg.norm(human_keypoints[TIP_HUMAN_INDICES] - human_keypoints[0], axis=1)
    valid = human_lengths > 0.04
    if valid.sum() < 3:
        return None

    ratios = robot_lengths[valid] / human_lengths[valid]
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) < 3:
        return None
    return float(np.clip(np.median(ratios), 0.4, 4.0))


def qpos_to_arc(qpos: np.ndarray, joint_names: Sequence[str], reserved_value: float) -> np.ndarray:
    qpos_dict = {name: float(value) for name, value in zip(joint_names, qpos)}
    out = np.zeros(20, dtype=np.float32)
    for i, joint_name in enumerate(G20_COMMAND_JOINT_NAMES):
        if joint_name is None:
            out[i] = float(reserved_value)
            continue
        value = qpos_dict[joint_name]
        out[i] = float(np.clip(value, G20_ARC_MIN[i], G20_ARC_MAX[i]))
    return out


def arc_to_raw(g20_arc: np.ndarray, reserved_raw_value: float) -> np.ndarray:
    g20_arc = np.asarray(g20_arc, dtype=np.float32)
    out = np.zeros(20, dtype=np.float32)
    for i, value in enumerate(g20_arc):
        if i in G20_RESERVED_INDICES:
            out[i] = float(np.clip(reserved_raw_value, 0.0, 255.0))
            continue

        lower = float(G20_ARC_MIN[i])
        upper = float(G20_ARC_MAX[i])
        if upper <= lower:
            out[i] = 0.0
            continue

        clipped = float(np.clip(value, lower, upper))
        normalized = (clipped - lower) / (upper - lower)
        if int(G20_RAW_DIRECTION[i]) == -1:
            raw = 255.0 * (1.0 - normalized)
        else:
            raw = 255.0 * normalized
        out[i] = float(np.clip(np.round(raw), 0.0, 255.0))
    return out


def summarize_vector(values: Sequence[float], names: Sequence[str], count: int = 6) -> str:
    return ", ".join(f"{name}={float(value):+.3f}" for name, value in zip(names[:count], values[:count]))


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
            canonical = apply_frame_alignment(
                prepare_npy_frame(frame, args.npy_frame),
                args.frame_alignment,
            )
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

    tic = time.perf_counter()
    for frame_idx, frame in enumerate(data[:max_frames]):
        if not _is_valid_keypoints(frame):
            continue
        canonical = apply_frame_alignment(
            prepare_npy_frame(frame, args.npy_frame),
            args.frame_alignment,
        )
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

    if not qpos_frames:
        raise RuntimeError("No valid frames were retargeted.")

    qpos_frames = np.asarray(qpos_frames, dtype=np.float32)
    g20_arc_frames = np.asarray(g20_arc_frames, dtype=np.float32)
    g20_raw_frames = np.asarray(g20_raw_frames, dtype=np.float32)

    wall_time = time.perf_counter() - tic
    solve_ms = 1000.0 * retargeting.accumulated_time / max(1, retargeting.num_retargeting)
    print(
        f"[offline] valid_frames={len(qpos_frames)} total_frames={min(len(data), max_frames)} "
        f"avg_solve_ms={solve_ms:.3f} wall_time_s={wall_time:.3f}"
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
            npy_frame=np.asarray([args.npy_frame]),
            frame_alignment=np.asarray([args.frame_alignment]),
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
                f"output_mode={args.output_mode}, frame_alignment={args.frame_alignment}, "
                f"scale={self.retargeting.optimizer.scaling:.4f}"
            )

        def glove_callback(self, msg: ManusGlove):
            keypoints = extract_semantic_keypoints(msg.raw_nodes)
            if keypoints is None or not _is_valid_keypoints(keypoints):
                self.skipped += 1
                return

            try:
                canonical = apply_frame_alignment(
                    hand_to_dex_frame(keypoints).astype(np.float32),
                    args.frame_alignment,
                )
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
        description="Retarget Manus right-hand keypoints to LinkerHand G20 with the official dex-retargeting flow."
    )
    parser.add_argument("--input", choices=["npy", "ros"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--scaling", type=float, default=None)
    parser.add_argument("--auto-scale", action="store_true")
    parser.add_argument("--print-every", type=int, default=120)

    parser.add_argument("--npy-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument(
        "--npy-frame",
        choices=["dex", "world", "geort"],
        default="dex",
        help="dex means already in dex-retargeting frame; world/geort will be converted with the official wrist estimator.",
    )
    parser.add_argument(
        "--frame-alignment",
        choices=["g20", "none"],
        default="g20",
        help="g20 aligns official dex/MANO right-hand points to the LinkerHand G20 URDF frame.",
    )
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
