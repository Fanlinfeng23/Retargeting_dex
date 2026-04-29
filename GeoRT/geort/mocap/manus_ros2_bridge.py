#!/usr/bin/env python3
"""
Bridge: manus_ros2 ManusGlove msgs → ZMQ canonical keypoints.

The official GeoRT pipeline expects a ZMQ PUB socket on port 8765 broadcasting
raw float32 bytes of shape (21, 3) in canonical wrist-local frame.

manus_mocap_core.py achieves this via a Windows Manus Core server + ROS topics
(/x_manus_rotations etc.).  This bridge achieves the same result directly from
the manus_ros2 Integrated SDK topic /manus_glove_<id>, which publishes
ManusGlove messages containing ManusRawNode[] raw_nodes with absolute 3D poses.

Node ID mapping (MediaPipe 21-keypoint convention used throughout GeoRT):
  0  = wrist
  1-4  = thumb (CMC → TIP)
  5-8  = index  (MCP → TIP)
  9-12 = middle (MCP → TIP)
  13-16= ring   (MCP → TIP)
  17-20= pinky  (MCP → TIP)

If your Manus SDK uses different node IDs, adjust NODE_ID_TO_MEDIAPIPE below.

Run:
  source /home/user/ros2_ws/install/setup.bash
  python geort/mocap/manus_ros2_bridge.py --glove_id 0 --port 8765
"""

import argparse
import numpy as np
import time
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from manus_ros2_msgs.msg import ManusGlove
import zmq


# ---------------------------------------------------------------------------
# Canonical frame transform matching dex-retargeting's SingleHandDetector.
# ---------------------------------------------------------------------------
OPERATOR2MANO_RIGHT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def hand_to_canonical(hand_point):
    """Convert world-frame 21 keypoints to dex-retargeting's right-hand frame."""
    keypoint_3d_array = np.asarray(hand_point, dtype=np.float64)
    if keypoint_3d_array.shape != (21, 3):
        raise ValueError(f"Expected (21, 3) keypoints, got {keypoint_3d_array.shape}")

    keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
    wrist_rot = estimate_frame_from_hand_points(keypoint_3d_array)
    return keypoint_3d_array @ wrist_rot @ OPERATOR2MANO_RIGHT


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
    x_axis = _normalize(x_axis, "official wrist->middle projection")
    z_axis = np.cross(x_axis, normal)
    z_axis = _normalize(z_axis, "official palm lateral")

    if np.sum(z_axis * (centered[1] - centered[2])) < 0:
        normal *= -1
        z_axis *= -1

    return np.stack([x_axis, normal, z_axis], axis=1)


def _is_valid_keypoints(keypoints: np.ndarray, eps: float = 1e-6) -> bool:
    if keypoints.shape != (21, 3):
        return False
    if not np.isfinite(keypoints).all():
        return False

    wrist = keypoints[0]
    max_radius = np.max(np.linalg.norm(keypoints - wrist, axis=1))
    if not np.isfinite(max_radius) or max_radius <= eps:
        return False

    return True


def _node_position(node) -> np.ndarray:
    p = node.pose.position
    return np.array([p.x, p.y, p.z], dtype=np.float64)


def _order_chain_nodes_by_joint_type(nodes):
    by_type = {}
    for node in nodes:
        by_type.setdefault(node.joint_type, []).append(node)

    def pick(*joint_types):
        for joint_type in joint_types:
            candidates = by_type.get(joint_type, [])
            if candidates:
                return min(candidates, key=lambda node: node.node_id)
        return None

    ordered = [
        pick("MCP"),
        pick("PIP"),
        pick("DIP", "IP"),
        pick("TIP"),
    ]
    if all(node is not None for node in ordered):
        return ordered

    priority = {"MCP": 0, "PIP": 1, "IP": 2, "DIP": 2, "TIP": 3}
    typed = [node for node in nodes if node.joint_type in priority]
    if len(typed) >= 4:
        return sorted(typed, key=lambda node: (priority[node.joint_type], node.node_id))[-4:]

    return []


def _order_chain_nodes(nodes):
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
        root = min(nodes, key=lambda node: node.node_id)

    ordered = []
    current = root
    visited = set()
    while current is not None and current.node_id not in visited:
        ordered.append(current)
        visited.add(current.node_id)
        children = sorted(child_map.get(current.node_id, []), key=lambda node: node.node_id)
        current = children[0] if len(children) > 0 else None

    return ordered


def _extract_semantic_keypoints(raw_nodes):
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
    if keypoints.shape != (21, 3):
        return None
    return keypoints


def _summarize_raw_nodes(raw_nodes):
    lines = []
    for node in sorted(raw_nodes, key=lambda n: (n.chain_type, n.node_id)):
        pos = _node_position(node)
        lines.append(
            f"id={node.node_id:>2} parent={node.parent_node_id:>2} "
            f"chain={node.chain_type:<6} joint={node.joint_type:<3} "
            f"pos=({pos[0]: .4f},{pos[1]: .4f},{pos[2]: .4f})"
        )
    return lines


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------
class ManusRos2Bridge(Node):
    def __init__(self, glove_id: int, zmq_port: int, debug_nodes: bool = False, debug_every: int = 200):
        super().__init__('manus_ros2_bridge')
        self._last_warn_time = {}
        self._debug_nodes = debug_nodes
        self._debug_every = max(1, debug_every)
        self._frame_count = 0

        # ZMQ publisher
        ctx = zmq.Context()
        self._context = ctx
        self._socket = ctx.socket(zmq.PUB)
        self._socket.bind(f"tcp://*:{zmq_port}")
        self.get_logger().info(f"ZMQ PUB bound on port {zmq_port}")

        self.create_subscription(
            ManusGlove,
            f'/manus_glove_{glove_id}',
            self._glove_cb,
            10,
        )
        self.get_logger().info(f"Subscribed to /manus_glove_{glove_id}")
        self._logged_semantic_mapping = False

    def _warn_throttled(self, key: str, message: str, interval_sec: float = 2.0):
        now = time.monotonic()
        last = self._last_warn_time.get(key, 0.0)
        if now - last >= interval_sec:
            self._last_warn_time[key] = now
            self.get_logger().warn(message)

    def _glove_cb(self, msg: ManusGlove):
        self._frame_count += 1
        keypoints = _extract_semantic_keypoints(msg.raw_nodes)
        if keypoints is None:
            # Fallback to node_id order for older SDK layouts.
            positions = {}
            for node in msg.raw_nodes:
                positions[node.node_id] = _node_position(node)

            if len(positions) < 21:
                return
            try:
                keypoints = np.array([positions[i] for i in range(21)], dtype=np.float64)
            except KeyError:
                self._warn_throttled("missing_ids", "Missing node IDs in raw_nodes; skipping frame.")
                return
        elif not self._logged_semantic_mapping:
            summary = []
            for chain_name in ["Hand", "Thumb", "Index", "Middle", "Ring", "Pinky"]:
                ids = [node.node_id for node in msg.raw_nodes if node.chain_type == chain_name]
                if len(ids) > 0:
                    summary.append(f"{chain_name}:{sorted(ids)}")
            self.get_logger().info("Using semantic Manus node ordering: " + ", ".join(summary))
            self._logged_semantic_mapping = True

        if self._debug_nodes and self._frame_count % self._debug_every == 1:
            self.get_logger().info(f"Debug raw_nodes frame={self._frame_count} count={len(msg.raw_nodes)}")
            for line in _summarize_raw_nodes(msg.raw_nodes):
                self.get_logger().info(line)

        if not _is_valid_keypoints(keypoints):
            self._warn_throttled(
                "invalid_keypoints",
                "Received invalid Manus keypoints (non-finite or collapsed pose); likely glove disconnected. Skipping frame.",
            )
            return

        try:
            canonical = hand_to_canonical(keypoints).astype(np.float32)  # (21, 3)
        except ValueError as exc:
            self._warn_throttled("degenerate_canonical", f"{exc}; skipping frame.")
            return

        if not np.isfinite(canonical).all():
            self._warn_throttled("nonfinite_canonical", "Canonical Manus frame contains NaN/Inf; skipping frame.")
            return

        self._socket.send(canonical.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove_id', type=int, default=0)
    parser.add_argument('--port',     type=int, default=8765)
    parser.add_argument('--debug_nodes', action='store_true', help='Periodically log raw node semantic info.')
    parser.add_argument('--debug_every', type=int, default=200, help='Log every N glove frames when --debug_nodes is enabled.')
    args = parser.parse_args()

    rclpy.init()
    node = ManusRos2Bridge(
        glove_id=args.glove_id,
        zmq_port=args.port,
        debug_nodes=args.debug_nodes,
        debug_every=args.debug_every,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        node._socket.close(0)
        node._context.term()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
