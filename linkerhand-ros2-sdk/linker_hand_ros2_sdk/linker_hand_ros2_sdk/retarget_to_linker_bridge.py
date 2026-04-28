#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from linker_hand_ros2_sdk.LinkerHand.utils.mapping import arc_to_range_right


class RetargetToLinkerBridge(Node):
    def __init__(self) -> None:
        super().__init__("retarget_to_linker_bridge")

        self.declare_parameter("input_topic", "/l20_dex_retarget/joint_states")
        self.declare_parameter("output_topic", "/cb_right_hand_control_cmd")
        self.declare_parameter("hand_joint", "L20")

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.hand_joint = str(self.get_parameter("hand_joint").value).upper()

        self.expected_joint_names: List[str] = [
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
        self.required_joint_names = self.expected_joint_names[:20]
        self.name_to_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.expected_joint_names)
        }

        self.publisher = self.create_publisher(JointState, self.output_topic, 10)
        self.subscription = self.create_subscription(
            JointState, self.input_topic, self._callback, 10
        )

        self.last_warn_ns = 0
        self.warn_interval_ns = int(2e9)

        self.get_logger().info(
            f"bridge started: {self.input_topic} (arc) -> {self.output_topic} (0~255)"
        )

    def _throttled_warn(self, text: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_warn_ns >= self.warn_interval_ns:
            self.get_logger().warn(text)
            self.last_warn_ns = now_ns

    def _callback(self, msg: JointState) -> None:
        if not msg.position:
            self._throttled_warn("received empty JointState.position; skip")
            return

        if msg.name:
            if len(msg.name) != len(msg.position):
                self._throttled_warn("name/position length mismatch; skip")
                return
            missing = [n for n in self.required_joint_names if n not in msg.name]
            if missing:
                self._throttled_warn(f"missing required joints: {missing}; skip")
                return
            input_map = {name: i for i, name in enumerate(msg.name)}
            ordered_arc = [float(msg.position[input_map[name]]) for name in self.required_joint_names]
        else:
            if len(msg.position) < len(self.required_joint_names):
                self._throttled_warn(
                    f"position length {len(msg.position)} < {len(self.required_joint_names)}; skip"
                )
                return
            ordered_arc = [float(x) for x in msg.position[: len(self.required_joint_names)]]

        try:
            range_vals = arc_to_range_right(ordered_arc, self.hand_joint)
        except Exception as exc:
            self._throttled_warn(f"arc_to_range_right failed: {exc}")
            return

        range_vals = [max(0.0, min(255.0, float(v))) for v in range_vals]

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.position = range_vals
        out.velocity = [0.0] * len(range_vals)
        out.effort = [0.0] * len(range_vals)
        self.publisher.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RetargetToLinkerBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
