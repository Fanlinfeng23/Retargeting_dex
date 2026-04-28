#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    can_arg = DeclareLaunchArgument("can", default_value="can0")
    glove_id_arg = DeclareLaunchArgument("glove_id", default_value="0")
    retarget_script_arg = DeclareLaunchArgument(
        "retarget_script",
        default_value="/home/user/workspace/linker_hand_ros2_sdk/src/Retargeting_dex/GeoRT/dex_retargeting/manus_l20_dex_retarget.py",
    )
    retarget_config_arg = DeclareLaunchArgument(
        "retarget_config",
        default_value="/home/user/workspace/linker_hand_ros2_sdk/src/Retargeting_dex/GeoRT/dex_retargeting/linkerhand_l20_right_vector.yml",
    )
    retarget_assets_arg = DeclareLaunchArgument(
        "retarget_assets",
        default_value="/home/user/workspace/linker_hand_ros2_sdk/src/Retargeting_dex/GeoRT/assets",
    )

    linker_sdk = Node(
        package="linker_hand_ros2_sdk",
        executable="linker_hand_sdk",
        name="linker_hand_sdk",
        output="screen",
        parameters=[
            {
                "hand_type": "right",
                "hand_joint": "L20",
                "is_touch": True,
                "can": LaunchConfiguration("can"),
                "modbus": "None",
            }
        ],
    )

    bridge = Node(
        package="linker_hand_ros2_sdk",
        executable="retarget_to_linker_bridge",
        name="retarget_to_linker_bridge",
        output="screen",
        parameters=[
            {
                "input_topic": "/l20_dex_retarget/joint_states",
                "output_topic": "/cb_right_hand_control_cmd",
                "hand_joint": "L20",
            }
        ],
    )

    retarget = ExecuteProcess(
        cmd=[
            "/usr/bin/python3",
            LaunchConfiguration("retarget_script"),
            "--input",
            "ros",
            "--glove-id",
            LaunchConfiguration("glove_id"),
            "--config",
            LaunchConfiguration("retarget_config"),
            "--asset-dir",
            LaunchConfiguration("retarget_assets"),
            "--joint-order",
            "geort",
            "--publish-topic",
            "/l20_dex_retarget/joint_states",
            "--print-every",
            "120",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            can_arg,
            glove_id_arg,
            retarget_script_arg,
            retarget_config_arg,
            retarget_assets_arg,
            linker_sdk,
            bridge,
            retarget,
        ]
    )
