#!/bin/bash
# Real-time retargeting visualization for LinkerHand L20.
# Requires: manus_data_publisher running + manus_ros2_bridge.py running (ZMQ port 8765).
#
# Usage:
#   bash evaluate_l20.sh                         # live mode (ZMQ)
#   bash evaluate_l20.sh --replay manus_data     # offline replay from recorded data
set -e

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Force NVIDIA Vulkan ICD — Intel Arc (Mesa) does not support 0x7d67 and will segfault.
NVIDIA_ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"
if [ -f "$NVIDIA_ICD" ]; then
    export VK_ICD_FILENAMES="$NVIDIA_ICD"
fi

TAG="${TAG:-manus_l20_fullhand}"
HAND="${HAND:-linkerhand_l20_right}"

if [ "$1" = "--replay" ]; then
    DATA="${2:-manus_data}"
    python geort/mocap/replay_evaluation.py \
        -hand "$HAND" \
        -ckpt_tag "$TAG" \
        -data "$DATA"
else
    # Do NOT source ROS2 setup.bash here — it injects library paths that conflict
    # with SAPIEN's renderer and cause segfaults.
    # The bridge node (manus_ros2_bridge.py) handles ROS2; evaluation uses ZMQ only.
    python geort/mocap/manus_evaluation.py \
        -hand "$HAND" \
        -ckpt_tag "$TAG"
fi
