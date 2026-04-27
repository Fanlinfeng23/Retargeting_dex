# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json 
from geort.utils.path import get_package_root
from pathlib import Path 
import os 
import numpy as np 


def save_json(data, filename):
    """
    Save a Python dictionary to a JSON file.
    
    Parameters:
    - data (dict): The data to be saved.
    - filename (str): Path to the file where data will be saved.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_json(filename):
    """
    Load data from a JSON file into a Python dictionary.
    
    Parameters:
    - filename (str): Path to the JSON file to be loaded.
    
    Returns:
    - dict: The loaded data.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_config(config_name):
    config_root = Path(get_package_root())  / "geort" / "config"
    all_configs = os.listdir(config_root)
    
    for config in all_configs:
        if config_name in config:
            return load_json(config_root / config)

    config_root_str = config_root.as_posix()
    assert False, f"Configuration {config_name}.json is not found in {config_root_str}"

def parse_config_keypoint_info(config):
    keypoint_links = []
    keypoint_offsets = []
    keypoint_human_ids = []
    keypoint_group_joints = []
    keypoint_group_sizes = []
    keypoint_group_names = []
    tip_indices = []

    joint_order = config["joint_order"]
    group_entries = config.get("retarget_groups")

    if group_entries is None:
        group_entries = []
        for info in config["fingertip_link"]:
            group_entries.append(
                {
                    "name": info["name"],
                    "links": [info["link"]],
                    "center_offsets": [info["center_offset"]],
                    "human_hand_ids": [info["human_hand_id"]],
                    "joint": info["joint"],
                }
            )

    for group_info in group_entries:
        group_links = group_info["links"]
        group_offsets = group_info.get("center_offsets", [[0.0, 0.0, 0.0] for _ in group_links])
        group_human_ids = group_info["human_hand_ids"]

        if not (len(group_links) == len(group_offsets) == len(group_human_ids)):
            raise ValueError(f"Invalid retarget group {group_info.get('name', 'unknown')}: inconsistent keypoint lengths.")

        group_joint = [joint_order.index(joint) for joint in group_info["joint"]]
        keypoint_group_joints.append(group_joint)
        keypoint_group_sizes.append(len(group_links))
        keypoint_group_names.append(group_info.get("name", f"group_{len(keypoint_group_names)}"))

        start_idx = len(keypoint_links)
        keypoint_links.extend(group_links)
        keypoint_offsets.extend(group_offsets)
        keypoint_human_ids.extend(group_human_ids)
        tip_indices.append(start_idx + len(group_links) - 1)

    out = {
        "link": keypoint_links,
        "offset": keypoint_offsets,
        "human_id": keypoint_human_ids,
        "group_joint": keypoint_group_joints,
        "group_size": keypoint_group_sizes,
        "group_name": keypoint_group_names,
        "tip_indices": tip_indices,
        # Backward-compatible alias for older fingertip-only code paths.
        "joint": keypoint_group_joints,
    }
    return out 


def parse_config_joint_limit(config):
    lower_limit = config["joint"]["lower"]
    upper_limit = config["joint"]["upper"]
    return np.array(lower_limit), np.array(upper_limit)
