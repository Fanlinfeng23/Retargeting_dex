"""
Simple demo: retarget synthetic hand poses to Allegro / Shadow / Leap robot hand.
No video, no webcam, no SAPIEN required.

MANO hand keypoint layout (21 joints):
  0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
"""

import numpy as np
from pathlib import Path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.constants import (
    RobotName, RetargetingType, HandType, get_default_config_path, ROBOT_NAME_MAP
)

# ── 1. Point to URDF assets ────────────────────────────────────────────────
robot_dir = Path(__file__).parent / "assets" / "robots" / "hands"
RetargetingConfig.set_default_urdf_dir(str(robot_dir))

# ── 2. Build a synthetic "open hand" pose (21 MANO keypoints, unit = metre) ─
def make_hand_pose(spread: float = 0.04) -> np.ndarray:
    """Returns (21, 3) array mimicking a slightly open right hand."""
    joints = np.zeros((21, 3))
    # Finger base positions along x-axis (knuckles)
    finger_offsets = [-spread * 2, -spread, 0, spread, spread * 2]
    for f, ox in enumerate(finger_offsets):
        base = f * 4 + 1  # joints 1,5,9,13,17
        for j in range(4):
            joints[base + j] = [ox, (j + 1) * 0.03, 0.0]
    return joints

# ── 3. Choose robots to demo ───────────────────────────────────────────────
DEMO_ROBOTS = [
    (RobotName.allegro, RetargetingType.vector,    HandType.right),
    (RobotName.leap,    RetargetingType.vector,    HandType.right),
    (RobotName.shadow,  RetargetingType.dexpilot,  HandType.right),
]

# ── 4. Run retargeting ─────────────────────────────────────────────────────
np.random.seed(42)
human_joints = make_hand_pose()  # shape (21, 3)

print("=" * 60)
print("  dex-retargeting  —  minimal demo")
print("  Input: synthetic open-hand pose (21 MANO keypoints)")
print("=" * 60)

for robot_name, retarget_type, hand_type in DEMO_ROBOTS:
    config_path = get_default_config_path(robot_name, retarget_type, hand_type)
    if config_path is None:
        print(f"\n[SKIP] No config for {robot_name.name} / {retarget_type.name}")
        continue

    retargeting = RetargetingConfig.load_from_file(config_path).build()
    optimizer   = retargeting.optimizer
    indices     = optimizer.target_link_human_indices  # shape (2, N) for vector

    # Build the reference value the optimizer expects
    if retarget_type == RetargetingType.position:
        ref = human_joints[indices, :]          # (N, 3)
    else:
        origin = human_joints[indices[0], :]   # wrist positions
        task   = human_joints[indices[1], :]   # fingertip positions
        ref    = task - origin                  # direction vectors

    robot_qpos = retargeting.retarget(ref)

    joint_names = retargeting.optimizer.robot.joint_names
    print(f"\n[{ROBOT_NAME_MAP[robot_name]}]  ({retarget_type.name})")
    print(f"  DoF: {len(robot_qpos)}")
    for name, val in zip(joint_names, robot_qpos):
        print(f"    {name:<35s}  {val:+.4f} rad")

# ── 5. Multi-frame demo with changing pose ─────────────────────────────────
print("\n" + "=" * 60)
print("  Multi-frame vector retargeting  (Allegro, 5 frames)")
print("=" * 60)

config_path = get_default_config_path(RobotName.allegro, RetargetingType.vector, HandType.right)
retargeting = RetargetingConfig.load_from_file(config_path).build()
optimizer   = retargeting.optimizer
indices     = optimizer.target_link_human_indices

for frame_i, spread in enumerate(np.linspace(0.01, 0.08, 5)):
    joints = make_hand_pose(spread=spread)
    origin = joints[indices[0], :]
    task   = joints[indices[1], :]
    ref    = task - origin
    qpos   = retargeting.retarget(ref)
    print(f"  frame {frame_i}  spread={spread:.3f}m  |  joint_sum={qpos.sum():.4f}  qpos[:4]={qpos[:4].round(4)}")

print("\nDemo complete.")
