#!/usr/bin/env /usr/bin/python3
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_geort_visual_g20")
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from manus_g20_dex_retarget import (
    DEFAULT_ASSET_DIR,
    DEFAULT_CONFIG_PATH,
    G20_COMMAND_NAMES,
    TIP_HUMAN_INDICES,
    apply_frame_alignment,
    arc_to_raw,
    build_retargeting,
    compute_retargeting_reference,
    compute_tip_scale_g20,
    prepare_npy_frame,
    qpos_to_arc,
)


FINGER_COLORS = {
    "thumb": "#ff6b6b",
    "index": "#ffd166",
    "middle": "#4ecdc4",
    "ring": "#5dade2",
    "little": "#c77dff",
}
BG = "#081019"
PANEL = "#0e1726"
FG = "#ecf3ff"
GRID = "#2a3550"

HUMAN_FINGER_CHAINS = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "little": [0, 17, 18, 19, 20],
}

ROBOT_FINGER_LINKS = {
    "thumb": [
        "wrist_base_link",
        "thumb_metacarpals_base1",
        "thumb_metacarpals_base2",
        "thumb_metacarpals",
        "thumb_proximal",
        "thumb_distal",
    ],
    "index": [
        "wrist_base_link",
        "index_metacarpals",
        "index_proximal",
        "index_middle",
        "index_distal",
    ],
    "middle": [
        "wrist_base_link",
        "middle_metacarpals",
        "middle_proximal",
        "middle_middle",
        "middle_distal",
    ],
    "ring": [
        "wrist_base_link",
        "ring_metacarpals",
        "ring_proximal",
        "ring_middle",
        "ring_distal",
    ],
    "little": [
        "wrist_base_link",
        "pinky_metacarpals",
        "pinky_proximal",
        "pinky_middle",
        "pinky_distal",
    ],
}


def _set_projection_axis_style(
    ax, title: str, xlim: float, ylim: Tuple[float, float], xlabel: str, ylabel: str
):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=FG, fontsize=12.5, pad=10, weight="bold")
    ax.tick_params(colors="#93a4c3", labelsize=8)
    ax.grid(color=GRID, alpha=0.7, linewidth=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, color="#93a4c3", fontsize=9)
    ax.set_ylabel(ylabel, color="#93a4c3", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _compute_selected_indices(
    num_frames: int, max_frames: int, frame_step: int, target_frames: int
) -> np.ndarray:
    if max_frames > 0:
        num_frames = min(num_frames, max_frames)
    if frame_step <= 0:
        frame_step = max(1, num_frames // max(1, target_frames))
    return np.arange(0, num_frames, frame_step, dtype=int)


def _robot_chain_positions(retargeting, qpos: np.ndarray) -> dict:
    robot = retargeting.optimizer.robot
    robot.compute_forward_kinematics(qpos)
    cache = {}
    for finger_name, link_names in ROBOT_FINGER_LINKS.items():
        positions = []
        for link_name in link_names:
            link_index = robot.get_link_index(link_name)
            positions.append(robot.get_link_pose(link_index)[:3, 3].copy())
        cache[finger_name] = np.asarray(positions, dtype=np.float32)
    return cache


def _retarget_frames(
    input_data: np.ndarray,
    config_path: Path,
    asset_dir: Path,
    scaling: float | None,
    auto_scale: bool,
    reserved_arc_value: float,
    reserved_raw_value: float,
):
    retargeting = build_retargeting(config_path, asset_dir, scaling)
    if auto_scale:
        for frame in input_data:
            scale = compute_tip_scale_g20(retargeting, frame)
            if scale is not None:
                retargeting.optimizer.scaling = scale
                break

    human_indices = retargeting.optimizer.target_link_human_indices
    qpos_frames: List[np.ndarray] = []
    g20_arc_frames: List[np.ndarray] = []
    g20_raw_frames: List[np.ndarray] = []
    robot_frames: List[dict] = []
    for frame in input_data:
        ref_value = compute_retargeting_reference(retargeting, frame)
        qpos = retargeting.retarget(ref_value).astype(np.float32)
        g20_arc = qpos_to_arc(qpos, retargeting.joint_names, reserved_arc_value)
        g20_raw = arc_to_raw(g20_arc, reserved_raw_value)
        qpos_frames.append(qpos)
        g20_arc_frames.append(g20_arc)
        g20_raw_frames.append(g20_raw)
        robot_frames.append(_robot_chain_positions(retargeting, qpos))

    return (
        np.asarray(qpos_frames, dtype=np.float32),
        np.asarray(g20_arc_frames, dtype=np.float32),
        np.asarray(g20_raw_frames, dtype=np.float32),
        robot_frames,
        retargeting,
    )


def _human_palm_points(frame: np.ndarray) -> np.ndarray:
    return frame[np.array([0, 5, 9, 13, 17, 0])]


def _robot_palm_points(chain_positions: dict) -> np.ndarray:
    return np.asarray(
        [
            chain_positions["thumb"][3],
            chain_positions["index"][1],
            chain_positions["middle"][1],
            chain_positions["ring"][1],
            chain_positions["little"][1],
            chain_positions["thumb"][3],
        ],
        dtype=np.float32,
    )


def _project(points: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "xy":
        return points[:, 0], points[:, 1]
    if mode == "yz":
        return points[:, 1], points[:, 2]
    raise ValueError(mode)


def _draw_projection(ax, chains: dict, trails: dict, mode: str, palm_points: np.ndarray):
    px, py = _project(palm_points, mode)
    ax.plot(px, py, color="#7f8fa6", linewidth=2.0, alpha=0.9)

    for finger_name, pts in chains.items():
        color = FINGER_COLORS[finger_name]
        x, y = _project(pts, mode)
        ax.plot(x, y, color=color, linewidth=4.0, alpha=0.98)
        ax.scatter(x, y, s=22, color="#e6eefc", alpha=0.9)
        ax.scatter(x[-1], y[-1], s=80, color=color, edgecolors="white", linewidths=0.8, zorder=5)
        trail = trails[finger_name]
        if len(trail) > 1:
            tx, ty = _project(trail, mode)
            ax.plot(tx, ty, color=color, linewidth=2.5, alpha=0.35)


def _prepare_trails_human(human_frames: np.ndarray, frame_idx: int, history: int) -> dict:
    start = max(0, frame_idx - history + 1)
    frames = human_frames[start : frame_idx + 1]
    out = {}
    for finger_name, tip_idx in zip(FINGER_COLORS.keys(), TIP_HUMAN_INDICES):
        out[finger_name] = frames[:, tip_idx, :]
    return out


def _prepare_trails_robot(robot_frames: List[dict], frame_idx: int, history: int) -> dict:
    start = max(0, frame_idx - history + 1)
    out = {}
    for finger_name in FINGER_COLORS.keys():
        trail = []
        for frame in robot_frames[start : frame_idx + 1]:
            trail.append(frame[finger_name][-1])
        out[finger_name] = np.asarray(trail, dtype=np.float32)
    return out


def _make_cover(fig: plt.Figure, output_path: Path):
    cover_path = output_path.with_suffix(".png")
    fig.savefig(cover_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return cover_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize Manus keypoints and retargeted G20 motion as an animated GIF."
    )
    parser.add_argument(
        "--input-path",
        default="/home/user/ros2_ws/manus-l20-retargeting-workspace/GeoRT/data/manus_data.npy",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument(
        "--output-path",
        default="/home/user/ros2_ws/manus-l20-retargeting-workspace/GeoRT/analysis/manus_g20_visualization.gif",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit the number of source frames before subsampling.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=0,
        help="Use every Nth frame. 0 means auto-select around target-frames.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=180,
        help="Used only when --frame-step=0.",
    )
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--history", type=int, default=12, help="Trail length in rendered frames.")
    parser.add_argument("--scaling", type=float, default=None)
    parser.add_argument("--auto-scale", action="store_true")
    parser.add_argument(
        "--npy-frame",
        choices=["dex", "geort", "world"],
        default="dex",
        help=(
            "Coordinate frame stored in --input-path: dex=new bridge/dex-retargeting "
            "canonical, geort=old GeoRT wrist-local canonical, world=raw world points."
        ),
    )
    parser.add_argument(
        "--frame-alignment",
        choices=["g20", "none"],
        default="g20",
        help="g20 aligns official dex/MANO right-hand points to the LinkerHand G20 URDF frame.",
    )
    parser.add_argument("--reserved-raw-value", type=float, default=255.0)
    parser.add_argument("--reserved-arc-value", type=float, default=0.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    human_all = np.load(input_path).astype(np.float32)
    if human_all.ndim != 3 or human_all.shape[1:] != (21, 3):
        raise ValueError(f"Expected input with shape (N, 21, 3), got {human_all.shape}")
    if args.npy_frame != "dex":
        human_all = np.asarray(
            [prepare_npy_frame(frame, args.npy_frame) for frame in human_all],
            dtype=np.float32,
        )
    human_all = np.asarray(
        [apply_frame_alignment(frame, args.frame_alignment) for frame in human_all],
        dtype=np.float32,
    )

    selected_indices = _compute_selected_indices(
        len(human_all), args.max_frames, args.frame_step, args.target_frames
    )
    human_frames = human_all[selected_indices]

    qpos_frames, g20_arc_frames, g20_raw_frames, robot_frames, retargeting = _retarget_frames(
        human_frames,
        Path(args.config).resolve(),
        Path(args.asset_dir).resolve(),
        args.scaling,
        args.auto_scale,
        args.reserved_arc_value,
        args.reserved_raw_value,
    )

    human_stack = human_frames.reshape(-1, 3)
    robot_stack = np.concatenate(
        [pts for frame in robot_frames for pts in frame.values()],
        axis=0,
    )
    human_xy_lim = max(0.10, float(np.max(np.abs(human_stack[:, :2])))) * 1.15
    robot_xy_lim = max(0.10, float(np.max(np.abs(robot_stack[:, :2])))) * 1.15
    human_yz_lim = (
        min(float(np.min(human_stack[:, 1])) * 1.15, -0.10),
        max(float(np.max(human_stack[:, 2])) * 1.15, 0.14),
    )
    robot_yz_lim = (
        min(float(np.min(robot_stack[:, 1])) * 1.15, -0.08),
        max(float(np.max(robot_stack[:, 2])) * 1.15, 0.18),
    )

    arc_norm = Normalize(
        vmin=float(np.min(g20_arc_frames)),
        vmax=float(np.max(g20_arc_frames)),
    )
    raw_norm = Normalize(vmin=0.0, vmax=255.0)

    fig = plt.figure(figsize=(16.0, 11.0), facecolor=BG)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.18, 1.18, 1.22], hspace=0.24, wspace=0.10)
    human_top_ax = fig.add_subplot(gs[0, 0])
    robot_top_ax = fig.add_subplot(gs[0, 1])
    human_side_ax = fig.add_subplot(gs[1, 0])
    robot_side_ax = fig.add_subplot(gs[1, 1])
    arc_ax = fig.add_subplot(gs[2, 0])
    raw_ax = fig.add_subplot(gs[2, 1])
    arc_ax.set_facecolor(PANEL)
    raw_ax.set_facecolor(PANEL)

    _set_projection_axis_style(
        human_top_ax, "Human Hand  •  XY Top View", human_xy_lim, (-human_xy_lim, human_xy_lim), "X", "Y"
    )
    _set_projection_axis_style(
        robot_top_ax, "G20 Robot  •  XY Top View", robot_xy_lim, (-robot_xy_lim, robot_xy_lim), "X", "Y"
    )
    _set_projection_axis_style(
        human_side_ax, "Human Hand  •  YZ Side View", human_xy_lim, human_yz_lim, "Y", "Z"
    )
    _set_projection_axis_style(
        robot_side_ax, "G20 Robot  •  YZ Side View", robot_xy_lim, robot_yz_lim, "Y", "Z"
    )

    arc_im = arc_ax.imshow(
        g20_arc_frames.T,
        aspect="auto",
        cmap="viridis",
        norm=arc_norm,
        interpolation="nearest",
    )
    raw_im = raw_ax.imshow(
        g20_raw_frames.T,
        aspect="auto",
        cmap="magma",
        norm=raw_norm,
        interpolation="nearest",
    )
    arc_cursor = arc_ax.axvline(0, color="#f8fafc", linewidth=2.0, alpha=0.95)
    raw_cursor = raw_ax.axvline(0, color="#f8fafc", linewidth=2.0, alpha=0.95)

    for ax, title in [
        (arc_ax, "G20 Arc Timeline"),
        (raw_ax, "G20 Raw Timeline"),
    ]:
        ax.set_title(title, color=FG, fontsize=13, pad=10, weight="bold")
        ax.set_xlabel("Rendered Frame", color="#93a4c3", fontsize=10)
        ax.set_ylabel("Command", color="#93a4c3", fontsize=10)
        ax.set_yticks(np.arange(len(G20_COMMAND_NAMES)))
        ax.set_yticklabels(G20_COMMAND_NAMES, fontsize=7, color="#c9d6ea")
        ax.tick_params(axis="x", colors="#93a4c3")
        ax.tick_params(axis="y", colors="#93a4c3")
        for spine in ax.spines.values():
            spine.set_color(GRID)

    arc_cbar = fig.colorbar(arc_im, ax=arc_ax, fraction=0.025, pad=0.02)
    raw_cbar = fig.colorbar(raw_im, ax=raw_ax, fraction=0.025, pad=0.02)
    for cbar, label in [
        (arc_cbar, "Arc Value (rad)"),
        (raw_cbar, "Raw Value (0-255)"),
    ]:
        cbar.ax.yaxis.set_tick_params(color="#93a4c3")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#93a4c3")
        cbar.set_label(label, color="#93a4c3")

    status_text = fig.text(
        0.5,
        0.965,
        "",
        color=FG,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.935,
        f"{input_path.name}  |  source frames={len(human_all)}  rendered frames={len(human_frames)}"
        f"  fps={args.fps}  scale={retargeting.optimizer.scaling:.3f}  alignment={args.frame_alignment}",
        color="#9fb3d1",
        ha="center",
        va="top",
        fontsize=10.5,
    )

    def update(render_idx: int):
        for ax, title, xlim, ylim, xlabel, ylabel in [
            (human_top_ax, "Human Hand  •  XY Top View", human_xy_lim, (-human_xy_lim, human_xy_lim), "X", "Y"),
            (robot_top_ax, "G20 Robot  •  XY Top View", robot_xy_lim, (-robot_xy_lim, robot_xy_lim), "X", "Y"),
            (human_side_ax, "Human Hand  •  YZ Side View", human_xy_lim, human_yz_lim, "Y", "Z"),
            (robot_side_ax, "G20 Robot  •  YZ Side View", robot_xy_lim, robot_yz_lim, "Y", "Z"),
        ]:
            ax.cla()
            _set_projection_axis_style(ax, title, xlim, ylim, xlabel, ylabel)

        human_frame = human_frames[render_idx]
        robot_frame = robot_frames[render_idx]
        human_trails = _prepare_trails_human(human_frames, render_idx, args.history)
        robot_trails = _prepare_trails_robot(robot_frames, render_idx, args.history)

        human_chain_points = {
            finger_name: human_frame[np.asarray(chain, dtype=int)]
            for finger_name, chain in HUMAN_FINGER_CHAINS.items()
        }
        _draw_projection(human_top_ax, human_chain_points, human_trails, "xy", _human_palm_points(human_frame))
        _draw_projection(human_side_ax, human_chain_points, human_trails, "yz", _human_palm_points(human_frame))
        _draw_projection(robot_top_ax, robot_frame, robot_trails, "xy", _robot_palm_points(robot_frame))
        _draw_projection(robot_side_ax, robot_frame, robot_trails, "yz", _robot_palm_points(robot_frame))
        arc_cursor.set_xdata([render_idx, render_idx])
        raw_cursor.set_xdata([render_idx, render_idx])

        tip_span = np.linalg.norm(human_frame[TIP_HUMAN_INDICES] - human_frame[0], axis=1).mean()
        raw_frame = g20_raw_frames[render_idx]
        saturated = [
            name
            for name, value in zip(G20_COMMAND_NAMES, raw_frame)
            if value <= 1.0 or value >= 254.0
        ]
        sat_text = ", ".join(saturated[:4]) if saturated else "none"
        status_text.set_text(
            f"Frame {render_idx + 1:03d}/{len(human_frames)}  |  mean human fingertip radius = {tip_span:.3f} m"
            f"  |  saturated raw channels: {sat_text}"
        )
        return [arc_cursor, raw_cursor, status_text]

    update(0)
    cover_path = _make_cover(fig, output_path)
    anim = FuncAnimation(fig, update, frames=len(human_frames), interval=1000 / args.fps, blit=False)
    writer = PillowWriter(fps=args.fps)
    anim.save(output_path, writer=writer, dpi=110)
    plt.close(fig)

    print(f"saved animation: {output_path}")
    print(f"saved cover: {cover_path}")
    print(f"rendered frames: {len(human_frames)} / source frames: {len(human_all)}")
    print(f"retarget joint order: {list(retargeting.joint_names)}")
    print(f"g20 command order: {list(G20_COMMAND_NAMES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
