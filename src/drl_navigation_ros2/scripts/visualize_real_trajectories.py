#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize evaluation trajectories generated during verification.

Data sources:
    - Trajectories:  src/drl_navigation_ros2/assets/trajectories_lightweight.pkl
    - Obstacles:     src/drl_navigation_ros2/assets/obstacle_map.json

Outputs:
    1. visualizations/real_trajectories_combined.png        (all trajectories)
    2. visualizations/real_trajectories/trajectory_XX.png   (individual plots)
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
VIS_DIR = PROJECT_ROOT / "visualizations"

# 与 Gazebo 世界保持一致的尺寸覆盖（obstacle1/2 在 SDF 中命名互换）
SIZE_OVERRIDES = {
    "obstacle1": {"shape": "box", "size": [0.3, 0.7, 1.0]},
    "obstacle2": {"shape": "box", "size": [1.0, 1.0, 1.0]},
}


def load_trajectories(pkl_path: Path):
    if not pkl_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        trajectories = pickle.load(f)

    if not isinstance(trajectories, list):
        raise ValueError("Trajectory file should contain a list object.")

    print(f"[OK] Loaded {len(trajectories)} trajectories")
    return trajectories


def load_obstacle_map(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(f"Obstacle map file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        obstacle_map = json.load(f)

    return obstacle_map


def draw_obstacles(ax, obstacle_map):
    colors = {
        "fixed": ("#d9534f", 0.8),
        "seeded": ("#f0ad4e", 0.8),
        "boundary": ("#607d8b", 0.4),
    }

    for obs in obstacle_map.get("obstacles", []):
        pos = obs["position"]
        obs_type = obs.get("type", "fixed")
        color, alpha = colors.get(obs_type, ("#cccccc", 0.5))

        override = SIZE_OVERRIDES.get(obs["name"])
        shape = override.get("shape") if override else obs.get("shape", "box")
        size = override.get("size") if override else obs.get("size", [1.0, 1.0, 1.0])

        if shape == "cylinder":
            radius = obs.get("radius", size[0] / 2.0)
            patch = Circle(
                (pos[0], pos[1]),
                radius=radius,
                facecolor=color,
                edgecolor="black",
                linewidth=1.5,
                alpha=alpha,
                zorder=1,
            )
        else:  # box
            patch = Rectangle(
                (pos[0] - size[0] / 2, pos[1] - size[1] / 2),
                width=size[0],
                height=size[1],
                facecolor=color,
                edgecolor="black",
                linewidth=1.5,
                alpha=alpha,
                zorder=1,
            )

        ax.add_patch(patch)

    safe_zone = obstacle_map["boundary"]["robot_safe_zone"]
    rect = Rectangle(
        (safe_zone["x_min"], safe_zone["y_min"]),
        safe_zone["x_max"] - safe_zone["x_min"],
        safe_zone["y_max"] - safe_zone["y_min"],
        linewidth=2,
        edgecolor="green",
        facecolor="none",
        linestyle="--",
        alpha=0.6,
        zorder=0,
    )
    ax.add_patch(rect)


def format_axes(ax, obstacle_map, title):
    boundary = obstacle_map["boundary"]
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(boundary["x_min"] - 0.2, boundary["x_max"] + 0.2)
    ax.set_ylim(boundary["y_min"] - 0.2, boundary["y_max"] + 0.2)


def plot_combined(trajectories, obstacle_map, output_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_obstacles(ax, obstacle_map)

    cmap = plt.get_cmap("tab10")

    for idx, traj in enumerate(trajectories):
        poses = np.array(traj["poses"])
        color = cmap(idx % 10)
        status = "Success" if traj["goal_reached"] else "Incomplete"
        label = f"Trajectory {idx+1:02d} ({status})"

        ax.plot(
            poses[:, 0],
            poses[:, 1],
            color=color,
            linewidth=2.5,
            label=label,
            zorder=5,
        )
        ax.scatter(
            poses[0, 0],
            poses[0, 1],
            color=color,
            edgecolor="white",
            s=70,
            marker="o",
            zorder=6,
        )
        ax.scatter(
            poses[-1, 0],
            poses[-1, 1],
            color=color,
            edgecolor="black",
            s=70,
            marker="X",
            zorder=6,
        )
        target = traj["target_pos"]
        ax.scatter(
            target[0],
            target[1],
            color=color,
            edgecolor="black",
            s=90,
            marker="*",
            zorder=7,
        )

    format_axes(ax, obstacle_map, "All 10 trajectories (combined view)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95, ncol=2)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")


def plot_individual(trajectories, obstacle_map, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")

    for idx, traj in enumerate(trajectories):
        poses = np.array(traj["poses"])
        color = cmap(idx % 10)

        fig, ax = plt.subplots(figsize=(8, 8))
        draw_obstacles(ax, obstacle_map)

        ax.plot(
            poses[:, 0],
            poses[:, 1],
            color=color,
            linewidth=3,
            label="Trajectory",
            zorder=5,
        )
        ax.scatter(
            poses[0, 0],
            poses[0, 1],
            color="green",
            edgecolor="white",
            s=100,
            marker="o",
            label="Start",
            zorder=6,
        )
        ax.scatter(
            poses[-1, 0],
            poses[-1, 1],
            color="red",
            edgecolor="black",
            s=100,
            marker="X",
            label="End",
            zorder=6,
        )
        target = traj["target_pos"]
        ax.scatter(
            target[0],
            target[1],
            color="gold",
            edgecolor="black",
            s=120,
            marker="*",
            label="Goal",
            zorder=7,
        )

        target_circle = Circle(target, radius=0.3, fill=False, linestyle="--", color="green")
        ax.add_patch(target_circle)

        status = "Success" if traj["goal_reached"] else "Incomplete"
        title = f"Trajectory {idx+1:02d} | Steps {traj['steps']} | {status}"
        format_axes(ax, obstacle_map, title)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

        fig.tight_layout()
        save_path = output_dir / f"trajectory_{idx+1:02d}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Saved individual plot: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize recorded evaluation trajectories")
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=ASSETS_DIR / "trajectories_lightweight.pkl",
        help="Trajectory data file (.pkl)",
    )
    parser.add_argument(
        "--obstacle-map",
        type=Path,
        default=ASSETS_DIR / "obstacle_map.json",
        help="Obstacle map (.json)",
    )
    parser.add_argument(
        "--mode",
        choices=["combined", "individual", "both"],
        default="both",
        help="Output mode: combined=single plot, individual=per-trajectory, both=all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VIS_DIR,
        help="Directory for generated figures",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    trajectories = load_trajectories(args.trajectories)
    obstacle_map = load_obstacle_map(args.obstacle_map)

    if args.mode in ("combined", "both"):
        combined_path = args.output_dir / "real_trajectories_combined.png"
        plot_combined(trajectories, obstacle_map, combined_path)

    if args.mode in ("individual", "both"):
        individuals_dir = args.output_dir / "real_trajectories"
        plot_individual(trajectories, obstacle_map, individuals_dir)

    print("\n[OK] Trajectory visualization finished.")
    print(f"[INFO] Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

