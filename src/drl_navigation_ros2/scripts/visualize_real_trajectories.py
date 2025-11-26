#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize evaluation trajectories scene-by-scene.

Data sources:
    - Trajectories:  src/drl_navigation_ros2/assets/trajectories_lightweight.pkl
    - Obstacles:     src/drl_navigation_ros2/assets/obstacle_map_scenario_XX.json
                     (fallback: obstacle_map.json)
    - Scenarios:     src/drl_navigation_ros2/assets/eval_scenarios.json

Outputs:
    - visualizations/real_trajectories/trajectory_sXX.png   (per scene)
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


def load_eval_scenarios(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_obstacles(ax, obstacle_map):
    colors = {
        "fixed": ("#d9534f", 0.8),
        "scenario_obstacle": ("#f0ad4e", 0.8),
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
            yaw = obs.get("yaw", 0.0)
            patch = Rectangle(
                (pos[0] - size[0] / 2, pos[1] - size[1] / 2),
                width=size[0],
                height=size[1],
                facecolor=color,
                edgecolor="black",
                linewidth=1.5,
                alpha=alpha,
                angle=np.degrees(yaw),
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


def plot_single_trajectory(idx, traj, obstacle_map, output_dir, scenario_id=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    poses = np.array(traj["poses"])
    color = plt.get_cmap("tab10")(idx % 10)

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
    sid_text = f"S{scenario_id:02d}" if scenario_id is not None else f"Idx {idx+1:02d}"
    title = f"Trajectory {sid_text} | Steps {traj['steps']} | {status}"
    format_axes(ax, obstacle_map, title)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    filename = f"trajectory_s{scenario_id:02d}.png" if scenario_id is not None else f"trajectory_{idx+1:02d}.png"
    save_path = output_dir / "real_trajectories" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved scene plot: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize recorded evaluation trajectories (per scenario)")
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
        "--eval-scenarios",
        type=Path,
        default=ASSETS_DIR / "eval_scenarios.json",
        help="Eval scenarios file (.json)",
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
    scenarios_data = load_eval_scenarios(args.eval_scenarios)
    scenarios = scenarios_data.get("scenarios", []) if scenarios_data else []

    for idx, traj in enumerate(trajectories):
        scenario = scenarios[idx] if idx < len(scenarios) else None
        scenario_id = scenario.get("scenario_id") if scenario else None
        map_path = args.obstacle_map
        if scenario_id is not None:
            candidate = ASSETS_DIR / f"obstacle_map_scenario_{scenario_id:02d}.json"
            if candidate.exists():
                map_path = candidate
        obstacle_map = load_obstacle_map(map_path)
        plot_single_trajectory(idx, traj, obstacle_map, args.output_dir, scenario_id=scenario_id)

    print("\n[OK] Trajectory visualization finished.")
    print(f"[INFO] Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

