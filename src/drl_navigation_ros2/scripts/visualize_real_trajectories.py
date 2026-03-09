#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize evaluation trajectories scene-by-scene.

使用方法：直接修改下面的配置变量即可
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
from matplotlib.transforms import Affine2D


# ============================================================================
# 配置区域 - 直接修改这里的路径即可
# ============================================================================
def _detect_project_root() -> Path:
    """
    优先使用当前工作目录（若包含 assets/visualizations），
    否则回退到脚本所在位置的父目录
    """
    cwd = Path.cwd()
    if (cwd / "assets").exists() and (cwd / "visualizations").exists():
        return cwd
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _detect_project_root()
ASSETS_DIR = PROJECT_ROOT / "assets"
VIS_DIR = PROJECT_ROOT / "visualizations"

# 配置：轨迹文件路径
TRAJECTORY_FILE = ASSETS_DIR / "Traj_Jan08_td3_lightweight_cvar_cpo_wc0.1_used" / "trajectories_lightweight_8_polar_td3_cvarcpo_v1.pkl"

# 配置：场景文件夹（包含 obstacle_map_scenario_XX.json 文件）
SCENARIO_DIR = ASSETS_DIR / "eval_scenarios_8_polar"

# 配置：场景配置文件（包含所有场景的起点、终点等信息）
SCENARIO_CONFIG_FILE = ASSETS_DIR / "eval_scenarios_8_polar.json"

# 配置：输出文件夹名称
OUTPUT_DIR_NAME = "real_trajectories_8_polar_td3_cvarcpo_wc0.1_v1"

# ============================================================================
# 多文件比较模式配置
# ============================================================================
# 配置：包含多个轨迹文件的文件夹路径（用于 --compare 模式）
MULTI_TRAJ_DIR = ASSETS_DIR / "Traj_wcsac_lightweight"

# 配置：多文件比较模式的输出文件夹名称
MULTI_OUTPUT_DIR_NAME = "real_trajectories_Traj_wcsac_lightweight_compare"
    
# ============================================================================

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
            transform = (
                Affine2D()
                .rotate_deg(np.degrees(yaw))
                .translate(pos[0], pos[1])
                + ax.transData
            )
            patch = Rectangle(
                (-size[0] / 2, -size[1] / 2),
                width=size[0],
                height=size[1],
                facecolor=color,
                edgecolor="black",
                linewidth=1.5,
                alpha=alpha,
                zorder=1,
                transform=transform,
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
    filename = (
        f"trajectory_s{scenario_id:02d}.png"
        if scenario_id is not None
        else f"trajectory_{idx:02d}.png"
    )
    save_path = output_dir / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved scene plot: {save_path}")


def plot_multi_trajectories(scenario_idx, trajs_list, file_labels, obstacle_map, output_dir, scenario_id=None):
    """
    绘制多个轨迹文件中同一场景的所有轨迹到一张图上。

    Args:
        scenario_idx: 场景索引
        trajs_list: 来自不同文件的轨迹列表，每个元素是一条轨迹
        file_labels: 对应每条轨迹的文件标签（如 v1, v2, ...）
        obstacle_map: 障碍物地图
        output_dir: 输出目录
        scenario_id: 场景ID（用于文件命名）
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_obstacles(ax, obstacle_map)

    # 用于记录起点和终点（只绘制一次）
    first_traj = trajs_list[0]
    target = first_traj["target_pos"]

    # 绘制所有轨迹（黑色线条）
    for i, traj in enumerate(trajs_list):
        poses = np.array(traj["poses"])
        ax.plot(
            poses[:, 0],
            poses[:, 1],
            color="black",
            linewidth=1.5,
            alpha=0.7,
            label="Trajectories" if i == 0 else None,
            zorder=5,
        )

    # 绘制起点（使用第一条轨迹的起点，假设所有轨迹起点相同）
    first_poses = np.array(first_traj["poses"])
    ax.scatter(
        first_poses[0, 0],
        first_poses[0, 1],
        color="green",
        edgecolor="white",
        s=100,
        marker="o",
        label="Start",
        zorder=6,
    )

    # 绘制目标点
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

    # 统计成功数
    success_count = sum(1 for traj in trajs_list if traj["goal_reached"])
    total_count = len(trajs_list)

    sid_text = f"S{scenario_id:02d}" if scenario_id is not None else f"Idx {scenario_idx+1:02d}"
    title = f"Scenario {sid_text} | {total_count} Trajectories | Success: {success_count}/{total_count}"
    format_axes(ax, obstacle_map, title)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    filename = (
        f"multi_trajectory_s{scenario_id:02d}.png"
        if scenario_id is not None
        else f"multi_trajectory_{scenario_idx:02d}.png"
    )
    save_path = output_dir / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved multi-trajectory plot: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize recorded evaluation trajectories (per scenario)")
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=TRAJECTORY_FILE,
        help="Trajectory data file (.pkl)",
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=SCENARIO_DIR,
        help="Scenario directory containing obstacle_map_scenario_XX.json files",
    )
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=SCENARIO_CONFIG_FILE,
        help="Scenario config file (.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VIS_DIR / OUTPUT_DIR_NAME,
        help="Directory for generated figures",
    )
    # 多文件比较模式参数
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Enable multi-file comparison mode: plot trajectories from multiple files (v1-v9) on the same scene",
    )
    parser.add_argument(
        "--multi-traj-dir",
        type=Path,
        default=MULTI_TRAJ_DIR,
        help="Directory containing multiple trajectory files (for --compare mode)",
    )
    parser.add_argument(
        "--multi-output-dir",
        type=Path,
        default=VIS_DIR / MULTI_OUTPUT_DIR_NAME,
        help="Output directory for multi-file comparison figures",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pkl",
        help="Glob pattern to match trajectory files in multi-traj-dir (default: *.pkl)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载场景配置
    scenarios_data = None
    if args.scenario_config.exists():
        with open(args.scenario_config, "r", encoding="utf-8") as f:
            scenarios_data = json.load(f)
    scenarios = scenarios_data.get("scenarios", []) if scenarios_data else []

    # ========================================================================
    # 多文件比较模式：将多个文件中同一场景的轨迹绘制到一张图
    # ========================================================================
    if args.compare:
        print("[INFO] Running in multi-file comparison mode...")

        # 查找所有轨迹文件
        traj_files = sorted(args.multi_traj_dir.glob(args.pattern))
        if not traj_files:
            print(f"[ERROR] No trajectory files found in {args.multi_traj_dir} with pattern '{args.pattern}'")
            return

        print(f"[OK] Found {len(traj_files)} trajectory files:")
        for f in traj_files:
            print(f"     - {f.name}")

        # 加载所有轨迹文件
        all_trajectories = []
        file_labels = []
        for traj_file in traj_files:
            trajs = load_trajectories(traj_file)
            all_trajectories.append(trajs)
            # 从文件名提取标签（如 v1, v2, ...）
            label = traj_file.stem  # 去掉扩展名的文件名
            file_labels.append(label)

        # 确定场景数量（使用第一个文件的轨迹数量）
        num_scenarios = len(all_trajectories[0])
        print(f"[INFO] Number of scenarios: {num_scenarios}")

        # 输出目录
        output_dir = args.multi_output_dir

        # 遍历每个场景
        for scenario_idx in range(num_scenarios):
            scenario = scenarios[scenario_idx] if scenario_idx < len(scenarios) else None
            scenario_id = scenario.get("scenario_id") if scenario else scenario_idx

            # 查找对应的障碍物地图文件
            obstacle_map_file = args.scenario_dir / f"obstacle_map_scenario_{scenario_id:02d}.json"

            if not obstacle_map_file.exists():
                print(f"[WARNING] Obstacle map not found: {obstacle_map_file}, skipping...")
                continue

            # 收集所有文件中该场景的轨迹
            trajs_for_scenario = []
            labels_for_scenario = []
            for file_idx, trajs in enumerate(all_trajectories):
                if scenario_idx < len(trajs):
                    trajs_for_scenario.append(trajs[scenario_idx])
                    labels_for_scenario.append(file_labels[file_idx])

            if not trajs_for_scenario:
                print(f"[WARNING] No trajectories found for scenario {scenario_id}, skipping...")
                continue

            # 加载障碍物地图并绘制多轨迹图
            obstacle_map = load_obstacle_map(obstacle_map_file)
            plot_multi_trajectories(
                scenario_idx,
                trajs_for_scenario,
                labels_for_scenario,
                obstacle_map,
                output_dir,
                scenario_id=scenario_id,
            )

        print("\n[OK] Multi-file comparison visualization finished.")
        print(f"[INFO] Output directory: {output_dir.resolve()}")
        return

    # ========================================================================
    # 单文件模式（原有功能）
    # ========================================================================
    # 加载轨迹
    trajectories = load_trajectories(args.trajectories)

    # 输出目录
    output_dir = args.output_dir

    # 遍历每条轨迹
    for idx, traj in enumerate(trajectories):
        scenario = scenarios[idx] if idx < len(scenarios) else None
        scenario_id = scenario.get("scenario_id") if scenario else idx

        # 查找对应的障碍物地图文件
        obstacle_map_file = args.scenario_dir / f"obstacle_map_scenario_{scenario_id:02d}.json"

        if not obstacle_map_file.exists():
            print(f"[WARNING] Obstacle map not found: {obstacle_map_file}, skipping...")
            continue

        # 加载障碍物地图并绘制
        obstacle_map = load_obstacle_map(obstacle_map_file)
        plot_single_trajectory(idx, traj, obstacle_map, output_dir, scenario_id=scenario_id)

    print("\n[OK] Trajectory visualization finished.")
    print(f"[INFO] Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

