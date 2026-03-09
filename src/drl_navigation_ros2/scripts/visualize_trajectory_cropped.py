#!/usr/bin/env python3
"""
在裁剪地图上叠加多轨迹可视化（compare 模式），论文小图风格。

使用方法：只需修改下面两行配置即可切换不同的轨迹文件夹和场景。
"""

import json
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.transforms import Affine2D

# ============================================================================
# >>> 只需修改这三行即可切换轨迹、场景和输出文件名 <<<
# ============================================================================
TRAJ_DIR = "Traj_lightweight_8_freeze_010"  # 轨迹文件夹名（在 assets/ 下）
# TRAJ_DIR = "Traj_Jan08_td3_lightweight_cvar_cpo_wc0.1_used"  # 轨迹文件夹名（在 assets/ 下）
# TRAJ_DIR = "Traj_Jan20_td3_lightweight_cvar_cpo_wc0.9"  # 轨迹文件夹名（在 assets/ 下）
# TRAJ_DIR = "Traj_Nov24_td3_lightweight_without_used"  # 轨迹文件夹名（在 assets/ 下）
# TRAJ_DIR = "Traj_rcpo"  # 轨迹文件夹名（在 assets/ 下）
# TRAJ_DIR = "Traj_sac_lagrangian"  # 轨迹文件夹名（在 assets/ 下）
# TRAJ_DIR = "Traj_Jan06_td3_lightweight_cvar_cpo_ori"  # 轨迹文件夹名（在 assets/ 下）
SCENARIO_ID = 1                                          # 场景编号
OUTPUT_NAME = "cvar_cpo_wc0.5_v2_cropped"    # 输出文件名（不含扩展名）
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
EVAL_SCENARIOS_DIR = PROJECT_ROOT / "eval_scenarios"
VIS_DIR = PROJECT_ROOT / "visualizations"

OBS_COLOR = "#4a4a4a"
BOUNDARY_COLOR = "#2b2b2b"
HATCH_PATTERN = "///"


def _setup_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
    })


def draw_obstacles(ax, obstacle_map):
    for obs in obstacle_map["obstacles"]:
        pos = obs["position"]
        size = obs["size"]

        if obs["type"] == "boundary":
            facecolor, edgecolor = BOUNDARY_COLOR, BOUNDARY_COLOR
            alpha, lw, hatch = 0.85, 0.8, None
        else:
            facecolor, edgecolor = OBS_COLOR, "black"
            alpha, lw, hatch = 0.75, 1.2, HATCH_PATTERN

        if obs["shape"] == "cylinder":
            radius = obs.get("radius", size[0] / 2)
            patch = patches.Circle(
                (pos[0], pos[1]),
                radius=radius,
                linewidth=lw,
                edgecolor=edgecolor,
                facecolor=facecolor,
                alpha=alpha,
                hatch=hatch,
                zorder=1,
            )
        else:
            yaw = obs.get("yaw", 0.0)
            transform = (
                Affine2D().rotate_deg(math.degrees(yaw)).translate(pos[0], pos[1])
                + ax.transData
            )
            patch = patches.Rectangle(
                (-size[0] / 2, -size[1] / 2),
                size[0],
                size[1],
                linewidth=lw,
                edgecolor=edgecolor,
                facecolor=facecolor,
                alpha=alpha,
                hatch=hatch,
                transform=transform,
                zorder=1,
            )
        ax.add_patch(patch)


def main():
    _setup_rc()

    # --- 路径推导 ---
    traj_dir = ASSETS_DIR / TRAJ_DIR
    obstacle_map_file = EVAL_SCENARIOS_DIR / f"obstacle_map_scenario_{SCENARIO_ID:02d}.json"
    output_dir = VIS_DIR / "trajectory_cropped"

    # --- 加载障碍物地图 ---
    with open(obstacle_map_file, "r", encoding="utf-8") as f:
        obstacle_map = json.load(f)

    # --- 加载所有 pkl 轨迹文件 ---
    traj_files = sorted(traj_dir.glob("*.pkl"))
    if not traj_files:
        print(f"[ERROR] No .pkl files found in {traj_dir}")
        return

    all_trajs = []
    for tf in traj_files:
        with open(tf, "rb") as f:
            trajs = pickle.load(f)
        if SCENARIO_ID < len(trajs):
            all_trajs.append(trajs[SCENARIO_ID])
    print(f"[OK] Loaded {len(all_trajs)} trajectories for scenario {SCENARIO_ID}")

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#f7f7f7")
    fig.patch.set_facecolor("white")

    draw_obstacles(ax, obstacle_map)

    # 绘制所有轨迹
    for traj in all_trajs:
        poses = np.array(traj["poses"])
        ax.plot(
            poses[:, 0], poses[:, 1],
            color="black", linewidth=1.2, alpha=0.65, zorder=5,
        )

    # 起点（取第一条轨迹）
    first_poses = np.array(all_trajs[0]["poses"])
    ax.scatter(
        first_poses[0, 0], first_poses[0, 1],
        color="#2ca02c", edgecolor="black", s=200, linewidths=1.5,
        marker="o", zorder=8,
    )

    # 目标点
    target = all_trajs[0]["target_pos"]
    ax.scatter(
        target[0], target[1],
        color="#FFD700", edgecolor="black", s=500, linewidths=1.5,
        marker="*", zorder=9,
    )
    ax.add_patch(patches.Circle(
        target, radius=0.3, fill=False, linestyle="--", color="#2ca02c",
        linewidth=1.2, zorder=6,
    ))

    # 裁剪视野 & 样式
    ax.set_xlim(-4.5, 1.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, linewidth=0.5, color="gray")
    ax.tick_params(labelbottom=False, labelleft=False)

    fig.tight_layout(pad=0.5)

    # --- 保存 ---
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{OUTPUT_NAME}.png"
    pdf_path = png_path.with_suffix(".pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
