#!/usr/bin/env python3
"""
裁剪视角可视化 obstacle_map（x: -4~2, y: -4~2），适合论文小图
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D


def visualize_obstacle_map_cropped(json_path: str, save_path: str = None):
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        obstacle_map = json.load(f)

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

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_facecolor("#f7f7f7")
    fig.patch.set_facecolor("white")

    OBS_COLOR = "#4a4a4a"
    BOUNDARY_COLOR = "#2b2b2b"
    HATCH_PATTERN = "///"

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
            )
        ax.add_patch(patch)

    ax.set_xlim(-5, 2)
    ax.set_ylim(-5, 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, linewidth=0.5, color="gray")
    ax.tick_params(labelbottom=False, labelleft=False)

    fig.tight_layout(pad=0.5)

    # 保存
    if save_path is None:
        save_path = json_path.parent / (json_path.stem + "_cropped.png")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    print(f"Saved: {save_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    default_json = (
        Path(__file__).resolve().parents[1]
        / "eval_scenarios"
        / "obstacle_map_scenario_06.json"
    )

    default_save = (
        Path(__file__).resolve().parents[1]
        / "visualizations"
        / "obstacle_map"
        / "obstacle_map_scenario_06_cropped.png"
    )

    import argparse
    parser = argparse.ArgumentParser(description="Visualize cropped obstacle map")
    parser.add_argument("--json", type=str, default=str(default_json),
                        help="Path to obstacle_map JSON file")
    parser.add_argument("--output", type=str, default=str(default_save),
                        help="Output PNG path")
    args = parser.parse_args()

    visualize_obstacle_map_cropped(args.json, args.output)
