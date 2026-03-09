#!/usr/bin/env python3
"""
轨迹回放脚本
从 pkl 文件中加载收集好的轨迹，在 Gazebo 模拟器中重新执行。

支持两种回放模式：
  - actions: 重建原始场景 + 重放动作（闭环，物理真实）
  - poses:   按帧 Teleport 小车到记录的位姿（开环，精确复现）

用法示例：
  # 回放所有轨迹（action 模式，默认）
  python3 replay_trajectory.py path/to/trajectories.pkl

  # 只回放第 0、2 条轨迹
  python3 replay_trajectory.py path/to/trajectories.pkl --traj_ids 0 2

  # 使用 pose 模式（Teleport）回放，每步间隔 0.15s
  python3 replay_trajectory.py path/to/trajectories.pkl --method poses --step_delay 0.15

  # 指定场景文件（action 模式需要）
  python3 replay_trajectory.py path/to/trajectories.pkl --scenarios path/to/eval_scenarios.json
"""

import sys
import argparse
import time
import pickle
import json
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ros_python import ROS_env
from utils import pos_data


# ─────────────────────────── 数据加载 ───────────────────────────

def load_trajectories(pkl_path: str):
    """从 pkl 文件中加载轨迹列表"""
    with open(pkl_path, "rb") as f:
        trajectories = pickle.load(f)
    return trajectories


def load_scenarios(json_path=None):
    """从 JSON 文件中加载评估场景（action 模式需要）"""
    if json_path is None:
        json_path = project_root / "assets" / "eval_scenarios_8_polar.json"
    json_path = Path(json_path)
    if not json_path.exists():
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    scenarios = []
    for scenario_dict in data["scenarios"]:
        scenario = []
        for elem_dict in scenario_dict["elements"]:
            elem = pos_data()
            elem.name  = elem_dict["name"]
            elem.x     = elem_dict["x"]
            elem.y     = elem_dict["y"]
            elem.angle = elem_dict["angle"]
            scenario.append(elem)
        scenarios.append(scenario)
    return scenarios


# ─────────────────────────── 回放模式 ───────────────────────────

def replay_with_actions(env, traj_data, scenario, step_delay: float = 0.0):
    """
    Action 回放（闭环）：
    1. 用 env.eval(scenario) 重建原始场景（含障碍物位置、机器人起点、目标）
    2. 逐步执行记录的 actions
    收集时的转换公式：a_in = [(raw[0] + 1) / 2, raw[1]]
    """
    actions = np.array(traj_data["actions"])  # (T-1, 2) raw agent output
    n_steps = len(actions)

    # ── 设置场景 ──
    env.eval(scenario)
    print(f"    场景已重建，开始回放 {n_steps} 步 action ...")

    for i, raw_action in enumerate(actions):
        # 还原为速度命令（与收集时相同的转换）
        lin = float((raw_action[0] + 1) / 2)
        ang = float(raw_action[1])

        # 执行
        _, distance, _, _, collision, goal, _, reward = env.step(
            lin_velocity=lin, ang_velocity=ang
        )

        print(
            f"    step {i+1:>4d}/{n_steps}: "
            f"lin={lin:.3f}  ang={ang:+.3f}  "
            f"dist={distance:.3f}  reward={reward:+.2f}  "
            f"{'[COLLISION]' if collision else ''}{'[GOAL]' if goal else ''}"
        )

        if step_delay > 0:
            time.sleep(step_delay)

        if collision or goal:
            result = "到达目标 ✓" if goal else "碰撞 ✗"
            print(f"\n    *** 终止：{result}（第 {i+1} 步）***")
            return collision, goal

    print(f"\n    *** 超时（{n_steps} 步未到终止）***")
    return False, False


def replay_with_poses(env, traj_data, step_delay: float = 0.12):
    """
    Pose 回放（开环 / Teleport）：
    按帧将机器人传送到每个记录的 (x, y, theta) 位置，精确复现轨迹形状。
    不依赖场景 JSON，仅需 pkl 数据。
    """
    poses      = np.array(traj_data["poses"])      # (T, 3)
    target_pos = traj_data["target_pos"]            # (x, y)
    n_steps    = len(poses)

    # ── 显示目标 marker ──
    env.target = [float(target_pos[0]), float(target_pos[1])]
    env.publish_target.publish(env.target[0], env.target[1])

    print(f"    目标位置: ({target_pos[0]:.3f}, {target_pos[1]:.3f})")
    print(f"    开始逐帧 Teleport，共 {n_steps} 帧 ...")

    env.physics_client.unpause_physics()

    for i, pose in enumerate(poses):
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        env.set_position("turtlebot3_waffle", x, y, theta)
        time.sleep(step_delay)
        print(
            f"    frame {i+1:>4d}/{n_steps}: "
            f"x={x:+.3f}  y={y:+.3f}  θ={theta:+.4f} rad"
        )

    env.physics_client.pause_physics()
    print(f"\n    *** Pose 回放完成 ***")


# ─────────────────────────── 主程序 ───────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="在 Gazebo 中回放 pkl 轨迹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pkl_path",
        type=str,
        help="pkl 轨迹文件路径",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        metavar="JSON",
        help="评估场景 JSON 文件路径（action 模式必需，默认: assets/eval_scenarios_8_polar.json）",
    )
    parser.add_argument(
        "--traj_ids",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="要回放的轨迹索引列表（默认: 全部），例如 --traj_ids 0 2 5",
    )
    parser.add_argument(
        "--method",
        choices=["actions", "poses"],
        default="actions",
        help="回放模式：actions（闭环重放动作） 或 poses（Teleport 位姿）（默认: actions）",
    )
    parser.add_argument(
        "--step_delay",
        type=float,
        default=0.0,
        metavar="SEC",
        help="每步额外等待时间（秒），用于放慢观察速度（默认: 0.0）",
    )
    parser.add_argument(
        "--pause_between",
        type=float,
        default=3.0,
        metavar="SEC",
        help="相邻轨迹之间的暂停时间（秒，默认: 3.0）",
    )
    return parser.parse_args()


def print_traj_summary(traj_data, traj_idx: int):
    print(f"\n{'─'*55}")
    print(f"  轨迹 #{traj_idx}")
    print(f"  步数:       {traj_data['steps']}")
    print(f"  到达目标:   {'是 ✓' if traj_data['goal_reached'] else '否'}")
    print(f"  碰撞:       {'是 ✗' if traj_data['collision']    else '否'}")
    print(f"  总奖励:     {traj_data['total_reward']:.2f}")
    print(f"  机器人起点: ({traj_data['robot_start'][0]:.3f}, {traj_data['robot_start'][1]:.3f})")
    print(f"  目标位置:   ({traj_data['target_pos'][0]:.3f}, {traj_data['target_pos'][1]:.3f})")
    print(f"{'─'*55}")


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  轨迹回放工具 (Trajectory Replay)")
    print("=" * 60)

    # ── 1. 加载轨迹 ──
    pkl_path = Path(args.pkl_path)
    if not pkl_path.exists():
        # 尝试相对于 assets 目录
        alt = project_root / "assets" / args.pkl_path
        if alt.exists():
            pkl_path = alt
        else:
            print(f"错误：找不到 pkl 文件: {args.pkl_path}")
            sys.exit(1)

    print(f"\n[1/3] 加载轨迹: {pkl_path}")
    trajectories = load_trajectories(str(pkl_path))
    print(f"  共 {len(trajectories)} 条轨迹")

    # ── 2. 加载场景（action 模式需要） ──
    scenarios = None
    if args.method == "actions":
        print(f"\n[2/3] 加载场景 JSON（action 模式必需）...")
        scenarios = load_scenarios(args.scenarios)
        if scenarios is None:
            print("  ⚠️  未找到场景文件，自动切换为 poses 模式")
            args.method = "poses"
        else:
            print(f"  已加载 {len(scenarios)} 个场景")
            if len(scenarios) != len(trajectories):
                print(
                    f"  ⚠️  警告：场景数量 ({len(scenarios)}) ≠ 轨迹数量 ({len(trajectories)})"
                )
    else:
        print(f"\n[2/3] Poses 模式无需场景文件，跳过")

    # ── 3. 初始化 ROS 环境 ──
    print(f"\n[3/3] 初始化 ROS 环境 ...")
    env = ROS_env(enable_random_obstacles=False)
    print("  ✅ ROS 环境就绪")

    # ── 确定要回放的轨迹 ──
    traj_ids = args.traj_ids if args.traj_ids is not None else list(range(len(trajectories)))

    print(f"\n回放模式:  {args.method}")
    print(f"目标轨迹:  {traj_ids}")
    print(f"步间延迟:  {args.step_delay}s")
    print(f"轨迹间隔:  {args.pause_between}s")

    # ── 回放循环 ──
    for n, traj_idx in enumerate(traj_ids):
        if traj_idx >= len(trajectories):
            print(f"\n⚠️  轨迹索引 {traj_idx} 超出范围（最大 {len(trajectories)-1}），跳过")
            continue

        traj_data = trajectories[traj_idx]
        if traj_data is None:
            print(f"\n⚠️  轨迹 #{traj_idx} 为 None，跳过")
            continue

        print_traj_summary(traj_data, traj_idx)

        if args.method == "actions":
            if scenarios is not None and traj_idx < len(scenarios):
                replay_with_actions(
                    env,
                    traj_data,
                    scenarios[traj_idx],
                    step_delay=args.step_delay,
                )
            else:
                print(f"  ⚠️  轨迹 #{traj_idx} 没有对应场景，跳过（action 模式需要场景）")
        else:
            replay_with_poses(
                env,
                traj_data,
                step_delay=args.step_delay if args.step_delay > 0 else 0.12,
            )

        # 两条轨迹之间暂停
        if n < len(traj_ids) - 1:
            print(f"\n  等待 {args.pause_between}s 后回放下一条轨迹 ...")
            time.sleep(args.pause_between)

    print("\n" + "=" * 60)
    print("  所有轨迹回放完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
