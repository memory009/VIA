#!/usr/bin/env python3
"""
Trajectory Collection Script

Runs the trained model in the Gazebo simulation environment to collect
trajectories for reachable set verification.

Supports:
  - TD3_Lightweight  (baseline)
  - TD3_VIA          (proposed method)

Output: a .pkl file saved under assets/, which can then be passed to
        reachable_set_verification.py for POLAR-based safety verification.

Usage:
  1. Edit the "User Configuration" section below.
  2. Run:  python scripts/collect_trajectories.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pickle
import json
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from TD3.TD3_lightweight import TD3 as TD3_Lightweight
from TD3.TD3_VIA import TD3_VIA
from ros_python import ROS_env
from utils import pos_data


def load_eval_scenarios(json_path=None):
    """Load evaluation scenarios from JSON file, or generate random ones."""
    if json_path is None:
        json_path = project_root / "assets" / "eval_scenarios_8_polar.json"

    if not json_path.exists():
        print(f"⚠️  Scenario file not found: {json_path}")
        print("   Generating random scenarios instead.")
        from utils import record_eval_positions
        scenarios = record_eval_positions(
            n_eval_scenarios=10,
            save_to_file=True,
            enable_random_obstacles=False
        )
        return scenarios

    with open(json_path, 'r') as f:
        data = json.load(f)

    scenarios = []
    for scenario_dict in data['scenarios']:
        scenario = []
        for element_dict in scenario_dict['elements']:
            element = pos_data()
            element.name = element_dict['name']
            element.x = element_dict['x']
            element.y = element_dict['y']
            element.angle = element_dict['angle']
            scenario.append(element)
        scenarios.append(scenario)

    return scenarios


def collect_single_trajectory(agent, env, scenario, max_steps=300,
                               model_type="TD3_Lightweight",
                               initial_e_t=0.0, gamma=0.99,
                               danger_threshold=0.5):
    """
    Collect one trajectory from a single scenario.

    Args:
        agent         : trained model instance
        env           : ROS environment
        scenario      : scenario configuration
        max_steps     : maximum steps per episode
        model_type    : "TD3_Lightweight" or "TD3_VIA"
        initial_e_t   : initial e_t value for TD3_VIA (read from model.var_u)
        gamma         : discount factor used to update e_t (TD3_VIA only)
        danger_threshold : cost threshold for binary cost (TD3_VIA only)
    """
    from squaternion import Quaternion

    trajectory = []
    actions = []
    rewards = []
    poses = []

    latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(scenario)

    robot_pos = (scenario[-2].x, scenario[-2].y, scenario[-2].angle)
    target_pos = (scenario[-1].x, scenario[-1].y)

    e_t = initial_e_t

    step_count = 0
    while step_count < max_steps:
        # Read real pose from Gazebo odometry
        latest_position = env.sensor_subscriber.latest_position
        latest_orientation = env.sensor_subscriber.latest_heading

        if latest_position is not None and latest_orientation is not None:
            odom_x = latest_position.x
            odom_y = latest_position.y
            quaternion = Quaternion(
                latest_orientation.w,
                latest_orientation.x,
                latest_orientation.y,
                latest_orientation.z,
            )
            euler = quaternion.to_euler(degrees=False)
            theta = euler[2]
            current_pose = [odom_x, odom_y, theta]
        else:
            current_pose = [robot_pos[0], robot_pos[1], robot_pos[2]]

        state, terminal = agent.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )
        trajectory.append(state)
        rewards.append(reward)
        poses.append(current_pose.copy())

        if terminal:
            break

        if model_type == "TD3_VIA":
            action = agent.act(state, e_t)
        else:
            action = agent.get_action(state, add_noise=False)
        actions.append(action)

        a_in = [(action[0] + 1) / 2, action[1]]
        latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )

        if model_type == "TD3_VIA":
            min_distance = min(latest_scan)
            cost = 1.0 if (collision or min_distance < danger_threshold) else 0.0
            e_t = (e_t - cost) / gamma

        step_count += 1

    return {
        'states': np.array(trajectory),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'poses': np.array(poses),
        'collision': collision,
        'goal_reached': goal,
        'steps': len(trajectory),
        'total_reward': sum(rewards),
        'robot_start': robot_pos,
        'target_pos': target_pos,
    }


def main():
    print("\n" + "=" * 70)
    print("Trajectory Collection Tool")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # ⚠️  User Configuration
    #
    # 1. Set model_type to "TD3_Lightweight" or "TD3_VIA".
    # 2. Set model_name to the filename prefix used when saving the model
    #    (e.g. "TD3_lightweight_best" or "TD3_VIA_best").
    # 3. Set model_dir to the run directory that contains the .pth weight files.
    # 4. Set output_name to a descriptive name for the output trajectory file.
    #    The file will be saved as: assets/<output_name>.pkl
    # =========================================================================
    model_type  = "TD3_Lightweight"       # "TD3_Lightweight" or "TD3_VIA"
    model_name  = "TD3_lightweight_best"  # filename prefix of the saved weights
    model_dir   = project_root / "models" / "TD3_lightweight" / "<your_run_id>"
    output_name = "trajectories_td3_lightweight_v1"  # output .pkl filename (no extension)
    # =========================================================================

    # --- Load model ---
    print(f"\n[1/4] Loading model '{model_name}' ({model_type}) ...")

    if model_type == "TD3_Lightweight":
        agent = TD3_Lightweight(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            load_model=True,
            model_name=model_name,
            load_directory=model_dir,
        )
        via_initial_e_t = None

    elif model_type == "TD3_VIA":
        agent = TD3_VIA(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            load_model=False,
            save_directory=model_dir,
            model_name=model_name,
            load_directory=model_dir,
        )
        agent.load(filename=model_name, directory=str(model_dir))
        via_initial_e_t = agent.var_u.item()
        print(f"  VIA initial e_t (from model.var_u): {via_initial_e_t:.4f}")

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'TD3_Lightweight' or 'TD3_VIA'.")

    print(f"  ✅ Model loaded successfully")

    # --- Initialize environment ---
    print("\n[2/4] Initializing ROS environment ...")
    env = ROS_env(enable_random_obstacles=False)
    print("  ✅ ROS environment ready")

    # --- Load scenarios ---
    print("\n[3/4] Loading evaluation scenarios ...")
    scenarios = load_eval_scenarios()
    n_scenarios = len(scenarios)
    print(f"  ✅ {n_scenarios} scenarios loaded")

    # --- Collect trajectories ---
    print(f"\n[4/4] Collecting trajectories for {n_scenarios} scenarios ...")
    all_trajectories = []

    for i, scenario in enumerate(tqdm(scenarios, desc="Collecting")):
        try:
            kwargs = dict(
                agent=agent,
                env=env,
                scenario=scenario,
                max_steps=300,
                model_type=model_type,
            )
            if model_type == "TD3_VIA":
                kwargs.update(
                    initial_e_t=via_initial_e_t,
                    gamma=0.99,
                    danger_threshold=0.5,
                )
            traj = collect_single_trajectory(**kwargs)
            all_trajectories.append(traj)

            status = "🎯 Goal" if traj['goal_reached'] else "💥 Collision"
            print(f"  Scenario {i+1}/{n_scenarios}: {status} | "
                  f"steps={traj['steps']} | reward={traj['total_reward']:.1f}")

        except Exception as e:
            print(f"  ❌ Scenario {i+1} failed: {e}")
            all_trajectories.append(None)

    # --- Save ---
    output_path = project_root / "assets" / f"{output_name}.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(all_trajectories, f)

    # --- Summary ---
    successful = [t for t in all_trajectories if t is not None]
    goal_count = sum(1 for t in successful if t['goal_reached'])
    col_count  = sum(1 for t in successful if t['collision'])
    avg_steps  = np.mean([t['steps'] for t in successful])
    avg_reward = np.mean([t['total_reward'] for t in successful])

    print("\n" + "=" * 70)
    print("Collection Summary")
    print(f"  Successful : {len(successful)}/{n_scenarios}")
    print(f"  Goal reached  : {goal_count}  ({goal_count/len(successful)*100:.1f}%)")
    print(f"  Collision     : {col_count}   ({col_count/len(successful)*100:.1f}%)")
    print(f"  Avg steps     : {avg_steps:.1f}")
    print(f"  Avg reward    : {avg_reward:.2f}")
    print("=" * 70)
    print(f"\n✅ Trajectories saved to: {output_path}")
    print(f"\nNext step: run reachable_set_verification.py with this trajectory file.")


if __name__ == "__main__":
    main()
