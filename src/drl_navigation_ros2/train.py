#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import socket
import argparse

from TD3.TD3_lightweight import TD3 as TD3_Lightweight
from ros_python import ROS_env
from replay_buffer import ReplayBuffer
import torch
import numpy as np
from utils import record_eval_positions
from pretrain_utils import Pretraining


def main(args=None):
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train TD3 for robot navigation')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Maximum number of training epochs (default: 100)')
    parser.add_argument('--episodes-per-epoch', type=int, default=70,
                        help='Number of episodes per epoch (default: 70)')
    cmd_args = parser.parse_args(args)

    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    hostname = socket.gethostname()
    run_id = f"{timestamp}_{hostname}"

    model_dir_name = "TD3"
    save_directory = Path("src/drl_navigation_ros2/models") / model_dir_name / run_id
    save_directory.mkdir(parents=True, exist_ok=True)

    model_name = "TD3"

    action_dim = 2          # number of actions
    max_action = 1          # maximum absolute action value
    state_dim = 25          # state vector length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nr_eval_episodes = 10
    max_epochs = cmd_args.max_epochs
    epoch = 0
    episodes_per_epoch = cmd_args.episodes_per_epoch
    episode = 0
    train_every_n = 2       # train every n episodes
    training_iterations = 500
    batch_size = 40
    max_steps = 300         # max steps per episode
    steps = 0
    load_saved_buffer = True
    pretrain = True
    pretraining_iterations = 50

    print("=" * 80)
    print(f"Run ID:       {run_id}")
    print(f"Save path:    {save_directory}")
    print(f"TensorBoard:  runs/{run_id}")
    print(f"Schedule:     {max_epochs} epochs x {episodes_per_epoch} episodes")
    print("=" * 80)

    model = TD3_Lightweight(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=100,
        load_model=False,
        save_directory=save_directory,
        model_name=model_name,
        run_id=run_id,
    )

    ros = ROS_env(
        enable_random_obstacles=True  # training: 4 fixed + 4 random obstacles
    )
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_eval_episodes,
        save_to_file=True,
        random_seed=42,
        enable_random_obstacles=True
    )

    if load_saved_buffer:
        pretraining = Pretraining(
            file_names=["src/drl_navigation_ros2/assets/data.yml"],
            model=model,
            replay_buffer=ReplayBuffer(buffer_size=5e3, random_seed=42),
            reward_function=ros.get_reward,
        )
        replay_buffer = pretraining.load_buffer()
        if pretrain:
            pretraining.train(
                pretraining_iterations=pretraining_iterations,
                replay_buffer=replay_buffer,
                iterations=training_iterations,
                batch_size=batch_size,
            )
    else:
        replay_buffer = ReplayBuffer(buffer_size=5e3, random_seed=42)

    latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
        lin_velocity=0.0, ang_velocity=0.0
    )

    best_metrics = {
        'success_rate': -1.0,
        'collision_rate': 2.0,
        'avg_reward': -float('inf'),
        'epoch': 0
    }
    epochs_since_improvement = 0

    while epoch < max_epochs:
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )
        action = model.get_action(state, True)
        a_in = [(action[0] + 1) / 2, action[1]]

        latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )
        next_state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )
        replay_buffer.add(state, action, reward, terminal, next_state)

        if terminal or steps == max_steps:
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
            episode += 1
            if episode % train_every_n == 0:
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )
            steps = 0
        else:
            steps += 1

        if (episode + 1) % episodes_per_epoch == 0:
            episode = 0
            epoch += 1

            current_metrics = eval(
                model=model,
                env=ros,
                scenarios=eval_scenarios,
                epoch=epoch,
                max_steps=max_steps,
                best_metrics=best_metrics,
                save_directory=save_directory,
                model_name=model_name,
            )

            if current_metrics['is_best']:
                best_metrics = current_metrics.copy()
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            print(f"Epochs since last improvement: {epochs_since_improvement}")
            print(f"Best from epoch {best_metrics['epoch']}: "
                  f"Success={best_metrics['success_rate']:.3f}, "
                  f"Collision={best_metrics['collision_rate']:.3f}, "
                  f"Reward={best_metrics['avg_reward']:.2f}")
            print("=" * 80 + "\n")

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best model from epoch {best_metrics['epoch']}")
    print(f"  Success Rate:   {best_metrics['success_rate']:.3f}")
    print(f"  Collision Rate: {best_metrics['collision_rate']:.3f}")
    print(f"  Avg Reward:     {best_metrics['avg_reward']:.2f}")
    print(f"Saved to: {save_directory}/{model_name}_best_*.pth")
    print("=" * 80)


def eval(model, env, scenarios, epoch, max_steps, best_metrics, save_directory, model_name):
    """Run evaluation and save the model if it is the best so far.

    Returns:
        dict: Current evaluation metrics including 'is_best' flag.
    """
    print("\n" + "=" * 80)
    print(f"Epoch {epoch} - Evaluating {len(scenarios)} scenarios")
    print("=" * 80)

    avg_reward = 0.0
    col = 0
    gl = 0

    for scenario in scenarios:
        count = 0
        latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(
            scenario=scenario
        )
        while count < max_steps:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            if terminal:
                break
            action = model.get_action(state, False)
            a_in = [(action[0] + 1) / 2, action[1]]
            latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            avg_reward += reward
            count += 1
            col += collision
            gl += goal

    avg_reward /= len(scenarios)
    avg_col = col / len(scenarios)
    avg_goal = gl / len(scenarios)

    print(f"   Success Rate:    {avg_goal:.3f} ({gl}/{len(scenarios)})")
    print(f"   Collision Rate:  {avg_col:.3f} ({col}/{len(scenarios)})")
    print(f"   Average Reward:  {avg_reward:.2f}")

    is_best = is_better_model(
        current_success=avg_goal,
        current_collision=avg_col,
        current_reward=avg_reward,
        best_success=best_metrics['success_rate'],
        best_collision=best_metrics['collision_rate'],
        best_reward=best_metrics['avg_reward']
    )

    if is_best:
        model.save(filename=f"{model_name}_best", directory=save_directory)
        print("NEW BEST MODEL saved.")
        if avg_goal > best_metrics['success_rate']:
            print(f"  Success: {best_metrics['success_rate']:.3f} -> {avg_goal:.3f}")
        elif avg_goal == best_metrics['success_rate'] and avg_col < best_metrics['collision_rate']:
            print(f"  Same success, lower collision: {best_metrics['collision_rate']:.3f} -> {avg_col:.3f}")
        elif avg_goal == best_metrics['success_rate'] and avg_col == best_metrics['collision_rate']:
            print(f"  Same success & collision, higher reward: {best_metrics['avg_reward']:.2f} -> {avg_reward:.2f}")
    else:
        print(f"Not best (best from epoch {best_metrics['epoch']})")

    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)

    print("=" * 80)

    return {
        'success_rate': avg_goal,   # TensorBoard key: eval/avg_goal
        'collision_rate': avg_col,  # TensorBoard key: eval/avg_col
        'avg_reward': avg_reward,
        'epoch': epoch,
        'is_best': is_best
    }


def is_better_model(current_success, current_collision, current_reward,
                    best_success, best_collision, best_reward):
    """Multi-criteria model comparison with tiebreaking.

    Priority:
      1. Higher success rate
      2. Lower collision rate (when success is equal)
      3. Higher average reward (when both are equal)
    """
    if current_success > best_success:
        return True
    elif current_success < best_success:
        return False

    if current_collision < best_collision:
        return True
    elif current_collision > best_collision:
        return False

    return current_reward > best_reward


if __name__ == "__main__":
    main()
