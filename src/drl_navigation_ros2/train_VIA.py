#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIA Training Script (Ablation Study)

Training procedure:
  1. Warm-up phase  : run one epoch of episodes (no network updates) to collect
                      cost statistics and initialize the VaR parameter u.
  2. Training phase : main loop for max_epochs epochs.
                      Each epoch: rollout -> update CVaR parameters -> evaluate.

Hyperparameters that match the baseline (TD3_Lightweight) are kept identical
for fair comparison. VIA-specific hyperparameters follow the paper settings.
"""

from pathlib import Path
from datetime import datetime
import socket
import argparse
import sys

import torch
import numpy as np

from TD3.TD3_VIA import TD3_VIA, BASELINE_BATCH_SIZE, BASELINE_BUFFER_SIZE, BASELINE_GAMMA, CVAR_ALPHA
from via_replay_buffer import VIAReplayBuffer


class TeeStream:
    """Duplicates writes to both a stream and a log file."""
    def __init__(self, stream, log_path):
        self.stream = stream
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "a", buffering=1, encoding="utf-8")

    def write(self, message):
        self.stream.write(message)
        self.log.write(message)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def compute_cost(laser_scan, collision, danger_threshold=0.5):
    """Binary cost signal: 1 if the robot enters a danger zone or collides, else 0.

    danger_threshold > collision_threshold, so:
      - entering the danger zone yields cost=1 but does not end the episode
      - an actual collision ends the episode (handled by the environment)
    """
    min_distance = min(laser_scan)
    if collision or min_distance < danger_threshold:
        return 1.0
    return 0.0


def run_warmup_phase(model, env, episodes_per_epoch, max_steps, gamma):
    """Collect cost statistics to initialize the VaR parameter u.

    Runs the same number of episodes as one training epoch.
    No transitions are stored and no network updates are performed.

    Returns:
        initial_var_u (float): alpha-quantile of episode costs, used to seed u.
        warmup_costs  (list) : per-episode cumulative costs from the warm-up.
    """
    print("\n" + "=" * 80)
    print("Warm-up Phase (does not count as a training epoch)")
    print(f"  Collecting {episodes_per_epoch} episodes to initialize VaR parameter u.")
    print("  No buffer storage, no network training.")
    print("=" * 80)

    warmup_costs = []

    for episode_idx in range(1, episodes_per_epoch + 1):
        latest_scan, distance, cos, sin, collision, goal, a, reward = env.reset()

        steps = 0
        episode_cost = 0.0
        e_t = 0.0  # placeholder; does not affect cost statistics

        while True:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            action = model.get_action(state, e_t, add_noise=True)
            a_in = [(action[0] + 1) / 2, action[1]]

            latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )

            cost = compute_cost(latest_scan, collision, danger_threshold=0.5)
            episode_cost += cost
            e_t = (e_t - cost) / gamma
            steps += 1

            if terminal or steps >= max_steps:
                break

        warmup_costs.append(episode_cost)

        if episode_idx % 10 == 0 or episode_idx == episodes_per_epoch:
            print(f"  [{episode_idx}/{episodes_per_epoch}] episode_cost={episode_cost:.1f}")

    avg_cost = np.mean(warmup_costs)
    min_cost = np.min(warmup_costs)
    max_cost = np.max(warmup_costs)

    # Initialize var_u as the alpha-quantile of the collected episode costs
    initial_var_u = np.percentile(warmup_costs, CVAR_ALPHA * 100)

    print(f"\n  Warm-up complete.")
    print(f"  Episode cost: avg={avg_cost:.2f}, min={min_cost:.2f}, max={max_cost:.2f}")
    print(f"  VaR init: alpha={CVAR_ALPHA}, var_u={initial_var_u:.4f} ({CVAR_ALPHA*100:.0f}th percentile)")
    print("=" * 80)

    return initial_var_u, warmup_costs


def main(args=None):
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train TD3 with VIA (Ablation Study)')
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--episodes-per-epoch', type=int, default=70)
    parser.add_argument('--train-every-n-episodes', type=int, default=2,
                        help='Train every N episodes (default: 2, same as baseline)')
    parser.add_argument('--train-iterations', type=int, default=500,
                        help='Training iterations per update (default: 500)')
    parser.add_argument('--batch-size', type=int, default=BASELINE_BATCH_SIZE,
                        help=f'Batch size (default: {BASELINE_BATCH_SIZE})')
    parser.add_argument('--buffer-size', type=int, default=BASELINE_BUFFER_SIZE,
                        help=f'Replay buffer size (default: {BASELINE_BUFFER_SIZE})')
    parser.add_argument('--max-steps', type=int, default=300,
                        help='Maximum steps per episode (default: 300)')
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    cmd_args = parser.parse_args(args)

    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    hostname = socket.gethostname()
    run_id = f"{timestamp}_{hostname}_via_ablation"

    model_dir_name = "TD3_VIA"
    model_name = "TD3_VIA"

    save_directory = Path("models") / model_dir_name / run_id
    save_directory.mkdir(parents=True, exist_ok=True)

    log_file = save_directory / "train_output.log"
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    action_dim = 2
    max_action = 1
    state_dim = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trains_per_epoch = cmd_args.episodes_per_epoch // cmd_args.train_every_n_episodes
    total_iterations_per_epoch = trains_per_epoch * cmd_args.train_iterations

    print("=" * 80)
    print("VIA Training (Ablation Study)")
    print("=" * 80)
    print(f"Run ID:     {run_id}")
    print(f"Save path:  {save_directory}")
    print(f"Schedule:   warm-up ({cmd_args.episodes_per_epoch} episodes) + "
          f"{cmd_args.max_epochs} epochs x {cmd_args.episodes_per_epoch} episodes")
    print(f"Per epoch:  {trains_per_epoch} updates x {cmd_args.train_iterations} iters "
          f"= {total_iterations_per_epoch:,} batches")
    print("=" * 80)

    model = TD3_VIA(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_directory=save_directory,
        model_name=model_name,
        run_id=run_id,
    )

    from ros_python import ROS_env
    from utils import record_eval_positions

    ros = ROS_env(enable_random_obstacles=True)
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=cmd_args.n_eval_episodes,
        save_to_file=True,
        random_seed=42,
        enable_random_obstacles=True
    )

    gamma = BASELINE_GAMMA

    # --- Warm-up phase ---
    initial_var_u, warmup_costs = run_warmup_phase(
        model=model,
        env=ros,
        episodes_per_epoch=cmd_args.episodes_per_epoch,
        max_steps=cmd_args.max_steps,
        gamma=gamma,
    )

    model.set_var_u(initial_var_u)
    print(f"\nvar_u initialized to: {model.var_u.item():.4f}")

    model.writer.add_scalar("warmup/avg_cost", np.mean(warmup_costs), 0)
    model.writer.add_scalar("warmup/max_cost", np.max(warmup_costs), 0)
    model.writer.add_scalar("warmup/min_cost", np.min(warmup_costs), 0)
    model.writer.add_scalar("warmup/initial_var_u", initial_var_u, 0)

    # Create replay buffer after warm-up to keep it clean
    replay_buffer = VIAReplayBuffer(buffer_size=cmd_args.buffer_size, random_seed=42)

    best_metrics = {
        'success_rate': -1.0,
        'collision_rate': 2.0,
        'avg_reward': -float('inf'),
        'avg_cost': float('inf'),
        'epoch': 0
    }
    epochs_since_improvement = 0

    # --- Training phase ---
    for epoch in range(cmd_args.max_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{cmd_args.max_epochs}")
        print(f"{'='*80}")

        # Phase 1: Rollout
        print(f"\n[Phase 1] Rollout ({cmd_args.episodes_per_epoch} episodes)")
        epoch_costs = []
        episode_count_in_epoch = 0

        for episode_idx in range(1, cmd_args.episodes_per_epoch + 1):
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()

            steps = 0
            episode_cost = 0.0
            episode_reward = 0.0

            # Paper eq: e_0 = u^k at the start of each episode
            e_t = model.var_u.item()
            episode_e_t_min = e_t

            while True:
                state, terminal = model.prepare_state(
                    latest_scan, distance, cos, sin, collision, goal, a
                )
                action = model.get_action(state, e_t, add_noise=True)
                a_in = [(action[0] + 1) / 2, action[1]]

                latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
                    lin_velocity=a_in[0], ang_velocity=a_in[1]
                )

                cost = compute_cost(latest_scan, collision, danger_threshold=0.5)
                episode_cost += cost
                episode_reward += reward

                # Paper eq: e_{t+1} = (e_t - C(s_t, a_t)) / gamma
                next_e_t = (e_t - cost) / gamma

                next_state, terminal = model.prepare_state(
                    latest_scan, distance, cos, sin, collision, goal, a
                )
                replay_buffer.add(
                    s=state, a=action, r=reward, t=terminal,
                    s2=next_state, c=cost, e_t=e_t, next_e_t=next_e_t
                )

                e_t = next_e_t
                episode_e_t_min = min(episode_e_t_min, e_t)
                steps += 1

                if terminal or steps >= cmd_args.max_steps:
                    break

            epoch_costs.append(episode_cost)
            episode_count_in_epoch += 1

            if episode_count_in_epoch % cmd_args.train_every_n_episodes == 0:
                if replay_buffer.size() >= cmd_args.batch_size:
                    model.train(
                        replay_buffer=replay_buffer,
                        iterations=cmd_args.train_iterations,
                        batch_size=cmd_args.batch_size,
                    )

            if episode_idx % 10 == 0 or episode_idx == cmd_args.episodes_per_epoch:
                print(f"  [{episode_idx}/{cmd_args.episodes_per_epoch}] "
                      f"cost={episode_cost:.1f}, reward={episode_reward:.1f}")

        avg_episode_cost = np.mean(epoch_costs)
        max_episode_cost = np.max(epoch_costs)
        min_episode_cost = np.min(epoch_costs)
        print(f"  Episode cost: avg={avg_episode_cost:.2f}, "
              f"min={min_episode_cost:.2f}, max={max_episode_cost:.2f}")

        # Phase 2: Update CVaR parameters (var_u and lambda_w)
        print(f"\n[Phase 2] Update CVaR parameters")
        old_var_u = model.var_u.item()
        old_lambda_w = model.lambda_w.item()

        model.update_var_and_lambda(avg_episode_cost, epoch_costs)

        print(f"  var_u:    {old_var_u:.4f} -> {model.var_u.item():.4f}")
        print(f"  lambda_w: {old_lambda_w:.4f} -> {model.lambda_w.item():.4f}")

        # Phase 3: Status
        print(f"\n[Phase 3] Training summary")
        print(f"  iter_count: {model.iter_count}  |  buffer: {replay_buffer.size()}/{cmd_args.buffer_size}")

        # Phase 4: Evaluation
        current_metrics = evaluate(
            model=model,
            env=ros,
            scenarios=eval_scenarios,
            epoch=epoch + 1,
            max_steps=cmd_args.max_steps,
            best_metrics=best_metrics,
            save_directory=save_directory,
            model_name=model_name,
            gamma=gamma,
        )

        model.save(filename=f"{model_name}_epoch_{epoch+1:03d}", directory=save_directory)

        if current_metrics['is_best']:
            best_metrics = current_metrics.copy()
            epochs_since_improvement = 0
            print(f"NEW BEST MODEL! (epoch {epoch + 1})")
        else:
            epochs_since_improvement += 1

        print("=" * 80)
        print(f"Epochs since last improvement: {epochs_since_improvement}")
        print(f"Best from epoch {best_metrics['epoch']}: "
              f"Success={best_metrics['success_rate']:.3f}, "
              f"Collision={best_metrics['collision_rate']:.3f}, "
              f"Reward={best_metrics['avg_reward']:.2f}, "
              f"Cost={best_metrics['avg_cost']:.2f}")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"  Warm-up phase + {cmd_args.max_epochs} training epochs")
    print(f"  Models saved to: {save_directory}")
    print("=" * 80)


def evaluate(model, env, scenarios, epoch, max_steps, best_metrics,
             save_directory, model_name, gamma):
    """Run deterministic evaluation and save the model if it is the best so far.

    Returns:
        dict: Evaluation metrics including 'is_best' flag.
    """
    print("\n" + "=" * 80)
    print(f"Epoch {epoch} - Evaluating {len(scenarios)} scenarios")
    print("=" * 80)

    avg_reward = 0.0
    avg_cost = 0.0
    col = 0
    gl = 0

    for scenario in scenarios:
        count = 0
        episode_cost = 0.0
        e_t = model.var_u.item()

        latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(scenario=scenario)

        while count < max_steps:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            if terminal:
                break

            action = model.act(state, e_t)
            a_in = [(action[0] + 1) / 2, action[1]]

            latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )

            cost = compute_cost(latest_scan, collision, danger_threshold=0.5)
            episode_cost += cost
            e_t = (e_t - cost) / gamma

            avg_reward += reward
            count += 1

        col += collision
        gl += goal
        avg_cost += episode_cost

    avg_reward /= len(scenarios)
    avg_cost /= len(scenarios)
    avg_col = col / len(scenarios)
    avg_goal = gl / len(scenarios)

    print(f"   Success Rate:    {avg_goal:.3f} ({gl}/{len(scenarios)})")
    print(f"   Collision Rate:  {avg_col:.3f} ({col}/{len(scenarios)})")
    print(f"   Average Reward:  {avg_reward:.2f}")
    print(f"   Average Cost:    {avg_cost:.2f}")

    is_best = is_better_model(
        avg_goal, avg_col, avg_reward, avg_cost,
        best_metrics['success_rate'], best_metrics['collision_rate'],
        best_metrics['avg_reward'], best_metrics['avg_cost']
    )

    if is_best:
        model.save(filename=f"{model_name}_best", directory=save_directory)

    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_cost", avg_cost, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)

    print("=" * 80)

    return {
        'success_rate': avg_goal,
        'collision_rate': avg_col,
        'avg_reward': avg_reward,
        'avg_cost': avg_cost,
        'epoch': epoch,
        'is_best': is_best
    }


def is_better_model(curr_success, curr_collision, curr_reward, curr_cost,
                    best_success, best_collision, best_reward, best_cost):
    """Multi-criteria model comparison with tiebreaking.

    Priority:
      1. Higher success rate
      2. Lower average cost (when success is equal)
      3. Lower collision rate (when cost is equal)
      4. Higher average reward (when all above are equal)
    """
    if curr_success > best_success:
        return True
    elif curr_success < best_success:
        return False

    if curr_cost < best_cost:
        return True
    elif curr_cost > best_cost:
        return False

    if curr_collision < best_collision:
        return True
    elif curr_collision > best_collision:
        return False

    return curr_reward > best_reward


if __name__ == "__main__":
    main()
