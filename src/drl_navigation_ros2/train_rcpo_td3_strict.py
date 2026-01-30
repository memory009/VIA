#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3-RCPO 训练脚本 (严格遵循论文版本)

论文: "Reward Constrained Policy Optimization" (Tessler et al., ICLR 2019)

============================================================================
与论文的对应关系
============================================================================

【论文原文】
- Backbone: A2C/PPO (on-policy)
- 实验: Mars Rover (A2C) + Mujoco Robotics (PPO)

【本实现】
- Backbone: TD3 (off-policy)
- 目的: 与CVaR-CPO进行公平的ablation比较

【关键适配】
1. on-policy → off-policy: 使用Replay Buffer
2. V function → Q function: 适配Actor-Critic到TD3框架
3. 保持核心不变: penalized reward, λ更新规则

============================================================================
"""

from pathlib import Path
from datetime import datetime
import socket
import argparse
import sys

import torch
import numpy as np

from TD3.TD3_rcpo_strict import TD3_RCPO_Strict, BASELINE_BATCH_SIZE, BASELINE_BUFFER_SIZE, BASELINE_GAMMA


class TeeStream:
    """同时输出到stdout和日志文件"""
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


class RCPOReplayBuffer:
    """
    RCPO Replay Buffer
    
    存储: (state, action, reward, cost, done, next_state)
    
    注意：与CVaR-CPO不同，RCPO不需要存储e_t
    """
    def __init__(self, buffer_size=500000, random_seed=None):
        self.buffer_size = int(buffer_size)
        self.count = 0
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'costs': [],
            'dones': [],
            'next_states': []
        }
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def add(self, s, a, r, t, s2, c):
        """添加transition"""
        if self.count < self.buffer_size:
            self.buffer['states'].append(s)
            self.buffer['actions'].append(a)
            self.buffer['rewards'].append(r)
            self.buffer['costs'].append(c)
            self.buffer['dones'].append(t)
            self.buffer['next_states'].append(s2)
            self.count += 1
        else:
            idx = self.count % self.buffer_size
            self.buffer['states'][idx] = s
            self.buffer['actions'][idx] = a
            self.buffer['rewards'][idx] = r
            self.buffer['costs'][idx] = c
            self.buffer['dones'][idx] = t
            self.buffer['next_states'][idx] = s2
            self.count += 1
    
    def size(self):
        return min(self.count, self.buffer_size)
    
    def sample_batch(self, batch_size):
        """采样batch"""
        current_size = self.size()
        indices = np.random.choice(current_size, min(batch_size, current_size), replace=False)
        
        return {
            'states': np.array([self.buffer['states'][i] for i in indices]),
            'actions': np.array([self.buffer['actions'][i] for i in indices]),
            'rewards': np.array([self.buffer['rewards'][i] for i in indices]).reshape(-1, 1),
            'costs': np.array([self.buffer['costs'][i] for i in indices]).reshape(-1, 1),
            'dones': np.array([self.buffer['dones'][i] for i in indices]).reshape(-1, 1),
            'next_states': np.array([self.buffer['next_states'][i] for i in indices])
        }


def compute_cost(laser_scan, collision, danger_threshold=0.5):
    """
    计算cost（与CVaR-CPO完全一致）
    """
    min_distance = min(laser_scan)
    
    if collision:
        cost = 1.0
    elif min_distance < danger_threshold:
        cost = 1.0
    else:
        cost = 0.0
    
    return cost


def main(args=None):
    """主训练函数"""
    parser = argparse.ArgumentParser(description='Train TD3 with RCPO (Strict Version)')
    
    # 训练设置
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--episodes-per-epoch', type=int, default=70)
    parser.add_argument('--train-every-n-episodes', type=int, default=2)
    parser.add_argument('--train-iterations', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=BASELINE_BATCH_SIZE)
    parser.add_argument('--buffer-size', type=int, default=BASELINE_BUFFER_SIZE)
    
    # 环境参数
    parser.add_argument('--safety-threshold', type=float, default=0.4)
    parser.add_argument('--max-steps', type=int, default=300)
    
    # 评估
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    
    cmd_args = parser.parse_args(args)
    
    # 运行ID和路径
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    hostname = socket.gethostname()
    run_id = f"{timestamp}_{hostname}_rcpo_strict"
    
    model_dir_name = "TD3_rcpo_strict"
    model_name = "TD3_rcpo_strict"
    
    save_directory = Path("models") / model_dir_name / run_id
    save_directory.mkdir(parents=True, exist_ok=True)

    log_file = save_directory / "train_output.log"
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    
    # 训练参数
    action_dim = 2
    max_action = 1
    state_dim = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trains_per_epoch = cmd_args.episodes_per_epoch // cmd_args.train_every_n_episodes
    total_iterations_per_epoch = trains_per_epoch * cmd_args.train_iterations
    
    print("=" * 80)
    print("🚀 TD3-RCPO训练 (严格遵循论文版本)")
    print("=" * 80)
    print(f"📁 运行ID: {run_id}")
    print(f"💾 保存路径: {save_directory}")
    print(f"\n📖 论文对应:")
    print(f"   原文Backbone: A2C/PPO (on-policy)")
    print(f"   本实现Backbone: TD3 (off-policy)")
    print(f"   核心保持不变: penalized reward, λ更新")
    print(f"\n📋 训练配置:")
    print(f"   {cmd_args.max_epochs} epochs × {cmd_args.episodes_per_epoch} episodes")
    print(f"   每epoch: {total_iterations_per_epoch:,} iterations")
    print("=" * 80)

    # 模型初始化
    model = TD3_RCPO_Strict(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_directory=save_directory,
        model_name=model_name,
        run_id=run_id,
    )

    # ROS环境
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
    
    # Replay Buffer
    replay_buffer = RCPOReplayBuffer(buffer_size=cmd_args.buffer_size, random_seed=42)
    
    # Best model追踪
    best_metrics = {
        'success_rate': -1.0,
        'collision_rate': 2.0,
        'avg_reward': -float('inf'),
        'avg_cost': float('inf'),
        'epoch': 0
    }
    epochs_since_improvement = 0
    
    # ===== Training Loop =====
    for epoch in range(cmd_args.max_epochs):
        print(f"\n{'='*80}")
        print(f"🎯 Epoch {epoch + 1}/{cmd_args.max_epochs}")
        print(f"{'='*80}")
        
        # ===== Phase 1: Rollout =====
        print(f"\n📍 Phase 1: Rollout ({cmd_args.episodes_per_epoch} episodes)")
        
        epoch_costs = []
        episode_count_in_epoch = 0
        
        for episode_idx in range(1, cmd_args.episodes_per_epoch + 1):
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
            
            steps = 0
            episode_cost = 0.0
            episode_reward = 0.0
            
            while True:
                state, terminal = model.prepare_state(
                    latest_scan, distance, cos, sin, collision, goal, a
                )
                
                action = model.get_action(state, add_noise=True)
                a_in = [(action[0] + 1) / 2, action[1]]
                
                latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
                    lin_velocity=a_in[0], ang_velocity=a_in[1]
                )
                
                cost = compute_cost(latest_scan, collision, danger_threshold=0.5)
                episode_cost += cost
                episode_reward += reward
                
                next_state, terminal = model.prepare_state(
                    latest_scan, distance, cos, sin, collision, goal, a
                )
                
                # 存储
                replay_buffer.add(
                    s=state, a=action, r=reward, t=terminal,
                    s2=next_state, c=cost
                )
                
                steps += 1
                
                if terminal or steps >= cmd_args.max_steps:
                    break
            
            epoch_costs.append(episode_cost)
            episode_count_in_epoch += 1
            
            # 训练
            if episode_count_in_epoch % cmd_args.train_every_n_episodes == 0:
                if replay_buffer.size() >= cmd_args.batch_size:
                    model.train(
                        replay_buffer=replay_buffer,
                        iterations=cmd_args.train_iterations,
                        batch_size=cmd_args.batch_size,
                    )
            
            if episode_idx % 10 == 0 or episode_idx == cmd_args.episodes_per_epoch:
                print(f"   Rollout: {episode_idx}/{cmd_args.episodes_per_epoch}, "
                      f"cost={episode_cost:.1f}, reward={episode_reward:.1f}")
        
        avg_episode_cost = np.mean(epoch_costs)
        print(f"   ✅ Phase 1 完成")
        print(f"   📊 Episode Cost: Avg={avg_episode_cost:.2f}, "
              f"Min={np.min(epoch_costs):.2f}, Max={np.max(epoch_costs):.2f}")
        
        # ===== Phase 2: 更新λ =====
        print(f"\n📍 Phase 2: 更新RCPO参数 (论文Line 10)")
        
        old_lambda = model.lambda_penalty.item()
        model.update_lambda(avg_episode_cost)
        print(f"   λ: {old_lambda:.6f} → {model.lambda_penalty.item():.6f}")
        
        # ===== Phase 3: 统计 =====
        print(f"\n🔧 Phase 3: Training Summary")
        print(f"   iter_count: {model.iter_count}")
        print(f"   Buffer: {replay_buffer.size()} / {cmd_args.buffer_size}")
        
        # ===== Phase 4: 评估 =====
        current_metrics = evaluate(
            model=model,
            env=ros,
            scenarios=eval_scenarios,
            epoch=epoch + 1,
            max_steps=cmd_args.max_steps,
            best_metrics=best_metrics,
            save_directory=save_directory,
            model_name=model_name,
        )
        
        model.save(filename=f"{model_name}_epoch_{epoch+1:03d}", directory=save_directory)
        
        if current_metrics['is_best']:
            best_metrics = current_metrics.copy()
            epochs_since_improvement = 0
            print(f"🌟 NEW BEST MODEL! (epoch {epoch + 1})")
        else:
            epochs_since_improvement += 1
        
        print("=" * 80)
        print(f"📈 Epochs since improvement: {epochs_since_improvement}")
        print(f"🏆 Best (epoch {best_metrics['epoch']}): "
              f"Success={best_metrics['success_rate']:.3f}, "
              f"Collision={best_metrics['collision_rate']:.3f}")
        print("=" * 80)
    
    print("\n" + "=" * 80)
    print("🎉 训练完成!")
    print(f"💾 模型保存在: {save_directory}")
    print("=" * 80)


def evaluate(model, env, scenarios, epoch, max_steps, best_metrics, 
             save_directory, model_name):
    """评估函数"""
    print("\n" + "=" * 80)
    print(f"📊 Epoch {epoch} - Evaluating {len(scenarios)} scenarios")
    print("=" * 80)

    avg_reward = 0.0
    avg_cost = 0.0
    col = 0
    gl = 0
    
    for scenario in scenarios:
        count = 0
        episode_cost = 0.0
        
        latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(scenario=scenario)
        
        while count < max_steps:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            if terminal:
                break
            
            action = model.act(state)
            a_in = [(action[0] + 1) / 2, action[1]]
            
            latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            
            cost = compute_cost(latest_scan, collision, danger_threshold=0.5)
            episode_cost += cost
            
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
    """多级判断标准"""
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