#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVaR-CPO训练脚本 (Ablation Study Version)

训练流程与baseline保持一致：
- 每2个episodes训练一次
- 每次训练500 iterations
- 每个iteration的batch_size为40
- 每个epoch 70 episodes，共35次训练 = 17,500 batches/epoch

CVaR-CPO特有：
- 每个epoch结束后更新VaR参数u和Lagrangian w
- 追踪真实的e_t演化（论文公式）
"""

from pathlib import Path
from datetime import datetime
import socket
import argparse
import sys

import torch
import numpy as np

# 导入模型和buffer
from TD3.TD3_cvar_cpo import TD3_CVaRCPO, BASELINE_BATCH_SIZE, BASELINE_BUFFER_SIZE, BASELINE_GAMMA
from cvar_replay_buffer import CVaRReplayBuffer


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


def compute_cost(laser_scan, collision, danger_threshold=0.5):
    """
    进入危险区域即产生cost（与论文hazard概念一致）
    danger_threshold > collision_threshold，这样：
    - 进入危险区域：cost=1，但不结束episode
    - 真正碰撞：episode结束
    """
    min_distance = min(laser_scan)
    
    if collision:
        cost = 1.0  # 碰撞也算一次cost
    elif min_distance < danger_threshold:
        cost = 1.0  # 进入危险区域
    else:
        cost = 0.0
    
    return cost


def main(args=None):
    """主训练函数"""
    parser = argparse.ArgumentParser(description='Train TD3 with CVaR-CPO (Ablation Study)')
    
    # 训练设置（与baseline一致）
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--episodes-per-epoch', type=int, default=70)
    parser.add_argument('--train-every-n-episodes', type=int, default=2, 
                        help='每N个episode训练一次（baseline: 2）')
    parser.add_argument('--train-iterations', type=int, default=500,
                        help='每次训练的iteration数（baseline: 500）')
    parser.add_argument('--batch-size', type=int, default=BASELINE_BATCH_SIZE,
                        help=f'Batch size（baseline: {BASELINE_BATCH_SIZE}）')
    parser.add_argument('--buffer-size', type=int, default=BASELINE_BUFFER_SIZE,
                        help=f'Buffer size（baseline: {BASELINE_BUFFER_SIZE}）')
    
    # 环境参数
    parser.add_argument('--safety-threshold', type=float, default=0.4,
                        help='Safety distance for cost computation (default: 0.4m)')
    parser.add_argument('--max-steps', type=int, default=300,
                        help='Maximum steps per episode')
    
    # 评估
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    
    cmd_args = parser.parse_args(args)
    
    # 运行ID和路径
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    hostname = socket.gethostname()
    run_id = f"{timestamp}_{hostname}_cvar_cpo_ablation"
    
    model_dir_name = "TD3_cvar_cpo"
    model_name = "TD3_cvar_cpo"
    
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
    
    # 计算每epoch的训练次数（与baseline一致）
    trains_per_epoch = cmd_args.episodes_per_epoch // cmd_args.train_every_n_episodes
    total_iterations_per_epoch = trains_per_epoch * cmd_args.train_iterations
    
    print("=" * 80)
    print("🚀 CVaR-CPO训练 (Ablation Study Version)")
    print("=" * 80)
    print(f"📁 运行ID: {run_id}")
    print(f"💾 保存路径: {save_directory}")
    print(f"\n📋 训练流程（与baseline一致）:")
    print(f"   每 {cmd_args.train_every_n_episodes} episodes 训练一次")
    print(f"   每次训练 {cmd_args.train_iterations} iterations")
    print(f"   每iteration batch_size = {cmd_args.batch_size}")
    print(f"   每epoch: {cmd_args.episodes_per_epoch} episodes → {trains_per_epoch} 次训练")
    print(f"   每epoch总batches: {total_iterations_per_epoch:,}")
    print(f"\n📋 CVaR-CPO特有参数（按论文设置）:")
    print(f"   在模型初始化时打印")
    print("=" * 80)

    # 模型初始化
    model = TD3_CVaRCPO(
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
    
    # Replay Buffer
    replay_buffer = CVaRReplayBuffer(buffer_size=cmd_args.buffer_size, random_seed=42)
    
    # Best model追踪
    best_metrics = {
        'success_rate': -1.0,
        'collision_rate': 2.0,
        'avg_reward': -float('inf'),
        'epoch': 0
    }
    epochs_since_improvement = 0
    
    gamma = BASELINE_GAMMA
    
    # ===== 主训练循环 =====
    for epoch in range(cmd_args.max_epochs):
        print(f"\n{'='*80}")
        print(f"🎯 Epoch {epoch + 1}/{cmd_args.max_epochs}")
        print(f"{'='*80}")
        
        # ===== Phase 1: Rollout =====
        print(f"\n📍 Phase 1: Rollout ({cmd_args.episodes_per_epoch} episodes)")
        
        epoch_costs = []  # 收集整个epoch的episode costs
        episode_count_in_epoch = 0
        
        for episode_idx in range(1, cmd_args.episodes_per_epoch + 1):
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
            
            steps = 0
            episode_cost = 0.0
            episode_reward = 0.0
            
            # 论文: e_0 = u^k（每个episode开始时初始化）
            e_t = model.var_u.item()
            episode_e_t_min = e_t
            
            while True:
                state, terminal = model.prepare_state(
                    latest_scan, distance, cos, sin, collision, goal, a
                )
                
                # 获取动作（使用当前e_t）
                action = model.get_action(state, e_t, add_noise=True)
                a_in = [(action[0] + 1) / 2, action[1]]
                
                # 执行动作
                latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
                    lin_velocity=a_in[0], ang_velocity=a_in[1]
                )
                
                # 计算cost（danger_threshold=0.5，大于碰撞阈值0.4）
                cost = compute_cost(latest_scan, collision, danger_threshold=0.5)
                episode_cost += cost
                episode_reward += reward
                
                # 论文公式: e_{t+1} = (e_t - C(s_t, a_t)) / γ
                next_e_t = (e_t - cost) / gamma
                
                next_state, terminal = model.prepare_state(
                    latest_scan, distance, cos, sin, collision, goal, a
                )
                
                # 存储（包含真实的e_t和next_e_t）
                replay_buffer.add(
                    s=state, a=action, r=reward, t=terminal,
                    s2=next_state, c=cost, e_t=e_t, next_e_t=next_e_t
                )
                
                # 更新e_t
                e_t = next_e_t
                episode_e_t_min = min(episode_e_t_min, e_t)
                steps += 1
                
                if terminal or steps >= cmd_args.max_steps:
                    break
            
            epoch_costs.append(episode_cost)
            episode_count_in_epoch += 1
            
            # 每N个episode训练一次（与baseline一致）
            if episode_count_in_epoch % cmd_args.train_every_n_episodes == 0:
                if replay_buffer.size() >= cmd_args.batch_size:
                    model.train(
                        replay_buffer=replay_buffer,
                        iterations=cmd_args.train_iterations,
                        batch_size=cmd_args.batch_size,
                    )
            
            # 进度显示
            if episode_idx % 10 == 0 or episode_idx == cmd_args.episodes_per_epoch:
                print(f"   Rollout: {episode_idx}/{cmd_args.episodes_per_epoch}, "
                      f"cost={episode_cost:.1f}, reward={episode_reward:.1f}")
        
        avg_episode_cost = np.mean(epoch_costs)
        max_episode_cost = np.max(epoch_costs)
        min_episode_cost = np.min(epoch_costs)
        
        print(f"   ✅ Phase 1 完成")
        print(f"   📊 Episode Cost: Avg={avg_episode_cost:.2f}, Min={min_episode_cost:.2f}, Max={max_episode_cost:.2f}")
        
        # ===== Phase 2: 更新CVaR参数 =====
        print(f"\n📍 Phase 2: 更新CVaR参数")
        
        old_var_u = model.var_u.item()
        old_lambda_w = model.lambda_w.item()
        
        model.update_var_and_lambda(avg_episode_cost, epoch_costs)
        
        print(f"   VaR u: {old_var_u:.4f} → {model.var_u.item():.4f}")
        print(f"   Lagrangian w: {old_lambda_w:.4f} → {model.lambda_w.item():.4f}")
        
        # ===== Phase 3: 统计信息 =====
        print(f"\n🔧 Phase 3: Network Training Summary")
        print(f"   ✅ 训练完成: {total_iterations_per_epoch} iterations (distributed)")
        print(f"   📊 当前 iter_count: {model.iter_count}")
        print(f"   📊 Buffer大小: {replay_buffer.size()} / {cmd_args.buffer_size}")
        
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
            safety_threshold=cmd_args.safety_threshold,
            gamma=gamma,
        )
        
        # 保存epoch checkpoint
        model.save(filename=f"{model_name}_epoch_{epoch+1:03d}", directory=save_directory)
        
        if current_metrics['is_best']:
            best_metrics = current_metrics.copy()
            epochs_since_improvement = 0
            print(f"💾 已保存: {model_name}_epoch_{epoch+1:03d}_*.pth")
            print(f"🌟 NEW BEST MODEL! (epoch {epoch + 1})")
        else:
            epochs_since_improvement += 1
            print(f"💾 已保存: {model_name}_epoch_{epoch+1:03d}_*.pth")
        
        print("=" * 80)
        print(f"📈 Epochs since last improvement: {epochs_since_improvement}")
        print(f"🏆 Current best from epoch {best_metrics['epoch']}: "
              f"Success={best_metrics['success_rate']:.3f}, "
              f"Collision={best_metrics['collision_rate']:.3f}, "
              f"Reward={best_metrics['avg_reward']:.2f}")
        print("=" * 80)
    
    print("\n" + "=" * 80)
    print("🎉 训练完成!")
    print(f"💾 模型保存在: {save_directory}")
    print("=" * 80)


def evaluate(model, env, scenarios, epoch, max_steps, best_metrics, 
             save_directory, model_name, safety_threshold, gamma):
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
        avg_goal, avg_col, avg_reward,
        best_metrics['success_rate'], best_metrics['collision_rate'], best_metrics['avg_reward']
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


def is_better_model(curr_success, curr_collision, curr_reward,
                    best_success, best_collision, best_reward):
    """多级判断标准（与baseline一致）"""
    if curr_success > best_success:
        return True
    elif curr_success < best_success:
        return False
    
    if curr_collision < best_collision:
        return True
    elif curr_collision > best_collision:
        return False
    
    return curr_reward > best_reward


if __name__ == "__main__":
    main()