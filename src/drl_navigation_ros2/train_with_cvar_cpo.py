#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVaR-CPO训练脚本（修正版）
基于论文 "CVaR-Constrained Policy Optimization for Safe Reinforcement Learning"

核心特点：
1. 状态扩展 s̄ = (s, e_t)
2. Quantile Regression Network估计cost分布（128 quantiles）
3. 交替更新VaR参数u和Lagrangian乘子w
"""

from pathlib import Path
from datetime import datetime
import socket
import argparse
import sys

from TD3.TD3_cvar_cpo import TD3_CVaRCPO
from cvar_replay_buffer import CVaRReplayBuffer
import torch
import numpy as np


class TeeStream:
    """同时将 stdout/stderr 输出写入日志文件。"""

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


def compute_cost(laser_scan, collision, safety_threshold=0.4):
    """
    计算即时cost（基于论文的连续cost定义）
    
    Args:
        laser_scan: 激光雷达数据
        collision: 是否碰撞
        safety_threshold: 安全距离阈值
    
    Returns:
        cost: 连续值 in [0, inf)
    """
    min_distance = min(laser_scan)
    
    if collision:
        # 碰撞：极高cost
        cost = 100.0
    elif min_distance < safety_threshold:
        # 接近障碍物：cost随距离递增
        cost = max(0, safety_threshold - min_distance) * 10
    else:
        # 安全：无cost
        cost = 0.0
    
    return cost


def main(args=None):
    """Main training function with CVaR-CPO"""
    parser = argparse.ArgumentParser(description='Train TD3 with CVaR-CPO')
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=100,
        help='Maximum number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--episodes-per-epoch',
        type=int,
        default=70,
        help='Number of episodes per epoch (default: 70)'
    )
    parser.add_argument(
        '--cvar-alpha',
        type=float,
        default=0.1,
        help='CVaR confidence level α (default: 0.1, top 10%% worst cases)'
    )
    parser.add_argument(
        '--cost-threshold',
        type=float,
        default=25.0,
        help='Cost constraint threshold b (default: 25.0)'
    )
    parser.add_argument(
        '--n-quantiles',
        type=int,
        default=128,  # ✅ 修正：默认值改为128
        help='Number of quantiles for distributional network (default: 128)'
    )
    parser.add_argument(
        '--safety-threshold',
        type=float,
        default=0.4,
        help='Safety distance threshold for cost computation (default: 0.4m)'
    )
    cmd_args = parser.parse_args(args)
    
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    hostname = socket.gethostname()
    run_id = f"{timestamp}_{hostname}"
    
    model_dir_name = "TD3_cvar_cpo"
    model_name = "TD3_cvar_cpo"
    
    save_directory = Path("models") / model_dir_name / run_id
    save_directory.mkdir(parents=True, exist_ok=True)

    log_file = save_directory / "train_output.log"
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    
    # 训练参数配置
    action_dim = 2
    max_action = 1
    state_dim = 25  # 原始状态维度（不包含e_t）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nr_eval_episodes = 10
    max_epochs = cmd_args.max_epochs
    episodes_per_epoch = cmd_args.episodes_per_epoch
    
    # CVaR-CPO特有参数
    cvar_alpha = cmd_args.cvar_alpha
    cost_threshold = cmd_args.cost_threshold
    n_quantiles = cmd_args.n_quantiles
    safety_threshold = cmd_args.safety_threshold
    
    # 训练频率：每个epoch训练一次（论文Algorithm 1的两阶段更新）
    training_iterations_per_epoch = 500 * (episodes_per_epoch // 2)
    batch_size = 40
    max_steps = 300
    
    print("=" * 80)
    print(f"🚀 开始CVaR-CPO训练（完全论文对齐版）")
    print(f"📁 运行ID: {run_id}")
    print(f"💾 模型保存路径: {save_directory}")
    print(f"📊 TensorBoard日志: runs/{run_id}")
    print(f"📝 日志文件: {log_file}")
    print(f"🎯 训练设置: {max_epochs} epochs × {episodes_per_epoch} episodes")
    print(f"🔄 训练频率: {training_iterations_per_epoch} iterations/epoch")
    print(f"\n📊 CVaR参数:")
    print(f"   α (confidence): {cvar_alpha}")
    print(f"   Cost threshold b: {cost_threshold}")
    print(f"   Quantiles: {n_quantiles}")
    print(f"   Safety threshold: {safety_threshold}m")
    print("=" * 80)

    # 模型初始化
    model = TD3_CVaRCPO(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        n_quantiles=n_quantiles,
        cvar_alpha=cvar_alpha,
        cost_threshold=cost_threshold,
        save_every=0,
        load_model=False,
        save_directory=save_directory,
        model_name=model_name,
        run_id=run_id,
    )

    # 导入ROS环境和eval场景
    from ros_python import ROS_env
    from utils import record_eval_positions
    # from eval_scenario_map_generator import generate_scenario_maps

    ros = ROS_env(enable_random_obstacles=True)
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_eval_episodes,
        save_to_file=True,
        random_seed=42,
        enable_random_obstacles=True
    )
    # generate_scenario_maps(eval_scenarios, scenario_tag="cvar_cpo")
    
    # Replay Buffer (CVaR-CPO使用26维状态，不支持从25维baseline预训练)
    replay_buffer = CVaRReplayBuffer(buffer_size=500000, random_seed=42)
    
    # Best model 追踪
    best_metrics = {
        'success_rate': -1.0,
        'collision_rate': 2.0,
        'avg_reward': -float('inf'),
        'epoch': 0
    }
    epochs_since_improvement = 0
    
    epoch = 0
    
    # ===== 主训练循环 (Epoch-level，按照论文Algorithm 1) =====
    while epoch < max_epochs:
        print(f"\n{'='*80}")
        print(f"🎯 Epoch {epoch + 1}/{max_epochs}")
        print(f"{'='*80}")
        
        # ===== Phase 1: Rollout Episodes =====
        print(f"\n📍 Phase 1: Rollout ({episodes_per_epoch} episodes)")
        
        epoch_costs = []  # 收集本epoch的所有episode cost
        
        for episode_in_epoch in range(1, episodes_per_epoch + 1):
            # Reset 环境
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()

            steps = 0
            episode_cost = 0.0

            # Episode rollout
            while True:
                # 准备原始状态（25维）
                state, terminal = model.prepare_state(
                    latest_scan, distance, cos, sin, collision, goal, a
                )

                # 获取动作（内部会自动扩展状态）
                action = model.get_action(state, model.var_u.item(), add_noise=True)
                a_in = [(action[0] + 1) / 2, action[1]]

                # 执行动作
                latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
                    lin_velocity=a_in[0], ang_velocity=a_in[1]
                )

                # 计算cost
                cost = compute_cost(latest_scan, collision, safety_threshold)
                episode_cost += cost

                # 准备next state
                next_state, terminal = model.prepare_state(
                    latest_scan, distance, cos, sin, collision, goal, a
                )

                # 存储到buffer（存储原始25维状态，不包含e_t）
                replay_buffer.add(state, action, reward, terminal, next_state, cost)

                steps += 1

                if terminal or steps >= max_steps:
                    break
            
            epoch_costs.append(episode_cost)
            
            # 进度显示
            if episode_in_epoch % 10 == 0 or episode_in_epoch == episodes_per_epoch:
                print(f"   Rollout 进度: {episode_in_epoch}/{episodes_per_epoch}")
        
        avg_episode_cost = np.mean(epoch_costs)
        max_episode_cost = np.max(epoch_costs)
        
        print(f"   ✅ Phase 1 完成")
        print(f"   📊 Episode Cost统计: Avg={avg_episode_cost:.2f}, Max={max_episode_cost:.2f}")
        
        # ===== Phase 2: 更新VaR和Lagrangian乘子（论文Algorithm 1的Step 2） =====
        print(f"\n📍 Phase 2: 更新CVaR参数")
        
        old_var_u = model.var_u.item()
        old_lambda_w = model.lambda_w.item()
        
        # ✅ 修正：传入完整的epoch_costs列表以准确估计P(C >= u_k)
        model.update_var_and_lambda(avg_episode_cost, epoch_costs)
        
        print(f"   VaR u: {old_var_u:.3f} → {model.var_u.item():.3f}")
        print(f"   Lagrangian w: {old_lambda_w:.3f} → {model.lambda_w.item():.3f}")
        
        # ===== Phase 3: 批量训练网络（论文Algorithm 1的Step 1） =====
        print(f"\n🔧 Phase 3: Network Training")
        
        if replay_buffer.size() >= batch_size:
            model.train(
                replay_buffer=replay_buffer,
                iterations=training_iterations_per_epoch,
                batch_size=batch_size,
            )
            print(f"   ✅ 训练完成: {training_iterations_per_epoch} iterations")
        else:
            print(f"   ⚠️  Buffer不足，跳过训练 ({replay_buffer.size()}/{batch_size})")
        
        print(f"   📊 当前 iter_count: {model.iter_count}")
        print(f"   📊 Buffer大小: {replay_buffer.size()} / 500000")
        
        # ===== Phase 4: Eval =====
        current_metrics = eval(
            model=model,
            env=ros,
            scenarios=eval_scenarios,
            epoch=epoch + 1,
            max_steps=max_steps,
            best_metrics=best_metrics,
            save_directory=save_directory,
            model_name=model_name,
            safety_threshold=safety_threshold,
        )

        if current_metrics['is_best']:
            best_metrics = current_metrics.copy()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        print(f"📈 Epochs since last improvement: {epochs_since_improvement}")
        print(f"🏆 Current best from epoch {best_metrics['epoch']}: "
              f"Success={best_metrics['success_rate']:.3f}, "
              f"Collision={best_metrics['collision_rate']:.3f}, "
              f"Reward={best_metrics['avg_reward']:.2f}")
        print("=" * 80 + "\n")
        
        # 保存模型
        model.save(
            filename=f"{model_name}_epoch_{epoch+1:03d}",
            directory=save_directory
        )
        
        epoch += 1

    # 训练结束
    print("\n" + "=" * 80)
    print("🎉 CVaR-CPO训练完成!")
    print(f"💾 模型保存在: {save_directory}")
    print("=" * 80)


def eval(model, env, scenarios, epoch, max_steps, best_metrics, save_directory, 
         model_name, safety_threshold):
    """
    Evaluation function for CVaR-CPO
    """
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
        
        latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(
            scenario=scenario
        )
        
        while count < max_steps:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            if terminal:
                break
            
            # 评估时使用确定性动作
            action = model.act(state, model.var_u.item())
            a_in = [(action[0] + 1) / 2, action[1]]
            
            latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            
            # 计算cost
            cost = compute_cost(latest_scan, collision, safety_threshold)
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
        current_success=avg_goal,
        current_collision=avg_col,
        current_reward=avg_reward,
        best_success=best_metrics['success_rate'],
        best_collision=best_metrics['collision_rate'],
        best_reward=best_metrics['avg_reward']
    )
    
    # 保存当前epoch模型
    model.save(
        filename=f"{model_name}_epoch_{epoch:03d}",
        directory=save_directory
    )
    print(f"💾 已保存: {model_name}_epoch_{epoch:03d}_*.pth")
    
    # 如果是best model
    if is_best:
        model.save(
            filename=f"{model_name}_best",
            directory=save_directory
        )
        print(f"🌟 NEW BEST MODEL! (epoch {epoch})")
    else:
        print(f"📊 Not best (Best from epoch {best_metrics['epoch']})")
    
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


def is_better_model(current_success, current_collision, current_reward,
                    best_success, best_collision, best_reward):
    """多级判断标准"""
    if current_success > best_success:
        return True
    elif current_success < best_success:
        return False
    
    if current_collision < best_collision:
        return True
    elif current_collision > best_collision:
        return False
    
    if current_reward > best_reward:
        return True
    else:
        return False


if __name__ == "__main__":
    main()