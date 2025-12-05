#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import socket
import argparse

from TD3.TD3 import TD3 as TD3_Original
from TD3.TD3_lightweight import TD3 as TD3_Lightweight
from SAC.SAC import SAC
from ros_python import ROS_env
from replay_buffer import ReplayBuffer
import torch
import numpy as np
from utils import record_eval_positions
from pretrain_utils import Pretraining

from eval_scenario_map_generator import generate_scenario_maps
from training_scenario_saver import save_training_episode_map
# from eval_trajectory_collector import collect_eval_trajectories
from training_trajectory_collector import TrainingTrajectoryCollector, collect_training_step
from reachability_verifier import verify_trajectories_reachability


def main(args=None):
    """Main training function"""
    # ✅ 解析命令行参数
    parser = argparse.ArgumentParser(description='Train TD3 for robot navigation')
    parser.add_argument(
        '--model-type',
        type=str,
        default='lightweight',
        choices=['TD3', 'lightweight'],
        help='Model type: "TD3" (original) or "lightweight" (default: lightweight)'
    )
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
    cmd_args = parser.parse_args(args)
    
    # 生成带时间戳的运行标识（与TensorBoard runs目录格式一致）
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    hostname = socket.gethostname()
    run_id = f"{timestamp}_{hostname}"
    
    # ✅ 根据命令行参数选择模型目录
    model_dir_name = "TD3" if cmd_args.model_type == "TD3" else "TD3_lightweight"
    save_directory = Path("src/drl_navigation_ros2/models") / model_dir_name / run_id
    save_directory.mkdir(parents=True, exist_ok=True)
    
    # ✅ 根据命令行参数选择模型类
    TD3_Class = TD3_Original if cmd_args.model_type == "TD3" else TD3_Lightweight
    model_name = "TD3" if cmd_args.model_type == "TD3" else "TD3_lightweight"
    
    # 训练参数配置
    action_dim = 2
    max_action = 1
    state_dim = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nr_eval_episodes = 10
    max_epochs = cmd_args.max_epochs
    epoch = 0
    episodes_per_epoch = cmd_args.episodes_per_epoch
    episode = 0
    train_every_n = 2
    training_iterations = 500
    batch_size = 40
    max_steps = 300
    steps = 0
    load_saved_buffer = True
    pretrain = True
    pretraining_iterations = 50
    
    print("=" * 80)
    print(f"🚀 开始新的训练运行")
    print(f"🔧 模型类型: {model_name}")
    print(f"📁 运行ID: {run_id}")
    print(f"💾 模型保存路径: {save_directory}")
    print(f"📊 TensorBoard日志: runs/{run_id}")
    print(f"🎯 训练设置: {max_epochs} epochs × {episodes_per_epoch} episodes")
    print("=" * 80)

    model = TD3_Class(
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

    ros = ROS_env(enable_random_obstacles=True)
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_eval_episodes,
        save_to_file=True,
        random_seed=42,
        enable_random_obstacles=True
    )

    # 生成评估场景地图
    generate_scenario_maps(eval_scenarios, scenario_tag="8_polar")

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
    
    epoch = 0
    episode = 0
    episode_in_epoch = 0

    # ✅ 用于跟踪 best model 的变量
    best_metrics = {
        'success_rate': -1.0,
        'collision_rate': 2.0,
        'avg_reward': -float('inf'),
        'epoch': 0
    }
    epochs_since_improvement = 0
    
    trajectory_collector = TrainingTrajectoryCollector(save_dir="training_trajectories_8_polar")

    # 保存第一个 episode 的初始场景
    episode_in_epoch = 1
    save_path = save_training_episode_map(
        env=ros,
        epoch=1,
        episode=episode_in_epoch,
        save_dir=f"train_scenario_8_polar"
    )
    print(f"   💾 已保存初始训练场景: epoch_001/episode_001.json")

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
        
        # ✅ 收集训练步骤
        collect_training_step(
            collector=trajectory_collector,
            state=state,
            action=action,
            reward=reward,
            done=terminal,
            latest_scan=latest_scan,
            distance=distance,
            cos=cos,
            sin=sin,
            collision=collision,
            goal=goal,
            prev_action=a
        )            

        if terminal or steps == max_steps:
            episode += 1
            
            # ✅ 关键修改：先用当前的 episode_in_epoch 保存轨迹
            current_epoch = epoch + 1
            current_episode = episode_in_epoch  # 使用当前值（第一次是1，后面会递增）
            
            if current_episode <= episodes_per_epoch:
                try:
                    traj_path = trajectory_collector.save_trajectory(
                        epoch=current_epoch,
                        episode=current_episode  # 使用保存的当前值
                    )
                    if current_episode == 1 or current_episode == episodes_per_epoch:
                        if traj_path:
                            print(f"   📊 已保存训练轨迹: epoch_{current_epoch:03d}/episode_{current_episode:03d}.pkl")
                except Exception as e:
                    print(f"   ⚠️  保存训练轨迹失败: {e}")
            
            # ✅ 重置轨迹收集器
            trajectory_collector.reset_trajectory()
            
            # ✅ 然后 reset 环境
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
            
            # ✅ 递增 episode_in_epoch（为下一个 episode 准备）
            episode_in_epoch += 1
            
            # ✅ 保存场景地图（reset 之后的场景，用于下一个 episode）
            if episode_in_epoch <= episodes_per_epoch:
                save_path = save_training_episode_map(
                    env=ros,
                    epoch=current_epoch,
                    episode=episode_in_epoch,
                    save_dir=f"train_scenario_8_polar"
                )
                if episode_in_epoch == 2 or episode_in_epoch == episodes_per_epoch:
                    print(f"   💾 已保存训练场景: epoch_{current_epoch:03d}/episode_{episode_in_epoch:03d}.json")

            if episode % train_every_n == 0:
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )

            steps = 0
            
            # ✅ 判断是否进入 eval
            if episode % episodes_per_epoch == 0:
                current_metrics = eval(
                    model=model,
                    env=ros,
                    scenarios=eval_scenarios,
                    epoch=epoch + 1,
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
                
                print(f"📈 Epochs since last improvement: {epochs_since_improvement}")
                print(f"🏆 Current best from epoch {best_metrics['epoch']}: "
                      f"Success={best_metrics['success_rate']:.3f}, "
                      f"Collision={best_metrics['collision_rate']:.3f}, "
                      f"Reward={best_metrics['avg_reward']:.2f}")
                print("=" * 80 + "\n")
                
                episode = 0
                epoch += 1
                episode_in_epoch = 1

                # ✅ eval 后重新 reset 到新的训练场景并保存
                latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
                if epoch < max_epochs:  # 确保不是最后一个 epoch
                    save_path = save_training_episode_map(
                        env=ros,
                        epoch=epoch + 1,
                        episode=episode_in_epoch,
                        save_dir=f"train_scenario_8_polar"
                    )
                    print(f"   💾 已保存训练场景: epoch_{epoch+1:03d}/episode_001.json")                
        else:
            steps += 1

    # ✅ 训练结束，打印最终统计
    print("\n" + "=" * 80)
    print("🎉 训练完成!")
    print(f"🏆 最佳模型来自 Epoch {best_metrics['epoch']}")
    print(f"   Success Rate: {best_metrics['success_rate']:.3f}")
    print(f"   Collision Rate: {best_metrics['collision_rate']:.3f}")
    print(f"   Avg Reward: {best_metrics['avg_reward']:.2f}")
    print(f"💾 最佳模型保存在: {save_directory}/{model_name}_best_*.pth")
    print("=" * 80)

def eval(model, env, scenarios, epoch, max_steps, best_metrics, save_directory, model_name):
    """
    Function to run evaluation
    
    Returns:
        dict: Current evaluation metrics including 'is_best' flag
    """
    print("\n" + "=" * 80)
    print(f"📊 Epoch {epoch} - Evaluating {len(scenarios)} scenarios")
    print("=" * 80)

    avg_reward = 0.0
    col = 0
    gl = 0
    
    # ✅ 使用原始的 eval 逻辑
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

    # 多级判断标准
    is_best = is_better_model(
        current_success=avg_goal,
        current_collision=avg_col,
        current_reward=avg_reward,
        best_success=best_metrics['success_rate'],
        best_collision=best_metrics['collision_rate'],
        best_reward=best_metrics['avg_reward']
    )
    
    # ✅ 只在是 best model 时才保存
    if is_best:
        model.save(
            filename=f"{model_name}_best",
            directory=save_directory
        )
        print(f"🌟 NEW BEST MODEL! 已保存")
        print(f"   Improvements:")
        if avg_goal > best_metrics['success_rate']:
            print(f"      Success: {best_metrics['success_rate']:.3f} → {avg_goal:.3f} ⬆️")
        elif avg_goal == best_metrics['success_rate'] and avg_col < best_metrics['collision_rate']:
            print(f"      Same success, lower collision: {best_metrics['collision_rate']:.3f} → {avg_col:.3f} ⬇️")
        elif avg_goal == best_metrics['success_rate'] and avg_col == best_metrics['collision_rate']:
            print(f"      Same success & collision, higher reward: {best_metrics['avg_reward']:.2f} → {avg_reward:.2f} ⬆️")
    else:
        print(f"📊 Not best (Best from epoch {best_metrics['epoch']})")
    
    # ✅ TensorBoard 记录 - 保持与原始代码完全一致的命名
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)
    
    print("=" * 80)
    
    return {
        'success_rate': avg_goal,      # 内部统一用success_rate，但TensorBoard用avg_goal
        'collision_rate': avg_col,      # 内部统一用collision_rate，但TensorBoard用avg_col
        'avg_reward': avg_reward,
        'epoch': epoch,
        'is_best': is_best
    }


def is_better_model(current_success, current_collision, current_reward,
                    best_success, best_collision, best_reward):
    """
    多级判断标准，打破平局
    
    优先级：
    1. Success rate 更高 → 更好
    2. Success rate 相同，collision rate 更低 → 更好
    3. Success rate 和 collision rate 都相同，avg reward 更高 → 更好
    """
    # 第一优先级：成功率
    if current_success > best_success:
        return True
    elif current_success < best_success:
        return False
    
    # 成功率相同，第二优先级：碰撞率（越低越好）
    if current_collision < best_collision:
        return True
    elif current_collision > best_collision:
        return False
    
    # 成功率和碰撞率都相同，第三优先级：平均奖励
    if current_reward > best_reward:
        return True
    else:
        return False


if __name__ == "__main__":
    main()