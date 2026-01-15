#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD7 Training Script for Robot Navigation

基于原始train.py适配，主要改动：
1. 使用TD7替代TD3
2. 使用TD7的checkpoint训练机制
3. 保持与TD3相同的训练频率（每2 episode训练）
4. 每个epoch保存模型 + best model保存
5. 移除预训练逻辑
"""
from pathlib import Path
from datetime import datetime
import socket
import argparse
import sys
from TD7.TD7_lightweight import TD7
from ros_python import ROS_env
import torch
import numpy as np
from utils import record_eval_positions

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

def main(args=None):
    """Main training function"""
    # ✅ 解析命令行参数
    parser = argparse.ArgumentParser(description='Train TD7 for robot navigation')
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
        '--hidden-dim',
        type=int,
        default=26,
        help='Hidden layer dimension (default: 26, same as TD3)'
    )
    cmd_args = parser.parse_args(args)
    
    # 生成带时间戳的运行标识（与TensorBoard runs目录格式一致）
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    hostname = socket.gethostname()
    run_id = f"{timestamp}_{hostname}"
    
    # 模型保存目录
    model_dir_name = "TD7_lightweight"
    save_directory = Path("src/drl_navigation_ros2/models") / model_dir_name / run_id
    save_directory.mkdir(parents=True, exist_ok=True)

    # 设置日志输出到文件
    log_file = save_directory / "train_output.log"
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    model_name = "TD7_lightweight"
    
    # 训练参数配置
    action_dim = 2          # 动作维度：线速度和角速度
    max_action = 1          # 最大动作值
    state_dim = 25          # 状态维度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nr_eval_episodes = 10   # 评估episode数量
    max_epochs = cmd_args.max_epochs
    epoch = 0
    episodes_per_epoch = cmd_args.episodes_per_epoch
    episode = 0
    train_every_n = 2       # 每N个episode训练一次（与TD3保持一致）
    max_steps = 300         # 单个episode最大步数
    steps = 0
    
    print("=" * 80)
    print(f"🚀 开始TD7训练运行")
    print(f"🔧 模型类型: {model_name}")
    print(f"📁 运行ID: {run_id}")
    print(f"💾 模型保存路径: {save_directory}")
    print(f"📊 TensorBoard日志: runs/{run_id}")
    print(f"🎯 训练设置: {max_epochs} epochs × {episodes_per_epoch} episodes")
    print(f"🔄 训练频率: 每 {train_every_n} episodes")
    print(f"🧠 隐藏层维度: {cmd_args.hidden_dim}")
    print("=" * 80)

    # 初始化TD7模型
    model = TD7(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        hidden_dim=cmd_args.hidden_dim,
        save_every=100,
        load_model=False,
        save_directory=save_directory,
        model_name=model_name,
        run_id=run_id,
    )

    # 初始化ROS环境
    ros = ROS_env(
        enable_random_obstacles=True
    )
    
    # 记录评估场景
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_eval_episodes,
        save_to_file=True,
        random_seed=42,
        enable_random_obstacles=True
    )

    # 初始化环境
    latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
        lin_velocity=0.0, ang_velocity=0.0
    )
    
    # ✅ 用于跟踪 best model 的变量
    best_metrics = {
        'success_rate': -1.0,
        'collision_rate': 2.0,
        'avg_reward': -float('inf'),
        'epoch': 0
    }
    epochs_since_improvement = 0
    
    # ✅ Episode return跟踪（TD7 checkpoint机制需要）
    ep_return = 0.0
    ep_timesteps = 0

    while epoch < max_epochs:
        # 准备状态
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )
        
        # 获取动作（TD7内部会根据training_steps决定是否使用checkpoint）
        action = model.get_action(state, add_noise=True)
        
        # 执行动作（线速度从[-1,1]映射到[0,1]）
        a_in = [(action[0] + 1) / 2, action[1]]
        latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )
        
        # 准备下一状态
        next_state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )
        
        # 添加到replay buffer
        model.replay_buffer.add(state, action, reward, terminal, next_state)
        
        # 累计episode return和timesteps
        ep_return += reward
        ep_timesteps += 1

        # Episode结束处理
        if terminal or steps == max_steps:
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
            episode += 1
            
            # ✅ TD7的checkpoint训练机制
            if episode % train_every_n == 0:
                model.maybe_train_and_checkpoint(ep_timesteps, ep_return)
                
                # TensorBoard记录训练统计
                model.writer.add_scalar("train/ep_return", ep_return, model.training_steps)
                model.writer.add_scalar("train/ep_timesteps", ep_timesteps, model.training_steps)
            
            # 重置episode统计
            ep_return = 0.0
            ep_timesteps = 0
            steps = 0
        else:
            steps += 1

        # Epoch结束处理
        if (episode + 1) % episodes_per_epoch == 0:
            episode = 0
            epoch += 1
            
            # ✅ 每个epoch都保存模型
            model.save(
                filename=f"{model_name}_epoch_{epoch}",
                directory=save_directory
            )
            print(f"💾 Epoch {epoch} 模型已保存")
            
            # 评估
            current_metrics = eval_model(
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
            
            print(f"📈 Epochs since last improvement: {epochs_since_improvement}")
            print(f"🏆 Current best from epoch {best_metrics['epoch']}: "
                  f"Success={best_metrics['success_rate']:.3f}, "
                  f"Collision={best_metrics['collision_rate']:.3f}, "
                  f"Reward={best_metrics['avg_reward']:.2f}")
            print(f"📊 TD7 Training steps: {model.training_steps}")
            print("=" * 80 + "\n")
    
    # ✅ 训练结束，打印最终统计
    print("\n" + "=" * 80)
    print("🎉 TD7训练完成!")
    print(f"🏆 最佳模型来自 Epoch {best_metrics['epoch']}")
    print(f"   Success Rate: {best_metrics['success_rate']:.3f}")
    print(f"   Collision Rate: {best_metrics['collision_rate']:.3f}")
    print(f"   Avg Reward: {best_metrics['avg_reward']:.2f}")
    print(f"💾 最佳模型保存在: {save_directory}/{model_name}_best_*.pth")
    print(f"📊 总训练步数: {model.training_steps}")
    print("=" * 80)


def eval_model(model, env, scenarios, epoch, max_steps, best_metrics, save_directory, model_name):
    """
    评估函数 - 与train.py的eval()保持一致
    
    Returns:
        dict: Current evaluation metrics including 'is_best' flag
    """
    print("\n" + "=" * 80)
    print(f"📊 Epoch {epoch} - Evaluating {len(scenarios)} scenarios")
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
            # 评估时不添加噪声
            action = model.get_action(state, add_noise=False)
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
    
    # ✅ 保存best model
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
    
    # ✅ TensorBoard记录 - 保持与TD3一致的命名
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)
    
    print("=" * 80)
    
    return {
        'success_rate': avg_goal,
        'collision_rate': avg_col,
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