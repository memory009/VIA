#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for TD3_lightweight_BFTQ

基于train.py的BFTQ训练脚本
主要修改:
1. 使用TD3_lightweight_BFTQ网络
2. 使用ReplayBufferBFTQ (支持budget和cost)
3. 每个episode采样不同的budget
4. 保存每个epoch的checkpoint + 最佳模型
5. 测试多个budget值的性能
"""

from pathlib import Path
from datetime import datetime
import socket
import argparse
import sys

from TD3.TD3_lightweight_BFTQ import TD3_BFTQ
from ros_python import ROS_env
from replay_buffer_bftq import ReplayBufferBFTQ
import torch
import numpy as np
from utils import record_eval_positions


class TeeStream:
    """重定向输出到文件和终端"""
    def __init__(self, stream, log_path):
        self.terminal = stream
        self.stream = stream
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "a", buffering=1, encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main(args=None):
    """Main training function for BFTQ"""
    # ✅ 解析命令行参数
    parser = argparse.ArgumentParser(description='Train TD3_BFTQ for robot navigation')
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
        '--save-every-n-epochs',
        type=int,
        default=1,
        help='Save checkpoint every N epochs (default: 1 = save all epochs)'
    )
    parser.add_argument(
        '--budget-min',
        type=float,
        default=0.0,
        help='Minimum budget value (default: 0.0)'
    )
    parser.add_argument(
        '--budget-max',
        type=float,
        default=1.0,
        help='Maximum budget value (default: 1.0)'
    )
    parser.add_argument(
        '--lambda-cost',
        type=float,
        default=1.0,
        help='Lagrangian cost penalty coefficient (default: 1.0)'
    )
    cmd_args = parser.parse_args(args)

    # 生成带时间戳的运行标识
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    hostname = socket.gethostname()
    run_id = f"{timestamp}_{hostname}_bftq"

    # 模型保存目录
    model_dir_name = "TD3_BFTQ_8obs"
    save_directory = Path("src/drl_navigation_ros2/models") / model_dir_name / run_id
    save_directory.mkdir(parents=True, exist_ok=True)

    # ✅ 重定向输出到log文件
    log_file = save_directory / "train_output.log"
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    model_name = "TD3_BFTQ"

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
    save_every_n_epochs = cmd_args.save_every_n_epochs

    # BFTQ特定参数
    budget_range = (cmd_args.budget_min, cmd_args.budget_max)

    print("=" * 80)
    print(f"🚀 开始新的BFTQ训练运行")
    print(f"🔧 模型类型: {model_name}")
    print(f"📁 运行ID: {run_id}")
    print(f"💾 模型保存路径: {save_directory}")
    print(f"📊 TensorBoard日志: runs/{run_id}")
    print(f"🎯 训练设置: {max_epochs} epochs × {episodes_per_epoch} episodes")
    print(f"💰 Budget范围: {budget_range}")
    print(f"⚖️  拉格朗日系数: {cmd_args.lambda_cost}")
    print(f"💾 Checkpoint保存频率: 每{save_every_n_epochs}个epoch")
    print("=" * 80)

    model = TD3_BFTQ(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        budget_range=budget_range,
        lambda_cost=cmd_args.lambda_cost,
        save_every=100,
        load_model=False,
        save_directory=save_directory,
        model_name=model_name,
        run_id=run_id,
    )

    ros = ROS_env(
        enable_random_obstacles=True
    )
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_eval_episodes,
        save_to_file=True,
        random_seed=42,
        enable_random_obstacles=True
    )

    # BFTQ使用新的replay buffer
    replay_buffer = ReplayBufferBFTQ(buffer_size=5e3, random_seed=42)

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

    # ✅ BFTQ核心: 每个episode采样不同的budget
    current_budget = model.sample_budget()
    print(f"\n📊 Episode {episode}: Budget = {current_budget:.3f}")

    while epoch < max_epochs:
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )

        # ✅ 使用当前episode的budget获取动作
        action = model.get_action(state, current_budget, add_noise=True)
        a_in = [(action[0] + 1) / 2, action[1]]

        latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )
        next_state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )

        # ✅ 计算成本信号 (碰撞 = 1, 无碰撞 = 0)
        cost = model.compute_cost(reward, collision)

        # ✅ 添加到replay buffer (包含budget和cost)
        replay_buffer.add(state, action, reward, terminal, next_state, current_budget, cost)

        if terminal or steps == max_steps:
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
            episode += 1

            # ✅ 每个新episode重新采样budget
            current_budget = model.sample_budget()
            if episode % 10 == 0:  # 每10个episode打印一次
                print(f"📊 Episode {episode}: Budget = {current_budget:.3f}")

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

            # ✅ 评估模型（测试多个budget值）
            current_metrics = eval_multi_budget(
                model=model,
                env=ros,
                scenarios=eval_scenarios,
                epoch=epoch,
                max_steps=max_steps,
                budget_test_values=[0.0, 0.25, 0.5, 0.75, 1.0],  # 测试5个不同的budget
            )

            # ✅ 判断是否为最佳模型
            is_best = is_better_model(
                current_success=current_metrics['success_rate'],
                current_collision=current_metrics['collision_rate'],
                current_reward=current_metrics['avg_reward'],
                best_success=best_metrics['success_rate'],
                best_collision=best_metrics['collision_rate'],
                best_reward=best_metrics['avg_reward']
            )

            # ✅ 保存checkpoint (每N个epoch)
            if epoch % save_every_n_epochs == 0:
                model.save(
                    filename=model_name,
                    directory=save_directory,
                    epoch=epoch
                )
                print(f"💾 Checkpoint saved: epoch_{epoch:03d}")

            # ✅ 保存最佳模型
            if is_best:
                model.save(
                    filename=f"{model_name}_best",
                    directory=save_directory
                )
                print(f"🌟 NEW BEST MODEL! 已保存")
                best_metrics = {
                    'success_rate': current_metrics['success_rate'],
                    'collision_rate': current_metrics['collision_rate'],
                    'avg_reward': current_metrics['avg_reward'],
                    'epoch': epoch
                }
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            print(f"📈 Epochs since last improvement: {epochs_since_improvement}")
            print(f"🏆 Current best from epoch {best_metrics['epoch']}: "
                  f"Success={best_metrics['success_rate']:.3f}, "
                  f"Collision={best_metrics['collision_rate']:.3f}, "
                  f"Reward={best_metrics['avg_reward']:.2f}")
            print("=" * 80 + "\n")

    # ✅ 训练结束，打印最终统计
    print("\n" + "=" * 80)
    print("🎉 BFTQ训练完成!")
    print(f"🏆 最佳模型来自 Epoch {best_metrics['epoch']}")
    print(f"   Success Rate: {best_metrics['success_rate']:.3f}")
    print(f"   Collision Rate: {best_metrics['collision_rate']:.3f}")
    print(f"   Avg Reward: {best_metrics['avg_reward']:.2f}")
    print(f"💾 最佳模型保存在: {save_directory}/{model_name}_best_{{actor,actor_target,critic,critic_target}}.pth")
    print(f"💾 所有checkpoint保存在: {save_directory}/checkpoint_epoch_XXX_{{actor,actor_target,critic,critic_target}}.pth")
    print("=" * 80)


def eval_multi_budget(model, env, scenarios, epoch, max_steps, budget_test_values):
    """
    评估模型在多个budget值下的性能

    Args:
        budget_test_values: 要测试的budget值列表

    Returns:
        dict: 平均性能指标
    """
    print("\n" + "=" * 80)
    print(f"📊 Epoch {epoch} - Evaluating with multiple budgets")
    print("=" * 80)

    all_rewards = []
    all_cols = []
    all_goals = []

    for budget in budget_test_values:
        print(f"\n💰 Testing with budget = {budget:.2f}")
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
                # ✅ 使用当前测试的budget
                action = model.get_action(state, budget, add_noise=False)
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

        print(f"   Budget={budget:.2f}: Success={avg_goal:.3f}, Collision={avg_col:.3f}, Reward={avg_reward:.2f}")

        all_rewards.append(avg_reward)
        all_cols.append(avg_col)
        all_goals.append(avg_goal)

        # ✅ TensorBoard 记录 (分budget记录)
        model.writer.add_scalar(f"eval/budget_{budget:.2f}/goal", avg_goal, epoch)
        model.writer.add_scalar(f"eval/budget_{budget:.2f}/col", avg_col, epoch)
        model.writer.add_scalar(f"eval/budget_{budget:.2f}/reward", avg_reward, epoch)

    # ✅ 计算所有budget的平均性能
    final_avg_reward = np.mean(all_rewards)
    final_avg_col = np.mean(all_cols)
    final_avg_goal = np.mean(all_goals)

    print(f"\n📊 Overall Average across all budgets:")
    print(f"   Success Rate:    {final_avg_goal:.3f}")
    print(f"   Collision Rate:  {final_avg_col:.3f}")
    print(f"   Average Reward:  {final_avg_reward:.2f}")

    # ✅ TensorBoard 记录整体平均（与baseline命名一致）
    model.writer.add_scalar("eval/avg_goal", final_avg_goal, epoch)
    model.writer.add_scalar("eval/avg_col", final_avg_col, epoch)
    model.writer.add_scalar("eval/avg_reward", final_avg_reward, epoch)

    print("=" * 80)

    return {
        'success_rate': final_avg_goal,
        'collision_rate': final_avg_col,
        'avg_reward': final_avg_reward,
        'epoch': epoch
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
