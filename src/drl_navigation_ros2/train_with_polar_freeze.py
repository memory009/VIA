#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_with_polar.py - 修改版

关键改动：
1. 添加 --load-baseline 和 --freeze-task-critic 命令行参数
2. 在模型初始化时加载baseline权重
3. 可选择冻结Task Critic
"""

from pathlib import Path
from datetime import datetime
import socket
import argparse
import sys

from TD3.TD3_lightweight_safety_critic_with_freeze import TD3_SafetyCritic
from ros_python import ROS_env
from replay_buffer import ReplayBuffer
import torch
import numpy as np
from utils import record_eval_positions
from pretrain_utils import Pretraining

from eval_scenario_map_generator import generate_scenario_maps
from training_scenario_saver import save_training_episode_map
from training_trajectory_collector import TrainingTrajectoryCollector, collect_training_step
from cvar_data_processor import CVaRDataProcessor


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


def main(args=None):
    """Main training function with CVaR-based Safe RL + Baseline loading"""
    parser = argparse.ArgumentParser(description='Train TD3 with CVaR Safe RL and Safety Critics')
    parser.add_argument(
        '--model-type',
        type=str,
        default='safety',
        choices=['safety'],
        help='Model type: only "safety" supported in this version'
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
    parser.add_argument(
        '--lambda-safe',
        type=float,
        default=50.0,
        help='Safety weight lambda for actor loss (default: 50.0)'
    )
    # ✅ 新增参数
    parser.add_argument(
        '--load-baseline',
        action='store_true',
        help='Load Task Critic and Actor from baseline model'
    )
    parser.add_argument(
        '--baseline-path',
        type=str,
        default=None,
        help='Path to baseline model directory (e.g., models/TD3_lightweight/best_run/)'
    )
    parser.add_argument(
        '--freeze-task-critic',
        action='store_true',
        help='Freeze Task Critic parameters (no updates during training)'
    )
    parser.add_argument(
        '--cost-scale-factor',
        type=float,
        default=500.0,
        help='Cost scale factor for Safety Critic (default: 500.0)'
    )
    
    cmd_args = parser.parse_args(args)
    
    # ✅ 验证baseline参数
    if cmd_args.load_baseline and not cmd_args.baseline_path:
        print("❌ Error: --load-baseline requires --baseline-path")
        return
    
    if cmd_args.baseline_path:
        baseline_path = Path(cmd_args.baseline_path)
        if not baseline_path.exists():
            print(f"❌ Error: Baseline path does not exist: {baseline_path}")
            return
    
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    hostname = socket.gethostname()
    
    # ✅ 在run_id中标记是否使用baseline
    run_suffix = "_from_baseline" if cmd_args.load_baseline else ""
    run_suffix += "_frozen" if cmd_args.freeze_task_critic else ""
    run_id = f"{timestamp}_{hostname}{run_suffix}"
    
    model_dir_name = "TD3_safety"
    model_name = "TD3_safety"
    
    save_directory = Path("src/drl_navigation_ros2/models") / model_dir_name / run_id
    save_directory.mkdir(parents=True, exist_ok=True)

    log_file = save_directory / "train_output.log"
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    
    # 训练参数配置
    action_dim = 2
    max_action = 1
    state_dim = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nr_eval_episodes = 10
    max_epochs = cmd_args.max_epochs
    episodes_per_epoch = cmd_args.episodes_per_epoch
    training_iterations_per_epoch = 17500
    batch_size = 40
    max_steps = 300
    load_saved_buffer = True
    pretrain = True
    pretraining_iterations = 50
    
    print("=" * 80)
    print(f"🚀 开始新的训练运行 (CVaR + Safety Critics + Baseline)")
    print(f"🔧 模型类型: {model_name}")
    print(f"🛡️  安全权重 λ: {cmd_args.lambda_safe}")
    print(f"💰 Cost scale factor: {cmd_args.cost_scale_factor}")
    if cmd_args.load_baseline:
        print(f"📦 加载Baseline: {cmd_args.baseline_path}")
    if cmd_args.freeze_task_critic:
        print(f"❄️  Task Critic: 冻结（不更新）")
    else:
        print(f"🔥 Task Critic: 正常训练")
    print(f"📁 运行ID: {run_id}")
    print(f"💾 模型保存路径: {save_directory}")
    print(f"📊 TensorBoard日志: runs/{run_id}")
    print(f"📝 日志文件: {log_file}")
    print(f"🎯 训练设置: {max_epochs} epochs × {episodes_per_epoch} episodes")
    print(f"🔄 训练频率: {training_iterations_per_epoch} iterations/epoch")
    print("=" * 80)

    # ✅ 模型初始化 - 添加baseline加载参数
    model_kwargs = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'device': device,
        'save_every': 0,
        'load_model': False,
        'save_directory': save_directory,
        'model_name': model_name,
        'run_id': run_id,
        'lambda_safe': cmd_args.lambda_safe,
        # ✅ 新增baseline参数
        'load_baseline': cmd_args.load_baseline,
        'baseline_path': cmd_args.baseline_path if cmd_args.load_baseline else None,
        'freeze_task_critic': cmd_args.freeze_task_critic,
    }
    
    model = TD3_SafetyCritic(**model_kwargs)

    ros = ROS_env(enable_random_obstacles=True)
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_eval_episodes,
        save_to_file=True,
        random_seed=42,
        enable_random_obstacles=True
    )

    generate_scenario_maps(eval_scenarios, scenario_tag="8_polar")

    # ✅ Pretraining策略调整
    if load_saved_buffer:
        pretraining_obj = Pretraining(
            file_names=["src/drl_navigation_ros2/assets/data.yml"],
            model=model,
            replay_buffer=ReplayBuffer(buffer_size=500000, random_seed=42),
            reward_function=ros.get_reward,
        )
        replay_buffer = pretraining_obj.load_buffer()
        
        # ✅ 如果加载了baseline，跳过预训练（因为已经有好的权重了）
        if pretrain and not cmd_args.load_baseline:
            print("\n⚡ 开始预训练...")
            pretraining_obj.train(
                pretraining_iterations=pretraining_iterations,
                replay_buffer=replay_buffer,
                iterations=500,
                batch_size=batch_size,
            )
        elif cmd_args.load_baseline:
            print("\n⏭️  跳过预训练（已加载baseline权重）")
    else:
        replay_buffer = ReplayBuffer(buffer_size=500000, random_seed=42)

    # ✅ 初始化 CVaR 处理器 - 使用命令行参数
    cvar_processor = CVaRDataProcessor(
        cvar_alpha=0.1, 
        penalty_scale=50.0,
        cost_scale_factor=cmd_args.cost_scale_factor
    )
    
    # Best model 追踪
    best_metrics = {
        'success_rate': -1.0,
        'collision_rate': 2.0,
        'avg_reward': -float('inf'),
        'epoch': 0
    }
    epochs_since_improvement = 0
    
    trajectory_collector = TrainingTrajectoryCollector(save_dir="training_trajectories_8_polar")

    epoch = 0
    
    # ===== 主训练循环 (Epoch-level) =====
    while epoch < max_epochs:
        print(f"\n{'='*80}")
        print(f"🎯 Epoch {epoch + 1}/{max_epochs}")
        if cmd_args.freeze_task_critic:
            print(f"❄️  Task Critic: 冻结状态")
        print(f"{'='*80}")
        
        # ===== Phase 1: Rollout 完整的 70 条轨迹 =====
        print(f"\n📍 Phase 1: Rollout ({episodes_per_epoch} episodes)")
        
        for episode_in_epoch in range(1, episodes_per_epoch + 1):
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
            save_training_episode_map(
                env=ros,
                epoch=epoch + 1,
                episode=episode_in_epoch,
                save_dir="train_scenario_8_polar"
            )
            
            steps = 0
            
            while True:
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
                
                steps += 1
                
                if terminal or steps >= max_steps:
                    trajectory_collector.save_trajectory(
                        epoch=epoch + 1,
                        episode=episode_in_epoch
                    )
                    trajectory_collector.reset_trajectory()
                    break
            
            if episode_in_epoch % 10 == 0 or episode_in_epoch == episodes_per_epoch:
                print(f"   Rollout 进度: {episode_in_epoch}/{episodes_per_epoch}")
        
        print(f"   ✅ Phase 1 完成: {episodes_per_epoch} episodes rolled out")
        
        # ===== Phase 2: CVaR 数据处理 =====
        processed_data = cvar_processor.process_epoch_trajectories(
            model=model,
            epoch=epoch + 1,
            traj_dir="training_trajectories_8_polar",
            scenario_dir="train_scenario_8_polar",
            observation_error=0.01
        )
        
        # ===== Phase 3: 批量加入 Replay Buffer =====
        print(f"\n📦 Phase 3: 更新 Replay Buffer")
        added_count = 0
        
        for traj_data in processed_data:
            for transition in traj_data['transitions']:
                state, action, reward_modified, cost, done, next_state, penalty = transition
                
                replay_buffer.add(state, action, reward_modified, done, next_state, cost)
                added_count += 1
        
        print(f"   ✅ 添加了 {added_count} 个 transitions")
        print(f"   📊 Buffer 当前大小: {replay_buffer.size()} / 500000")
        
        # ===== Phase 4: 批量训练 =====
        print(f"\n🔧 Phase 4: Network Training")
        if cmd_args.freeze_task_critic:
            print(f"   ❄️  Task Critic: 冻结（不更新）")
            print(f"   🔥 Safety Critic: 训练中")
            print(f"   🎭 Actor: 训练中（受Task + Safety双重指导）")
        
        model.train(
            replay_buffer=replay_buffer,
            iterations=training_iterations_per_epoch,
            batch_size=batch_size,
        )
        print(f"   ✅ 训练完成: {training_iterations_per_epoch} iterations")
        print(f"   📊 当前 iter_count: {model.iter_count}")
        print(f"   📊 当前 lambda_safe: {model.lambda_safe}")
        
        # ===== Phase 5: Eval =====
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
        
        epoch += 1

    # 训练结束
    print("\n" + "=" * 80)
    print("🎉 训练完成!")
    print(f"🏆 最佳模型来自 Epoch {best_metrics['epoch']}")
    print(f"   Success Rate: {best_metrics['success_rate']:.3f}")
    print(f"   Collision Rate: {best_metrics['collision_rate']:.3f}")
    print(f"   Avg Reward: {best_metrics['avg_reward']:.2f}")
    print(f"💾 最佳模型: {save_directory}/{model_name}_best_*.pth")
    print("=" * 80)


def eval(model, env, scenarios, epoch, max_steps, best_metrics, save_directory, model_name):
    """评估函数（与原版相同）"""
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
    
    model.save(
        filename=f"{model_name}_epoch_{epoch:03d}",
        directory=save_directory
    )
    print(f"💾 已保存: {model_name}_epoch_{epoch:03d}_*.pth")
    
    if is_best:
        model.save(
            filename=f"{model_name}_best",
            directory=save_directory
        )
        print(f"🌟 NEW BEST MODEL! (epoch {epoch})")
        print(f"   Improvements:")
        if avg_goal > best_metrics['success_rate']:
            print(f"      Success: {best_metrics['success_rate']:.3f} → {avg_goal:.3f} ⬆️")
        elif avg_goal == best_metrics['success_rate'] and avg_col < best_metrics['collision_rate']:
            print(f"      Same success, lower collision: {best_metrics['collision_rate']:.3f} → {avg_col:.3f} ⬇️")
        elif avg_goal == best_metrics['success_rate'] and avg_col == best_metrics['collision_rate']:
            print(f"      Same success & collision, higher reward: {best_metrics['avg_reward']:.2f} → {avg_reward:.2f} ⬆️")
    else:
        print(f"📊 Not best (Best from epoch {best_metrics['epoch']})")
    
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