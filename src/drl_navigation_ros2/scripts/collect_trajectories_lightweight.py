#!/usr/bin/env python3
"""
轨迹收集脚本
在本地 Gazebo 环境中运行，收集所有评估场景的轨迹
输出：保存到 assets/trajectories_lightweight_8_polar_freeze.pkl
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pickle
import json
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from TD3.TD3 import TD3
from TD3.TD3_lightweight import TD3 as TD3_Lightweight
from TD3.TD3_lightweight_safety_critic import TD3_SafetyCritic as TD3_SafetyCritic_Base
from TD3.TD3_lightweight_safety_critic_with_freeze import TD3_SafetyCritic as TD3_SafetyCritic_Freeze
from ros_python import ROS_env
from utils import pos_data


def load_eval_scenarios(json_path=None):
    """加载评估场景"""
    if json_path is None:
        json_path = Path(__file__).parent.parent / "assets" / "eval_scenarios_8_polar.json"
    
    if not json_path.exists():
        print(f"⚠️  场景文件不存在: {json_path}")
        print("   将生成随机场景")
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


def collect_single_trajectory(agent, env, scenario, max_steps=300):
    """
    收集单个场景的轨迹（添加位姿追踪）
    修复：使用真实位姿而非估计值
    """
    from squaternion import Quaternion
    
    trajectory = []
    actions = []
    rewards = []
    poses = []  # 保存每一步的真实位姿
    
    # 重置环境
    latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(scenario)
    
    # 记录初始信息
    robot_pos = (scenario[-2].x, scenario[-2].y, scenario[-2].angle)
    target_pos = (scenario[-1].x, scenario[-1].y)
    
    step_count = 0
    while step_count < max_steps:
        # ===== 获取真实位姿（从Gazebo传感器） =====
        latest_position = env.sensor_subscriber.latest_position
        latest_orientation = env.sensor_subscriber.latest_heading
        
        if latest_position is not None and latest_orientation is not None:
            # 提取真实的x, y坐标
            odom_x = latest_position.x
            odom_y = latest_position.y
            # 提取真实的theta角度
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
            # 如果传感器数据不可用，使用初始位姿
            current_pose = [robot_pos[0], robot_pos[1], robot_pos[2]]
        # ===== 真实位姿获取结束 =====
        
        # 准备状态
        state, terminal = agent.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )
        trajectory.append(state)
        rewards.append(reward)
        poses.append(current_pose.copy())  # 保存真实位姿
        
        if terminal:
            break
        
        # 获取动作
        action = agent.get_action(state, add_noise=False)
        actions.append(action)
        a_in = [(action[0] + 1) / 2, action[1]]
        
        # 执行动作
        latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )
        
        step_count += 1
    
    # 打包数据
    trajectory_data = {
        'states': np.array(trajectory),  # (T, 25)
        'actions': np.array(actions),     # (T-1, 2)
        'rewards': np.array(rewards),     # (T,)
        'poses': np.array(poses),         # ← 新增：(T, 3) - (x, y, θ)
        'collision': collision,
        'goal_reached': goal,
        'steps': len(trajectory),
        'total_reward': sum(rewards),
        'robot_start': robot_pos,
        'target_pos': target_pos,
    }
    
    return trajectory_data


def main():
    """主函数：收集所有场景的轨迹"""
    print("\n" + "="*70)
    print("轨迹收集工具")
    print("="*70)
    
    # ===== 1. 加载模型 =====
    print("\n[1/4] 加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 配置要加载的模型 =====
    # 可选模型类型:
    # - "TD3_Lightweight": 基础轻量级TD3模型
    # - "TD3_SafetyCritic": 安全批评家模型（基础版，来自 TD3_lightweight_safety_critic.py）
    # - "TD3_SafetyCritic_Freeze": 安全批评家模型（支持冻结Task Critic，来自 TD3_lightweight_safety_critic_with_freeze.py）

    model_type = "TD3_SafetyCritic_Freeze"  # ← 修改这里选择模型类型
    model_name = "TD3_safety_epoch_011"     # 要加载的权重文件名（不含后缀）
    model_dir = Path("models/TD3_safety/Dec09_20-20-51_cheeson_from_baseline_frozen")

    # 根据模型类型加载
    if model_type == "TD3_SafetyCritic_Freeze":
        print(f"  加载 TD3_SafetyCritic_Freeze 模型 (支持冻结): {model_name}")
        print(f"  来源: TD3_lightweight_safety_critic_with_freeze.py")
        agent = TD3_SafetyCritic_Freeze(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            save_every=0,
            load_model=False,  # 先不自动加载
            save_directory=model_dir,
            model_name=model_name,
            run_id="eval_mode",
            lambda_safe=50.0,  # 这些参数在推理时不重要
            load_baseline=False,
            baseline_path=None,
            freeze_task_critic=False,
        )
        # 手动加载权重
        agent.load(filename=model_name, directory=str(model_dir))
        print(f"  ✅ TD3_SafetyCritic_Freeze 模型加载成功")

    elif model_type == "TD3_SafetyCritic":
        print(f"  加载 TD3_SafetyCritic 模型 (基础版): {model_name}")
        print(f"  来源: TD3_lightweight_safety_critic.py")
        agent = TD3_SafetyCritic_Base(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            save_every=0,
            load_model=False,
            save_directory=model_dir,
            model_name=model_name,
            run_id="eval_mode",
            lambda_safe=50.0,
        )
        agent.load(filename=model_name, directory=str(model_dir))
        print(f"  ✅ TD3_SafetyCritic 模型加载成功")

    elif model_type == "TD3_Lightweight":
        print(f"  加载 TD3_Lightweight 模型: {model_name}")
        agent = TD3_Lightweight(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            load_model=True,
            model_name=model_name,
            load_directory=model_dir,
        )
        print(f"  ✅ TD3_Lightweight 模型加载成功")

    else:
        raise ValueError(f"未知的模型类型: {model_type}。请选择: TD3_Lightweight, TD3_SafetyCritic, 或 TD3_SafetyCritic_Freeze")
    
    # ===== 2. 初始化环境 =====
    print("\n[2/4] 初始化 ROS 环境...")
    env = ROS_env(enable_random_obstacles=False)
    print("  ✅ ROS 环境初始化成功")
    
    # ===== 3. 加载场景 =====
    print("\n[3/4] 加载评估场景...")
    scenarios = load_eval_scenarios()
    n_scenarios = len(scenarios)
    print(f"  ✅ 加载 {n_scenarios} 个场景")
    
    # ===== 4. 收集所有轨迹 =====
    print(f"\n[4/4] 收集 {n_scenarios} 个场景的轨迹...")
    print("  (这可能需要几分钟，请耐心等待...)\n")
    
    all_trajectories = []
    
    for i, scenario in enumerate(tqdm(scenarios, desc="收集轨迹")):
        try:
            trajectory_data = collect_single_trajectory(
                agent, env, scenario, max_steps=300
            )
            all_trajectories.append(trajectory_data)
            
            # 打印简要信息
            status = "🎯 目标" if trajectory_data['goal_reached'] else "💥 碰撞"
            print(f"  场景 {i+1}/{n_scenarios}: {status} | "
                  f"步数={trajectory_data['steps']} | "
                  f"奖励={trajectory_data['total_reward']:.1f}")
        
        except Exception as e:
            print(f"  ❌ 场景 {i+1} 失败: {e}")
            all_trajectories.append(None)
    
    # ===== 5. 保存轨迹 =====
    output_path = Path(__file__).parent.parent / "assets" / "trajectories_lightweight_8_polar_freeze_011.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    print(f"\n✅ 所有轨迹已保存到: {output_path}")
    
    # ===== 6. 统计信息 =====
    print("\n" + "="*70)
    print("收集统计:")
    
    successful = [t for t in all_trajectories if t is not None]
    print(f"  成功收集: {len(successful)}/{n_scenarios}")
    
    goal_count = sum(1 for t in successful if t['goal_reached'])
    collision_count = sum(1 for t in successful if t['collision'])
    
    print(f"  到达目标: {goal_count} ({goal_count/len(successful)*100:.1f}%)")
    print(f"  发生碰撞: {collision_count} ({collision_count/len(successful)*100:.1f}%)")
    
    total_steps = sum(t['steps'] for t in successful)
    avg_steps = total_steps / len(successful)
    print(f"  总步数: {total_steps}")
    print(f"  平均步数: {avg_steps:.1f}")
    
    avg_reward = np.mean([t['total_reward'] for t in successful])
    print(f"  平均奖励: {avg_reward:.2f}")
    
    print("="*70)
    print("\n🎉 轨迹收集完成！")
    print(f"现在可以将 {output_path} 复制到服务器进行批量验证")


if __name__ == "__main__":
    main()