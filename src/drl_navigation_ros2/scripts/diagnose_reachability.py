#!/usr/bin/env python3
"""
可达性验证诊断脚本
使用真实评估轨迹测试 TD3 模型的可达集计算
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from TD3.TD3 import TD3
from ros_python import ROS_env
from verification.polar_verifier import compute_reachable_set, verify_safety
from utils import pos_data


def load_eval_scenarios(json_path=None):
    """
    加载评估场景
    
    Args:
        json_path: 场景文件路径，默认使用 assets/eval_scenarios.json
    
    Returns:
        scenarios: 评估场景列表
    """
    if json_path is None:
        json_path = Path(__file__).parent.parent / "assets" / "eval_scenarios.json"
    
    if not json_path.exists():
        print(f"⚠️  场景文件不存在: {json_path}")
        print("   将使用随机场景")
        return None
    
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
    
    print(f"✅ 加载 {len(scenarios)} 个评估场景")
    return scenarios


def collect_trajectory(agent, env, scenario, max_steps=300, verbose=False):
    """
    收集单个场景的完整轨迹
    
    Args:
        agent: TD3 对象
        env: ROS_env 对象
        scenario: 评估场景
        max_steps: 最大步数
        verbose: 是否打印详细信息
    
    Returns:
        trajectory: list of states
        rewards: list of rewards
        collision: bool
        goal_reached: bool
    """
    trajectory = []
    rewards = []
    
    # 重置环境到指定场景
    latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(scenario)
    
    if verbose:
        print(f"\n场景初始化:")
        print(f"  机器人位置: ({scenario[-2].x:.2f}, {scenario[-2].y:.2f})")
        print(f"  目标位置: ({scenario[-1].x:.2f}, {scenario[-1].y:.2f})")
        print(f"  初始距离: {distance:.3f} m")
    
    step_count = 0
    while step_count < max_steps:
        # 准备状态
        state, terminal = agent.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )
        trajectory.append(state)
        rewards.append(reward)
        
        if terminal:
            if verbose:
                status = "🎯 到达目标" if goal else "💥 发生碰撞"
                print(f"  步数 {step_count}: {status}")
            break
        
        # 获取动作（无噪声）
        action = agent.get_action(state, add_noise=False)
        a_in = [(action[0] + 1) / 2, action[1]]  # 线速度映射到 [0, 1]
        
        # 执行动作
        latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )
        
        step_count += 1
    
    if verbose:
        print(f"  轨迹长度: {len(trajectory)} 步")
        print(f"  累积奖励: {sum(rewards):.2f}")
    
    return trajectory, rewards, collision, goal


def test_trajectory_reachability(
    agent, 
    trajectory, 
    observation_error=0.01,
    sample_interval=10,
    verbose=True
):
    """
    测试轨迹中采样点的可达集
    
    Args:
        agent: TD3 对象
        trajectory: 状态轨迹列表
        observation_error: 观测误差
        sample_interval: 采样间隔（每隔 N 步采样一次）
        verbose: 是否打印详细信息
    
    Returns:
        results: dict 包含统计信息
    """
    sampled_states = trajectory[::sample_interval]
    n_samples = len(sampled_states)
    
    if verbose:
        print("\n" + "="*70)
        print(f"测试轨迹可达集 (采样 {n_samples} 个状态, 间隔={sample_interval})")
        print("="*70)
    
    safe_count = 0
    widths_v = []
    widths_omega = []
    all_results = []
    
    for i, state in enumerate(sampled_states):
        # 计算可达集
        is_safe, action_ranges = verify_safety(
            agent, 
            state, 
            observation_error=observation_error,
            bern_order=1,
            error_steps=4000,
        )
        
        # 计算确定性动作
        det_action = agent.get_action(state, add_noise=False)
        
        width_v = action_ranges[0][1] - action_ranges[0][0]
        width_omega = action_ranges[1][1] - action_ranges[1][0]
        
        if is_safe:
            safe_count += 1
        
        widths_v.append(width_v)
        widths_omega.append(width_omega)
        
        result = {
            'step': i * sample_interval,
            'state': state,
            'det_action': det_action,
            'action_ranges': action_ranges,
            'is_safe': is_safe,
            'width_v': width_v,
            'width_omega': width_omega,
            'min_laser': np.min(state[:20]),
            'distance': state[20],
        }
        all_results.append(result)
        
        if verbose:
            status = "✅" if is_safe else "❌"
            print(f"  [{i+1}/{n_samples}] 步数={result['step']:3d} | "
                  f"安全={status} | "
                  f"激光={result['min_laser']:.3f}m | "
                  f"距离={result['distance']:.3f}m | "
                  f"宽度=[Δv={width_v:.4f}, Δω={width_omega:.4f}]")
    
    # 统计结果
    summary = {
        'n_samples': n_samples,
        'safe_count': safe_count,
        'safety_rate': safe_count / n_samples,
        'avg_width_v': np.mean(widths_v),
        'std_width_v': np.std(widths_v),
        'avg_width_omega': np.mean(widths_omega),
        'std_width_omega': np.std(widths_omega),
        'max_width_v': np.max(widths_v),
        'max_width_omega': np.max(widths_omega),
        'all_results': all_results,
    }
    
    if verbose:
        print("\n" + "-"*70)
        print("统计结果:")
        print(f"  安全率: {summary['safety_rate']*100:.1f}% ({safe_count}/{n_samples})")
        print(f"  平均可达集宽度:")
        print(f"    线速度: {summary['avg_width_v']:.6f} ± {summary['std_width_v']:.6f}")
        print(f"    角速度: {summary['avg_width_omega']:.6f} ± {summary['std_width_omega']:.6f}")
        print(f"  最大可达集宽度:")
        print(f"    线速度: {summary['max_width_v']:.6f}")
        print(f"    角速度: {summary['max_width_omega']:.6f}")
        print("="*70)
    
    return summary


def test_detailed_state(agent, state, observation_error=0.01):
    """
    详细测试单个状态（用于诊断）
    
    Args:
        agent: TD3 对象
        state: 状态向量
        observation_error: 观测误差
    """
    print("\n" + "="*70)
    print("详细状态诊断")
    print("="*70)
    
    # 1. 状态信息
    print("\n状态信息:")
    laser_data = state[:20]
    print(f"  激光雷达:")
    print(f"    最小值: {np.min(laser_data):.3f} m")
    print(f"    最大值: {np.max(laser_data):.3f} m")
    print(f"    平均值: {np.mean(laser_data):.3f} m")
    print(f"  目标距离: {state[20]:.3f} m")
    print(f"  方向: cos={state[21]:.3f}, sin={state[22]:.3f}")
    print(f"  上一步动作: [v={state[23]:.3f}, ω={state[24]:.3f}]")
    
    # 2. 确定性动作
    det_action = agent.get_action(state, add_noise=False)
    print("\n确定性动作 (无噪声):")
    print(f"  线速度:  {det_action[0]:.6f}")
    print(f"  角速度:  {det_action[1]:.6f}")
    
    # 3. 可达集
    is_safe, action_ranges = verify_safety(
        agent, 
        state, 
        observation_error=observation_error,
        bern_order=1,
        error_steps=4000,
    )
    
    print(f"\n可达集 (观测误差 ±{observation_error}):")
    print(f"  线速度:  [{action_ranges[0][0]:.6f}, {action_ranges[0][1]:.6f}]")
    print(f"  角速度:  [{action_ranges[1][0]:.6f}, {action_ranges[1][1]:.6f}]")
    
    width_v = action_ranges[0][1] - action_ranges[0][0]
    width_omega = action_ranges[1][1] - action_ranges[1][0]
    print(f"\n可达集宽度:")
    print(f"  Δv = {width_v:.6f}")
    print(f"  Δω = {width_omega:.6f}")
    
    print(f"\n安全性判断: {'✅ 安全' if is_safe else '❌ 不安全'}")
    print("="*70)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("POLAR 可达性验证诊断工具 (使用真实评估轨迹)")
    print("="*70)
    
    # ===== 1. 加载模型 =====
    print("\n[1/5] 加载模型...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 25
    action_dim = 2
    max_action = 1.0
    
    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        load_model=True,
        model_name="TD3",
        load_directory=Path("models/TD3/Nov17_06-22-08_archived"),
    )
    
    print(f"  ✅ 模型加载成功 (设备: {device})")
    
    # ===== 2. 初始化 ROS 环境 =====
    print("\n[2/5] 初始化 ROS 环境...")
    
    env = ROS_env(enable_random_obstacles=False)
    print("  ✅ ROS 环境初始化成功")
    
    # ===== 3. 加载评估场景 =====
    print("\n[3/5] 加载评估场景...")
    
    scenarios = load_eval_scenarios()
    if scenarios is None or len(scenarios) == 0:
        print("  ⚠️  使用随机场景")
        from utils import record_eval_positions
        scenarios = record_eval_positions(
            n_eval_scenarios=3,
            save_to_file=False,
            enable_random_obstacles=False
        )
    
    # 选择第一个场景进行测试
    test_scenario = scenarios[0]
    print(f"  ✅ 使用场景 #0")
    
    # ===== 4. 收集轨迹 =====
    print("\n[4/5] 收集评估轨迹...")
    
    trajectory, rewards, collision, goal = collect_trajectory(
        agent, env, test_scenario, max_steps=300, verbose=True
    )
    
    # ===== 5. 测试可达集 =====
    print("\n[5/5] 测试轨迹可达集...")
    
    # 5.1 详细测试第一个状态
    if len(trajectory) > 0:
        test_detailed_state(agent, trajectory[0], observation_error=0.01)
    
    # 5.2 批量测试轨迹
    summary = test_trajectory_reachability(
        agent,
        trajectory,
        observation_error=0.01,
        sample_interval=10,  # 每 10 步采样一次
        verbose=True
    )
    
    # ===== 6. 保存结果（可选）=====
    output_path = Path(__file__).parent.parent / "assets" / "reachability_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换 numpy 类型为 Python 原生类型以便 JSON 序列化
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    summary_serializable = convert_to_serializable({
        'n_samples': summary['n_samples'],
        'safe_count': summary['safe_count'],
        'safety_rate': summary['safety_rate'],
        'avg_width_v': summary['avg_width_v'],
        'std_width_v': summary['std_width_v'],
        'avg_width_omega': summary['avg_width_omega'],
        'std_width_omega': summary['std_width_omega'],
    })
    
    with open(output_path, 'w') as f:
        json.dump(summary_serializable, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")
    
    print("\n" + "="*70)
    print("诊断完成！")
    print("="*70)


if __name__ == "__main__":
    main()