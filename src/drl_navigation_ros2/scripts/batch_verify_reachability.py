#!/usr/bin/env python3
"""
批量可达性验证脚本
在服务器（无 Gazebo）运行，加载保存的轨迹并批量计算可达集
输入：assets/trajectories.pkl
输出：assets/reachability_results.json
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pickle
import json
from tqdm import tqdm
import time

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from TD3.TD3 import TD3
from verification.polar_verifier import verify_safety


def load_trajectories(pkl_path=None):
    """加载保存的轨迹"""
    if pkl_path is None:
        pkl_path = Path(__file__).parent.parent / "assets" / "trajectories.pkl"
    
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"轨迹文件不存在: {pkl_path}\n"
            f"请先在本地运行 collect_trajectories.py"
        )
    
    with open(pkl_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    # 过滤掉失败的轨迹
    valid_trajectories = [t for t in trajectories if t is not None]
    
    return valid_trajectories


def verify_single_trajectory(
    agent,
    trajectory_data,
    observation_error=0.01,
    sample_interval=10,
    verbose=False
):
    """
    验证单个轨迹的可达集
    
    Returns:
        trajectory_results: dict
    """
    states = trajectory_data['states']
    sampled_states = states[::sample_interval]
    
    results = []
    safe_count = 0
    
    for i, state in enumerate(sampled_states):
        step_idx = i * sample_interval
        
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
        
        result = {
            'step': step_idx,
            'det_action': det_action.tolist(),
            'action_ranges': action_ranges,
            'is_safe': is_safe,
            'width_v': float(width_v),
            'width_omega': float(width_omega),
            'min_laser': float(np.min(state[:20])),
            'distance': float(state[20]),
        }
        results.append(result)
    
    # 统计
    n_samples = len(sampled_states)
    trajectory_summary = {
        'n_samples': n_samples,
        'safe_count': safe_count,
        'safety_rate': safe_count / n_samples if n_samples > 0 else 0,
        'collision': trajectory_data['collision'],
        'goal_reached': trajectory_data['goal_reached'],
        'steps': trajectory_data['steps'],
        'total_reward': float(trajectory_data['total_reward']),
        'results': results,
    }
    
    return trajectory_summary


def main():
    """主函数：批量验证所有轨迹"""
    print("\n" + "="*70)
    print("批量可达性验证工具")
    print("="*70)
    
    # ===== 1. 加载模型 =====
    print("\n[1/3] 加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = TD3(
        state_dim=25,
        action_dim=2,
        max_action=1.0,
        device=device,
        load_model=True,
        model_name="TD3",
        load_directory=Path("models/TD3/Nov17_06-22-08_archived"),
    )
    print(f"  ✅ 模型加载成功 (设备: {device})")
    
    # ===== 2. 加载轨迹 =====
    print("\n[2/3] 加载保存的轨迹...")
    trajectories = load_trajectories()
    n_trajectories = len(trajectories)
    print(f"  ✅ 加载 {n_trajectories} 条轨迹")
    
    # 统计轨迹信息
    total_states = sum(t['steps'] for t in trajectories)
    print(f"  总状态数: {total_states}")
    print(f"  平均长度: {total_states/n_trajectories:.1f} 步/轨迹")
    
    # ===== 3. 批量验证 =====
    print(f"\n[3/3] 批量验证可达集...")
    print(f"  观测误差: ±0.01")
    print(f"  采样间隔: 每10步")
    print(f"  Bernstein阶数: 1")
    print()
    
    all_results = []
    start_time = time.time()
    
    for i, trajectory_data in enumerate(tqdm(trajectories, desc="验证进度")):
        try:
            trajectory_summary = verify_single_trajectory(
                agent,
                trajectory_data,
                observation_error=0.01,
                sample_interval=10,
                verbose=False
            )
            all_results.append(trajectory_summary)
            
        except Exception as e:
            print(f"\n  ❌ 轨迹 {i+1} 验证失败: {e}")
            all_results.append(None)
    
    elapsed_time = time.time() - start_time
    
    # ===== 4. 汇总统计 =====
    print("\n" + "="*70)
    print("验证统计:")
    print("="*70)
    
    valid_results = [r for r in all_results if r is not None]
    
    # 4.1 整体统计
    total_samples = sum(r['n_samples'] for r in valid_results)
    total_safe = sum(r['safe_count'] for r in valid_results)
    overall_safety_rate = total_safe / total_samples if total_samples > 0 else 0
    
    print(f"\n整体可达集安全性:")
    print(f"  总采样点: {total_samples}")
    print(f"  安全点数: {total_safe}")
    print(f"  安全率: {overall_safety_rate*100:.1f}%")
    
    # 4.2 轨迹分类统计
    goal_trajectories = [r for r in valid_results if r['goal_reached']]
    collision_trajectories = [r for r in valid_results if r['collision']]
    
    print(f"\n按轨迹结果分类:")
    print(f"  到达目标的轨迹: {len(goal_trajectories)}")
    if goal_trajectories:
        goal_safety = np.mean([r['safety_rate'] for r in goal_trajectories])
        print(f"    平均安全率: {goal_safety*100:.1f}%")
    
    print(f"  碰撞的轨迹: {len(collision_trajectories)}")
    if collision_trajectories:
        collision_safety = np.mean([r['safety_rate'] for r in collision_trajectories])
        print(f"    平均安全率: {collision_safety*100:.1f}%")
    
    # 4.3 可达集宽度统计
    all_widths_v = []
    all_widths_omega = []
    
    for result in valid_results:
        for r in result['results']:
            all_widths_v.append(r['width_v'])
            all_widths_omega.append(r['width_omega'])
    
    print(f"\n可达集宽度统计:")
    print(f"  线速度:")
    print(f"    平均: {np.mean(all_widths_v):.6f}")
    print(f"    标准差: {np.std(all_widths_v):.6f}")
    print(f"    最大: {np.max(all_widths_v):.6f}")
    
    print(f"  角速度:")
    print(f"    平均: {np.mean(all_widths_omega):.6f}")
    print(f"    标准差: {np.std(all_widths_omega):.6f}")
    print(f"    最大: {np.max(all_widths_omega):.6f}")
    
    # 4.4 性能统计
    print(f"\n性能统计:")
    print(f"  总耗时: {elapsed_time:.2f} 秒")
    print(f"  平均每轨迹: {elapsed_time/n_trajectories:.2f} 秒")
    print(f"  平均每采样点: {elapsed_time/total_samples:.4f} 秒")
    
    # ===== 5. 保存结果 =====
    output_path = Path(__file__).parent.parent / "assets" / "reachability_results_batch.json"
    
    output_data = {
        'metadata': {
            'n_trajectories': n_trajectories,
            'total_samples': total_samples,
            'observation_error': 0.01,
            'sample_interval': 10,
            'elapsed_time': elapsed_time,
        },
        'summary': {
            'overall_safety_rate': overall_safety_rate,
            'total_safe': total_safe,
            'total_samples': total_samples,
            'goal_trajectories': len(goal_trajectories),
            'collision_trajectories': len(collision_trajectories),
        },
        'trajectories': all_results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")
    print("="*70)
    print("\n🎉 批量验证完成！")


if __name__ == "__main__":
    main()