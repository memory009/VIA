#!/usr/bin/env python3
"""
并行可达性验证脚本
使用 multiprocessing 并行处理多条轨迹
保持完全相同的计算精度
"""

import sys
try:
    import distutils.version
except AttributeError:
    import distutils
    from packaging import version as packaging_version
    distutils.version = type('version', (), {
        'LooseVersion': packaging_version.Version,
        'StrictVersion': packaging_version.Version
    })
from pathlib import Path
import numpy as np
import torch
import pickle
import json
import time
from multiprocessing import Pool, cpu_count
from functools import partial

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
            f"请先运行 collect_trajectories.py"
        )
    
    with open(pkl_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    valid_trajectories = [t for t in trajectories if t is not None]
    return valid_trajectories


def verify_single_trajectory_worker(args):
    """
    单个轨迹的验证函数（添加位姿支持）
    """
    trajectory_idx, trajectory_data, model_path, observation_error, sample_interval = args
    
    # 加载模型...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3(
        state_dim=25,
        action_dim=2,
        max_action=1.0,
        device=device,
        load_model=True,
        model_name="TD3",
        load_directory=model_path,
    )
    
    # ===== 提取状态和位姿并采样 =====
    states = trajectory_data['states']
    poses = trajectory_data['poses']  # ← 新增
    
    sampled_states = states[::sample_interval]
    sampled_poses = poses[::sample_interval]  # ← 新增
    n_samples = len(sampled_states)
    
    print(f"[进程 {trajectory_idx+1}] 开始验证 {n_samples} 个采样点...")
    
    results = []
    safe_count = 0
    start_time = time.time()
    
    # ===== 逐点验证（添加位姿） =====
    for i, (state, pose) in enumerate(zip(sampled_states, sampled_poses)):
        step_idx = i * sample_interval
        
        if i % max(1, n_samples // 4) == 0:
            elapsed = time.time() - start_time
            print(f"[进程 {trajectory_idx+1}] 进度: {i+1}/{n_samples} "
                  f"({i/n_samples*100:.0f}%) | 已用时: {elapsed/60:.1f}分钟")
        
        # ← 修改：传递位姿
        is_safe, action_ranges = verify_safety(
            agent,
            state,
            tuple(pose),  # ← 新增：(x, y, θ)
            observation_error=observation_error,
            bern_order=1,
            error_steps=4000,
        )
        
        # 后续处理不变...
        det_action = agent.get_action(state, add_noise=False)
        width_v = action_ranges[0][1] - action_ranges[0][0]
        width_omega = action_ranges[1][1] - action_ranges[1][0]
        
        if is_safe:
            safe_count += 1
        
        result = {
            'step': step_idx,
            'pose': pose.tolist(),  # ← 新增：保存位姿用于调试
            'det_action': det_action.tolist(),
            'action_ranges': action_ranges,
            'is_safe': is_safe,
            'width_v': float(width_v),
            'width_omega': float(width_omega),
            'min_laser': float(np.min(state[:20])),
            'distance': float(state[20]),
        }
        results.append(result)
    
    # 统计部分不变...
    elapsed_time = time.time() - start_time
    trajectory_summary = {
        'trajectory_idx': trajectory_idx,
        'n_samples': n_samples,
        'safe_count': safe_count,
        'safety_rate': safe_count / n_samples if n_samples > 0 else 0,
        'collision': trajectory_data['collision'],
        'goal_reached': trajectory_data['goal_reached'],
        'steps': trajectory_data['steps'],
        'total_reward': float(trajectory_data['total_reward']),
        'compute_time': elapsed_time,
        'results': results,
    }
    
    print(f"[进程 {trajectory_idx+1}] ✅ 完成！安全率: {trajectory_summary['safety_rate']*100:.1f}% | "
          f"耗时: {elapsed_time/60:.1f}分钟")
    
    return (trajectory_idx, trajectory_summary)


def main():
    """主函数：并行验证所有轨迹"""
    print("\n" + "="*70)
    print("🚀 并行可达性验证工具")
    print("="*70)
    
    # ===== 1. 检测 CPU 核心数 =====
    n_cores = cpu_count()
    print(f"\n检测到 CPU 核心数: {n_cores}")
    
    # ===== 2. 加载轨迹 =====
    print("\n[1/3] 加载保存的轨迹...")
    trajectories = load_trajectories()
    n_trajectories = len(trajectories)
    print(f"  ✅ 加载 {n_trajectories} 条轨迹")
    
    total_states = sum(t['steps'] for t in trajectories)
    print(f"  总状态数: {total_states}")
    print(f"  平均长度: {total_states/n_trajectories:.1f} 步/轨迹")
    
    # ===== 3. 准备并行参数 =====
    print("\n[2/3] 准备并行计算...")
    
    model_path = project_root / "models" / "TD3" / "Nov17_06-22-08_archived"
    observation_error = 0.01
    sample_interval = 10  # 保持原始采样间隔
    
    # 决定并行进程数
    n_workers = min(n_trajectories, n_cores // 2)  # 每个进程用 2 核
    print(f"  并行进程数: {n_workers}")
    print(f"  观测误差: ±{observation_error}")
    print(f"  采样间隔: 每 {sample_interval} 步")
    print(f"  Bernstein 阶数: 1")
    print(f"  Bernstein 采样: 4000 步（保持原始精度）")
    
    # 构造参数列表
    args_list = [
        (i, traj, model_path, observation_error, sample_interval)
        for i, traj in enumerate(trajectories)
    ]
    
    # ===== 4. 并行执行 =====
    print(f"\n[3/3] 启动 {n_workers} 个并行进程...")
    print("="*70)
    
    start_time = time.time()
    
    # 使用进程池，添加异常处理
    try:
        with Pool(processes=n_workers) as pool:
            # map 会自动分配任务到各个进程
            results = pool.map(verify_single_trajectory_worker, args_list)
    except Exception as e:
        print(f"\n❌ 并行验证过程中出现错误: {e}")
        print("尝试保存已完成的部分结果...")
        import traceback
        traceback.print_exc()
        raise
    
    total_elapsed = time.time() - start_time
    
    # ===== 5. 汇总结果 =====
    print("\n" + "="*70)
    print("验证统计:")
    print("="*70)
    
    # 按轨迹索引排序
    results = sorted(results, key=lambda x: x[0])
    all_results = [r[1] for r in results]
    
    # 5.1 整体统计
    total_samples = sum(r['n_samples'] for r in all_results)
    total_safe = sum(r['safe_count'] for r in all_results)
    overall_safety_rate = total_safe / total_samples if total_samples > 0 else 0
    
    print(f"\n整体可达集安全性:")
    print(f"  总采样点: {total_samples}")
    print(f"  安全点数: {total_safe}")
    print(f"  安全率: {overall_safety_rate*100:.1f}%")
    
    # 5.2 轨迹分类统计
    goal_trajectories = [r for r in all_results if r['goal_reached']]
    collision_trajectories = [r for r in all_results if r['collision']]
    
    print(f"\n按轨迹结果分类:")
    print(f"  到达目标的轨迹: {len(goal_trajectories)}")
    if goal_trajectories:
        goal_safety = np.mean([r['safety_rate'] for r in goal_trajectories])
        print(f"    平均安全率: {goal_safety*100:.1f}%")
    
    print(f"  碰撞的轨迹: {len(collision_trajectories)}")
    if collision_trajectories:
        collision_safety = np.mean([r['safety_rate'] for r in collision_trajectories])
        print(f"    平均安全率: {collision_safety*100:.1f}%")
    
    # 5.3 可达集宽度统计
    all_widths_v = []
    all_widths_omega = []
    
    for result in all_results:
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
    
    # 5.4 性能统计
    print(f"\n性能统计:")
    print(f"  总耗时: {total_elapsed/60:.1f} 分钟 ({total_elapsed/3600:.2f} 小时)")
    print(f"  平均每轨迹: {total_elapsed/n_trajectories:.1f} 秒")
    print(f"  平均每采样点: {total_elapsed/total_samples:.2f} 秒")
    
    # 计算加速比
    avg_traj_time = np.mean([r['compute_time'] for r in all_results])
    serial_time = avg_traj_time * n_trajectories
    speedup = serial_time / total_elapsed
    
    print(f"\n并行加速:")
    print(f"  串行预计耗时: {serial_time/60:.1f} 分钟 ({serial_time/3600:.2f} 小时)")
    print(f"  并行实际耗时: {total_elapsed/60:.1f} 分钟 ({total_elapsed/3600:.2f} 小时)")
    print(f"  加速比: {speedup:.1f}x")
    print(f"  并行效率: {speedup/n_workers*100:.1f}%")
    
    # ===== 6. 保存结果 =====
    output_path = Path(__file__).parent.parent / "assets" / "reachability_results_parallel.json"
    
    output_data = {
        'metadata': {
            'n_trajectories': n_trajectories,
            'total_samples': total_samples,
            'observation_error': observation_error,
            'sample_interval': sample_interval,
            'bern_order': 1,
            'error_steps': 4000,
            'n_workers': n_workers,
            'n_cores': n_cores,
            'elapsed_time': total_elapsed,
            'speedup': speedup,
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
    
    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✅ 结果已保存到: {output_path}")
        print(f"   文件大小: {output_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"\n❌ 保存结果失败: {e}")
        print(f"   目标路径: {output_path}")
        import traceback
        traceback.print_exc()
        raise
    
    print("="*70)
    print("\n🎉 并行验证完成！")


if __name__ == "__main__":
    main()