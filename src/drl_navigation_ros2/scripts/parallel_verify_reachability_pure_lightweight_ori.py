#!/usr/bin/env python3
"""
并行可达性验证脚本 - 纯POLAR版本 (TD3_lightweight)
移除光线投射，完全遵循论文方法
使用轻量级网络加速计算
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

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ← 修改1: 导入轻量级TD3
from TD3.TD3_lightweight import TD3 as TD3_Lightweight


def compute_reachable_set_pure_polar(
    actor,
    state,
    observation_error=0.01,
    bern_order=1,
    error_steps=4000,
    max_action=1.0,
):
    """
    纯POLAR可达性验证 - 与clearpath_rl_polar完全一致
    支持动态网络结构（自动适配隐藏层维度）
    """
    import sympy as sym
    from verification.taylor_model import (
        TaylorModel,
        TaylorArithmetic,
        BernsteinPolynomial,
        compute_tm_bounds,
        apply_activation,
    )
    
    # 1. 提取Actor权重
    weights = []
    biases = []
    
    with torch.no_grad():
        for name, param in actor.named_parameters():
            if 'weight' in name:
                weights.append(param.cpu().numpy())
            elif 'bias' in name:
                biases.append(param.cpu().numpy())
    
    # 验证网络结构
    state_dim = len(state)
    assert weights[0].shape[1] == state_dim, \
        f"输入维度不匹配: 期望 {state_dim}, 实际 {weights[0].shape[1]}"
    assert weights[-1].shape[0] == 2, \
        f"输出维度不匹配: 期望 2, 实际 {weights[-1].shape[0]}"
    
    # 自动检测隐藏层维度
    hidden_dim = weights[0].shape[0]
    # print(f"[POLAR] 检测到网络结构: {state_dim} → {hidden_dim} → {hidden_dim} → 2")
    
    # 2. 创建符号变量
    z_symbols = [sym.Symbol(f'z{i}') for i in range(state_dim)]
    
    # 3. 构造输入Taylor模型
    TM_state = []
    for i in range(state_dim):
        poly = sym.Poly(
            observation_error * z_symbols[i] + state[i], 
            *z_symbols
        )
        TM_state.append(TaylorModel(poly, [0.0, 0.0]))
    
    # 4. 逐层传播
    TM_input = TM_state
    TA = TaylorArithmetic()
    BP = BernsteinPolynomial(error_steps=error_steps)
    
    num_layers = len(biases)
    
    for layer_idx in range(num_layers):
        TM_temp = []
        W = weights[layer_idx]
        b = biases[layer_idx]
        
        num_neurons = len(b)
        
        for neuron_idx in range(num_neurons):
            # 线性变换
            tm_neuron = TA.weighted_sumforall(
                TM_input,
                W[neuron_idx],
                b[neuron_idx]
            )
            
            # 激活函数
            is_hidden = (layer_idx < num_layers - 1)
            
            if is_hidden:
                # ReLU (使用论文Equation 8的优化)
                a, b_bound = compute_tm_bounds(tm_neuron)
                
                if a >= 0:
                    # 情况1: 完全激活
                    TM_after = tm_neuron
                elif b_bound <= 0:
                    # 情况2: 完全不激活
                    zero_poly = sym.Poly(0, *z_symbols)
                    TM_after = TaylorModel(zero_poly, [0, 0])
                else:
                    # 情况3: 跨越零点，使用Bernstein多项式
                    bern_poly = BP.approximate(a, b_bound, bern_order, 'relu')
                    bern_error = BP.compute_error(a, b_bound, 'relu')
                    TM_after = apply_activation(
                        tm_neuron, bern_poly, bern_error, bern_order
                    )
            else:
                # 输出层: Tanh
                a, b_bound = compute_tm_bounds(tm_neuron)
                bern_poly = BP.approximate(a, b_bound, bern_order, 'tanh')
                bern_error = BP.compute_error(a, b_bound, 'tanh')
                TM_after = apply_activation(
                    tm_neuron, bern_poly, bern_error, bern_order
                )
                # 缩放到动作空间
                TM_after = TA.constant_product(TM_after, max_action)
            
            TM_temp.append(TM_after)
        
        TM_input = TM_temp
    
    # 5. 计算动作可达集
    action_ranges = []
    for tm in TM_input:
        a, b = compute_tm_bounds(tm)
        action_ranges.append([a, b])
    
    return action_ranges


def check_action_safety_simple(action_ranges, state):
    """
    简单的安全性检查 - 与clearpath_rl_polar一致
    只基于可达集宽度和激光雷达数据
    """
    # 1. 检查可达集宽度
    for i, (min_val, max_val) in enumerate(action_ranges):
        range_width = max_val - min_val
        if range_width > 1.5:
            return False
    
    # 2. 检查碰撞风险（基于激光雷达）
    laser_readings = state[2:10]  # 8个激光数据（已归一化）
    min_laser = np.min(laser_readings)
    
    if min_laser < 0.05:  # 很近的障碍物
        linear_vel_range = action_ranges[0]
        if linear_vel_range[1] > 0.3:  # 可能前进
            return False
    
    # 3. 检查动作范围
    if action_ranges[0][0] < -0.6 or action_ranges[0][1] > 0.6:
        return False
    if action_ranges[1][0] < -1.1 or action_ranges[1][1] > 1.1:
        return False
    
    return True


def verify_single_trajectory_worker(args):
    """单个轨迹的验证函数（纯POLAR + lightweight版本）"""
    trajectory_idx, trajectory_data, model_path, observation_error, sample_interval = args
    
    # ← 修改2: 加载轻量级模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3_Lightweight(
        state_dim=25,
        action_dim=2,
        max_action=1.0,
        device=device,
        hidden_dim=26,  # ← 轻量级网络的隐藏层维度
        load_model=True,
        model_name="TD3_lightweight_best",  # ← 轻量级模型名称
        load_directory=model_path,
    )
    
    # 提取状态并采样
    states = trajectory_data['states']
    poses = trajectory_data['poses']
    
    sampled_states = states[::sample_interval]
    sampled_poses = poses[::sample_interval]
    n_samples = len(sampled_states)
    
    print(f"[进程 {trajectory_idx+1}] 开始验证 {n_samples} 个采样点（纯POLAR-Lightweight）...")
    
    results = []
    safe_count = 0
    start_time = time.time()
    
    for i, (state, pose) in enumerate(zip(sampled_states, sampled_poses)):
        step_idx = i * sample_interval
        
        if i % max(1, n_samples // 4) == 0:
            elapsed = time.time() - start_time
            print(f"[进程 {trajectory_idx+1}] 进度: {i+1}/{n_samples} "
                  f"({i/n_samples*100:.0f}%) | 已用时: {elapsed/60:.1f}分钟")
        
        # 纯POLAR计算可达集
        action_ranges = compute_reachable_set_pure_polar(
            agent.actor,
            state,
            observation_error=observation_error,
            bern_order=1,
            error_steps=4000,
            max_action=1.0,
        )
        
        # 简单安全性检查
        is_safe = check_action_safety_simple(action_ranges, state)
        
        det_action = agent.get_action(state, add_noise=False)
        width_v = action_ranges[0][1] - action_ranges[0][0]
        width_omega = action_ranges[1][1] - action_ranges[1][0]
        
        if is_safe:
            safe_count += 1
        
        result = {
            'step': step_idx,
            'pose': pose.tolist(),
            'det_action': det_action.tolist(),
            'action_ranges': action_ranges,
            'is_safe': is_safe,
            'width_v': float(width_v),
            'width_omega': float(width_omega),
            'min_laser': float(np.min(state[:20])),
            'distance': float(state[20]),
        }
        results.append(result)
    
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


def load_trajectories(pkl_path=None):
    """加载保存的轨迹"""
    if pkl_path is None:
        # ← 修改3: 使用lightweight的轨迹文件
        pkl_path = Path(__file__).parent.parent / "assets" / "trajectories_lightweight.pkl"
    
    if not pkl_path.exists():
        raise FileNotFoundError(f"轨迹文件不存在: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    valid_trajectories = [t for t in trajectories if t is not None]
    return valid_trajectories


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🚀 纯POLAR并行验证工具 (TD3_Lightweight)")
    print("="*70)
    
    n_cores = cpu_count()
    print(f"\n检测到 CPU 核心数: {n_cores}")
    
    print("\n[1/3] 加载轨迹...")
    trajectories = load_trajectories()
    n_trajectories = len(trajectories)
    print(f"  ✅ 加载 {n_trajectories} 条轨迹")
    
    total_states = sum(t['steps'] for t in trajectories)
    print(f"  总状态数: {total_states}")
    
    print("\n[2/3] 准备并行计算...")
    
    # ← 修改4: 使用lightweight模型路径
    model_path = project_root / "models" / "TD3_lightweight" / "Nov19_01-37-30_cheeson"
    observation_error = 0.01
    sample_interval = 1
    
    n_workers = min(n_trajectories, n_cores // 2)
    print(f"  模型: TD3_Lightweight (26神经元)")
    print(f"  模型路径: {model_path}")
    print(f"  并行进程数: {n_workers}")
    print(f"  观测误差: ±{observation_error}")
    print(f"  采样间隔: 每 {sample_interval} 步")
    
    args_list = [
        (i, traj, model_path, observation_error, sample_interval)
        for i, traj in enumerate(trajectories)
    ]
    
    print(f"\n[3/3] 启动 {n_workers} 个并行进程...")
    print("="*70)
    
    start_time = time.time()
    
    try:
        with Pool(processes=n_workers) as pool:
            results = pool.map(verify_single_trajectory_worker, args_list)
    except Exception as e:
        print(f"\n❌ 并行验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("验证统计:")
    print("="*70)
    
    results = sorted(results, key=lambda x: x[0])
    all_results = [r[1] for r in results]
    
    total_samples = sum(r['n_samples'] for r in all_results)
    total_safe = sum(r['safe_count'] for r in all_results)
    overall_safety_rate = total_safe / total_samples if total_samples > 0 else 0
    
    print(f"\n整体可达集安全性:")
    print(f"  总采样点: {total_samples}")
    print(f"  安全点数: {total_safe}")
    print(f"  安全率: {overall_safety_rate*100:.1f}%")
    
    # 轨迹分类统计
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
    
    # 可达集宽度统计
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
    
    print(f"\n性能统计:")
    print(f"  总耗时: {total_elapsed/60:.1f} 分钟 ({total_elapsed/3600:.2f} 小时)")
    print(f"  平均每轨迹: {total_elapsed/n_trajectories:.1f} 秒")
    print(f"  平均每采样点: {total_elapsed/total_samples:.2f} 秒")
    
    # 计算加速比
    avg_traj_time = np.mean([r['compute_time'] for r in all_results])
    serial_time = avg_traj_time * n_trajectories
    speedup = serial_time / total_elapsed
    
    print(f"\n并行加速:")
    print(f"  串行预计耗时: {serial_time/60:.1f} 分钟")
    print(f"  并行实际耗时: {total_elapsed/60:.1f} 分钟")
    print(f"  加速比: {speedup:.1f}x")
    print(f"  并行效率: {speedup/n_workers*100:.1f}%")
    
    # ← 修改5: 保存到lightweight专用文件
    output_path = Path(__file__).parent.parent / "assets" / "reachability_results_pure_polar_lightweight_ori.json"
    
    output_data = {
        'metadata': {
            'method': 'pure_polar',
            'model': 'TD3_lightweight',
            'hidden_dim': 26,
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
        import traceback
        traceback.print_exc()
        raise
    
    print("="*70)
    print("\n🎉 纯POLAR验证完成（Lightweight版本）！")
    print(f"💡 提示: 轻量级网络参数量减少约99.8%，计算速度更快")


if __name__ == "__main__":
    main()