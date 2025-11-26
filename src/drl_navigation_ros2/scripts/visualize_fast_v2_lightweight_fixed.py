#!/usr/bin/env python3
"""
POLAR 可达集可视化 - TD3_Lightweight版本（修正版）
直接使用 poses 数据，100%精确，无坐标系转换误差
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_data(use_pure_polar=True):
    """加载数据 - Lightweight版本"""
    traj_path = project_root / "assets" / "trajectories_lightweight.pkl"
    
    if use_pure_polar:
        result_path = project_root / "assets" / "reachability_results_pure_polar_lightweight.json"
    else:
        result_path = project_root / "assets" / "reachability_results_parallel_lightweight.json"
    
    if not traj_path.exists():
        raise FileNotFoundError(f"轨迹文件不存在: {traj_path}")
    
    if not result_path.exists():
        raise FileNotFoundError(f"验证结果文件不存在: {result_path}")
    
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    model_info = results.get('metadata', {})
    model_name = model_info.get('model', 'TD3_lightweight')
    hidden_dim = model_info.get('hidden_dim', 26)
    
    print(f"✅ 加载 {len(trajectories)} 条轨迹")
    print(f"✅ 使用 {'纯POLAR' if use_pure_polar else '增强版'} 验证结果")
    print(f"📊 模型: {model_name} (隐藏层: {hidden_dim} 神经元)")
    
    return trajectories, results


def simulate_reachable_tube(pos, yaw, ranges, T=15, dt=0.1):
    """
    模拟可达路径
    与clearpath_rl_polar完全一致的采样策略
    """
    v_range = ranges[0][1] - ranges[0][0]
    omega_range = ranges[1][1] - ranges[1][0]
    
    # 自适应采样密度
    if v_range < 0.001:
        n_v = 50
    elif v_range < 0.01:
        n_v = 40
    else:
        n_v = 30
    
    if omega_range < 0.01:
        n_omega = 50
    elif omega_range < 0.1:
        n_omega = 40
    else:
        n_omega = 30
    
    v_samples = np.linspace(ranges[0][0], ranges[0][1], n_v)
    omega_samples = np.linspace(ranges[1][0], ranges[1][1], n_omega)
    
    all_paths = []
    
    for v in v_samples:
        for omega in omega_samples:
            path = [pos.copy()]
            p = pos.copy()
            theta = yaw
            
            for _ in range(T):
                # 🔧 修正：与collect_trajectories.py保持一致
                # v的范围是[-1, 1]，需要先映射到[0, 1]再乘0.5
                v_real = ((v + 1) / 2) * 0.5  # ← 关键修正
                p = p + dt * np.array([v_real * np.cos(theta), v_real * np.sin(theta)])
                theta += omega * dt
                path.append(p.copy())
            
            all_paths.append(np.array(path))
    
    return all_paths


def visualize_single_trajectory(traj_idx, trajectory_data, verification_result, 
                                step_interval=1, model_name='TD3_lightweight'):
    """
    可视化单条轨迹 - 修正版
    直接使用 poses 数据，无坐标系转换
    """
    print(f"\n可视化轨迹 {traj_idx+1}...")
    
    # ✅ 直接使用 poses 数据（精确的位姿）
    poses = trajectory_data['poses']  # (T, 3) - (x, y, θ)
    actions = trajectory_data['actions']
    robot_start = trajectory_data['robot_start']
    target_pos = trajectory_data['target_pos']
    
    # 从验证结果提取可达集
    verified_steps = {r['step']: r for r in verification_result['results']}
    
    print(f"  轨迹长度: {len(poses)} 步")
    print(f"  验证采样点: {len(verified_steps)} 个")
    
    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # ===== 左图：密集可达管道 =====
    ax_main = axes[0]
    ax_main.set_title(f'Dense Reachable Tube ({model_name}, error=1.0%, every {step_interval} steps)', 
                      fontsize=14, fontweight='bold', pad=15)
    ax_main.set_xlabel('X Position (m)', fontsize=12)
    ax_main.set_ylabel('Y Position (m)', fontsize=12)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.set_aspect('equal')
    
    # 收集所有可达点
    all_reachable_points = []
    total_steps = 0
    available_steps = sorted(verified_steps.keys())
    
    for step_idx in available_steps:
        if step_idx >= len(poses):  # ✅ 修正：使用 poses 的长度
            break
        
        result_data = verified_steps[step_idx]
        ranges = result_data['action_ranges']
        
        # ✅ 修正：直接使用 poses（真实位姿）
        pos = poses[step_idx][:2]  # (x, y)
        yaw = poses[step_idx][2]   # θ
        
        # 模拟可达路径
        paths = simulate_reachable_tube(pos, yaw, ranges, T=15, dt=0.1)
        
        # 收集可达终点
        step_reachable_points = np.array([p[-1] for p in paths])
        all_reachable_points.append(step_reachable_points)
        
        # 绘制路径（半透明）
        for path in paths[::2]:
            ax_main.plot(path[:, 0], path[:, 1], 
                        color='lightgreen', alpha=0.03, linewidth=0.3, zorder=1)
        
        # ✅ 标记采样点（现在应该在轨迹上了）
        if step_idx % (step_interval * 1) == 0:
            ax_main.plot(pos[0], pos[1], 'o', color='orange', 
                        markersize=6, alpha=0.6, zorder=8)
        
        total_steps += 1
        if total_steps % 10 == 0:
            print(f"  ✓ 已处理 {total_steps} 个时间步...")
    
    # ✅ 绘制真实轨迹（直接使用poses）
    ax_main.plot(poses[:, 0], poses[:, 1], 
                'b-', linewidth=3.5, label='Actual Trajectory', zorder=10, alpha=0.9)
    
    # 起点、终点
    ax_main.plot(poses[0, 0], poses[0, 1], 
                'go', markersize=15, label='Start', zorder=11,
                markeredgecolor='darkgreen', markeredgewidth=2)
    ax_main.plot(poses[-1, 0], poses[-1, 1], 
                'ro', markersize=15, label='End', zorder=11,
                markeredgecolor='darkred', markeredgewidth=2)
    
    # 目标点
    ax_main.plot(target_pos[0], target_pos[1], 'g*', markersize=25, 
                label='Goal', zorder=12,
                markeredgecolor='darkgreen', markeredgewidth=2)
    goal_circle = Circle(target_pos, 0.3, fill=False, edgecolor='green', 
                        linestyle='--', linewidth=2, alpha=0.5)
    ax_main.add_patch(goal_circle)
    
    ax_main.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    # 自动调整坐标轴范围
    x_min = min(poses[:, 0].min(), target_pos[0]) - 0.5
    x_max = max(poses[:, 0].max(), target_pos[0]) + 0.5
    y_min = min(poses[:, 1].min(), target_pos[1]) - 0.5
    y_max = max(poses[:, 1].max(), target_pos[1]) + 0.5
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    
    # ===== 右图：累积可达边界 =====
    ax_tube = axes[1]
    ax_tube.set_title(f'Accumulated Reachable Envelope ({model_name}, error=1.0%)', 
                     fontsize=14, fontweight='bold', pad=15)
    ax_tube.set_xlabel('X Position (m)', fontsize=12)
    ax_tube.set_ylabel('Y Position (m)', fontsize=12)
    ax_tube.grid(True, alpha=0.3, linestyle='--')
    ax_tube.set_aspect('equal')
    
    print(f"  ✓ 绘制累积可达边界...")
    
    # 绘制整体外包络
    if len(all_reachable_points) > 0:
        all_points_combined = np.vstack(all_reachable_points)
        if len(all_points_combined) > 3:
            try:
                hull_global = ConvexHull(all_points_combined)
                hull_points_global = all_points_combined[hull_global.vertices]
                hull_points_global = np.vstack([hull_points_global, hull_points_global[0]])
                
                ax_tube.fill(hull_points_global[:, 0], hull_points_global[:, 1], 
                           color='lightgreen', alpha=0.2, zorder=1, label='Overall Envelope')
                ax_tube.plot(hull_points_global[:, 0], hull_points_global[:, 1], 
                           color='darkgreen', linewidth=2.5, alpha=0.6, zorder=2)
            except Exception as e:
                print(f"  ⚠️  绘制全局凸包失败: {e}")
    
    # 绘制管道内部散点
    all_tube_points = []
    for step_idx in available_steps:
        if step_idx >= len(poses):
            break
        
        result_data = verified_steps[step_idx]
        ranges = result_data['action_ranges']
        
        pos = poses[step_idx][:2]
        yaw = poses[step_idx][2]
        
        paths = simulate_reachable_tube(pos, yaw, ranges, T=15, dt=0.1)
        
        for path in paths[::3]:
            all_tube_points.extend(path[::2])
    
    if len(all_tube_points) > 100:
        all_tube_points = np.array(all_tube_points)
        
        ax_tube.scatter(all_tube_points[:, 0], all_tube_points[:, 1], 
                       c='green', s=1, alpha=0.02, zorder=1)
        
        try:
            hull_tube = ConvexHull(all_tube_points)
            hull_points_tube = all_tube_points[hull_tube.vertices]
            hull_points_tube = np.vstack([hull_points_tube, hull_points_tube[0]])
            ax_tube.plot(hull_points_tube[:, 0], hull_points_tube[:, 1], 
                        color='darkgreen', linewidth=2, alpha=0.8, 
                        label='Reachable Tube Boundary', zorder=3)
        except Exception as e:
            print(f"  ⚠️  绘制管道边界失败: {e}")
    
    # 真实轨迹
    ax_tube.plot(poses[:, 0], poses[:, 1], 
                'b-', linewidth=3.5, label='Actual Trajectory', zorder=10, alpha=0.9)
    
    # 起点、终点、目标
    ax_tube.plot(poses[0, 0], poses[0, 1], 
                'go', markersize=15, label='Start', zorder=11,
                markeredgecolor='darkgreen', markeredgewidth=2)
    ax_tube.plot(poses[-1, 0], poses[-1, 1], 
                'ro', markersize=15, label='End', zorder=11,
                markeredgecolor='darkred', markeredgewidth=2)
    ax_tube.plot(target_pos[0], target_pos[1], 'g*', markersize=25, 
                label='Goal', zorder=12,
                markeredgecolor='darkgreen', markeredgewidth=2)
    goal_circle2 = Circle(target_pos, 0.3, fill=False, edgecolor='green', 
                         linestyle='--', linewidth=2, alpha=0.5)
    ax_tube.add_patch(goal_circle2)
    
    ax_tube.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax_tube.set_xlim(x_min, x_max)
    ax_tube.set_ylim(y_min, y_max)
    
    # 总标题
    success_text = "✅ SUCCESS" if trajectory_data['goal_reached'] else "⚠️ INCOMPLETE"
    plt.suptitle(
        f'Dense Reachability Tube Visualization ({model_name}) - {success_text}\n' +
        f'Sampled {len(all_reachable_points)} time steps (every {step_interval} steps) | ' +
        f'Observation Error: 1.0%',
        fontsize=15,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout()
    
    filename = f'trajectory_{traj_idx+1:02d}_dense_tube_lightweight_fixed.png'
    save_path = project_root / "visualizations" / filename
    save_path.parent.mkdir(exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ 保存: {filename}")


def visualize_comparison(trajectories, results, selected_indices=None):
    """批量可视化多条轨迹"""
    if selected_indices is None:
        selected_indices = list(range(len(trajectories)))
    
    model_info = results.get('metadata', {})
    model_name = model_info.get('model', 'TD3_lightweight')
    hidden_dim = model_info.get('hidden_dim', 26)
    
    print(f"\n将可视化 {len(selected_indices)} 条轨迹")
    print(f"模型: {model_name} ({hidden_dim}神经元)")
    
    import time
    start = time.time()
    
    for idx in selected_indices:
        if idx >= len(trajectories):
            print(f"⚠️  跳过索引 {idx}（超出范围）")
            continue
        
        if trajectories[idx] is None:
            print(f"⚠️  跳过索引 {idx}（数据为空）")
            continue
        
        visualize_single_trajectory(
            idx, 
            trajectories[idx], 
            results['trajectories'][idx],
            step_interval=1,
            model_name=model_name
        )
    
    elapsed = time.time() - start
    
    print("\n" + "="*70)
    print(f"✅ 完成！耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"📁 可视化结果保存在: {project_root}/visualizations/")
    print(f"   文件命名格式: trajectory_XX_dense_tube_lightweight_fixed.png")
    print("="*70)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🎨 POLAR 可达集可视化 - TD3_Lightweight版本（修正版）")
    print("="*70)
    
    try:
        trajectories, results = load_data(use_pure_polar=True)
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("\n💡 提示:")
        print("   1. 请先运行 parallel_verify_reachability_pure_lightweight.py")
        print("   2. 确保生成了 reachability_results_pure_polar_lightweight.json")
        print("   3. 确保存在 trajectories_lightweight.pkl")
        return
    
    # 可视化所有轨迹
    visualize_comparison(trajectories, results)


if __name__ == "__main__":
    main()