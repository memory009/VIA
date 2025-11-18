#!/usr/bin/env python3
"""
POLAR 可达集可视化脚本 - 完全修复版
修复内容：
1. 统一参数名为 observation_error
2. 强制使用CPU避免显存溢出
3. 添加完整的错误处理
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull
from multiprocessing import Pool, cpu_count
import time
import traceback

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from TD3.TD3 import TD3
from verification.polar_verifier import verify_safety


# ============================================================================
# 数据加载
# ============================================================================

def load_trajectories_and_results(
    trajectories_path="assets/trajectories.pkl",
    results_path="assets/reachability_results_parallel.json"
):
    """加载轨迹数据和验证结果"""
    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"✅ 加载 {len(trajectories)} 条轨迹")
    print(f"✅ 加载 {len(results['trajectories'])} 条验证结果")
    
    return trajectories, results


def extract_world_trajectory(trajectory_data, sample_interval=1):
    """从状态序列重建世界坐标轨迹"""
    states = trajectory_data['states']
    robot_start = trajectory_data['robot_start']
    target_pos = trajectory_data['target_pos']
    
    world_traj = []
    x0, y0, yaw0 = robot_start
    world_traj.append([x0, y0, yaw0])
    
    if 'actions' in trajectory_data:
        actions = trajectory_data['actions']
        dt = 0.1
        
        x, y, yaw = x0, y0, yaw0
        
        for action in actions[::sample_interval]:
            v = (action[0] + 1) / 2
            omega = action[1]
            
            x += v * np.cos(yaw) * dt
            y += v * np.sin(yaw) * dt
            yaw += omega * dt
            
            world_traj.append([x, y, yaw])
    
    world_traj = np.array(world_traj)
    return world_traj, target_pos


# ============================================================================
# 可达集计算
# ============================================================================

def compute_reachable_set_at_step(agent, state, observation_error=0.01):
    """计算某一步的可达集"""
    is_safe, ranges = verify_safety(
        agent,
        state,
        observation_error=observation_error,
        bern_order=1,
        error_steps=4000,
    )
    return is_safe, ranges


def simulate_reachable_tube(pos, yaw, ranges, T=10, dt=0.1):
    """模拟可达管道"""
    v_range = ranges[0][1] - ranges[0][0]
    omega_range = ranges[1][1] - ranges[1][0]
    
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
                v_real = v * 0.5  # TurtleBot3 max_vel
                p = p + dt * np.array([v_real * np.cos(theta), v_real * np.sin(theta)])
                theta += omega * dt
                path.append(p.copy())
            
            all_paths.append(np.array(path))
    
    return all_paths


# ============================================================================
# 模式A：关键时刻可视化
# ============================================================================

def create_keymoments_visualization(
    agent,
    trajectory_data,
    verification_results,
    save_path="keymoments.png"
):
    """生成关键时刻可视化"""
    print(f"    [模式A] 生成关键时刻可视化...")
    
    try:
        states = trajectory_data['states']
        actions = trajectory_data['actions']
        world_traj, target_pos = extract_world_trajectory(trajectory_data)
        
        n_steps = len(states)
        key_steps = [0, n_steps//3, 2*n_steps//3, n_steps-1]
        
        fig = plt.figure(figsize=(20, 12))
        
        # 主图
        ax_main = plt.subplot(2, 3, (1, 4))
        ax_main.set_title('Reachable Sets at Key Moments', fontsize=14, fontweight='bold')
        ax_main.set_xlabel('X Position (m)', fontsize=12)
        ax_main.set_ylabel('Y Position (m)', fontsize=12)
        ax_main.grid(True, alpha=0.3, linestyle='--')
        ax_main.set_aspect('equal')
        
        # 真实轨迹
        ax_main.plot(world_traj[:, 0], world_traj[:, 1], 
                    'b-', linewidth=3.5, label='Actual Trajectory', zorder=10, alpha=0.9)
        
        # 关键时刻可达集
        colors = ['orange', 'purple', 'cyan', 'magenta']
        
        for idx, (step_idx, color) in enumerate(zip(key_steps, colors)):
            if step_idx >= len(states):
                continue
            
            state = states[step_idx]
            pos = world_traj[step_idx, :2]
            yaw = world_traj[step_idx, 2]
            
            # ✅ 使用正确的参数名
            is_safe, ranges = compute_reachable_set_at_step(agent, state, observation_error=0.01)
            
            paths = simulate_reachable_tube(pos, yaw, ranges, T=25, dt=0.1)
            
            for path in paths[::5]:
                ax_main.plot(path[:, 0], path[:, 1], 
                            color=color, alpha=0.05, linewidth=0.5, zorder=1)
            
            all_points = np.vstack([p[-1] for p in paths])
            
            if len(all_points) > 3:
                try:
                    hull = ConvexHull(all_points)
                    hull_points = all_points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    ax_main.fill(hull_points[:, 0], hull_points[:, 1], 
                                color=color, alpha=0.2, zorder=2)
                    ax_main.plot(hull_points[:, 0], hull_points[:, 1], 
                                color=color, linewidth=2.5, alpha=0.8,
                                label=f'Step {step_idx}')
                except:
                    pass
            
            ax_main.plot(pos[0], pos[1], 'o', color=color, 
                        markersize=12, zorder=11,
                        markeredgecolor='black', markeredgewidth=2)
        
        # 起点、终点、目标
        ax_main.plot(world_traj[0, 0], world_traj[0, 1], 
                    'go', markersize=15, label='Start', zorder=11,
                    markeredgecolor='darkgreen', markeredgewidth=2)
        ax_main.plot(world_traj[-1, 0], world_traj[-1, 1], 
                    'ro', markersize=15, label='End', zorder=11,
                    markeredgecolor='darkred', markeredgewidth=2)
        ax_main.plot(target_pos[0], target_pos[1], 'g*', markersize=25, 
                    label='Goal', zorder=12,
                    markeredgecolor='darkgreen', markeredgewidth=2)
        
        goal_circle = Circle(target_pos, 0.3, fill=False, edgecolor='green', 
                            linestyle='--', linewidth=2, alpha=0.5)
        ax_main.add_patch(goal_circle)
        
        ax_main.legend(loc='upper left', fontsize=10, framealpha=0.95)
        
        x_min = min(world_traj[:, 0].min(), target_pos[0]) - 1
        x_max = max(world_traj[:, 0].max(), target_pos[0]) + 1
        y_min = min(world_traj[:, 1].min(), target_pos[1]) - 1
        y_max = max(world_traj[:, 1].max(), target_pos[1]) + 1
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)
        
        # 速度曲线
        ax_vel = plt.subplot(2, 3, 2)
        ax_vel.set_title('Action History', fontsize=12, fontweight='bold')
        ax_vel.plot(actions[:, 0], 'b-', linewidth=2, label='Linear Vel')
        ax_vel.plot(actions[:, 1], 'r-', linewidth=2, label='Angular Vel')
        ax_vel.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax_vel.set_xlabel('Step', fontsize=10)
        ax_vel.set_ylabel('Action', fontsize=10)
        ax_vel.legend(fontsize=9)
        ax_vel.grid(True, alpha=0.3)
        
        # 距离演化
        ax_dist = plt.subplot(2, 3, 3)
        ax_dist.set_title('Distance to Goal', fontsize=12, fontweight='bold')
        distances = np.sqrt((world_traj[:, 0] - target_pos[0])**2 + 
                           (world_traj[:, 1] - target_pos[1])**2)
        ax_dist.plot(distances, 'g-', linewidth=2.5)
        ax_dist.axhline(0.5, color='red', linestyle='--', 
                       linewidth=2, alpha=0.5, label='Goal Threshold')
        ax_dist.fill_between(range(len(distances)), 0, 0.5, color='green', alpha=0.1)
        ax_dist.set_xlabel('Step', fontsize=10)
        ax_dist.set_ylabel('Distance (m)', fontsize=10)
        ax_dist.legend(fontsize=9)
        ax_dist.grid(True, alpha=0.3)
        
        # 可达集宽度
        ax_width = plt.subplot(2, 3, 5)
        ax_width.set_title('Reachable Set Width Evolution', fontsize=12, fontweight='bold')
        
        if verification_results:
            steps = [r['step'] for r in verification_results['results']]
            widths_v = [r['width_v'] for r in verification_results['results']]
            widths_omega = [r['width_omega'] for r in verification_results['results']]
            
            ax_width.plot(steps, widths_v, 'b-o', linewidth=2, markersize=6, label='Linear Vel Width')
            ax_width.plot(steps, widths_omega, 'r-s', linewidth=2, markersize=6, label='Angular Vel Width')
            ax_width.set_xlabel('Step', fontsize=10)
            ax_width.set_ylabel('Width', fontsize=10)
            ax_width.legend(fontsize=9)
            ax_width.grid(True, alpha=0.3)
        
        # 统计信息
        ax_stats = plt.subplot(2, 3, 6)
        ax_stats.axis('off')
        
        success_text = "✅ GOAL" if trajectory_data['goal_reached'] else "💥 COLLISION"
        
        stats_text = f"""
TRAJECTORY STATISTICS
{'='*35}

Total Steps:        {trajectory_data['steps']}
Result:             {success_text}
Total Reward:       {trajectory_data['total_reward']:.2f}

Start Position:     ({world_traj[0, 0]:.2f}, {world_traj[0, 1]:.2f})
End Position:       ({world_traj[-1, 0]:.2f}, {world_traj[-1, 1]:.2f})
Goal Position:      ({target_pos[0]:.2f}, {target_pos[1]:.2f})

Final Distance:     {distances[-1]:.3f} m

Avg Linear Vel:     {np.mean(actions[:, 0]):.3f}
Avg Angular Vel:    {np.mean(np.abs(actions[:, 1])):.3f} rad/s

POLAR Safety:       {verification_results['safety_rate']*100:.0f}%
Safe Samples:       {verification_results['safe_count']}/{verification_results['n_samples']}
        """
        
        ax_stats.text(0.1, 0.9, stats_text, 
                     transform=ax_stats.transAxes,
                     fontsize=10,
                     verticalalignment='top',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=1', 
                              facecolor='lightblue', 
                              alpha=0.3,
                              edgecolor='blue',
                              linewidth=2))
        
        plt.suptitle(
            f'POLAR Reachable Set Analysis - Trajectory #{verification_results["trajectory_idx"]+1}\n' +
            f'{success_text} | Steps: {trajectory_data["steps"]} | Final Distance: {distances[-1]:.3f}m',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"      ✓ 保存: {save_path}")
        return True
        
    except Exception as e:
        print(f"      ✗ 失败: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# 模式B：密集管道可视化
# ============================================================================

def create_dense_tube_visualization(
    agent,
    trajectory_data,
    verification_results,
    step_interval=5,
    observation_error=0.01,
    save_path="dense.png"
):
    """生成密集管道可视化"""
    print(f"    [模式B] 生成密集管道可视化 (interval={step_interval})...")
    
    try:
        states = trajectory_data['states']
        world_traj, target_pos = extract_world_trajectory(trajectory_data, sample_interval=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        
        # 左图：密集路径
        ax_main = axes[0]
        ax_main.set_title(f'Dense Reachable Tube (error={observation_error*100:.1f}%, every {step_interval} steps)', 
                          fontsize=14, fontweight='bold', pad=15)
        ax_main.set_xlabel('X Position (m)', fontsize=12)
        ax_main.set_ylabel('Y Position (m)', fontsize=12)
        ax_main.grid(True, alpha=0.3, linestyle='--')
        ax_main.set_aspect('equal')
        
        all_reachable_points = []
        
        for step_idx in range(0, len(states), step_interval):
            if step_idx >= len(world_traj) - 1:
                break
            
            state = states[step_idx]
            pos = world_traj[step_idx, :2]
            yaw = world_traj[step_idx, 2]
            
            # ✅ 使用正确的参数名
            is_safe, ranges = compute_reachable_set_at_step(agent, state, observation_error)
            
            paths = simulate_reachable_tube(pos, yaw, ranges, T=15, dt=0.1)
            
            step_points = np.array([p[-1] for p in paths])
            all_reachable_points.append(step_points)
            
            for path in paths[::3]:
                ax_main.plot(path[:, 0], path[:, 1], 
                            color='lightgreen', alpha=0.03, linewidth=0.3, zorder=1)
            
            if step_idx % (step_interval * 3) == 0:
                ax_main.plot(pos[0], pos[1], 'o', color='orange', 
                            markersize=6, alpha=0.6, zorder=8)
        
        # 真实轨迹
        ax_main.plot(world_traj[:, 0], world_traj[:, 1], 
                    'b-', linewidth=3.5, label='Actual Trajectory', zorder=10, alpha=0.9)
        
        # 起点、终点、目标
        ax_main.plot(world_traj[0, 0], world_traj[0, 1], 
                    'go', markersize=15, label='Start', zorder=11,
                    markeredgecolor='darkgreen', markeredgewidth=2)
        ax_main.plot(world_traj[-1, 0], world_traj[-1, 1], 
                    'ro', markersize=15, label='End', zorder=11,
                    markeredgecolor='darkred', markeredgewidth=2)
        ax_main.plot(target_pos[0], target_pos[1], 'g*', markersize=25, 
                    label='Goal', zorder=12,
                    markeredgecolor='darkgreen', markeredgewidth=2)
        
        goal_circle = Circle(target_pos, 0.5, fill=False, edgecolor='green', 
                            linestyle='--', linewidth=2, alpha=0.5)
        ax_main.add_patch(goal_circle)
        
        ax_main.legend(loc='upper left', fontsize=10, framealpha=0.95)
        
        x_min = min(world_traj[:, 0].min(), target_pos[0]) - 1
        x_max = max(world_traj[:, 0].max(), target_pos[0]) + 1
        y_min = min(world_traj[:, 1].min(), target_pos[1]) - 1
        y_max = max(world_traj[:, 1].max(), target_pos[1]) + 1
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)
        
        # 右图：累积边界
        ax_tube = axes[1]
        ax_tube.set_title(f'Accumulated Reachable Envelope', 
                         fontsize=14, fontweight='bold', pad=15)
        ax_tube.set_xlabel('X Position (m)', fontsize=12)
        ax_tube.set_ylabel('Y Position (m)', fontsize=12)
        ax_tube.grid(True, alpha=0.3, linestyle='--')
        ax_tube.set_aspect('equal')
        
        # 全局凸包
        if len(all_reachable_points) > 0:
            all_points_combined = np.vstack(all_reachable_points)
            if len(all_points_combined) > 3:
                try:
                    hull_global = ConvexHull(all_points_combined)
                    hull_points = all_points_combined[hull_global.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    
                    ax_tube.fill(hull_points[:, 0], hull_points[:, 1], 
                               color='lightgreen', alpha=0.2, zorder=1, label='Overall Envelope')
                    ax_tube.plot(hull_points[:, 0], hull_points[:, 1], 
                               color='darkgreen', linewidth=2.5, alpha=0.6, zorder=2)
                except:
                    pass
        
        # 散点
        if len(all_reachable_points) > 0:
            all_points = np.vstack(all_reachable_points)
            ax_tube.scatter(all_points[:, 0], all_points[:, 1], 
                           c='green', alpha=0.05, s=10, zorder=1)
        
        # 真实轨迹、起点、终点、目标
        ax_tube.plot(world_traj[:, 0], world_traj[:, 1], 
                    'b-', linewidth=3.5, label='Actual Trajectory', zorder=10, alpha=0.9)
        ax_tube.plot(world_traj[0, 0], world_traj[0, 1], 
                    'go', markersize=15, label='Start', zorder=11,
                    markeredgecolor='darkgreen', markeredgewidth=2)
        ax_tube.plot(world_traj[-1, 0], world_traj[-1, 1], 
                    'ro', markersize=15, label='End', zorder=11,
                    markeredgecolor='darkred', markeredgewidth=2)
        ax_tube.plot(target_pos[0], target_pos[1], 'g*', markersize=25, 
                    label='Goal', zorder=12,
                    markeredgecolor='darkgreen', markeredgewidth=2)
        
        ax_tube.legend(loc='upper left', fontsize=10, framealpha=0.95)
        ax_tube.set_xlim(x_min, x_max)
        ax_tube.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"      ✓ 保存: {save_path}")
        return True
        
    except Exception as e:
        print(f"      ✗ 失败: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# 并行处理
# ============================================================================

def visualize_single_trajectory_worker(args):
    """单条轨迹可视化（并行）"""
    traj_idx, traj_data, verif_result, model_path, mode, params = args
    
    print(f"\n[进程 {traj_idx+1}] 开始可视化...")
    
    try:
        # ✅ 强制CPU
        device = torch.device("cpu")
        
        agent = TD3(
            state_dim=25,
            action_dim=2,
            max_action=1.0,
            device=device,
            load_model=True,
            model_name="TD3",
            load_directory=model_path,
        )
        
        success = True
        
        if mode == 'keymoments' or mode == 'both':
            save_path = f"visualizations/trajectory_{traj_idx+1:02d}_keymoments.png"
            Path("visualizations").mkdir(exist_ok=True)
            success = create_keymoments_visualization(agent, traj_data, verif_result, save_path)
        
        if success and (mode == 'dense' or mode == 'both'):
            save_path = f"visualizations/trajectory_{traj_idx+1:02d}_dense.png"
            Path("visualizations").mkdir(exist_ok=True)
            success = create_dense_tube_visualization(
                agent, traj_data, verif_result, 
                step_interval=params['step_interval'],
                observation_error=params['observation_error'],  # ✅ 使用正确的参数名
                save_path=save_path
            )
        
        if success:
            print(f"[进程 {traj_idx+1}] ✅ 完成！")
        else:
            print(f"[进程 {traj_idx+1}] ⚠️  部分失败")
        
        return traj_idx
        
    except Exception as e:
        print(f"[进程 {traj_idx+1}] ❌ 失败: {e}")
        traceback.print_exc()
        return -1


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("\n" + "="*70)
    print("🎨 POLAR 可达集可视化工具 (完全修复版)")
    print("="*70)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    trajectories, results = load_trajectories_and_results()
    
    # 2. 选择模式
    print("\n[2/4] 选择可视化模式:")
    print("  1. 关键时刻对比")
    print("  2. 密集可达管道")
    print("  3. 两种都生成")
    
    mode_choice = 3
    mode = {1: 'keymoments', 2: 'dense', 3: 'both'}[mode_choice]
    print(f"  ✓ 选择模式: {mode}")
    
    # 3. 设置参数
    params = {
        'step_interval': 5,
        'observation_error': 0.01,  # ✅ 使用正确的参数名
    }
    
    model_path = Path("models/TD3/Nov17_06-22-08_archived")
    
    # 4. 选择轨迹
    print("\n[3/4] 选择要可视化的轨迹:")
    print(f"  总共 {len(trajectories)} 条轨迹")
    
    # 测试：只可视化第一条轨迹
    selected_indices = [0]
    # 可视化所有：selected_indices = list(range(len(trajectories)))
    # 可视化指定：selected_indices = [0, 2, 5]
    
    print(f"  ✓ 将可视化 {len(selected_indices)} 条轨迹: {selected_indices}")
    
    # 5. 并行可视化
    print("\n[4/4] 开始并行可视化...")
    
    n_cores = cpu_count()
    n_workers = min(len(selected_indices), max(1, n_cores // 2))
    print(f"  使用 {n_workers} 个并行进程（CPU模式）")
    
    args_list = []
    for idx in selected_indices:
        traj_data = trajectories[idx]
        verif_result = results['trajectories'][idx]
        args_list.append((idx, traj_data, verif_result, model_path, mode, params))
    
    start_time = time.time()
    
    with Pool(processes=n_workers) as pool:
        results_list = pool.map(visualize_single_trajectory_worker, args_list)
    
    elapsed = time.time() - start_time
    
    successful = sum(1 for r in results_list if r >= 0)
    
    print("\n" + "="*70)
    print(f"✅ 可视化完成！")
    print(f"  成功: {successful}/{len(selected_indices)}")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    print("="*70)
    print("\n生成的文件位于 visualizations/ 目录:")
    
    if mode == 'keymoments' or mode == 'both':
        print("  - trajectory_XX_keymoments.png (关键时刻对比)")
    if mode == 'dense' or mode == 'both':
        print("  - trajectory_XX_dense.png (密集可达管道)")


if __name__ == "__main__":
    main()