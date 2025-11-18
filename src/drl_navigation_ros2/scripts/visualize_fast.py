#!/usr/bin/env python3
"""
POLAR 可达集可视化 - 极速版
关键优化：直接使用已计算的可达集，不重新计算！
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


def load_data():
    """加载数据"""
    with open("assets/trajectories.pkl", 'rb') as f:
        trajectories = pickle.load(f)
    
    with open("assets/reachability_results_parallel.json", 'r') as f:
        results = json.load(f)
    
    print(f"✅ 加载 {len(trajectories)} 条轨迹")
    return trajectories, results


def extract_world_trajectory(trajectory_data):
    """重建世界坐标轨迹"""
    robot_start = trajectory_data['robot_start']
    target_pos = trajectory_data['target_pos']
    actions = trajectory_data['actions']
    
    world_traj = []
    x, y, yaw = robot_start
    world_traj.append([x, y, yaw])
    
    dt = 0.1
    for action in actions:
        v = (action[0] + 1) / 2
        omega = action[1]
        
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        yaw += omega * dt
        
        world_traj.append([x, y, yaw])
    
    return np.array(world_traj), target_pos


def simulate_paths(pos, yaw, ranges, T=20, dt=0.1):
    """模拟可达路径"""
    v_min, v_max = ranges[0]
    omega_min, omega_max = ranges[1]
    
    # 简化采样
    n_v = 20
    n_omega = 20
    
    v_samples = np.linspace(v_min, v_max, n_v)
    omega_samples = np.linspace(omega_min, omega_max, n_omega)
    
    paths = []
    for v in v_samples:
        for omega in omega_samples:
            path = [pos.copy()]
            p = pos.copy()
            theta = yaw
            
            for _ in range(T):
                v_real = v * 0.5
                p = p + dt * np.array([v_real * np.cos(theta), v_real * np.sin(theta)])
                theta += omega * dt
                path.append(p.copy())
            
            paths.append(np.array(path))
    
    return paths


def visualize_single(traj_idx, trajectory_data, verification_result):
    """可视化单条轨迹（极速版）"""
    print(f"\n可视化轨迹 {traj_idx+1}...")
    
    states = trajectory_data['states']
    actions = trajectory_data['actions']
    world_traj, target_pos = extract_world_trajectory(trajectory_data)
    
    # 从验证结果提取可达集（已计算好的！）
    verified_steps = {r['step']: r['action_ranges'] for r in verification_result['results']}
    
    print(f"  使用 {len(verified_steps)} 个已计算的可达集")
    
    # 创建图
    fig = plt.figure(figsize=(20, 12))
    
    # ===== 主图 =====
    ax_main = plt.subplot(2, 3, (1, 4))
    ax_main.set_title('Reachable Sets at Key Moments', fontsize=14, fontweight='bold')
    ax_main.set_xlabel('X Position (m)', fontsize=12)
    ax_main.set_ylabel('Y Position (m)', fontsize=12)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.set_aspect('equal')
    
    # 真实轨迹
    ax_main.plot(world_traj[:, 0], world_traj[:, 1], 
                'b-', linewidth=3.5, label='Actual Trajectory', zorder=10, alpha=0.9)
    
    # 关键时刻
    n_steps = len(states)
    key_steps = [0, n_steps//3, 2*n_steps//3, n_steps-1]
    colors = ['orange', 'purple', 'cyan', 'magenta']
    
    available_steps = sorted(verified_steps.keys())
    
    for idx, (target_step, color) in enumerate(zip(key_steps, colors)):
        # 找最接近的已验证步数
        closest_step = min(available_steps, key=lambda x: abs(x - target_step))
        ranges = verified_steps[closest_step]
        
        pos = world_traj[closest_step, :2]
        yaw = world_traj[closest_step, 2]
        
        # 模拟路径
        paths = simulate_paths(pos, yaw, ranges, T=25)
        
        # 绘制路径
        for path in paths[::5]:
            ax_main.plot(path[:, 0], path[:, 1], 
                        color=color, alpha=0.05, linewidth=0.5, zorder=1)
        
        # 凸包
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
                            label=f'Step {closest_step}')
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
    
    # ===== 速度 =====
    ax_vel = plt.subplot(2, 3, 2)
    ax_vel.set_title('Action History', fontsize=12, fontweight='bold')
    ax_vel.plot(actions[:, 0], 'b-', linewidth=2, label='Linear Vel')
    ax_vel.plot(actions[:, 1], 'r-', linewidth=2, label='Angular Vel')
    ax_vel.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax_vel.set_xlabel('Step')
    ax_vel.set_ylabel('Action')
    ax_vel.legend()
    ax_vel.grid(True, alpha=0.3)
    
    # ===== 距离 =====
    ax_dist = plt.subplot(2, 3, 3)
    ax_dist.set_title('Distance to Goal', fontsize=12, fontweight='bold')
    distances = np.sqrt((world_traj[:, 0] - target_pos[0])**2 + 
                       (world_traj[:, 1] - target_pos[1])**2)
    ax_dist.plot(distances, 'g-', linewidth=2.5)
    ax_dist.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Threshold')
    ax_dist.fill_between(range(len(distances)), 0, 0.5, color='green', alpha=0.1)
    ax_dist.set_xlabel('Step')
    ax_dist.set_ylabel('Distance (m)')
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    
    # ===== 宽度 =====
    ax_width = plt.subplot(2, 3, 5)
    ax_width.set_title('Reachable Set Width', fontsize=12, fontweight='bold')
    steps = [r['step'] for r in verification_result['results']]
    widths_v = [r['width_v'] for r in verification_result['results']]
    widths_omega = [r['width_omega'] for r in verification_result['results']]
    ax_width.plot(steps, widths_v, 'b-o', linewidth=2, label='Linear Vel Width')
    ax_width.plot(steps, widths_omega, 'r-s', linewidth=2, label='Angular Vel Width')
    ax_width.set_xlabel('Step')
    ax_width.set_ylabel('Width')
    ax_width.legend()
    ax_width.grid(True, alpha=0.3)
    
    # ===== 统计 =====
    ax_stats = plt.subplot(2, 3, 6)
    ax_stats.axis('off')
    
    success_text = "✅ GOAL" if trajectory_data['goal_reached'] else "💥 COLLISION"
    
    stats_text = f"""
TRAJECTORY STATISTICS
{'='*35}

Total Steps:        {trajectory_data['steps']}
Result:             {success_text}
Total Reward:       {trajectory_data['total_reward']:.2f}

Start:              ({world_traj[0, 0]:.2f}, {world_traj[0, 1]:.2f})
End:                ({world_traj[-1, 0]:.2f}, {world_traj[-1, 1]:.2f})
Goal:               ({target_pos[0]:.2f}, {target_pos[1]:.2f})

Final Distance:     {distances[-1]:.3f} m

Avg Linear Vel:     {np.mean(actions[:, 0]):.3f}
Avg Angular Vel:    {np.mean(np.abs(actions[:, 1])):.3f} rad/s

POLAR Safety:       {verification_result['safety_rate']*100:.0f}%
Safe Samples:       {verification_result['safe_count']}/{verification_result['n_samples']}
    """
    
    ax_stats.text(0.1, 0.9, stats_text, 
                 transform=ax_stats.transAxes,
                 fontsize=10,
                 verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', 
                          alpha=0.3, edgecolor='blue', linewidth=2))
    
    plt.suptitle(
        f'POLAR Reachable Set Analysis - Trajectory #{traj_idx+1}\n' +
        f'{success_text} | Steps: {trajectory_data["steps"]} | Final Distance: {distances[-1]:.3f}m',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    
    save_path = f"visualizations/trajectory_{traj_idx+1:02d}_fast.png"
    Path("visualizations").mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ 保存: {save_path}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🎨 POLAR 可达集可视化 - 极速版")
    print("="*70)
    
    trajectories, results = load_data()
    
    # 可视化所有轨迹
    selected = list(range(len(trajectories)))
    # 或只可视化指定轨迹：selected = [0, 1, 5]
    
    print(f"\n将可视化 {len(selected)} 条轨迹")
    
    import time
    start = time.time()
    
    for idx in selected:
        visualize_single(idx, trajectories[idx], results['trajectories'][idx])
    
    elapsed = time.time() - start
    
    print("\n" + "="*70)
    print(f"✅ 完成！耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print("="*70)


if __name__ == "__main__":
    main()