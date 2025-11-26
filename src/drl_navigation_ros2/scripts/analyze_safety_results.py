#!/usr/bin/env python3
"""
分析可达集安全验证结果
读取 JSON 并生成详细报告
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def analyze_safety_results(json_path=None):
    """分析安全验证结果"""
    if json_path is None:
        json_path = Path("assets/reachability_results_pure_polar_lightweight.json")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*70)
    print("🔍 可达集安全验证详细分析")
    print("="*70)
    
    # 1. 总体统计
    summary = data['summary']
    print(f"\n📊 总体统计:")
    print(f"  总轨迹数: {data['metadata']['n_trajectories']}")
    print(f"  总采样点: {summary['total_samples']}")
    print(f"  总安全点: {summary['total_safe']}")
    print(f"  整体安全率: {summary['overall_safety_rate']*100:.1f}%")
    print(f"  到达目标轨迹: {summary['goal_trajectories']}")
    print(f"  碰撞轨迹: {summary['collision_trajectories']}")
    
    # 2. 按轨迹结果分类
    trajectories = data['trajectories']
    
    goal_trajs = [t for t in trajectories if t['goal_reached']]
    collision_trajs = [t for t in trajectories if t['collision']]
    incomplete_trajs = [t for t in trajectories if not t['goal_reached'] and not t['collision']]
    
    print(f"\n🎯 按结果分类:")
    
    if goal_trajs:
        goal_safety_rates = [t['safety_rate'] for t in goal_trajs]
        print(f"  到达目标 ({len(goal_trajs)}条):")
        print(f"    平均安全率: {np.mean(goal_safety_rates)*100:.1f}%")
        print(f"    最高: {np.max(goal_safety_rates)*100:.1f}%")
        print(f"    最低: {np.min(goal_safety_rates)*100:.1f}%")
    
    if collision_trajs:
        collision_safety_rates = [t['safety_rate'] for t in collision_trajs]
        print(f"  碰撞轨迹 ({len(collision_trajs)}条):")
        print(f"    平均安全率: {np.mean(collision_safety_rates)*100:.1f}%")
        print(f"    最高: {np.max(collision_safety_rates)*100:.1f}%")
        print(f"    最低: {np.min(collision_safety_rates)*100:.1f}%")
    
    if incomplete_trajs:
        incomplete_safety_rates = [t['safety_rate'] for t in incomplete_trajs]
        print(f"  未完成轨迹 ({len(incomplete_trajs)}条):")
        print(f"    平均安全率: {np.mean(incomplete_safety_rates)*100:.1f}%")
    
    # 3. 不安全段分析
    all_unsafe_segments = []
    for traj in trajectories:
        if 'unsafe_segments' in traj:
            all_unsafe_segments.extend(traj['unsafe_segments'])
    
    if all_unsafe_segments:
        segment_lengths = [seg['length'] for seg in all_unsafe_segments]
        print(f"\n⚠️  不安全段分析:")
        print(f"  总不安全段数: {len(all_unsafe_segments)}")
        print(f"  平均长度: {np.mean(segment_lengths):.1f} 步")
        print(f"  最长段: {np.max(segment_lengths)} 步")
        print(f"  最短段: {np.min(segment_lengths)} 步")
    
    # 4. 不安全原因统计
    all_reasons = {}
    for traj in trajectories:
        if 'unsafe_reasons_count' in traj:
            for reason, count in traj['unsafe_reasons_count'].items():
                all_reasons[reason] = all_reasons.get(reason, 0) + count
    
    if all_reasons:
        print(f"\n🚨 不安全原因统计:")
        total_unsafe = sum(all_reasons.values())
        for reason, count in sorted(all_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count} 次 ({count/total_unsafe*100:.1f}%)")
    
    # 5. 找出最危险的轨迹
    print(f"\n⚠️  最危险的5条轨迹:")
    sorted_trajs = sorted(trajectories, key=lambda t: t['safety_rate'])
    for i, traj in enumerate(sorted_trajs[:5]):
        status = "碰撞" if traj['collision'] else "到达" if traj['goal_reached'] else "未完成"
        print(f"  {i+1}. 轨迹 {traj['trajectory_idx']+1}: "
              f"安全率 {traj['safety_rate']*100:.1f}% | {status}")
    
    # 6. 逐轨迹详细输出
    print(f"\n📝 逐轨迹详细信息:")
    print("="*70)
    
    for traj in trajectories:
        idx = traj['trajectory_idx']
        status = "🎯 到达" if traj['goal_reached'] else "💥 碰撞" if traj['collision'] else "⏸️  未完成"
        
        print(f"\n轨迹 {idx+1}: {status}")
        print(f"  采样点: {traj['n_samples']}")
        print(f"  安全率: {traj['safety_rate']*100:.1f}% ({traj['safe_count']}/{traj['n_samples']})")
        print(f"  总步数: {traj['steps']}")
        print(f"  总奖励: {traj['total_reward']:.1f}")
        
        if 'unsafe_segments' in traj and len(traj['unsafe_segments']) > 0:
            print(f"  不安全段: {len(traj['unsafe_segments'])} 段")
            for seg in traj['unsafe_segments'][:2]:
                print(f"    步骤 {seg['start_step']}~{seg['end_step']} ({seg['length']}步)")
            if len(traj['unsafe_segments']) > 2:
                print(f"    ...还有 {len(traj['unsafe_segments'])-2} 段")
        
        if 'unsafe_reasons_count' in traj and traj['unsafe_reasons_count']:
            print(f"  不安全原因:")
            for reason, count in sorted(traj['unsafe_reasons_count'].items(), key=lambda x: -x[1])[:3]:
                print(f"    {reason}: {count} 次")
    
    print("\n" + "="*70)
    
    # 7. 生成可视化
    plot_safety_distribution(trajectories)


def plot_safety_distribution(trajectories):
    """绘制安全率分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 安全率直方图
    ax1 = axes[0, 0]
    safety_rates = [t['safety_rate'] for t in trajectories]
    ax1.hist(safety_rates, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Safety Rate')
    ax1.set_ylabel('Number of Trajectories')
    ax1.set_title('Distribution of Safety Rates')
    ax1.axvline(np.mean(safety_rates), color='r', linestyle='--', 
                label=f'Mean: {np.mean(safety_rates)*100:.1f}%')
    ax1.legend()
    
    # 2. 按结果分类的安全率箱线图
    ax2 = axes[0, 1]
    goal_rates = [t['safety_rate'] for t in trajectories if t['goal_reached']]
    collision_rates = [t['safety_rate'] for t in trajectories if t['collision']]
    
    data_to_plot = []
    labels = []
    if goal_rates:
        data_to_plot.append(goal_rates)
        labels.append(f'Goal\n(n={len(goal_rates)})')
    if collision_rates:
        data_to_plot.append(collision_rates)
        labels.append(f'Collision\n(n={len(collision_rates)})')
    
    ax2.boxplot(data_to_plot, labels=labels)
    ax2.set_ylabel('Safety Rate')
    ax2.set_title('Safety Rate by Trajectory Outcome')
    ax2.grid(True, alpha=0.3)
    
    # 3. 不安全原因饼图
    ax3 = axes[1, 0]
    all_reasons = {}
    for traj in trajectories:
        if 'unsafe_reasons_count' in traj:
            for reason, count in traj['unsafe_reasons_count'].items():
                all_reasons[reason] = all_reasons.get(reason, 0) + count
    
    if all_reasons:
        labels_pie = list(all_reasons.keys())
        sizes = list(all_reasons.values())
        ax3.pie(sizes, labels=labels_pie, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Unsafe Reasons Distribution')
    else:
        ax3.text(0.5, 0.5, 'No unsafe points', ha='center', va='center')
        ax3.set_title('Unsafe Reasons Distribution')
    
    # 4. 轨迹安全率时间序列
    ax4 = axes[1, 1]
    traj_indices = [t['trajectory_idx'] for t in trajectories]
    colors = ['green' if t['goal_reached'] else 'red' if t['collision'] else 'gray' 
              for t in trajectories]
    ax4.scatter(traj_indices, safety_rates, c=colors, alpha=0.6, s=100)
    ax4.set_xlabel('Trajectory Index')
    ax4.set_ylabel('Safety Rate')
    ax4.set_title('Safety Rate per Trajectory')
    ax4.axhline(np.mean(safety_rates), color='b', linestyle='--', alpha=0.5, 
                label=f'Mean: {np.mean(safety_rates)*100:.1f}%')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Goal Reached'),
        Patch(facecolor='red', alpha=0.6, label='Collision'),
        Patch(facecolor='gray', alpha=0.6, label='Incomplete')
    ]
    ax4.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    output_path = Path("visualizations/safety_analysis.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化图表已保存到: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    analyze_safety_results()