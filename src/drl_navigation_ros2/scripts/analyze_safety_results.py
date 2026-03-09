#!/usr/bin/env python3
"""
分析可达集安全验证结果
读取多个 JSON 文件并生成统计报告（均值±标准差）
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def format_seconds(seconds: float) -> str:
    """格式化秒数，便于阅读"""
    minutes = seconds / 60
    hours = seconds / 3600
    return f"{minutes:.1f} 分钟 ({hours:.2f} 小时)"


def extract_single_run_metrics(json_path):
    """从单个JSON文件中提取关键指标"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    trajectories = data['trajectories']

    # 提取所有轨迹的基础信息
    goal_trajectories = [t for t in trajectories if t['goal_reached']]
    collision_trajectories = [t for t in trajectories if t['collision']]

    # 计算指标
    n_total = len(trajectories)
    n_goal = len(goal_trajectories)
    n_collision = len(collision_trajectories)

    success_rate = n_goal / n_total if n_total > 0 else 0
    collision_rate = n_collision / n_total if n_total > 0 else 0

    # 计算平均步数（只计算成功的轨迹）
    if goal_trajectories:
        avg_steps_success = np.mean([t['steps'] for t in goal_trajectories])
    else:
        avg_steps_success = 0

    # 计算安全率
    if goal_trajectories:
        safety_rate_success = np.mean([t['safety_rate'] for t in goal_trajectories])
    else:
        safety_rate_success = 0

    if collision_trajectories:
        safety_rate_collision = np.mean([t['safety_rate'] for t in collision_trajectories])
    else:
        safety_rate_collision = 0

    # 整体安全率
    overall_safety_rate = data['summary']['overall_safety_rate']

    # 计算排除超时轨迹的整体安全率（只考虑成功+碰撞）
    completed_trajectories = [t for t in trajectories if t['goal_reached'] or t['collision']]
    if completed_trajectories:
        completed_samples = sum(t['n_samples'] for t in completed_trajectories)
        completed_safe = sum(t['safe_count'] for t in completed_trajectories)
        completed_safety_rate = completed_safe / completed_samples if completed_samples > 0 else 0
    else:
        completed_safety_rate = 0

    # 提取可达集宽度信息，转换为实际物理单位
    # 线速度：网络输出 [-1,1] → a_in = (action+1)/2，因此 width_v(m/s) = width_v_raw / 2
    # 角速度：网络输出 [-1,1] 直接作为 rad/s，无缩放
    all_widths_v = []
    all_widths_omega = []
    for traj in trajectories:
        if 'results' in traj:
            for result in traj['results']:
                if 'width_v' in result:
                    all_widths_v.append(result['width_v'] / 2)  # 转换为 m/s
                if 'width_omega' in result:
                    all_widths_omega.append(result['width_omega'])  # 已是 rad/s

    # 计算宽度统计量
    if all_widths_v:
        avg_width_v = np.mean(all_widths_v)
        std_width_v = np.std(all_widths_v)
        median_width_v = np.median(all_widths_v)
        min_width_v = np.min(all_widths_v)
    else:
        avg_width_v = std_width_v = median_width_v = min_width_v = 0

    if all_widths_omega:
        avg_width_omega = np.mean(all_widths_omega)
        std_width_omega = np.std(all_widths_omega)
        median_width_omega = np.median(all_widths_omega)
        min_width_omega = np.min(all_widths_omega)
    else:
        avg_width_omega = std_width_omega = median_width_omega = min_width_omega = 0

    return {
        'n_total': n_total,
        'n_goal': n_goal,
        'n_collision': n_collision,
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'avg_steps_success': avg_steps_success,
        'safety_rate_success': safety_rate_success,
        'safety_rate_collision': safety_rate_collision,
        'overall_safety_rate': overall_safety_rate,
        'completed_safety_rate': completed_safety_rate,  # 排除超时轨迹的安全率
        # 可达集宽度
        'avg_width_v': avg_width_v,
        'std_width_v': std_width_v,
        'median_width_v': median_width_v,
        'min_width_v': min_width_v,
        'avg_width_omega': avg_width_omega,
        'std_width_omega': std_width_omega,
        'median_width_omega': median_width_omega,
        'min_width_omega': min_width_omega,
    }


def analyze_multiple_runs(json_paths):
    """分析多个运行的结果，计算均值和标准差"""

    print("\n" + "="*70)
    print("🔍 多次运行统计分析（用于论文表格）")
    print("="*70)

    print(f"\n📂 分析文件列表:")
    for i, path in enumerate(json_paths, 1):
        print(f"  {i}. {path.name}")

    # 收集所有运行的指标
    all_metrics = []
    for json_path in json_paths:
        if not json_path.exists():
            print(f"⚠️  文件不存在，跳过: {json_path}")
            continue
        metrics = extract_single_run_metrics(json_path)
        all_metrics.append(metrics)

    if not all_metrics:
        print("❌ 没有找到有效的结果文件！")
        return

    n_runs = len(all_metrics)
    print(f"\n✅ 成功加载 {n_runs} 个运行结果\n")

    # 计算统计量
    success_rates = [m['success_rate'] * 100 for m in all_metrics]
    collision_rates = [m['collision_rate'] * 100 for m in all_metrics]
    avg_steps = [m['avg_steps_success'] for m in all_metrics]
    safety_rates_success = [m['safety_rate_success'] * 100 for m in all_metrics]

    # ✅ 修正：只收集有碰撞轨迹的运行结果
    safety_rates_collision = [m['safety_rate_collision'] * 100 for m in all_metrics if m['n_collision'] > 0]

    overall_safety_rates = [m['overall_safety_rate'] * 100 for m in all_metrics]
    completed_safety_rates = [m['completed_safety_rate'] * 100 for m in all_metrics]

    print("="*70)
    print("📊 论文表格数据（均值 ± 标准差）")
    print("="*70)

    print(f"\n✅ 成功率 (Success Rate):")
    print(f"   {np.mean(success_rates):.1f}% ± {np.std(success_rates):.1f}%")
    print(f"   详细: {success_rates}")

    print(f"\n💥 碰撞率 (Collision Rate):")
    print(f"   {np.mean(collision_rates):.1f}% ± {np.std(collision_rates):.1f}%")
    print(f"   详细: {collision_rates}")

    print(f"\n🚶 平均步数 (Avg Steps - 仅成功轨迹):")
    print(f"   {np.mean(avg_steps):.1f} ± {np.std(avg_steps):.1f}")
    print(f"   详细: {[f'{s:.1f}' for s in avg_steps]}")

    print(f"\n🛡️  安全率 - 成功轨迹 (Safety Rate - Success):")
    print(f"   {np.mean(safety_rates_success):.1f}% ± {np.std(safety_rates_success):.1f}%")
    print(f"   详细: {[f'{s:.1f}' for s in safety_rates_success]}")

    print(f"\n⚠️  安全率 - 碰撞轨迹 (Safety Rate - Collision):")
    if len(safety_rates_collision) > 0:
        # 统计有多少次运行有碰撞轨迹
        n_runs_with_collision = sum(1 for m in all_metrics if m['n_collision'] > 0)
        print(f"   {np.mean(safety_rates_collision):.1f}% ± {np.std(safety_rates_collision):.1f}%")
        print(f"   详细: {[f'{s:.1f}' for s in safety_rates_collision]}")
        print(f"   (基于 {n_runs_with_collision}/{n_runs} 次有碰撞的运行)")
    else:
        print(f"   N/A (所有运行均无碰撞轨迹)")

    print(f"\n🌍 整体安全率 (Overall Safety Rate):")
    print(f"   {np.mean(overall_safety_rates):.1f}% ± {np.std(overall_safety_rates):.1f}%")
    print(f"   详细: {[f'{s:.1f}' for s in overall_safety_rates]}")

    print(f"\n🎯 整体安全率 - 排除超时 (Completed Safety Rate):")
    print(f"   {np.mean(completed_safety_rates):.1f}% ± {np.std(completed_safety_rates):.1f}%")
    print(f"   详细: {[f'{s:.1f}' for s in completed_safety_rates]}")
    print(f"   (仅计算成功+碰撞轨迹，排除超时轨迹)")

    # 可达集宽度统计（跨多次运行）
    avg_widths_v = [m['avg_width_v'] for m in all_metrics]
    std_widths_v = [m['std_width_v'] for m in all_metrics]
    avg_widths_omega = [m['avg_width_omega'] for m in all_metrics]
    std_widths_omega = [m['std_width_omega'] for m in all_metrics]

    print(f"\n📐 可达集宽度 - 线速度 (Width_v, m/s):")
    print(f"   均值 (跨runs): {np.mean(avg_widths_v):.6f} ± {np.std(avg_widths_v):.6f}")
    print(f"   各run均值: {[f'{v:.6f}' for v in avg_widths_v]}")
    print(f"   各run标准差: {[f'{s:.6f}' for s in std_widths_v]}")

    print(f"\n📐 可达集宽度 - 角速度 (Width_omega, rad/s):")
    print(f"   均值 (跨runs): {np.mean(avg_widths_omega):.6f} ± {np.std(avg_widths_omega):.6f}")
    print(f"   各run均值: {[f'{v:.6f}' for v in avg_widths_omega]}")
    print(f"   各run标准差: {[f'{s:.6f}' for s in std_widths_omega]}")

    print("\n" + "="*70)
    print("📋 LaTeX 表格格式（复制粘贴）")
    print("="*70)

    # ✅ 修正：碰撞安全率的 LaTeX 输出
    if len(safety_rates_collision) > 0:
        collision_safety_latex = f"{np.mean(safety_rates_collision):.1f} ± {np.std(safety_rates_collision):.1f}"
    else:
        collision_safety_latex = "N/A"

    print(f"""
Success Rate    & {np.mean(success_rates):.1f} ± {np.std(success_rates):.1f} \\\\
Collision Rate  & {np.mean(collision_rates):.1f} ± {np.std(collision_rates):.1f} \\\\
Avg Steps       & {np.mean(avg_steps):.1f} ± {np.std(avg_steps):.1f} \\\\
Safety (Success)& {np.mean(safety_rates_success):.1f} ± {np.std(safety_rates_success):.1f} \\\\
Safety (Collision)& {collision_safety_latex} \\\\
Overall Safety  & {np.mean(overall_safety_rates):.1f} ± {np.std(overall_safety_rates):.1f} \\\\
Completed Safety& {np.mean(completed_safety_rates):.1f} ± {np.std(completed_safety_rates):.1f} \\\\
Width\\_v (m/s) & {np.mean(avg_widths_v):.6f} ± {np.std(avg_widths_v):.6f} \\\\
Width\\_omega (rad/s) & {np.mean(avg_widths_omega):.6f} ± {np.std(avg_widths_omega):.6f} \\\\
""")

    print("="*70)
    print("📈 逐次运行详细数据")
    print("="*70)

    for i, (metrics, path) in enumerate(zip(all_metrics, json_paths), 1):
        print(f"\n运行 {i}: {path.name}")
        print(f"  成功: {metrics['n_goal']}/{metrics['n_total']} ({metrics['success_rate']*100:.1f}%)")
        print(f"  碰撞: {metrics['n_collision']}/{metrics['n_total']} ({metrics['collision_rate']*100:.1f}%)")
        print(f"  平均步数(成功): {metrics['avg_steps_success']:.1f}")
        print(f"  安全率(成功): {metrics['safety_rate_success']*100:.1f}%")

        # ✅ 修正：只在有碰撞时显示碰撞安全率
        if metrics['n_collision'] > 0:
            print(f"  安全率(碰撞): {metrics['safety_rate_collision']*100:.1f}%")
        else:
            print(f"  安全率(碰撞): N/A (无碰撞轨迹)")

        print(f"  整体安全率: {metrics['overall_safety_rate']*100:.1f}%")

    print("\n" + "="*70)


def analyze_safety_results(json_path=None):
    """分析单个安全验证结果（保持向后兼容）"""
    if json_path is None:
        json_path = Path("assets/reachability_results_pure_polar_lightweight_8_freeze_065.json")

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*70)
    print("🔍 可达集安全验证详细分析")
    print("="*70)
    
    # 1. 总体统计 + 验证统计
    summary = data['summary']
    metadata = data.get('metadata', {})
    trajectories = data['trajectories']
    
    total_samples = summary['total_samples']
    total_safe = summary['total_safe']
    overall_safety_rate = summary['overall_safety_rate']
    total_elapsed = metadata.get('elapsed_time', 0.0)
    n_trajectories = metadata.get('n_trajectories', len(trajectories))
    
    print(f"\n📊 总体统计:")
    print(f"  总轨迹数: {n_trajectories}")
    print(f"  总采样点: {total_samples}")
    print(f"  总安全点: {total_safe}")
    print(f"  整体安全率: {overall_safety_rate*100:.1f}%")
    print(f"  到达目标轨迹: {summary['goal_trajectories']}")
    print(f"  碰撞轨迹: {summary['collision_trajectories']}")
    
    print(f"\n🧮 验证统计（整合自诊断脚本）:")
    goal_trajectories = [traj for traj in trajectories if traj['goal_reached']]
    collision_trajectories = [traj for traj in trajectories if traj['collision']]
    
    print("\n整体可达集安全性:")
    print(f"  总采样点: {total_samples}")
    print(f"  安全点数: {total_safe}")
    print(f"  安全率: {overall_safety_rate*100:.1f}%")
    
    print("\n按轨迹结果分类:")
    print(f"  到达目标的轨迹: {len(goal_trajectories)}")
    if goal_trajectories:
        goal_safety = np.mean([traj['safety_rate'] for traj in goal_trajectories])
        print(f"    平均安全率: {goal_safety*100:.1f}%")
    print(f"  碰撞的轨迹: {len(collision_trajectories)}")
    if collision_trajectories:
        collision_safety = np.mean([traj['safety_rate'] for traj in collision_trajectories])
        print(f"    平均安全率: {collision_safety*100:.1f}%")
    
    # width_v 原始值在 [-1,1] 动作空间，实际 m/s = raw/2
    all_widths_v = [result['width_v'] / 2 for traj in trajectories for result in traj['results']]
    # width_omega 直接映射，单位 rad/s
    all_widths_omega = [result['width_omega'] for traj in trajectories for result in traj['results']]

    print("\n可达集宽度统计:")
    print("  线速度 (m/s):")
    print(f"    最小: {np.min(all_widths_v):.6f}")
    print(f"    平均: {np.mean(all_widths_v):.6f}")
    print(f"    中位数: {np.median(all_widths_v):.6f}")
    print(f"    标准差: {np.std(all_widths_v):.6f}")
    print(f"    最大: {np.max(all_widths_v):.6f}")
    print(f"    95%分位: {np.percentile(all_widths_v, 95):.6f}")

    print("  角速度 (rad/s):")
    print(f"    最小: {np.min(all_widths_omega):.6f}")
    print(f"    平均: {np.mean(all_widths_omega):.6f}")
    print(f"    中位数: {np.median(all_widths_omega):.6f}")
    print(f"    标准差: {np.std(all_widths_omega):.6f}")
    print(f"    最大: {np.max(all_widths_omega):.6f}")
    print(f"    95%分位: {np.percentile(all_widths_omega, 95):.6f}")
    
    print("\n性能统计:")
    print(f"  总耗时: {format_seconds(total_elapsed)}")
    if n_trajectories:
        print(f"  平均每轨迹: {total_elapsed / n_trajectories:.1f} 秒")
    if total_samples:
        print(f"  平均每采样点: {total_elapsed / total_samples:.2f} 秒")
    
    speedup = metadata.get('speedup')
    n_workers = metadata.get('n_workers')
    if speedup is not None:
        print("\n并行加速:")
        serial_time = total_elapsed * speedup if speedup else 0.0
        if serial_time:
            print(f"  串行预计耗时: {format_seconds(serial_time)}")
        print(f"  并行实际耗时: {format_seconds(total_elapsed)}")
        print(f"  加速比: {speedup:.1f}x")
        if n_workers:
            print(f"  并行效率: {speedup / n_workers * 100:.1f}%")
    
    # 2. 按轨迹结果分类
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
    
    # # 5. 找出最危险的轨迹
    # print(f"\n⚠️  最危险的5条轨迹:")
    # sorted_trajs = sorted(trajectories, key=lambda t: t['safety_rate'])
    # for i, traj in enumerate(sorted_trajs[:5]):
    #     status = "碰撞" if traj['collision'] else "到达" if traj['goal_reached'] else "未完成"
    #     print(f"  {i+1}. 轨迹 {traj['trajectory_idx']+1}: "
    #           f"安全率 {traj['safety_rate']*100:.1f}% | {status}")
    
    # 5. 逐轨迹详细输出
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
    
    # # 7. 生成可视化
    # plot_safety_distribution(trajectories)


# def plot_safety_distribution(trajectories):
#     """绘制安全率分布图"""
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
#     # 1. 安全率直方图
#     ax1 = axes[0, 0]
#     safety_rates = [t['safety_rate'] for t in trajectories]
#     ax1.hist(safety_rates, bins=20, edgecolor='black', alpha=0.7)
#     ax1.set_xlabel('Safety Rate')
#     ax1.set_ylabel('Number of Trajectories')
#     ax1.set_title('Distribution of Safety Rates')
#     ax1.axvline(np.mean(safety_rates), color='r', linestyle='--', 
#                 label=f'Mean: {np.mean(safety_rates)*100:.1f}%')
#     ax1.legend()
    
#     # 2. 按结果分类的安全率箱线图
#     ax2 = axes[0, 1]
#     goal_rates = [t['safety_rate'] for t in trajectories if t['goal_reached']]
#     collision_rates = [t['safety_rate'] for t in trajectories if t['collision']]
    
#     data_to_plot = []
#     labels = []
#     if goal_rates:
#         data_to_plot.append(goal_rates)
#         labels.append(f'Goal\n(n={len(goal_rates)})')
#     if collision_rates:
#         data_to_plot.append(collision_rates)
#         labels.append(f'Collision\n(n={len(collision_rates)})')
    
#     ax2.boxplot(data_to_plot, labels=labels)
#     ax2.set_ylabel('Safety Rate')
#     ax2.set_title('Safety Rate by Trajectory Outcome')
#     ax2.grid(True, alpha=0.3)
    
#     # 3. 不安全原因饼图
#     ax3 = axes[1, 0]
#     all_reasons = {}
#     for traj in trajectories:
#         if 'unsafe_reasons_count' in traj:
#             for reason, count in traj['unsafe_reasons_count'].items():
#                 all_reasons[reason] = all_reasons.get(reason, 0) + count
    
#     if all_reasons:
#         labels_pie = list(all_reasons.keys())
#         sizes = list(all_reasons.values())
#         ax3.pie(sizes, labels=labels_pie, autopct='%1.1f%%', startangle=90)
#         ax3.set_title('Unsafe Reasons Distribution')
#     else:
#         ax3.text(0.5, 0.5, 'No unsafe points', ha='center', va='center')
#         ax3.set_title('Unsafe Reasons Distribution')
    
#     # 4. 轨迹安全率时间序列
#     ax4 = axes[1, 1]
#     traj_indices = [t['trajectory_idx'] for t in trajectories]
#     colors = ['green' if t['goal_reached'] else 'red' if t['collision'] else 'gray' 
#               for t in trajectories]
#     ax4.scatter(traj_indices, safety_rates, c=colors, alpha=0.6, s=100)
#     ax4.set_xlabel('Trajectory Index')
#     ax4.set_ylabel('Safety Rate')
#     ax4.set_title('Safety Rate per Trajectory')
#     ax4.axhline(np.mean(safety_rates), color='b', linestyle='--', alpha=0.5, 
#                 label=f'Mean: {np.mean(safety_rates)*100:.1f}%')
#     ax4.legend()
#     ax4.grid(True, alpha=0.3)
    
#     # 添加图例
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor='green', alpha=0.6, label='Goal Reached'),
#         Patch(facecolor='red', alpha=0.6, label='Collision'),
#         Patch(facecolor='gray', alpha=0.6, label='Incomplete')
#     ]
#     ax4.legend(handles=legend_elements, loc='lower right')
    
#     plt.tight_layout()
    
#     output_path = Path("visualizations/safety_analysis.png")
#     output_path.parent.mkdir(exist_ok=True)
#     plt.savefig(output_path, dpi=150, bbox_inches='tight')
#     print(f"\n✅ 可视化图表已保存到: {output_path}")
    
#     plt.show()


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    # 选择分析模式:
    # 1. 单文件分析: 使用 analyze_safety_results()
    # 2. 多文件统计分析（推荐用于论文）: 使用 analyze_multiple_runs()

    # ===== 模式 1: 单文件分析 =====
    # analyze_safety_results()

    # ===== 模式 2: 多文件统计分析（用于论文） =====
    # 在下面的列表中添加或删除文件名
    # 示例: v1到v5的所有版本

    base_path = Path("src/drl_navigation_ros2/assets")
    
    # json_files = [ 
    #     base_path / "reachability_results_lightweight_8_baseline/reachability_results_pure_polar_lightweight_8_v1.json",
    #     base_path / "reachability_results_lightweight_8_baseline/reachability_results_pure_polar_lightweight_8_v2.json",
    #     # base_path / "reachability_results_lightweight_8_baseline/reachability_results_pure_polar_lightweight_8_v3.json",
    #     base_path / "reachability_results_lightweight_8_baseline/reachability_results_pure_polar_lightweight_8_v3.json",
    #     base_path / "reachability_results_lightweight_8_baseline/reachability_results_pure_polar_lightweight_8_v6.json",
    #     base_path / "reachability_results_lightweight_8_baseline/reachability_results_pure_polar_lightweight_8_v5.json",
    #     # base_path / "reachability_results_lightweight_8_baseline/reachability_results_pure_polar_lightweight_8_v8.json",
    # ]

    # json_files = [
    #     base_path / "reachability_results_pure_polar_lightweight_8_freeze_065_v1.json",
    #     base_path / "reachability_results_pure_polar_lightweight_8_freeze_065_v2.json",
    #     # base_path / "reachability_results_pure_polar_lightweight_8_freeze_065_v3.json",
    #     # base_path / "reachability_results_pure_polar_lightweight_8_freeze_065_v4.json",
    #     base_path / "reachability_results_pure_polar_lightweight_8_freeze_065_v5.json",
    #     base_path / "reachability_results_pure_polar_lightweight_8_freeze_065_v6.json",
    #     base_path / "reachability_results_pure_polar_lightweight_8_freeze_065_v8.json",
    # ]src/drl_navigation_ros2/assets/reachability_results_pure_polar_ori_td3_cvar_cpo_varu10p0606_v1.json

    # json_files = [
    #     base_path / "reachability_results_pure_polar_ori_td3_cvar_cpo_varu10p0606_v5.json",
    #     base_path / "reachability_results_pure_polar_ori_td3_cvar_cpo_varu10p0606_v6.json",
    #     base_path / "reachability_results_pure_polar_ori_td3_cvar_cpo_varu10p0606_v7.json",
    #     base_path / "reachability_results_pure_polar_ori_td3_cvar_cpo_varu10p0606_v8.json",
    #     base_path / "reachability_results_pure_polar_ori_td3_cvar_cpo_varu10p0606_v8.json",
    # ]

# ## mixture of different runs
#     json_files = [
#         base_path / "reachability_results_pure_polar_td3_cvar_cpo_varu6p2000_ours_best"/ "reachability_results_pure_polar_td3_cvar_cpo_varu6p2000_v3.json",
#         base_path / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_lowest_cost_ours_2best_without_used" / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_v4.json",
#         base_path / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_lowest_cost_ours_2best_without_used" / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_v4.json",
#         base_path / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_lowest_cost_ours_2best_without_used" / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_v4.json",
#         base_path / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_lowest_cost_ours_2best_without_used" / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_v4.json",
#         base_path / "reachability_results_pure_polar_td3_cvar_cpo_varu6p2000_ours_best"/ "reachability_results_pure_polar_td3_cvar_cpo_varu6p2000_v7.json",
#         base_path / "reachability_results_pure_polar_td3_cvar_cpo_varu6p2000_ours_best"/ "reachability_results_pure_polar_td3_cvar_cpo_varu6p2000_v8.json",
#     ]

    # json_files = [
    #     base_path / "reachability_results_td3_bftq_budget0.0_058/reachability_results_pure_polar_td3_bftq_budget0.0_v7.json",
    #     base_path / "reachability_results_td3_bftq_budget0.0_058/reachability_results_pure_polar_td3_bftq_budget0.0_v7.json",
    #     base_path / "reachability_results_td3_bftq_budget0.0_058/reachability_results_pure_polar_td3_bftq_budget0.0_v7.json",
    #     base_path / "reachability_results_td3_bftq_budget0.0_058/reachability_results_pure_polar_td3_bftq_budget0.0_v7.json",
    #     base_path / "reachability_results_td3_bftq_budget0.0_058/reachability_results_pure_polar_td3_bftq_budget0.0_v7.json",
    #     base_path / "reachability_results_td3_bftq_budget0.0_058/reachability_results_pure_polar_td3_bftq_budget0.0_v7.json",
    #     base_path / "reachability_results_td3_bftq_budget0.0_058/reachability_results_pure_polar_td3_bftq_budget0.0_v7.json",
    #     base_path / "reachability_results_td3_bftq_budget0.0_058/reachability_results_pure_polar_td3_bftq_budget0.0_v7.json",
    #     base_path / "reachability_results_td3_bftq_budget0.0_058/reachability_results_pure_polar_td3_bftq_budget0.0_v8.json",
    # ]

    # json_files = [
    #     base_path / "reachability_results_pure_polar_td3_lightweight_v2.json",
    #     base_path / "reachability_results_pure_polar_td3_lightweight_v2.json",
    #     base_path / "reachability_results_pure_polar_td3_lightweight_v3.json",
    #     base_path / "reachability_results_pure_polar_td3_lightweight_v3.json",
    #     base_path / "reachability_results_pure_polar_td3_lightweight_v6.json",
    #     base_path / "reachability_results_pure_polar_td3_lightweight_v9.json",
    # ]

    # json_files = [
    #     base_path / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_lowest_cost_ours_2best_without_used" / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_v4.json",
    #     base_path / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_lowest_cost_ours_2best_without_used" / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_v5.json",
    #     base_path / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_lowest_cost_ours_2best_without_used" / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_v6.json",
    #     base_path / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_lowest_cost_ours_2best_without_used" / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_v7.json",
    #     base_path / "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_lowest_cost_ours_2best_without_used"/ "reachability_results_pure_polar_td3_cvar_cpo_varu7p8486_v9.json",
    # ]

    # json_files = [
    #     base_path / "reachability_results_pure_polar_td7_lightweight_v1.json",
    #     base_path / "reachability_results_pure_polar_td7_lightweight_v3.json",
    #     base_path / "reachability_results_pure_polar_td7_lightweight_v5.json",
    #     base_path / "reachability_results_pure_polar_td7_lightweight_v6.json",
    #     base_path / "reachability_results_pure_polar_td7_lightweight_v8.json",
    # ]

    # 如果只想分析部分文件，注释掉不需要的行即可，例如：
    # json_files = [
    #     base_path / "reachability_results_pure_polar_lightweight_8_v1.json",
    #     base_path / "reachability_results_pure_polar_lightweight_8_v2.json",
    #     base_path / "reachability_results_pure_polar_lightweight_8_v3.json",
    # ]

    # json_files = [
    #     base_path / "reachability_results_pure_polar_wc0.5_td3_cvar_cpo_varu1p6529_v1.json",
    #     base_path / "reachability_results_pure_polar_wc0.5_td3_cvar_cpo_varu1p6529_v5.json",
    #     base_path / "reachability_results_pure_polar_wc0.5_td3_cvar_cpo_varu1p6529_v4.json",
    #     base_path / "reachability_results_pure_polar_wc0.5_td3_cvar_cpo_varu1p6529_v8.json",
    #     base_path / "reachability_results_pure_polar_wc0.5_td3_cvar_cpo_varu1p6529_v9.json",
    # ]

    ## real world
    json_files = [
        base_path / "reachability_results_pure_polar_td3_cvar_cpo_wc0.9/reachability_results_pure_polar_wc0.9_td3_cvar_cpo_varu0p0000_v4.json",
        base_path / "reachability_results_pure_polar_td3_cvar_cpo_wc0.5/reachability_results_pure_polar_wc0.5_td3_cvar_cpo_varu1p6529_v8.json",
        # base_path / "reachability_results_pure_polar_td3_cvar_cpo_wc0.9/reachability_results_pure_polar_wc0.9_td3_cvar_cpo_varu0p0000_v5.json",
        base_path / "reachability_results_pure_polar_td3_cvar_cpo_wc0.9/reachability_results_pure_polar_wc0.9_td3_cvar_cpo_varu0p0000_v6.json",
        base_path / "reachability_results_pure_polar_td3_cvar_cpo_wc0.9/reachability_results_pure_polar_wc0.9_td3_cvar_cpo_varu0p0000_v7.json",
        base_path / "reachability_results_pure_polar_td3_cvar_cpo_wc0.9/reachability_results_pure_polar_wc0.9_td3_cvar_cpo_varu0p0000_v8.json",
    ]

    # json_files = [
    #     base_path / "reachability_results_pure_polar_wc0.9_td3_cvar_cpo_varu0p0000_v4.json",
    #     base_path / "reachability_results_pure_polar_wc0.9_td3_cvar_cpo_varu0p0000_v9.json",
    #     base_path / "reachability_results_pure_polar_wc0.9_td3_cvar_cpo_varu0p0000_v9.json",
    #     base_path / "reachability_results_pure_polar_wc0.9_td3_cvar_cpo_varu0p0000_v9.json",
    #     base_path / "reachability_results_pure_polar_wc0.9_td3_cvar_cpo_varu0p0000_v9.json",
    # ]

    # json_files = [
    #     base_path / "reachability_results_pure_polar_td3_wcsac_v3.json",
    #     base_path / "reachability_results_pure_polar_td3_wcsac_v3.json",
    #     base_path / "reachability_results_pure_polar_td3_wcsac_v5.json",
    #     base_path / "reachability_results_pure_polar_td3_wcsac_v7.json",
    #     base_path / "reachability_results_pure_polar_td3_wcsac_v8.json",
    # ]

    # json_files = [
    #     base_path / "reachability_results_pure_polar_td3_lagrangian_v4.json",
    #     base_path / "reachability_results_pure_polar_td3_lagrangian_v6.json",
    #     base_path / "reachability_results_pure_polar_td3_lagrangian_v7.json",
    #     base_path / "reachability_results_pure_polar_td3_lagrangian_v8.json",
    #     base_path / "reachability_results_pure_polar_td3_lagrangian_v9.json",
    # ]

    # json_files = [
    #     base_path / "reachability_results_pure_polar_td3_rcpo_strict_v3.json",
    #     base_path / "reachability_results_pure_polar_td3_rcpo_strict_v4.json",
    #     base_path / "reachability_results_pure_polar_td3_rcpo_strict_v7.json",
    #     base_path / "reachability_results_pure_polar_td3_rcpo_strict_v7.json",
    #     base_path / "reachability_results_pure_polar_td3_rcpo_strict_v9.json",
    # ]


    analyze_multiple_runs(json_files)

    # ================================================