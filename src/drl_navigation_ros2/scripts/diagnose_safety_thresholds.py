#!/usr/bin/env python3
"""
诊断安全检查阈值是否合理
"""

import json
import numpy as np
from pathlib import Path

def diagnose_thresholds():
    """诊断各个安全检查条件的触发情况"""
    
    json_path = Path("assets/reachability_results_pure_polar_lightweight.json")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*70)
    print("🔍 安全阈值诊断")
    print("="*70)
    
    # 收集所有采样点的数据
    all_widths_v = []
    all_widths_omega = []
    all_min_lasers = []
    all_is_safe = []
    
    # 统计每个条件的违反次数
    violations = {
        'width_v_exceeded': 0,      # 线速度宽度超标
        'width_omega_exceeded': 0,  # 角速度宽度超标
        'collision_risk': 0,        # 碰撞风险
        'action_range': 0,          # 动作范围超限
        'total_unsafe': 0,          # 总不安全点
    }
    
    for traj in data['trajectories']:
        for result in traj['results']:
            all_widths_v.append(result['width_v'])
            all_widths_omega.append(result['width_omega'])
            all_min_lasers.append(result['min_laser'])
            all_is_safe.append(result['is_safe'])
            
            if not result['is_safe']:
                violations['total_unsafe'] += 1
                
                # 检查具体原因
                width_v = result['width_v']
                width_omega = result['width_omega']
                min_laser = result['min_laser']
                action_ranges = result['action_ranges']
                
                if width_v > 0.3:
                    violations['width_v_exceeded'] += 1
                if width_omega > 0.6:
                    violations['width_omega_exceeded'] += 1
                if min_laser < 0.45:
                    actual_v_max = (action_ranges[0][1] + 1) / 2
                    if actual_v_max > 0.05:
                        predicted_min = min_laser - actual_v_max * 0.1
                        if predicted_min < 0.4:
                            violations['collision_risk'] += 1
                if action_ranges[0][0] < -1.0 or action_ranges[0][1] > 1.0:
                    violations['action_range'] += 1
                if action_ranges[1][0] < -1.0 or action_ranges[1][1] > 1.0:
                    violations['action_range'] += 1
    
    # 统计
    total_points = len(all_widths_v)
    safe_points = sum(all_is_safe)
    unsafe_points = total_points - safe_points
    
    print(f"\n📊 总体统计:")
    print(f"  总采样点: {total_points}")
    print(f"  安全点: {safe_points} ({safe_points/total_points*100:.1f}%)")
    print(f"  不安全点: {unsafe_points} ({unsafe_points/total_points*100:.1f}%)")
    
    print(f"\n🚨 违反条件统计:")
    print(f"  总不安全点: {violations['total_unsafe']}")
    print(f"  线速度宽度超标: {violations['width_v_exceeded']} "
          f"({violations['width_v_exceeded']/violations['total_unsafe']*100:.1f}%)")
    print(f"  角速度宽度超标: {violations['width_omega_exceeded']} "
          f"({violations['width_omega_exceeded']/violations['total_unsafe']*100:.1f}%)")
    print(f"  碰撞风险: {violations['collision_risk']} "
          f"({violations['collision_risk']/violations['total_unsafe']*100:.1f}%)")
    print(f"  动作范围超限: {violations['action_range']} "
          f"({violations['action_range']/violations['total_unsafe']*100:.1f}%)")
    
    print(f"\n📏 可达集宽度分布:")
    print(f"  线速度宽度:")
    print(f"    平均: {np.mean(all_widths_v):.6f}")
    print(f"    中位数: {np.median(all_widths_v):.6f}")
    print(f"    最大: {np.max(all_widths_v):.6f}")
    print(f"    90%分位: {np.percentile(all_widths_v, 90):.6f}")
    print(f"    95%分位: {np.percentile(all_widths_v, 95):.6f}")
    print(f"    99%分位: {np.percentile(all_widths_v, 99):.6f}")
    print(f"    >0.3的比例: {sum(1 for w in all_widths_v if w > 0.3)/len(all_widths_v)*100:.1f}%")
    
    print(f"  角速度宽度:")
    print(f"    平均: {np.mean(all_widths_omega):.6f}")
    print(f"    中位数: {np.median(all_widths_omega):.6f}")
    print(f"    最大: {np.max(all_widths_omega):.6f}")
    print(f"    90%分位: {np.percentile(all_widths_omega, 90):.6f}")
    print(f"    95%分位: {np.percentile(all_widths_omega, 95):.6f}")
    print(f"    99%分位: {np.percentile(all_widths_omega, 99):.6f}")
    print(f"    >0.6的比例: {sum(1 for w in all_widths_omega if w > 0.6)/len(all_widths_omega)*100:.1f}%")
    
    print(f"\n📡 最小激光距离分布:")
    print(f"    平均: {np.mean(all_min_lasers):.3f}m")
    print(f"    中位数: {np.median(all_min_lasers):.3f}m")
    print(f"    最小: {np.min(all_min_lasers):.3f}m")
    print(f"    <0.45m的比例: {sum(1 for l in all_min_lasers if l < 0.45)/len(all_min_lasers)*100:.1f}%")
    print(f"    <0.4m的比例: {sum(1 for l in all_min_lasers if l < 0.4)/len(all_min_lasers)*100:.1f}%")
    
    print("\n" + "="*70)
    
    # 建议新阈值
    print(f"\n💡 建议的新阈值:")
    
    # 线速度宽度：使用95%分位
    suggested_width_v = np.percentile(all_widths_v, 95)
    print(f"  MAX_WIDTH_LINEAR: {suggested_width_v:.3f} (当前: 0.3)")
    
    # 角速度宽度：使用95%分位
    suggested_width_omega = np.percentile(all_widths_omega, 95)
    print(f"  MAX_WIDTH_ANGULAR: {suggested_width_omega:.3f} (当前: 0.6)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    diagnose_thresholds()