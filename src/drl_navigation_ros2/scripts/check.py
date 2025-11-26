#!/usr/bin/env python3
"""
验证可达集是否包含实际动作
"""

import json
import numpy as np

# 加载验证结果
with open('assets/reachability_results_pure_polar_lightweight.json', 'r') as f:
    data = json.load(f)

print("="*70)
print("验证：可达集是否包含实际执行的动作")
print("="*70)

total_checks = 0
contained = 0
not_contained = 0

violations = []

for traj in data['trajectories']:
    for result in traj['results']:
        total_checks += 1
        
        # 实际执行的动作
        actual_action = result['det_action']
        
        # 可达集范围
        action_ranges = result['action_ranges']
        
        # 检查线速度
        v_contained = (action_ranges[0][0] <= actual_action[0] <= action_ranges[0][1])
        
        # 检查角速度
        omega_contained = (action_ranges[1][0] <= actual_action[1] <= action_ranges[1][1])
        
        if v_contained and omega_contained:
            contained += 1
        else:
            not_contained += 1
            violations.append({
                'trajectory': traj['trajectory_idx'],
                'step': result['step'],
                'actual_action': actual_action,
                'action_ranges': action_ranges,
                'v_violation': not v_contained,
                'omega_violation': not omega_contained,
                'v_diff': min(actual_action[0] - action_ranges[0][1], 
                              action_ranges[0][0] - actual_action[0]) if not v_contained else 0,
                'omega_diff': min(actual_action[1] - action_ranges[1][1], 
                                  action_ranges[1][0] - actual_action[1]) if not omega_contained else 0,
            })

print(f"\n总检查点数: {total_checks}")
print(f"✅ 实际动作在可达集内: {contained} ({contained/total_checks*100:.2f}%)")
print(f"❌ 实际动作在可达集外: {not_contained} ({not_contained/total_checks*100:.2f}%)")

if not_contained > 0:
    print(f"\n⚠️ 发现 {not_contained} 个违规点！")
    print("\n前10个违规详情:")
    for i, v in enumerate(violations[:10]):
        print(f"\n违规 {i+1}:")
        print(f"  轨迹: {v['trajectory']}, 步骤: {v['step']}")
        print(f"  实际动作: v={v['actual_action'][0]:.6f}, ω={v['actual_action'][1]:.6f}")
        print(f"  可达集: v=[{v['action_ranges'][0][0]:.6f}, {v['action_ranges'][0][1]:.6f}]")
        print(f"           ω=[{v['action_ranges'][1][0]:.6f}, {v['action_ranges'][1][1]:.6f}]")
        if v['v_violation']:
            print(f"  ❌ 线速度超出范围 {v['v_diff']:.6f}")
        if v['omega_violation']:
            print(f"  ❌ 角速度超出范围 {v['omega_diff']:.6f}")
    
    # 统计违规幅度
    v_diffs = [v['v_diff'] for v in violations if v['v_violation']]
    omega_diffs = [v['omega_diff'] for v in violations if v['omega_violation']]
    
    print(f"\n违规幅度统计:")
    if v_diffs:
        print(f"  线速度:")
        print(f"    平均: {np.mean(v_diffs):.6f}")
        print(f"    最大: {np.max(v_diffs):.6f}")
        print(f"    最小: {np.min(v_diffs):.6f}")
    
    if omega_diffs:
        print(f"  角速度:")
        print(f"    平均: {np.mean(omega_diffs):.6f}")
        print(f"    最大: {np.max(omega_diffs):.6f}")
        print(f"    最小: {np.min(omega_diffs):.6f}")
else:
    print("\n✅ 所有实际动作都在可达集内！")
    print("   可达集计算正确，符合POLAR理论保证。")

print("="*70)