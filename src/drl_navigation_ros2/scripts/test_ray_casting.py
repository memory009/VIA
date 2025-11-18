#!/usr/bin/env python3
"""测试光线投射是否正确"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from verification.ray_casting import ObstacleMap


def test_ray_casting():
    """测试光线投射"""
    print("="*70)
    print("测试光线投射模块")
    print("="*70)
    
    # 加载地图
    obstacle_map = ObstacleMap()
    print(f"\n✅ 加载地图成功")
    print(f"   激光雷达参数: 范围={obstacle_map.laser_range}m, 光束数={obstacle_map.n_beams}")
    print(f"   障碍物数量: {len(obstacle_map.obstacles)}")
    
    # 测试案例1：原点朝向 0°（对照 JSON 中的 examples）
    print("\n" + "-"*70)
    print("测试案例1: 原点朝向 0°")
    laser1 = obstacle_map.predict_laser_scan(0.0, 0.0, 0.0)
    print(f"   预测: min={np.min(laser1):.3f}, max={np.max(laser1):.3f}")
    print(f"   JSON:  min=3.5, max=3.5")
    print(f"   误差: {abs(np.min(laser1) - 3.5):.6f}")
    
    # 测试案例2：(-2, 2) 朝向 45°
    print("\n" + "-"*70)
    print("测试案例2: (-2, 2) 朝向 45°")
    laser2 = obstacle_map.predict_laser_scan(-2.0, 2.0, np.deg2rad(45))
    print(f"   预测: min={np.min(laser2):.3f}, max={np.max(laser2):.3f}")
    print(f"   JSON:  min=0.823, max=3.5")
    print(f"   误差: {abs(np.min(laser2) - 0.823):.3f}")
    
    # 测试案例3：(2, -2) 朝向 -90°
    print("\n" + "-"*70)
    print("测试案例3: (2, -2) 朝向 -90°")
    laser3 = obstacle_map.predict_laser_scan(2.0, -2.0, np.deg2rad(-90))
    print(f"   预测: min={np.min(laser3):.3f}, max={np.max(laser3):.3f}")
    print(f"   JSON:  min=0.658, max=3.5")
    print(f"   误差: {abs(np.min(laser3) - 0.658):.3f}")
    
    # 可视化（可选）
    print("\n" + "-"*70)
    print("完整读数预览（案例2）:")
    print(f"   预测: {laser2}")
    
    print("\n" + "="*70)
    print("✅ 测试完成！")
    print("="*70)


if __name__ == "__main__":
    test_ray_casting()