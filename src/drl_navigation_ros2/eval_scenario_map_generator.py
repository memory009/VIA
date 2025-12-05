#!/usr/bin/env python3
"""
辅助函数：为评估场景批量生成障碍物地图和可视化
供 train.py 调用
"""

import json
from pathlib import Path
import sys

# 添加项目路径，以便导入 export_gazebo_map_polar
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def generate_scenario_maps(eval_scenarios, scenario_tag="8_polar"):
    """
    为所有评估场景生成障碍物地图和可视化
    
    Args:
        eval_scenarios: 从 record_eval_positions 返回的场景列表
        scenario_tag: 场景标签（用于目录命名）
    """
    # 导入必要的函数（延迟导入，避免循环依赖）
    try:
        # 从 scripts 目录导入（假设已经在项目根目录）
        from scripts.export_gazebo_map  import (
            save_obstacle_map,
            visualize_map,
            export_obstacle_map
        )
    except ImportError:
        print("⚠️  无法导入 export_gazebo_map_polar，跳过地图生成")
        print("    请确保 scripts/export_gazebo_map_polar.py 存在")
        return
    
    # 设置输出目录
    base_dir = Path("src/drl_navigation_ros2")
    assets_dir = base_dir / "assets" / f"eval_scenarios_{scenario_tag}"
    vis_dir = base_dir / "visualizations" / f"obstacle_{scenario_tag}_map"
    
    assets_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 输出目录:")
    print(f"   JSON: {assets_dir}")
    print(f"   PNG:  {vis_dir}")
    print()
    
    # 为每个场景生成地图
    n_scenarios = len(eval_scenarios)
    print(f"开始生成 {n_scenarios} 个场景的障碍物地图...")
    print("-" * 70)
    
    for scenario_idx, scenario in enumerate(eval_scenarios):
        print(f"\n场景 {scenario_idx + 1}/{n_scenarios}:")
        
        # 构造场景字典（模拟从 JSON 加载的格式）
        scenario_dict = {
            'scenario_id': scenario_idx,
            'elements': []
        }
        
        for element in scenario:
            scenario_dict['elements'].append({
                'name': element.name,
                'x': float(element.x),
                'y': float(element.y),
                'angle': float(element.angle)
            })
        
        # 生成 JSON 文件路径
        json_path = assets_dir / f"obstacle_map_scenario_{scenario_idx:02d}.json"
        png_path = vis_dir / f"obstacle_map_scenario_{scenario_idx:02d}.png"
        
        try:
            # 1. 保存障碍物地图 JSON
            obstacle_map = save_obstacle_map(
                output_path=json_path,
                scenario=scenario_dict,
                scenario_tag=scenario_tag
            )
            print(f"   ✅ JSON: {json_path.name}")
            
            # 2. 生成可视化 PNG
            visualize_map(
                obstacle_map=obstacle_map,
                save_path=png_path,
                scenario_tag=scenario_tag
            )
            print(f"   ✅ PNG:  {png_path.name}")
            
        except Exception as e:
            print(f"   ❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "-" * 70)
    print(f"✅ 完成！共生成 {n_scenarios} 个场景的地图文件")
    print(f"   JSON 文件: {assets_dir}")
    print(f"   PNG 文件:  {vis_dir}")


if __name__ == "__main__":
    # 测试代码（可选）
    print("这是一个辅助模块，应该从 train.py 导入使用")
    print("示例用法:")
    print("  from scenario_map_generator import generate_scenario_maps")
    print("  generate_scenario_maps(eval_scenarios, scenario_tag='8_polar')")