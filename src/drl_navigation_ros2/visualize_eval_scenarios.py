#!/usr/bin/env python3
"""
可视化评估场景的工具
用于检查生成的起点、终点和障碍物位置是否合理
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np


def visualize_scenarios(json_path=None, show_all=True, scenario_ids=None):
    """
    可视化评估场景
    
    Args:
        json_path: JSON文件路径，默认为 assets/eval_scenarios.json
        show_all: 是否显示所有场景
        scenario_ids: 要显示的场景ID列表，如果show_all=False
    """
    if json_path is None:
        json_path = Path(__file__).parent / "assets" / "eval_scenarios.json"
    else:
        json_path = Path(json_path)
    
    if not json_path.exists():
        print(f"❌ 文件不存在: {json_path}")
        print("请先运行训练程序生成评估场景")
        return
    
    # 读取场景数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 评估场景信息:")
    print(f"   - 场景数量: {data['n_scenarios']}")
    print(f"   - 随机种子: {data['random_seed']}")
    print(f"   - 随机障碍物: {'启用' if data.get('enable_random_obstacles', True) else '禁用'}")
    print(f"   - 障碍物总数: {data.get('n_obstacles', 8)}")
    print(f"   - 最小距离: {data['min_distance']}m")
    print()
    
    scenarios = data['scenarios']
    
    # 确定要显示的场景
    if not show_all and scenario_ids is not None:
        scenarios_to_show = [s for s in scenarios if s['scenario_id'] in scenario_ids]
    else:
        scenarios_to_show = scenarios
    
    # 计算子图布局
    n_scenarios = len(scenarios_to_show)
    n_cols = min(3, n_scenarios)
    n_rows = (n_scenarios + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_scenarios == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_scenarios > 1 else axes
    
    # 障碍物尺寸定义（中心点到边界的距离）
    obstacle_sizes = {
        'obstacle1': (0.15, 0.35),  # 0.3 x 0.7
        'obstacle2': (0.5, 0.5),     # 1 x 1
        'obstacle3': (0.125, 0.125), # 0.25 x 0.25
        'obstacle4': (1.0, 0.75),    # 2 x 1.5
        'obstacle5': (0.15, 0.15),   # 假设 0.3 x 0.3
        'obstacle6': (0.15, 0.15),   # 假设 0.3 x 0.3
        'obstacle7': (0.15, 0.15),   # 假设 0.3 x 0.3
        'obstacle8': (0.15, 0.15),   # 假设 0.3 x 0.3
    }
    robot_size = 0.265 / 2  # TurtleBot3 Waffle 半径
    target_size = 0.2  # 目标点显示半径
    
    for idx, scenario in enumerate(scenarios_to_show):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        scenario_id = scenario['scenario_id']
        robot_start = scenario['robot_start']
        target = scenario['target']
        
        # 设置坐标轴
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'场景 {scenario_id}')
        
        # 绘制固定障碍物（前4个）
        fixed_obstacles = [[-2.93, 3.17], [2.86, -3.0], [-2.77, -0.96], [2.83, 2.93]]
        fixed_names = ['obstacle1', 'obstacle2', 'obstacle3', 'obstacle4']
        
        for pos, name in zip(fixed_obstacles, fixed_names):
            size = obstacle_sizes.get(name, (0.5, 0.5))
            rect = patches.Rectangle(
                (pos[0] - size[0], pos[1] - size[1]),
                size[0] * 2, size[1] * 2,
                linewidth=2, edgecolor='gray', facecolor='gray', alpha=0.6,
                label='固定障碍物' if name == 'obstacle1' else ''
            )
            ax.add_patch(rect)
            ax.text(pos[0], pos[1], name[-1], ha='center', va='center', 
                   fontsize=8, color='white', weight='bold')
        
        # 绘制可移动障碍物（obstacle5-8）- 仅在启用时存在
        enable_random = data.get('enable_random_obstacles', True)
        if enable_random:
            for element in scenario['elements']:
                name = element['name']
                if name.startswith('obstacle') and int(name[-1]) >= 5:
                    x, y = element['x'], element['y']
                    size = obstacle_sizes.get(name, (0.15, 0.15))
                    
                    circle = patches.Circle(
                        (x, y), max(size), 
                        linewidth=2, edgecolor='orange', facecolor='orange', alpha=0.5,
                        label='随机障碍物' if name == 'obstacle5' else ''
                    )
                    ax.add_patch(circle)
                    ax.text(x, y, name[-1], ha='center', va='center', 
                           fontsize=8, color='white', weight='bold')
        
        # 绘制机器人起点
        robot_circle = patches.Circle(
            (robot_start['x'], robot_start['y']), robot_size,
            linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7,
            label='机器人起点'
        )
        ax.add_patch(robot_circle)
        
        # 绘制机器人朝向
        angle = robot_start['angle']
        arrow_length = 0.4
        dx = arrow_length * np.cos(angle)
        dy = arrow_length * np.sin(angle)
        ax.arrow(robot_start['x'], robot_start['y'], dx, dy,
                head_width=0.15, head_length=0.1, fc='blue', ec='blue')
        
        # 绘制目标点
        target_circle = patches.Circle(
            (target['x'], target['y']), target_size,
            linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.7,
            label='目标点'
        )
        ax.add_patch(target_circle)
        ax.plot(target['x'], target['y'], 'g*', markersize=15)
        
        # 绘制从起点到终点的直线距离
        distance = np.sqrt((target['x'] - robot_start['x'])**2 + 
                          (target['y'] - robot_start['y'])**2)
        ax.plot([robot_start['x'], target['x']], 
               [robot_start['y'], target['y']], 
               'k--', alpha=0.3, linewidth=1)
        
        # 显示距离信息
        mid_x = (robot_start['x'] + target['x']) / 2
        mid_y = (robot_start['y'] + target['y']) / 2
        ax.text(mid_x, mid_y, f'{distance:.2f}m', 
               fontsize=9, ha='center', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        # 添加图例（只在第一个子图）
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
        
        # 显示起点和终点坐标
        info_text = f"起点: ({robot_start['x']:.2f}, {robot_start['y']:.2f})\n"
        info_text += f"终点: ({target['x']:.2f}, {target['y']:.2f})\n"
        info_text += f"直线距离: {distance:.2f}m"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 隐藏多余的子图
    for idx in range(len(scenarios_to_show), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = json_path.parent / "eval_scenarios_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化图片已保存到: {output_path}")
    
    plt.show()


def print_scenario_details(json_path=None, scenario_id=None):
    """
    打印场景详细信息
    
    Args:
        json_path: JSON文件路径
        scenario_id: 场景ID，None表示打印所有场景
    """
    if json_path is None:
        json_path = Path(__file__).parent / "assets" / "eval_scenarios.json"
    else:
        json_path = Path(json_path)
    
    if not json_path.exists():
        print(f"❌ 文件不存在: {json_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenarios = data['scenarios']
    
    if scenario_id is not None:
        scenarios = [s for s in scenarios if s['scenario_id'] == scenario_id]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"场景 ID: {scenario['scenario_id']}")
        print(f"{'='*60}")
        
        robot = scenario['robot_start']
        target = scenario['target']
        
        print(f"\n🤖 机器人起点:")
        print(f"   位置: ({robot['x']:.3f}, {robot['y']:.3f})")
        print(f"   朝向: {robot['angle']:.3f} rad ({np.degrees(robot['angle']):.1f}°)")
        
        print(f"\n🎯 目标点:")
        print(f"   位置: ({target['x']:.3f}, {target['y']:.3f})")
        
        distance = np.sqrt((target['x'] - robot['x'])**2 + 
                          (target['y'] - robot['y'])**2)
        print(f"\n📏 直线距离: {distance:.3f}m")
        
        print(f"\n📦 所有元素:")
        for element in scenario['elements']:
            print(f"   - {element['name']}: ({element['x']:.3f}, {element['y']:.3f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化评估场景')
    parser.add_argument('--json', type=str, help='JSON文件路径')
    parser.add_argument('--scenario', type=int, help='显示特定场景ID')
    parser.add_argument('--details', action='store_true', help='打印详细信息')
    parser.add_argument('--no-plot', action='store_true', help='不显示图形')
    
    args = parser.parse_args()
    
    if args.details:
        print_scenario_details(args.json, args.scenario)
    
    if not args.no_plot:
        if args.scenario is not None:
            visualize_scenarios(args.json, show_all=False, scenario_ids=[args.scenario])
        else:
            visualize_scenarios(args.json)

