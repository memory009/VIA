#!/usr/bin/env python3
"""
导出 Gazebo 环境地图信息
用于可达性分析中的激光雷达预测
"""

import json
import numpy as np
from pathlib import Path


def export_obstacle_map():
    """
    导出 Gazebo 环境中的障碍物信息
    
    Returns:
        obstacle_map: 字典，包含所有障碍物的几何信息
    """
    
    # ===== 固定障碍物（从 ros_python.py 第37-42行提取） =====
    # 这4个障碍物始终存在，位置固定
    fixed_obstacles = [
        {
            'name': 'obstacle1',
            'position': [-2.93, 3.17],
            'shape': 'box',
            'size': [1.0, 1.0, 1.0],  # 1x1x1米立方体（从model.sdf获取）
            'height': 0.5,  # z坐标（底部中心）
            'type': 'fixed'
        },
        {
            'name': 'obstacle2',
            'position': [2.86, -3.0],
            'shape': 'box',
            'size': [1.0, 1.0, 1.0],
            'height': 0.5,
            'type': 'fixed'
        },
        {
            'name': 'obstacle3',
            'position': [-2.77, -0.96],
            'shape': 'box',
            'size': [1.0, 1.0, 1.0],
            'height': 0.5,
            'type': 'fixed'
        },
        {
            'name': 'obstacle4',
            'position': [2.83, 2.93],
            'shape': 'box',
            'size': [1.0, 1.0, 1.0],
            'height': 0.5,
            'type': 'fixed'
        }
    ]
    
    # ===== 环境边界 =====
    # 从 ros_python.py 第116-127行：机器人活动范围 [-4.0, 4.0] x [-4.0, 4.0]
    # 实际世界是 10x10 米，边界墙壁在外围
    boundary = {
        'x_min': -5.0,
        'x_max': 5.0,
        'y_min': -5.0,
        'y_max': 5.0,
        'robot_safe_zone': {
            'x_min': -4.0,
            'x_max': 4.0,
            'y_min': -4.0,
            'y_max': 4.0
        }
    }
    
    # ===== 边界墙壁（10x10米环境） =====
    # 从 10by10 模型定义推断
    boundary_walls = [
        {
            'name': 'wall_north',
            'position': [0.0, 5.0],
            'shape': 'box',
            'size': [10.0, 0.1, 1.0],  # 长x宽x高
            'type': 'boundary'
        },
        {
            'name': 'wall_south',
            'position': [0.0, -5.0],
            'shape': 'box',
            'size': [10.0, 0.1, 1.0],
            'type': 'boundary'
        },
        {
            'name': 'wall_east',
            'position': [5.0, 0.0],
            'shape': 'box',
            'size': [0.1, 10.0, 1.0],
            'type': 'boundary'
        },
        {
            'name': 'wall_west',
            'position': [-5.0, 0.0],
            'shape': 'box',
            'size': [0.1, 10.0, 1.0],
            'type': 'boundary'
        }
    ]
    
    # ===== 合并所有障碍物 =====
    obstacle_map = {
        'metadata': {
            'environment': 'turtlebot3_drl',
            'world_size': '10x10 meters',
            'robot_model': 'turtlebot3_waffle',
            'laser_range': 3.5,  # TurtleBot3 激光雷达最大范围
            'laser_beams': 20,   # 使用的激光束数量
            'laser_fov': 180,    # 视野角度（度）
        },
        'boundary': boundary,
        'obstacles': fixed_obstacles + boundary_walls,
        'total_obstacles': len(fixed_obstacles) + len(boundary_walls)
    }
    
    return obstacle_map


def compute_distance_to_obstacle(robot_pos, robot_yaw, beam_angle, obstacles):
    """
    计算给定光线与障碍物的交点距离
    
    Args:
        robot_pos: (x, y) 机器人位置
        robot_yaw: float, 机器人朝向（弧度）
        beam_angle: float, 光线相对朝向的角度（弧度）
        obstacles: list of obstacle dicts
    
    Returns:
        distance: float, 最近障碍物的距离（米）
    """
    # 光线的全局角度
    ray_angle = robot_yaw + beam_angle
    ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])
    
    min_distance = 3.5  # 激光雷达最大范围
    
    for obs in obstacles:
        if obs['shape'] == 'box':
            # 计算光线与矩形障碍物的交点
            dist = ray_box_intersection(
                robot_pos,
                ray_dir,
                obs['position'],
                obs['size']
            )
            if dist is not None and dist < min_distance:
                min_distance = dist
    
    return min_distance


def ray_box_intersection(ray_origin, ray_dir, box_center, box_size):
    """
    计算光线与2D矩形的交点（2D光线投射）
    
    Args:
        ray_origin: (x, y) 光线起点
        ray_dir: (dx, dy) 光线方向（单位向量）
        box_center: (x, y) 矩形中心
        box_size: [width, height, _] 矩形尺寸
    
    Returns:
        distance: float or None
    """
    # 将矩形转换为边界
    half_width = box_size[0] / 2.0
    half_height = box_size[1] / 2.0
    
    box_min = np.array([
        box_center[0] - half_width,
        box_center[1] - half_height
    ])
    box_max = np.array([
        box_center[0] + half_width,
        box_center[1] + half_height
    ])
    
    # 使用 slab method 计算交点
    t_min = -np.inf
    t_max = np.inf
    
    for i in range(2):  # x, y 两个维度
        if abs(ray_dir[i]) < 1e-8:  # 光线平行于该轴
            if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                return None  # 光线不会相交
        else:
            t1 = (box_min[i] - ray_origin[i]) / ray_dir[i]
            t2 = (box_max[i] - ray_origin[i]) / ray_dir[i]
            
            if t1 > t2:
                t1, t2 = t2, t1
            
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            
            if t_min > t_max:
                return None  # 不相交
    
    if t_min < 0:
        return None  # 交点在光线起点后方
    
    return t_min


def predict_laser_scan(robot_pos, robot_yaw, obstacle_map, n_beams=20):
    """
    预测给定位姿下的激光雷达扫描
    
    Args:
        robot_pos: (x, y) 机器人位置
        robot_yaw: float, 机器人朝向（弧度）
        obstacle_map: 障碍物地图字典
        n_beams: 激光束数量
    
    Returns:
        laser_scan: array of shape (n_beams,), 每个光束的距离（米）
    """
    # TurtleBot3 激光雷达参数
    fov = np.pi  # 180度视野
    beam_angles = np.linspace(-fov/2, fov/2, n_beams)
    
    laser_scan = []
    obstacles = obstacle_map['obstacles']
    
    for angle in beam_angles:
        distance = compute_distance_to_obstacle(
            robot_pos,
            robot_yaw,
            angle,
            obstacles
        )
        laser_scan.append(distance)
    
    return np.array(laser_scan)


def save_obstacle_map(output_path=None):
    """
    保存障碍物地图到JSON文件
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent / "assets" / "obstacle_map.json"
    
    obstacle_map = export_obstacle_map()
    
    # 添加一些示例预测
    examples = []
    test_positions = [
        ([0.0, 0.0], 0.0),
        ([-2.0, 2.0], np.pi/4),
        ([2.0, -2.0], -np.pi/2)
    ]
    
    for pos, yaw in test_positions:
        laser = predict_laser_scan(pos, yaw, obstacle_map)
        examples.append({
            'position': pos,
            'yaw': float(yaw),
            'yaw_deg': float(np.degrees(yaw)),
            'laser_scan': laser.tolist(),
            'min_laser': float(np.min(laser))
        })
    
    obstacle_map['examples'] = examples
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(obstacle_map, f, indent=2)
    
    print(f"✅ 障碍物地图已保存到: {output_path}")
    print(f"\n地图信息:")
    print(f"  - 固定障碍物: 4个")
    print(f"  - 边界墙壁: 4个")
    print(f"  - 环境大小: 10x10 米")
    print(f"  - 机器人活动范围: 8x8 米")
    
    print(f"\n固定障碍物位置:")
    for obs in obstacle_map['obstacles'][:4]:
        print(f"  - {obs['name']}: {obs['position']}")
    
    print(f"\n示例激光雷达预测:")
    for ex in examples:
        print(f"  位置 {ex['position']}, 朝向 {ex['yaw_deg']:.1f}°: min_laser = {ex['min_laser']:.3f}m")
    
    return output_path


def visualize_map(obstacle_map=None, save_path=None):
    """
    可视化障碍物地图（可选）
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("⚠️  需要 matplotlib 进行可视化，跳过")
        return
    
    if obstacle_map is None:
        obstacle_map = export_obstacle_map()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制障碍物
    for obs in obstacle_map['obstacles']:
        pos = obs['position']
        size = obs['size']
        
        if obs['type'] == 'fixed':
            color = 'red'
            alpha = 0.7
        else:  # boundary
            color = 'gray'
            alpha = 0.5
        
        rect = patches.Rectangle(
            (pos[0] - size[0]/2, pos[1] - size[1]/2),
            size[0], size[1],
            linewidth=2, edgecolor='black', facecolor=color, alpha=alpha
        )
        ax.add_patch(rect)
        
        # 添加标签
        if obs['type'] == 'fixed':
            ax.text(pos[0], pos[1], obs['name'], 
                   ha='center', va='center', fontsize=8, weight='bold')
    
    # 绘制机器人安全区域
    safe_zone = obstacle_map['boundary']['robot_safe_zone']
    rect = patches.Rectangle(
        (safe_zone['x_min'], safe_zone['y_min']),
        safe_zone['x_max'] - safe_zone['x_min'],
        safe_zone['y_max'] - safe_zone['y_min'],
        linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
    )
    ax.add_patch(rect)
    
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Gazebo Environment - Obstacle Map')
    ax.legend(['Fixed Obstacles', 'Boundary Walls', 'Robot Safe Zone'])
    
    if save_path is None:
        save_path = Path(__file__).parent.parent / "visualizations" / "obstacle_map.png"
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 地图可视化已保存到: {save_path}")
    
    return fig


def main():
    """主函数"""
    print("=" * 70)
    print("🗺️  Gazebo 环境地图导出工具")
    print("=" * 70)
    
    # 1. 导出并保存地图
    output_path = save_obstacle_map()
    
    # 2. 可视化地图
    print("\n" + "=" * 70)
    print("🎨 生成地图可视化...")
    obstacle_map = export_obstacle_map()
    visualize_map(obstacle_map)
    
    print("\n" + "=" * 70)
    print("✅ 完成！")
    print("\n使用方法:")
    print("  1. 地图数据: src/drl_navigation_ros2/assets/obstacle_map.json")
    print("  2. 可视化: src/drl_navigation_ros2/visualizations/obstacle_map.png")
    print("\n在代码中使用:")
    print("  >>> from scripts.export_gazebo_map import predict_laser_scan")
    print("  >>> laser = predict_laser_scan((0, 0), 0.0, obstacle_map)")
    print("=" * 70)


if __name__ == "__main__":
    main()

