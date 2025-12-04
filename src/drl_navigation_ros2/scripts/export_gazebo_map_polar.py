#!/usr/bin/env python3
"""
导出 Gazebo 环境地图信息
用于可达性分析中的激光雷达预测
"""

import json
import xml.etree.ElementTree as ET
import numpy as np
import math
from pathlib import Path

MODELS_DIR = (
    Path(__file__).resolve().parents[2]
    / "turtlebot3_simulations"
    / "turtlebot3_gazebo"
    / "models"
)

# Gazebo world 和模型目录中的 model.sdf 命名不完全一致（obstacle1/2 尺寸互换）。
# 为保持与实际仿真一致，强制覆盖已知障碍尺寸。
SIZE_OVERRIDES = {
    "obstacle1": {"shape": "box", "size": [0.3, 0.7, 1.0], "height": 1.0},
    "obstacle2": {"shape": "box", "size": [1.0, 1.0, 1.0], "height": 1.0},
}


def load_obstacle_specs(obstacle_names=None):
    """
    直接从 Gazebo model.sdf 解析障碍物几何信息，确保尺寸与仿真一致
    """
    if obstacle_names is None:
        obstacle_names = []
        for path in MODELS_DIR.iterdir():
            if path.is_dir() and path.name.startswith("obstacle"):
                obstacle_names.append(path.name)
        if not obstacle_names:
            obstacle_names = [f"obstacle{i}" for i in range(1, 9)]
        else:
            def _sort_key(name):
                suffix = name.replace("obstacle", "")
                return int(suffix) if suffix.isdigit() else float("inf")
            obstacle_names = sorted(obstacle_names, key=_sort_key)

    specs = {}
    for name in obstacle_names:
        sdf_path = MODELS_DIR / name / "model.sdf"
        if not sdf_path.exists():
            continue

        try:
            tree = ET.parse(sdf_path)
            geometry = tree.find(".//collision/geometry")
        except ET.ParseError:
            continue

        if geometry is None:
            continue

        entry = {'shape': 'box', 'size': [1.0, 1.0, 1.0]}
        box = geometry.find("box")
        cylinder = geometry.find("cylinder")

        if box is not None and box.find("size") is not None:
            size_vals = [float(v) for v in box.find("size").text.split()]
            entry = {'shape': 'box', 'size': size_vals}
        elif cylinder is not None and cylinder.find("radius") is not None:
            radius = float(cylinder.find("radius").text)
            height = float(cylinder.find("length").text)
            entry = {
                'shape': 'cylinder',
                'radius': radius,
                'height': height,
                'size': [2 * radius, 2 * radius, height]
            }

        specs[name] = entry

    # 应用尺寸覆盖，确保与 Gazebo 中的实体一致
    for name, override in SIZE_OVERRIDES.items():
        if name in specs:
            specs[name].update(override)
        else:
            specs[name] = override.copy()

    return specs


OBSTACLE_SPECS = load_obstacle_specs()


def _resolve_scenario_tag(scenario_path: Path) -> str:
    """根据文件名推断场景标签（例如 eval_scenarios_8_polar -> 8_polar）"""
    stem = scenario_path.stem
    prefix = "eval_scenarios_"
    if stem.startswith(prefix):
        tag = stem[len(prefix):]
        return tag.lstrip("_") or "default"
    parts = stem.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return part
    return stem or "default"


def load_eval_scenarios(path=None):
    """
    加载 eval_scenarios JSON，并返回 (data, scenario_tag)
    scenario_tag 根据文件名推断，用于输出目录划分
    """
    if path is None:
        path = (
            Path(__file__).parent.parent
            / "assets"
            / "eval_scenarios_8_polar.json"
        )

    scenario_path = Path(path)
    scenario_tag = _resolve_scenario_tag(scenario_path)

    if not scenario_path.exists():
        return None, scenario_tag

    with open(scenario_path, "r", encoding="utf-8") as f:
        return json.load(f), scenario_tag


def build_obstacle_entry(name, position, obs_type, yaw=None):
    """
    根据 SDF 规格构建障碍物条目
    """
    spec = OBSTACLE_SPECS.get(name, {'shape': 'box', 'size': [1.0, 1.0, 1.0]})
    entry = {
        'name': name,
        'position': position,
        'type': obs_type,
        'shape': spec.get('shape', 'box'),
        'size': spec.get('size', [1.0, 1.0, 1.0]),
        'height': spec.get('size', [1.0, 1.0, 1.0])[2] if spec.get('shape') == 'box' else spec.get('height', 1.0),
    }

    if yaw is not None:
        entry['yaw'] = yaw

    if entry['shape'] == 'cylinder':
        entry['radius'] = spec.get('radius', entry['size'][0] / 2.0)
        entry['height'] = spec.get('height', entry['size'][2])

    return entry


def generate_seeded_obstacles(
    base_positions,
    seed=42,
    count=4,
    min_dist=1.2,
    area_limits=(-4.0, 4.0),
):
    """
    使用固定随机种子生成与训练环境一致的“随机”障碍物
    """
    rng = np.random.RandomState(seed)
    element_positions = [pos[:] for pos in base_positions]
    seeded_obstacles = []

    for idx in range(count):
        yaw = float(rng.uniform(-np.pi, np.pi))  # 保持与 utils.set_random_position 一致

        while True:
            x = float(rng.uniform(area_limits[0], area_limits[1]))
            y = float(rng.uniform(area_limits[0], area_limits[1]))

            candidate = np.array([x, y])
            distances = [
                np.linalg.norm(np.array(existing) - candidate)
                for existing in element_positions
            ]
            if all(dist >= min_dist for dist in distances):
                break

        element_positions.append([x, y])
        name = f'obstacle{idx + 5}'
        entry = build_obstacle_entry(name, [x, y], obs_type='seeded', yaw=yaw)
        entry['seed'] = seed
        entry['min_distance'] = min_dist
        seeded_obstacles.append(entry)

    return seeded_obstacles


def export_obstacle_map(scenario=None):
    """
    导出 Gazebo 环境中的障碍物信息
    
    Returns:
        obstacle_map: 字典，包含所有障碍物的几何信息
    """
    
    # ===== 固定障碍物（从 ros_python.py 第37-42行提取） =====
    # 这4个障碍物始终存在，位置固定
    fixed_positions = [
        ('obstacle1', [-2.93, 3.17]),
        ('obstacle2', [2.86, -3.0]),
        ('obstacle3', [-2.77, -0.96]),
        ('obstacle4', [2.83, 2.93]),
    ]
    fixed_obstacles = [
        build_obstacle_entry(name, pos, obs_type='fixed')
        for name, pos in fixed_positions
    ]

    scenario_obstacles = []
    scenario_id = None
    if scenario is not None:
        scenario_id = scenario.get("scenario_id")
        for element in scenario.get("elements", []):
            name = element.get("name", "")
            if name.startswith("obstacle"):
                suffix = name.replace("obstacle", "")
                if suffix.isdigit() and int(suffix) >= 5:
                    pos = [element.get("x", 0.0), element.get("y", 0.0)]
                    yaw = element.get("angle")
                    entry = build_obstacle_entry(name, pos, obs_type="scenario_obstacle", yaw=yaw)
                    entry["scenario_id"] = scenario_id
                    scenario_obstacles.append(entry)
    else:
        scenario_obstacles = generate_seeded_obstacles(
            base_positions=[obs['position'][:] for obs in fixed_obstacles],
            seed=42,
            count=4,
            min_dist=1.2,
            area_limits=(-4.0, 4.0)
        )
    
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
    all_obstacles = fixed_obstacles + scenario_obstacles + boundary_walls
    counts = {
        'fixed': len(fixed_obstacles),
        'scenario_obstacles': len(scenario_obstacles),
        'boundary': len(boundary_walls)
    }

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
        'obstacles': all_obstacles,
        'counts': counts,
        'total_obstacles': len(all_obstacles)
    }
    if scenario is None:
        obstacle_map['metadata']['seeded_obstacles'] = {
            'seed': 42,
            'count': len(scenario_obstacles),
            'min_distance': 1.2,
            'area': {
                'x_range': [-4.0, 4.0],
                'y_range': [-4.0, 4.0]
            }
        }
    else:
        obstacle_map['metadata']['scenario'] = {
            'scenario_id': scenario_id,
            'elements': len(scenario.get("elements", [])),
            'source': 'eval_scenarios.json'
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
        dist = None
        if obs['shape'] == 'box':
            dist = ray_box_intersection(
                robot_pos,
                ray_dir,
                obs['position'],
                obs['size']
            )
        elif obs['shape'] == 'cylinder':
            dist = ray_circle_intersection(
                robot_pos,
                ray_dir,
                obs['position'],
                obs.get('radius', obs['size'][0] / 2.0)
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


def ray_circle_intersection(ray_origin, ray_dir, circle_center, radius):
    """
    计算光线与圆（圆柱截面）的交点，返回距离或 None
    """
    oc = np.array(ray_origin) - np.array(circle_center)
    b = 2.0 * np.dot(ray_dir, oc)
    c = np.dot(oc, oc) - radius ** 2
    discriminant = b ** 2 - 4 * c  # ray_dir 已归一化 => a = 1

    if discriminant < 0:
        return None

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    for t in (t1, t2):
        if t >= 0:
            return t

    return None


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


def save_obstacle_map(output_path=None, scenario=None, scenario_tag=None):
    """
    保存障碍物地图到JSON文件
    """
    if output_path is None:
        assets_dir = Path(__file__).parent.parent / "assets"
        if scenario_tag:
            assets_dir = assets_dir / f"eval_scenarios_{scenario_tag}"
        output_path = assets_dir / "obstacle_map_8_polar.json"
    
    obstacle_map = export_obstacle_map(scenario=scenario)
    
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
    counts = obstacle_map.get('counts', {})
    print(f"  - 固定障碍物: {counts.get('fixed', 0)}个")
    if scenario is None:
        seeded_meta = obstacle_map['metadata'].get('seeded_obstacles', {})
        print(f"  - 种子障碍物: {counts.get('scenario_obstacles', 0)}个 (seed={seeded_meta.get('seed')})")
    else:
        print(f"  - 场景障碍物: {counts.get('scenario_obstacles', 0)}个 (scenario={scenario.get('scenario_id')})")
    print(f"  - 环境大小: 10x10 米")
    print(f"  - 机器人活动范围: 8x8 米")
    
    print(f"\n固定障碍物位置:")
    for obs in obstacle_map['obstacles']:
        if obs['type'] == 'fixed':
            print(f"  - {obs['name']}: {obs['position']}")
    
    if scenario is None:
        seeded_meta = obstacle_map['metadata'].get('seeded_obstacles', {})
        print(f"\nSeed={seeded_meta.get('seed')} 障碍物位置:")
        for obs in obstacle_map['obstacles']:
            if obs['type'] == 'scenario_obstacle':
                print(f"  - {obs['name']}: {obs['position']} (yaw={obs['yaw']:.3f} rad)")
    else:
        print(f"\nScenario {scenario.get('scenario_id')} 障碍物位置:")
        for obs in obstacle_map['obstacles']:
            if obs['type'] == 'scenario_obstacle':
                print(f"  - {obs['name']}: {obs['position']} (yaw={obs.get('yaw', 0):.3f} rad)")
    
    print(f"\n示例激光雷达预测:")
    for ex in examples:
        print(f"  位置 {ex['position']}, 朝向 {ex['yaw_deg']:.1f}°: min_laser = {ex['min_laser']:.3f}m")
    
    return obstacle_map


def visualize_map(obstacle_map=None, save_path=None, scenario_tag=None):
    """
    可视化障碍物地图（可选）
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.transforms import Affine2D
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
        elif obs['type'] == 'scenario_obstacle':
            color = 'orange'
            alpha = 0.7
        else:  # boundary
            color = 'gray'
            alpha = 0.5

        if obs['shape'] == 'cylinder':
            radius = obs.get('radius', size[0] / 2)
            patch = patches.Circle(
                (pos[0], pos[1]),
                radius=radius,
                linewidth=2,
                edgecolor='black',
                facecolor=color,
                alpha=alpha
            )
        else:
            yaw = obs.get('yaw', 0.0)
            angle_deg = math.degrees(yaw)
            transform = (
                Affine2D()
                .rotate_deg(angle_deg)
                .translate(pos[0], pos[1])
                + ax.transData
            )
            patch = patches.Rectangle(
                (-size[0] / 2, -size[1] / 2),
                size[0], size[1],
                linewidth=2, edgecolor='black', facecolor=color, alpha=alpha,
                transform=transform
            )
        ax.add_patch(patch)

        # 添加标签（固定与种子障碍物）
        if obs['type'] in {'fixed', 'scenario_obstacle'}:
            ax.text(
                pos[0],
                pos[1],
                obs['name'],
                ha='center',
                va='center',
                fontsize=8,
                weight='bold'
            )
    
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
    legend_handles = [
        patches.Patch(facecolor='red', edgecolor='black', alpha=0.7, label='Fixed Obstacles'),
        patches.Patch(facecolor='orange', edgecolor='black', alpha=0.7, label='Scenario Obstacles'),
        patches.Patch(facecolor='gray', edgecolor='black', alpha=0.5, label='Boundary Walls'),
        patches.Patch(facecolor='none', edgecolor='green', linestyle='--', label='Robot Safe Zone')
    ]
    ax.legend(handles=legend_handles, loc='upper right')
    
    if save_path is None:
        vis_dir = Path(__file__).parent.parent / "visualizations"
        if scenario_tag:
            save_path = vis_dir / f"obstacle_{scenario_tag}_map" / "obstacle_map.png"
        else:
            save_path = vis_dir / "obstacle_map.png"
    
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
    eval_data, scenario_tag = load_eval_scenarios()
    base_dir = Path(__file__).parent.parent
    assets_dir = base_dir / "assets"
    visuals_dir = base_dir / "visualizations"

    if scenario_tag:
        assets_dir = assets_dir / f"eval_scenarios_{scenario_tag}"
        vis_dir = visuals_dir / f"obstacle_{scenario_tag}_map"
    else:
        vis_dir = visuals_dir / "obstacle_map"

    if eval_data and eval_data.get("enable_random_obstacles"):
        print("\n检测到 eval_scenarios 中启用了随机障碍，将为每个场景生成独立地图...")
        vis_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)
        for scenario in eval_data.get("scenarios", []):
            scenario_id = scenario.get("scenario_id", 0)
            json_path = assets_dir / f"obstacle_map_scenario_{scenario_id:02d}.json"
            png_path = vis_dir / f"obstacle_map_scenario_{scenario_id:02d}.png"
            obstacle_map = save_obstacle_map(
                output_path=json_path,
                scenario=scenario,
                scenario_tag=scenario_tag,
            )
            print("\n" + "-" * 60)
            print(f"🎨 生成场景 {scenario_id} 地图可视化...")
            visualize_map(
                obstacle_map=obstacle_map,
                save_path=png_path,
                scenario_tag=scenario_tag,
            )
        print("\n" + "=" * 70)
        print("✅ 所有场景地图已生成！")
        asset_hint = (
            f"src/drl_navigation_ros2/assets/eval_scenarios_{scenario_tag}/obstacle_map_scenario_XX.json"
            if scenario_tag
            else "src/drl_navigation_ros2/assets/obstacle_map_scenario_XX.json"
        )
        vis_hint = (
            f"src/drl_navigation_ros2/visualizations/obstacle_{scenario_tag}_map/obstacle_map_scenario_XX.png"
            if scenario_tag
            else "src/drl_navigation_ros2/visualizations/obstacle_map/obstacle_map_scenario_XX.png"
        )
        print(f"地图目录: {asset_hint}")
        print(f"可视化目录: {vis_hint}")
        print("=" * 70)
        return
    
    obstacle_map = save_obstacle_map(scenario_tag=scenario_tag)
    print("\n" + "=" * 70)
    print("🎨 生成地图可视化...")
    visualize_map(obstacle_map, scenario_tag=scenario_tag)
    
    print("\n" + "=" * 70)
    print("✅ 完成！")
    print("\n使用方法:")
    asset_default = (
        f"src/drl_navigation_ros2/assets/eval_scenarios_{scenario_tag}/obstacle_map.json"
        if scenario_tag
        else "src/drl_navigation_ros2/assets/obstacle_map.json"
    )
    vis_default = (
        f"src/drl_navigation_ros2/visualizations/obstacle_{scenario_tag}_map/obstacle_map.png"
        if scenario_tag
        else "src/drl_navigation_ros2/visualizations/obstacle_map.png"
    )
    print(f"  1. 地图数据: {asset_default}")
    print(f"  2. 可视化: {vis_default}")
    print("\n在代码中使用:")
    print("  >>> from scripts.export_gazebo_map import predict_laser_scan")
    print("  >>> laser = predict_laser_scan((0, 0), 0.0, obstacle_map)")
    print("=" * 70)


if __name__ == "__main__":
    main()

