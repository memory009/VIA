#!/usr/bin/env python3
"""
辅助函数：在训练过程中保存每个 episode 的场景信息
供 train.py 调用
"""

from pathlib import Path
import json
import sys

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def save_training_episode_map(
    env,
    epoch,
    episode,
    save_dir="train_scenario_8_polar"
):
    """
    保存当前训练 episode 的场景地图
    
    Args:
        env: ROS_env 环境实例
        epoch: 当前 epoch 编号
        episode: 当前 episode 编号
        save_dir: 保存目录名称
    
    Returns:
        str: 保存的文件路径
    """
    # 延迟导入以避免循环依赖
    try:
        from scripts.export_gazebo_map_polar import (
            export_obstacle_map,
            load_obstacle_specs
        )
    except ImportError:
        print("⚠️  无法导入 export_gazebo_map_polar")
        return None
    
    # 设置保存路径
    base_dir = Path("src/drl_navigation_ros2/assets")
    epoch_dir = base_dir / save_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件名：episode_XXX.json
    save_path = epoch_dir / f"episode_{episode:03d}.json"
    
    # 从 ROS_env 提取当前场景信息
    scenario_dict = extract_scenario_from_env(env)
    
    # 生成障碍物地图
    obstacle_map = export_obstacle_map(scenario=scenario_dict)
    
    # 添加训练信息
    obstacle_map['training_info'] = {
        'epoch': epoch,
        'episode': episode,
        'is_training': True
    }
    
    # 保存
    with open(save_path, 'w') as f:
        json.dump(obstacle_map, f, indent=2)
    
    return str(save_path)


def extract_scenario_from_env(env):
    """
    从 ROS_env 实例中提取当前场景信息
    
    Args:
        env: ROS_env 实例
    
    Returns:
        dict: 场景信息字典
    """
    scenario_dict = {
        'scenario_id': -1,  # 训练场景没有固定 ID
        'elements': []
    }
    
    # 从 env.element_positions 重建场景信息
    # 注意：env.element_positions 只包含 [x, y]，没有 name 和 angle
    
    # 固定障碍物（索引 0-3）
    fixed_obstacles = [
        ('obstacle1', [-2.93, 3.17]),
        ('obstacle2', [2.86, -3.0]),
        ('obstacle3', [-2.77, -0.96]),
        ('obstacle4', [2.83, 2.93]),
    ]
    
    for name, (x, y) in fixed_obstacles:
        scenario_dict['elements'].append({
            'name': name,
            'x': float(x),
            'y': float(y),
            'angle': 0.0  # 固定障碍物角度为0
        })
    
    # 随机障碍物（索引 4-7，如果存在）
    # 从 env.element_positions 中提取
    if hasattr(env, 'element_positions') and len(env.element_positions) > 4:
        # 随机障碍物从索引4开始
        for i in range(4, min(8, len(env.element_positions))):
            if i < len(env.element_positions):
                pos = env.element_positions[i]
                obstacle_name = f'obstacle{i + 1}'
                
                scenario_dict['elements'].append({
                    'name': obstacle_name,
                    'x': float(pos[0]),
                    'y': float(pos[1]),
                    'angle': 0.0  # 训练时的随机障碍物角度信息未保存
                })
    
    # 机器人位置（倒数第二个）
    if hasattr(env, 'element_positions') and len(env.element_positions) >= 2:
        robot_pos = env.element_positions[-2]
        scenario_dict['elements'].append({
            'name': 'turtlebot3_waffle',
            'x': float(robot_pos[0]),
            'y': float(robot_pos[1]),
            'angle': 0.0  # 角度信息未保存在 element_positions 中
        })
    
    # 目标位置（最后一个）
    if hasattr(env, 'target'):
        scenario_dict['elements'].append({
            'name': 'target',
            'x': float(env.target[0]),
            'y': float(env.target[1]),
            'angle': 0.0
        })
    
    return scenario_dict


def save_training_episode_map_enhanced(
    env,
    epoch,
    episode,
    robot_pose=None,
    obstacle_angles=None,
    save_dir="train_scenario_8_polar"
):
    """
    增强版本：如果能获取到更详细的信息（角度等），可以保存更完整的场景
    
    Args:
        env: ROS_env 环境实例
        epoch: 当前 epoch 编号
        episode: 当前 episode 编号
        robot_pose: 机器人位姿 [x, y, theta]（如果可用）
        obstacle_angles: 障碍物角度字典 {obstacle_name: angle}（如果可用）
        save_dir: 保存目录名称
    
    Returns:
        str: 保存的文件路径
    """
    try:
        from scripts.export_gazebo_map_polar import export_obstacle_map
    except ImportError:
        print("⚠️  无法导入 export_gazebo_map_polar")
        return None
    
    base_dir = Path("src/drl_navigation_ros2/assets")
    epoch_dir = base_dir / save_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = epoch_dir / f"episode_{episode:03d}.json"
    
    # 提取场景信息
    scenario_dict = extract_scenario_from_env(env)
    
    # 如果提供了额外信息，更新场景
    if robot_pose is not None and len(scenario_dict['elements']) > 0:
        # 找到机器人元素并更新
        for elem in scenario_dict['elements']:
            if elem['name'] == 'turtlebot3_waffle':
                elem['x'] = float(robot_pose[0])
                elem['y'] = float(robot_pose[1])
                if len(robot_pose) > 2:
                    elem['angle'] = float(robot_pose[2])
    
    if obstacle_angles is not None:
        # 更新障碍物角度
        for elem in scenario_dict['elements']:
            if elem['name'] in obstacle_angles:
                elem['angle'] = float(obstacle_angles[elem['name']])
    
    # 生成障碍物地图
    obstacle_map = export_obstacle_map(scenario=scenario_dict)
    
    # 添加训练信息
    obstacle_map['training_info'] = {
        'epoch': epoch,
        'episode': episode,
        'is_training': True,
        'has_robot_pose': robot_pose is not None,
        'has_obstacle_angles': obstacle_angles is not None
    }
    
    # 保存
    with open(save_path, 'w') as f:
        json.dump(obstacle_map, f, indent=2)
    
    return str(save_path)


if __name__ == "__main__":
    print("这是一个辅助模块，应该从 train.py 导入使用")
    print("示例用法:")
    print("  from training_scenario_saver import save_training_episode_map")
    print("  save_training_episode_map(env, epoch, episode)")