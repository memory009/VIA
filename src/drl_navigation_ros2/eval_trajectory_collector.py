#!/usr/bin/env python3
"""
辅助函数：在训练评估阶段收集轨迹数据
供 train.py 的 eval 函数调用
"""

import pickle
from pathlib import Path
import torch
import numpy as np


def collect_eval_trajectories(model, env, scenarios, epoch, max_steps, save_dir="trajectories_lightweight_8_polar"):
    """
    在评估阶段收集轨迹数据并保存
    
    Args:
        model: 训练的模型（TD3或其他）
        env: ROS环境
        scenarios: 评估场景列表
        epoch: 当前epoch编号
        max_steps: 每个场景的最大步数
        save_dir: 保存目录名称（相对于assets文件夹）
    
    Returns:
        str: 保存的文件路径
    """
    # 设置保存路径
    base_dir = Path("src/drl_navigation_ros2/assets")
    trajectory_dir = base_dir / save_dir
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件名格式：trajectories_lightweight_8_polar_epoch[i].pkl
    save_path = trajectory_dir / f"{save_dir}_epoch{epoch:03d}.pkl"
    
    print(f"📊 收集 Epoch {epoch} 的轨迹数据...")
    
    # 收集所有场景的轨迹
    all_trajectories = []
    
    for scenario_idx, scenario in enumerate(scenarios):
        trajectory = {
            'epoch': epoch,
            'scenario_id': scenario_idx,
            'states': [],
            'actions': [],
            'observations': [],
            'rewards': [],
            'dones': [],
            'info': {
                'reached_goal': False,
                'collision': False,
                'timeout': False,
                'steps': 0
            }
        }
        
        # 重置环境到该场景
        latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(scenario=scenario)
        
        step_count = 0
        done = False
        
        while step_count < max_steps and not done:
            # 准备状态
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            
            # 保存观测信息
            trajectory['observations'].append({
                'laser_scan': latest_scan.copy() if hasattr(latest_scan, 'copy') else latest_scan,
                'distance': float(distance),
                'cos': float(cos),
                'sin': float(sin),
                'collision': bool(collision),
                'goal': bool(goal),
                'prev_action': a.copy() if hasattr(a, 'copy') else a
            })
            
            # 保存状态
            if isinstance(state, torch.Tensor):
                trajectory['states'].append(state.cpu().numpy())
            else:
                trajectory['states'].append(state)
            
            # 检查终止条件
            if terminal:
                done = True
                trajectory['dones'].append(True)
                trajectory['info']['reached_goal'] = bool(goal)
                trajectory['info']['collision'] = bool(collision)
                break
            
            # 获取动作（evaluation模式，无噪声）
            action = model.get_action(state, add_noise=False)
            
            # 保存动作
            if isinstance(action, torch.Tensor):
                trajectory['actions'].append(action.cpu().numpy())
            else:
                trajectory['actions'].append(action)
            
            # 执行动作
            a_in = [(action[0] + 1) / 2, action[1]]
            latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            
            # 保存奖励和done标志
            trajectory['rewards'].append(float(reward))
            trajectory['dones'].append(False)
            
            step_count += 1
        
        # 如果达到最大步数
        if step_count >= max_steps and not done:
            trajectory['dones'][-1] = True
            trajectory['info']['timeout'] = True
        
        trajectory['info']['steps'] = step_count
        
        # 转换为numpy数组以便保存
        trajectory['states'] = np.array(trajectory['states'])
        trajectory['actions'] = np.array(trajectory['actions'])
        trajectory['rewards'] = np.array(trajectory['rewards'])
        trajectory['dones'] = np.array(trajectory['dones'])
        
        all_trajectories.append(trajectory)
    
    # 保存所有轨迹
    with open(save_path, 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    # 统计信息
    n_success = sum(1 for t in all_trajectories if t['info']['reached_goal'])
    n_collision = sum(1 for t in all_trajectories if t['info']['collision'])
    n_timeout = sum(1 for t in all_trajectories if t['info']['timeout'])
    avg_steps = np.mean([t['info']['steps'] for t in all_trajectories])
    
    print(f"   ✅ 轨迹已保存: {save_path.name}")
    print(f"   📈 统计: Success={n_success}/{len(scenarios)}, "
          f"Collision={n_collision}, Timeout={n_timeout}, "
          f"Avg Steps={avg_steps:.1f}")
    
    return str(save_path)


def load_trajectories(trajectory_path):
    """
    加载保存的轨迹数据
    
    Args:
        trajectory_path: 轨迹文件路径
    
    Returns:
        list: 轨迹数据列表
    """
    with open(trajectory_path, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories


if __name__ == "__main__":
    print("这是一个辅助模块，应该从 train.py 导入使用")
    print("示例用法:")
    print("  from trajectory_collector import collect_eval_trajectories")
    print("  collect_eval_trajectories(model, env, scenarios, epoch, max_steps)")