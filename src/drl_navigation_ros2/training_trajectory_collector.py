#!/usr/bin/env python3
"""
辅助函数：在训练过程中收集每个 episode 的轨迹数据
供 train.py 调用
"""

import pickle
from pathlib import Path
import torch
import numpy as np


class TrainingTrajectoryCollector:
    """训练轨迹收集器"""
    
    def __init__(self, save_dir="training_trajectories_8_polar"):
        """
        初始化收集器
        
        Args:
            save_dir: 保存目录名称（相对于assets文件夹）
        """
        self.save_dir = save_dir
        self.current_trajectory = None
        self.reset_trajectory()
        
        # 设置保存路径
        base_dir = Path("src/drl_navigation_ros2/assets")
        self.trajectory_dir = base_dir / save_dir
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
    
    def reset_trajectory(self):
        """重置当前轨迹"""
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'observations': [],
        }
    
    def add_step(self, state, action, reward, done, observation_dict):
        """
        添加一个训练步骤的数据
        
        Args:
            state: 处理后的状态（用于模型输入）
            action: 执行的动作
            reward: 获得的奖励
            done: 是否终止
            observation_dict: 原始观测信息字典，包含：
                - laser_scan: 激光雷达数据
                - distance: 到目标的距离
                - cos: 方向余弦
                - sin: 方向正弦
                - collision: 是否碰撞
                - goal: 是否到达目标
                - prev_action: 上一个动作
        """
        # 保存状态
        if isinstance(state, torch.Tensor):
            self.current_trajectory['states'].append(state.cpu().numpy())
        else:
            self.current_trajectory['states'].append(np.array(state))
        
        # 保存动作
        if isinstance(action, torch.Tensor):
            self.current_trajectory['actions'].append(action.cpu().numpy())
        else:
            self.current_trajectory['actions'].append(np.array(action))
        
        # 保存奖励和done标志
        self.current_trajectory['rewards'].append(float(reward))
        self.current_trajectory['dones'].append(bool(done))
        
        # 保存原始观测信息
        self.current_trajectory['observations'].append({
            'laser_scan': observation_dict['laser_scan'].copy() if hasattr(observation_dict['laser_scan'], 'copy') else observation_dict['laser_scan'],
            'distance': float(observation_dict['distance']),
            'cos': float(observation_dict['cos']),
            'sin': float(observation_dict['sin']),
            'collision': bool(observation_dict['collision']),
            'goal': bool(observation_dict['goal']),
            'prev_action': observation_dict['prev_action'].copy() if hasattr(observation_dict['prev_action'], 'copy') else observation_dict['prev_action']
        })
    
    def save_trajectory(self, epoch, episode):
        """
        保存当前轨迹到文件
        
        Args:
            epoch: 当前 epoch 编号
            episode: 当前 episode 编号
        
        Returns:
            str: 保存的文件路径
        """
        if len(self.current_trajectory['states']) == 0:
            print(f"⚠️  轨迹为空，跳过保存")
            return None
        
        # 转换为numpy数组
        trajectory_data = {
            'states': np.array(self.current_trajectory['states']),
            'actions': np.array(self.current_trajectory['actions']),
            'rewards': np.array(self.current_trajectory['rewards']),
            'dones': np.array(self.current_trajectory['dones']),
            'observations': self.current_trajectory['observations'],
            'info': {
                'epoch': epoch,
                'episode': episode,
                'steps': len(self.current_trajectory['states']),
                'total_reward': float(np.sum(self.current_trajectory['rewards'])),
                'collision': any(obs['collision'] for obs in self.current_trajectory['observations']),
                'reached_goal': any(obs['goal'] for obs in self.current_trajectory['observations']),
                'timeout': len(self.current_trajectory['states']) >= 300,  # 假设 max_steps=300
            }
        }
        
        # 构造保存路径
        epoch_dir = self.trajectory_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = epoch_dir / f"episode_{episode:03d}.pkl"
        
        # 保存
        with open(save_path, 'wb') as f:
            pickle.dump(trajectory_data, f)
        
        return str(save_path)


def collect_training_step(
    collector,
    state,
    action,
    reward,
    done,
    latest_scan,
    distance,
    cos,
    sin,
    collision,
    goal,
    prev_action
):
    """
    便捷函数：收集一个训练步骤
    
    Args:
        collector: TrainingTrajectoryCollector 实例
        state: 状态
        action: 动作
        reward: 奖励
        done: 是否终止
        latest_scan: 激光雷达数据
        distance: 到目标的距离
        cos: 方向余弦
        sin: 方向正弦
        collision: 是否碰撞
        goal: 是否到达目标
        prev_action: 上一个动作
    """
    observation_dict = {
        'laser_scan': latest_scan,
        'distance': distance,
        'cos': cos,
        'sin': sin,
        'collision': collision,
        'goal': goal,
        'prev_action': prev_action
    }
    
    collector.add_step(state, action, reward, done, observation_dict)


def load_training_trajectory(trajectory_path):
    """
    加载保存的训练轨迹数据
    
    Args:
        trajectory_path: 轨迹文件路径
    
    Returns:
        dict: 轨迹数据
    """
    with open(trajectory_path, 'rb') as f:
        trajectory = pickle.load(f)
    return trajectory


if __name__ == "__main__":
    print("这是一个辅助模块，应该从 train.py 导入使用")
    print("示例用法:")
    print("  from training_trajectory_collector import TrainingTrajectoryCollector")
    print("  collector = TrainingTrajectoryCollector()")
    print("  collector.add_step(state, action, reward, done, observation_dict)")
    print("  collector.save_trajectory(epoch, episode)")