"""
CVaR-CPO专用Replay Buffer (Ablation Study Version)

关键实现：
- 存储完整的 (s, a, r, t, s2, c, e_t, next_e_t) 8元组
- e_t 在rollout时追踪，按照论文公式: e_{t+1} = (e_t - C(s_t, a_t)) / γ
- 采样时直接使用存储的真实e_t值

论文: "CVaR-Constrained Policy Optimization for Safe Reinforcement Learning"
Zhang et al., IEEE TNNLS 2025
"""

import random
from collections import deque
import numpy as np


class CVaRReplayBuffer(object):
    """
    CVaR-CPO专用Replay Buffer
    
    存储格式: (s, a, r, t, s2, c, e_t, next_e_t)
    
    其中:
    - s: 原始状态 (state_dim,)，不包含e_t
    - a: 动作 (action_dim,)
    - r: 奖励 (scalar)
    - t: terminal flag (0 or 1)
    - s2: 下一状态 (state_dim,)，不包含e_t
    - c: 即时cost (scalar)
    - e_t: 当前累积cost阈值 (scalar)
    - next_e_t: 下一步累积cost阈值 (scalar)，= (e_t - c) / γ
    
    论文公式参考:
    - 状态扩展: s̄_t = (s_t, e_t)
    - e_t演化: e_0 = u^k, e_{t+1} = (e_t - C(s_t, a_t)) / γ
    """
    
    def __init__(self, buffer_size, random_seed=123):
        """
        初始化Buffer
        
        Args:
            buffer_size: 最大存储容量
            random_seed: 随机种子
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)
        
        print(f"📦 CVaRReplayBuffer 初始化")
        print(f"   Buffer大小: {buffer_size:,}")
        print(f"   存储格式: (s, a, r, t, s2, c, e_t, next_e_t)")

    def add(self, s, a, r, t, s2, c, e_t, next_e_t):
        """
        添加经验到buffer
        
        Args:
            s: 原始状态 (不含e_t)
            a: 动作
            r: 奖励
            t: terminal (0 or 1)
            s2: 下一状态 (不含e_t)
            c: 即时cost
            e_t: 当前累积cost阈值
            next_e_t: 下一步累积cost阈值 = (e_t - c) / γ
        """
        experience = (s, a, r, t, s2, c, e_t, next_e_t)
        
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        """返回当前buffer大小"""
        return self.count

    def sample_batch_with_augmented_state(self, batch_size):
        """
        采样并返回扩展状态
        
        返回的states和next_states已经扩展为 s̄ = (s, e_t)
        
        Args:
            batch_size: 批次大小
        
        Returns:
            dict with keys:
                - states: (batch, state_dim+1) - 扩展状态 s̄_t
                - next_states: (batch, state_dim+1) - 扩展状态 s̄_{t+1}
                - actions: (batch, action_dim)
                - rewards: (batch, 1)
                - costs: (batch, 1)
                - dones: (batch, 1)
                - e_t: (batch, 1)
                - next_e_t: (batch, 1)
        """
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        # 解包8元组
        s_batch = np.array([exp[0] for exp in batch])
        a_batch = np.array([exp[1] for exp in batch])
        r_batch = np.array([exp[2] for exp in batch]).reshape(-1, 1)
        t_batch = np.array([exp[3] for exp in batch]).reshape(-1, 1)
        s2_batch = np.array([exp[4] for exp in batch])
        c_batch = np.array([exp[5] for exp in batch]).reshape(-1, 1)
        e_t_batch = np.array([exp[6] for exp in batch]).reshape(-1, 1)
        next_e_t_batch = np.array([exp[7] for exp in batch]).reshape(-1, 1)
        
        # 扩展状态: s̄ = (s, e_t)
        augmented_s = np.concatenate([s_batch, e_t_batch], axis=1)
        augmented_s2 = np.concatenate([s2_batch, next_e_t_batch], axis=1)
        
        return {
            'states': augmented_s,
            'next_states': augmented_s2,
            'actions': a_batch,
            'rewards': r_batch,
            'costs': c_batch,
            'dones': t_batch,
            'e_t': e_t_batch,
            'next_e_t': next_e_t_batch,
        }

    def sample_batch_with_cost(self, batch_size):
        """
        普通采样方法（兼容性）
        返回: (s, a, r, c, t, s2)
        """
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([exp[0] for exp in batch])
        a_batch = np.array([exp[1] for exp in batch])
        r_batch = np.array([exp[2] for exp in batch]).reshape(-1, 1)
        t_batch = np.array([exp[3] for exp in batch]).reshape(-1, 1)
        s2_batch = np.array([exp[4] for exp in batch])
        c_batch = np.array([exp[5] for exp in batch]).reshape(-1, 1)

        return s_batch, a_batch, r_batch, c_batch, t_batch, s2_batch

    def get_statistics(self):
        """获取buffer统计信息（调试用）"""
        if self.count == 0:
            return {'count': 0}
        
        costs = [exp[5] for exp in self.buffer]
        e_ts = [exp[6] for exp in self.buffer]
        
        return {
            'count': self.count,
            'cost_mean': np.mean(costs),
            'cost_std': np.std(costs),
            'cost_min': np.min(costs),
            'cost_max': np.max(costs),
            'e_t_mean': np.mean(e_ts),
            'e_t_std': np.std(e_ts),
            'e_t_min': np.min(e_ts),
            'e_t_max': np.max(e_ts),
        }

    def clear(self):
        """清空buffer"""
        self.buffer.clear()
        self.count = 0