# TD3 with CVaR-Constrained Policy Optimization (CVar-CPO)_v1
import random
from collections import deque
import numpy as np


class CVaRReplayBuffer(object):
    """
    CVaR-CPO专用Replay Buffer
    支持状态扩展 s̄ = (s, e_t)
    """
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2, c=0.0):
        """
        添加经验到 buffer
        
        Args:
            s: state (不包含e_t，原始25维)
            a: action
            r: reward
            t: terminal
            s2: next_state (不包含e_t，原始25维)
            c: cost
        """
        experience = (s, a, r, t, s2, c)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch_with_cost(self, batch_size):
        """
        普通采样方法（兼容性）
        返回: (s, a, r, c, t, s2)
        """
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])
        c_batch = np.array([_[5] for _ in batch]).reshape(-1, 1)

        return s_batch, a_batch, r_batch, c_batch, t_batch, s2_batch

    def sample_batch_with_augmented_state(self, batch_size, var_u, discount=0.99):
        """
        CVaR-CPO专用采样方法
        返回扩展状态 s̄ = (s, e_t)，其中 e_t 根据cumulative cost计算
        
        Args:
            batch_size: 批次大小
            var_u: 当前VaR参数值
            discount: 折扣因子γ
        
        Returns:
            dict with keys:
                - states: (batch, state_dim+1) - 扩展状态 s̄_t
                - next_states: (batch, state_dim+1) - 扩展状态 s̄_{t+1}
                - actions: (batch, action_dim)
                - rewards: (batch, 1)
                - costs: (batch, 1)
                - dones: (batch, 1)
                - e_t: (batch, 1) - 当前累积cost阈值
                - next_e_t: (batch, 1) - 下一步累积cost阈值
        """
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        s_batch = np.array([_[0] for _ in batch])  # (batch, state_dim)
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])  # (batch, state_dim)
        c_batch = np.array([_[5] for _ in batch]).reshape(-1, 1)
        
        # 计算 e_t 和 next_e_t（论文公式中的状态扩展）
        # e_0 = var_u
        # e_{t+1} = (e_t - C(s_t, a_t)) / γ
        
        # 简化版本：e_t = var_u（初始阈值）
        e_t_batch = np.full((len(batch), 1), var_u, dtype=np.float32)
        
        # next_e_t = (e_t - cost) / discount
        next_e_t_batch = (e_t_batch - c_batch) / discount
        
        # 扩展状态: s̄ = (s, e_t)
        augmented_s_batch = np.concatenate([s_batch, e_t_batch], axis=1)  # (batch, state_dim+1)
        augmented_s2_batch = np.concatenate([s2_batch, next_e_t_batch], axis=1)  # (batch, state_dim+1)
        
        return {
            'states': augmented_s_batch,
            'next_states': augmented_s2_batch,
            'actions': a_batch,
            'rewards': r_batch,
            'costs': c_batch,
            'dones': t_batch,
            'e_t': e_t_batch,
            'next_e_t': next_e_t_batch,
        }

    def clear(self):
        self.buffer.clear()
        self.count = 0
