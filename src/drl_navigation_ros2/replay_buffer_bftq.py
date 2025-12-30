"""
Replay Buffer for BFTQ (Budgeted Fitted-Q)

扩展原始ReplayBuffer以支持:
1. Budget (预算) - 碰撞风险预算
2. Cost (成本) - 碰撞成本信号

经验格式: (state, action, reward, done, next_state, budget, cost)
"""

import random
from collections import deque
import numpy as np


class ReplayBufferBFTQ(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        BFTQ Replay Buffer

        Args:
            buffer_size: 缓冲区大小
            random_seed: 随机种子
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2, budget, cost):
        """
        添加经验到buffer

        Args:
            s: state
            a: action
            r: reward
            t: terminal (done)
            s2: next_state
            budget: 碰撞风险预算 ∈ [0, 1]
            cost: 碰撞成本 (1 if collision else 0)
        """
        experience = (s, a, r, t, s2, budget, cost)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch_bftq(self, batch_size):
        """
        采样batch (BFTQ版本)

        Returns:
            s_batch, a_batch, r_batch, t_batch, s2_batch, budget_batch, cost_batch
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
        budget_batch = np.array([_[5] for _ in batch]).reshape(-1, 1)
        cost_batch = np.array([_[6] for _ in batch]).reshape(-1, 1)

        return s_batch, a_batch, r_batch, t_batch, s2_batch, budget_batch, cost_batch

    def return_buffer(self):
        """返回完整buffer"""
        s = np.array([_[0] for _ in self.buffer])
        a = np.array([_[1] for _ in self.buffer])
        r = np.array([_[2] for _ in self.buffer]).reshape(-1, 1)
        t = np.array([_[3] for _ in self.buffer]).reshape(-1, 1)
        s2 = np.array([_[4] for _ in self.buffer])
        budget = np.array([_[5] for _ in self.buffer]).reshape(-1, 1)
        cost = np.array([_[6] for _ in self.buffer]).reshape(-1, 1)

        return s, a, r, t, s2, budget, cost

    def clear(self):
        self.buffer.clear()
        self.count = 0
