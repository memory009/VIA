"""
Replay buffer for VIA training.

Stores 8-tuples (s, a, r, t, s2, c, e_t, next_e_t) where e_t is the
cumulative cost threshold tracked per rollout step:
    e_0 = u^k,  e_{t+1} = (e_t - C(s_t, a_t)) / gamma

Augmented state s̄ = (s, e_t) is assembled on the fly during sampling.
"""

import random
from collections import deque
import numpy as np


class VIAReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2, c, e_t, next_e_t):
        """Store one transition. Evicts the oldest entry when the buffer is full."""
        experience = (s, a, r, t, s2, c, e_t, next_e_t)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch_with_augmented_state(self, batch_size):
        """
        Sample a random batch and return augmented states s̄ = (s, e_t).

        Returns a dict with keys:
            states, next_states  — shape (batch, state_dim + 1)
            actions              — shape (batch, action_dim)
            rewards, costs, dones, e_t, next_e_t  — shape (batch, 1)
        """
        batch = random.sample(self.buffer, min(batch_size, self.count))

        s_batch      = np.array([exp[0] for exp in batch])
        a_batch      = np.array([exp[1] for exp in batch])
        r_batch      = np.array([exp[2] for exp in batch]).reshape(-1, 1)
        t_batch      = np.array([exp[3] for exp in batch]).reshape(-1, 1)
        s2_batch     = np.array([exp[4] for exp in batch])
        c_batch      = np.array([exp[5] for exp in batch]).reshape(-1, 1)
        e_t_batch    = np.array([exp[6] for exp in batch]).reshape(-1, 1)
        next_e_t_batch = np.array([exp[7] for exp in batch]).reshape(-1, 1)

        # Augment: s̄ = (s, e_t)
        augmented_s  = np.concatenate([s_batch, e_t_batch], axis=1)
        augmented_s2 = np.concatenate([s2_batch, next_e_t_batch], axis=1)

        return {
            'states':      augmented_s,
            'next_states': augmented_s2,
            'actions':     a_batch,
            'rewards':     r_batch,
            'costs':       c_batch,
            'dones':       t_batch,
            'e_t':         e_t_batch,
            'next_e_t':    next_e_t_batch,
        }

    def sample_batch_with_cost(self, batch_size):
        """Return (s, a, r, c, t, s2) without e_t augmentation."""
        batch = random.sample(self.buffer, min(batch_size, self.count))
        s_batch  = np.array([exp[0] for exp in batch])
        a_batch  = np.array([exp[1] for exp in batch])
        r_batch  = np.array([exp[2] for exp in batch]).reshape(-1, 1)
        t_batch  = np.array([exp[3] for exp in batch]).reshape(-1, 1)
        s2_batch = np.array([exp[4] for exp in batch])
        c_batch  = np.array([exp[5] for exp in batch]).reshape(-1, 1)
        return s_batch, a_batch, r_batch, c_batch, t_batch, s2_batch

    def get_statistics(self):
        """Return cost and e_t statistics for debugging."""
        if self.count == 0:
            return {'count': 0}
        costs = [exp[5] for exp in self.buffer]
        e_ts  = [exp[6] for exp in self.buffer]
        return {
            'count':    self.count,
            'cost_mean': np.mean(costs),
            'cost_std':  np.std(costs),
            'cost_min':  np.min(costs),
            'cost_max':  np.max(costs),
            'e_t_mean':  np.mean(e_ts),
            'e_t_std':   np.std(e_ts),
            'e_t_min':   np.min(e_ts),
            'e_t_max':   np.max(e_ts),
        }

    def clear(self):
        self.buffer.clear()
        self.count = 0
