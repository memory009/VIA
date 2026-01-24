"""
WCSAC Replay Buffer

与官方WCSAC实现保持一致的简单FIFO经验回放缓冲区。
存储：(obs, act, rew, next_obs, done, cost)

注意：WCSAC不需要像CVaR-CPO那样的状态扩展(e_t)，
因为WCSAC使用分布式Safety Critic直接从batch数据中学习cost分布。
"""

import numpy as np


class WCSACReplayBuffer:
    """
    与官方WCSAC一致的简单FIFO经验回放缓冲区
    
    官方实现参考：wcsac.py第184-215行
    """

    def __init__(self, state_dim, action_dim, buffer_size=500000, random_seed=None):
        """
        初始化缓冲区
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            buffer_size: 缓冲区大小
            random_seed: 随机种子
        """
        self.buffer_size = int(buffer_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 预分配内存（与官方代码一致）
        self.states = np.zeros([self.buffer_size, state_dim], dtype=np.float32)
        self.next_states = np.zeros([self.buffer_size, state_dim], dtype=np.float32)
        self.actions = np.zeros([self.buffer_size, action_dim], dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.costs = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self._size = 0
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        print(f"🔹 WCSAC Replay Buffer: size={buffer_size}, state_dim={state_dim}, action_dim={action_dim}")

    def add(self, state, action, reward, done, next_state, cost):
        """
        添加一条经验（与官方代码store方法一致）
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            done: 是否终止（0或1）
            next_state: 下一状态
            cost: 安全成本
        """
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.costs[self.ptr] = cost
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def sample_batch(self, batch_size=32):
        """
        随机采样一个batch（与官方代码一致）
        
        Args:
            batch_size: 批次大小
            
        Returns:
            dict: 包含states, next_states, actions, rewards, costs, dones的字典
        """
        idxs = np.random.randint(0, self._size, size=batch_size)
        
        return dict(
            states=self.states[idxs],
            next_states=self.next_states[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            costs=self.costs[idxs],
            dones=self.dones[idxs]
        )

    def size(self):
        """返回当前缓冲区中的经验数量"""
        return self._size

    def is_full(self):
        """检查缓冲区是否已满"""
        return self._size >= self.buffer_size


class WCSACReplayBufferSimple:
    """
    更简洁的版本，接口与baseline ReplayBuffer保持一致
    
    用于快速集成到现有训练脚本
    """
    
    def __init__(self, buffer_size=500000, random_seed=None):
        """
        初始化（延迟分配内存，在第一次add时确定维度）
        """
        self.buffer_size = int(buffer_size)
        self.initialized = False
        
        self.states = None
        self.next_states = None
        self.actions = None
        self.rewards = None
        self.costs = None
        self.dones = None
        
        self.ptr = 0
        self._size = 0
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        print(f"🔹 WCSAC Replay Buffer (Simple): size={buffer_size}")

    def _initialize(self, state_dim, action_dim):
        """首次添加时初始化数组"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.states = np.zeros([self.buffer_size, state_dim], dtype=np.float32)
        self.next_states = np.zeros([self.buffer_size, state_dim], dtype=np.float32)
        self.actions = np.zeros([self.buffer_size, action_dim], dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.costs = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        
        self.initialized = True
        print(f"   Buffer initialized: state_dim={state_dim}, action_dim={action_dim}")

    def add(self, s, a, r, t, s2, c):
        """
        添加经验（接口与baseline兼容）
        
        Args:
            s: state
            a: action
            r: reward
            t: terminal (done)
            s2: next_state
            c: cost
        """
        s = np.array(s, dtype=np.float32)
        a = np.array(a, dtype=np.float32)
        s2 = np.array(s2, dtype=np.float32)
        
        if not self.initialized:
            self._initialize(len(s), len(a))
        
        self.states[self.ptr] = s
        self.next_states[self.ptr] = s2
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.costs[self.ptr] = c
        self.dones[self.ptr] = t
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def sample_batch(self, batch_size=32):
        """随机采样"""
        idxs = np.random.randint(0, self._size, size=batch_size)
        
        return dict(
            states=self.states[idxs],
            next_states=self.next_states[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            costs=self.costs[idxs],
            dones=self.dones[idxs]
        )

    def size(self):
        """返回当前大小"""
        return self._size