"""
LAP (Prioritized) Replay Buffer for TD7
基于原始TD7的buffer.py实现，仅做接口适配以兼容现有训练代码
"""
import numpy as np
import torch


class LAP(object):
    """
    Prioritized Replay Buffer with LAP (Learned Adaptive Prioritization)
    
    完全保留TD7原始实现，仅适配接口：
    - add() 参数顺序适配为 (state, action, reward, done, next_state)
    - 添加 sample_batch() 作为 sample() 的别名
    - 默认参数适配为导航任务配置
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_size=5e3,           # 适配：默认5000（与TD3配置一致）
        batch_size=40,          # 适配：默认40（与TD3配置一致）
        max_action=1,
        normalize_actions=False, # 适配：默认False（动作已在[-1,1]范围）
        prioritized=True
    ):
        max_size = int(max_size)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.device = device
        self.batch_size = batch_size

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.prioritized = prioritized
        if prioritized:
            self.priority = torch.zeros(max_size, device=device)
            self.max_priority = 1

        self.normalize_actions = max_action if normalize_actions else 1

    def add(self, state, action, reward, done, next_state):
        """
        添加经验到buffer
        
        参数顺序适配为与现有训练代码一致：
        原始TD7: add(state, action, next_state, reward, done)
        适配后:  add(state, action, reward, done, next_state)
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action / self.normalize_actions
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        
        if self.prioritized:
            self.priority[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        """
        采样一个batch
        
        返回: (state, action, next_state, reward, not_done)
        注意：返回的是 not_done 而不是 done
        """
        if self.prioritized:
            csum = torch.cumsum(self.priority[:self.size], 0)
            val = torch.rand(size=(self.batch_size,), device=self.device) * csum[-1]
            self.ind = torch.searchsorted(csum, val).cpu().data.numpy()
        else:
            self.ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.not_done[self.ind], dtype=torch.float, device=self.device)
        )

    def sample_batch(self, batch_size=None):
        """
        sample() 的别名，兼容现有训练代码接口
        
        Args:
            batch_size: 如果提供，临时覆盖默认batch_size
            
        Returns:
            (states, actions, next_states, rewards, not_dones) - 所有都是torch.Tensor
        """
        if batch_size is not None:
            old_batch_size = self.batch_size
            self.batch_size = batch_size
            result = self.sample()
            self.batch_size = old_batch_size
            return result
        return self.sample()

    def update_priority(self, priority):
        """更新LAP优先级"""
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        """重置最大优先级"""
        self.max_priority = float(self.priority[:self.size].max())

    def load_D4RL(self, dataset):
        """加载D4RL数据集（保留原始实现，虽然导航任务不使用）"""
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]
        
        if self.prioritized:
            self.priority = torch.ones(self.size).to(self.device)