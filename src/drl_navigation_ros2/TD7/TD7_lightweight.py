"""
TD7 Lightweight - 轻量级TD7实现
基于原始TD7论文实现，适配导航任务

完整保留TD7的所有机制：
- Encoder (状态编码器 + 状态-动作编码器)
- LAP (Learned Adaptive Prioritization)
- Checkpoint机制
- 值裁剪 (Value Clipping)
- Hard target update
"""
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from TD7.buffer_lap import LAP


@dataclass
class Hyperparameters:
    """TD7超参数配置 - 使用TD7论文默认值"""
    # Generic
    batch_size: int = 40            # 适配：与TD3配置一致
    buffer_size: int = 5e3          # 适配：与TD3配置一致
    discount: float = 0.99          # TD7默认
    target_update_rate: int = 250   # TD7默认
    exploration_noise: float = 0.1  # TD7默认
    
    # TD3
    target_policy_noise: float = 0.2  # TD7默认
    noise_clip: float = 0.5           # TD7默认
    policy_freq: int = 2              # TD7默认
    
    # LAP
    alpha: float = 0.4        # TD7默认
    min_priority: float = 1   # TD7默认
    
    # TD3+BC (offline RL, 不使用)
    lmbda: float = 0.1
    
    # Checkpointing - TD7默认值
    max_eps_when_checkpointing: int = 20
    steps_before_checkpointing: int = 75e4
    reset_weight: float = 0.9
    
    # Encoder Model - 适配轻量级
    zs_dim: int = 26              # 与hidden_dim一致
    enc_hdim: int = 26            # 轻量级
    enc_activ: Callable = F.elu   # TD7默认
    encoder_lr: float = 3e-4      # TD7默认
    
    # Critic Model - 适配轻量级
    critic_hdim: int = 26         # 轻量级
    critic_activ: Callable = F.elu  # TD7默认
    critic_lr: float = 3e-4       # TD7默认
    
    # Actor Model - 适配轻量级
    actor_hdim: int = 26          # 轻量级
    actor_activ: Callable = F.relu  # TD7默认
    actor_lr: float = 3e-4        # TD7默认


def AvgL1Norm(x, eps=1e-8):
    """TD7的归一化函数 - 完全保留原实现"""
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def LAP_huber(x, min_priority=1):
    """TD7的LAP Huber损失 - 完全保留原实现"""
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


class Actor(nn.Module):
    """TD7 Actor网络 - 轻量级版本"""
    def __init__(self, state_dim, action_dim, zs_dim=26, hdim=26, activ=F.relu):
        super(Actor, self).__init__()

        self.activ = activ

        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)
        
        # 打印网络结构
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🔹 TD7 Actor: {state_dim} → {hdim} → concat(zs:{zs_dim}) → {hdim} → {hdim} → {action_dim}")
        print(f"   参数量: {total_params:,}")

    def forward(self, state, zs):
        a = AvgL1Norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        return torch.tanh(self.l3(a))


class Encoder(nn.Module):
    """TD7 Encoder网络 - 轻量级版本"""
    def __init__(self, state_dim, action_dim, zs_dim=26, hdim=26, activ=F.elu):
        super(Encoder, self).__init__()

        self.activ = activ

        # state encoder
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)
        
        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)
        
        # 打印网络结构
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🔹 TD7 Encoder:")
        print(f"   State encoder: {state_dim} → {hdim} → {hdim} → {zs_dim}")
        print(f"   State-action encoder: {zs_dim + action_dim} → {hdim} → {hdim} → {zs_dim}")
        print(f"   参数量: {total_params:,}")

    def zs(self, state):
        """状态编码"""
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = AvgL1Norm(self.zs3(zs))
        return zs

    def zsa(self, zs, action):
        """状态-动作编码"""
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa


class Critic(nn.Module):
    """TD7 Critic网络 - 轻量级双Q版本"""
    def __init__(self, state_dim, action_dim, zs_dim=26, hdim=26, activ=F.elu):
        super(Critic, self).__init__()

        self.activ = activ
        
        # Q1网络
        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        # Q2网络
        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)
        
        # 打印网络结构
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🔹 TD7 Critic (双Q):")
        print(f"   Input: (state:{state_dim} + action:{action_dim}) → {hdim} → concat(zsa:{zs_dim}, zs:{zs_dim})")
        print(f"   → {hdim} → {hdim} → 1")
        print(f"   参数量: {total_params:,}")

    def forward(self, state, action, zsa, zs):
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1 = AvgL1Norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], 1)
        q1 = self.activ(self.q1(q1))
        q1 = self.activ(self.q2(q1))
        q1 = self.q3(q1)

        q2 = AvgL1Norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], 1)
        q2 = self.activ(self.q4(q2))
        q2 = self.activ(self.q5(q2))
        q2 = self.q6(q2)
        
        return torch.cat([q1, q2], 1)


class TD7(object):
    """
    TD7 Agent - 轻量级版本
    
    完整保留TD7的所有机制：
    - Encoder (状态编码器 + 状态-动作编码器)
    - LAP (Learned Adaptive Prioritization)  
    - Checkpoint机制 (用于探索时选择稳定策略)
    - 值裁剪 (Value Clipping)
    - Hard target update
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        hidden_dim=26,
        save_every=0,
        load_model=False,
        save_directory=Path("src/drl_navigation_ros2/models/TD7_lightweight"),
        model_name="TD7_lightweight",
        load_directory=Path("src/drl_navigation_ros2/models/TD7_lightweight"),
        run_id=None,
        hp=None,  # 可选：自定义超参数
    ):
        print("\n" + "=" * 80)
        print("🚀 初始化轻量级TD7网络")
        print("=" * 80)
        
        self.device = device
        
        # 使用自定义超参数或默认超参数
        if hp is None:
            hp = Hyperparameters()
            # 确保隐藏层维度一致
            hp.zs_dim = hidden_dim
            hp.enc_hdim = hidden_dim
            hp.critic_hdim = hidden_dim
            hp.actor_hdim = hidden_dim
        self.hp = hp
        
        # 保存维度信息
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # ==================== 初始化网络 ====================
        # Actor
        self.actor = Actor(
            state_dim, action_dim, hp.zs_dim, hp.actor_hdim, hp.actor_activ
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        # Critic
        self.critic = Critic(
            state_dim, action_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ
        ).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        # Encoder
        self.encoder = Encoder(
            state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ
        ).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        # Checkpoint副本 (TD7核心机制)
        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        # ==================== Replay Buffer ====================
        self.replay_buffer = LAP(
            state_dim, action_dim, self.device, 
            hp.buffer_size, hp.batch_size,
            max_action, normalize_actions=False, prioritized=True
        )

        # ==================== 训练状态跟踪 ====================
        self.training_steps = 0
        self.offline = False  # 在线学习模式

        # Checkpointing tracked values (TD7核心机制)
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # Value clipping tracked values (TD7核心机制)
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0
        self.min_target = 0
        
        # ==================== TensorBoard ====================
        if run_id:
            tensorboard_log_dir = f"runs/{run_id}"
            self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        else:
            self.writer = SummaryWriter()
        
        # ==================== 保存/加载配置 ====================
        self.iter_count = 0
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        
        if load_model:
            self.load(filename=model_name, directory=load_directory)
        
        # ==================== 打印统计信息 ====================
        total_params = (
            sum(p.numel() for p in self.actor.parameters()) +
            sum(p.numel() for p in self.critic.parameters()) +
            sum(p.numel() for p in self.encoder.parameters())
        )
        print(f"\n✅ TD7网络总参数量: {total_params:,}")
        print(f"📍 设备: {device}")
        print(f"🎯 隐藏层维度: {hidden_dim}")
        print(f"🔧 zs_dim: {hp.zs_dim}")
        print(f"⚙️ TD7超参数:")
        print(f"   - target_update_rate: {hp.target_update_rate}")
        print(f"   - steps_before_checkpointing: {int(hp.steps_before_checkpointing)}")
        print(f"   - max_eps_when_checkpointing: {hp.max_eps_when_checkpointing}")
        print("=" * 80 + "\n")

    def get_action(self, obs, add_noise):
        """
        获取动作 - 适配现有训练代码接口
        
        Args:
            obs: 观测状态 (numpy array)
            add_noise: 是否添加探索噪声
            
        Returns:
            action: 动作 (numpy array)
        """
        # 使用TD7的select_action逻辑
        # 注意：TD7使用checkpoint来决定使用哪个策略
        return self.select_action(
            np.array(obs), 
            use_checkpoint=self.hp.steps_before_checkpointing <= self.training_steps,
            use_exploration=add_noise
        )

    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        """
        TD7原始的动作选择方法 - 完全保留原实现
        """
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                action = self.checkpoint_actor(state, zs)
            else:
                zs = self.fixed_encoder.zs(state)
                action = self.actor(state, zs)
            
            if use_exploration:
                action = action + torch.randn_like(action) * self.hp.exploration_noise

            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    def train_step(self):
        """
        TD7单步训练 - 完全保留原实现
        
        包含:
        1. Update Encoder
        2. Update Critic
        3. Update LAP priority
        4. Update Actor (delayed)
        5. Update targets (periodic hard update)
        """
        self.training_steps += 1

        state, action, next_state, reward, not_done = self.replay_buffer.sample()

        # ==================== Update Encoder ====================
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)

        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        # ==================== Update Critic ====================
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)

            noise = (torch.randn_like(action) * self.hp.target_policy_noise).clamp(
                -self.hp.noise_clip, self.hp.noise_clip
            )
            next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(-1, 1)
            
            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)

            Q_target = self.critic_target(
                next_state, next_action, fixed_target_zsa, fixed_target_zs
            ).min(1, keepdim=True)[0]
            Q_target = reward + not_done * self.hp.discount * Q_target.clamp(
                self.min_target, self.max_target
            )

            self.max = max(self.max, float(Q_target.max()))
            self.min = min(self.min, float(Q_target.min()))

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

        Q = self.critic(state, action, fixed_zsa, fixed_zs)
        td_loss = (Q - Q_target).abs()
        critic_loss = LAP_huber(td_loss, self.hp.min_priority)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ==================== Update LAP ====================
        priority = td_loss.max(1)[0].clamp(min=self.hp.min_priority).pow(self.hp.alpha)
        self.replay_buffer.update_priority(priority)

        # ==================== Update Actor ====================
        actor_loss = None
        if self.training_steps % self.hp.policy_freq == 0:
            actor = self.actor(state, fixed_zs)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor)
            Q = self.critic(state, actor, fixed_zsa, fixed_zs)

            actor_loss = -Q.mean()
            if self.offline:
                actor_loss = actor_loss + self.hp.lmbda * Q.abs().mean().detach() * F.mse_loss(actor, action)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # ==================== Update Targets (Hard Update) ====================
        if self.training_steps % self.hp.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())
            
            self.replay_buffer.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min
        
        return encoder_loss.item(), critic_loss.item(), actor_loss.item() if actor_loss else None

    def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
        """
        TD7的checkpoint训练机制 - 完全保留原实现
        
        在每个episode结束时调用，用于：
        1. 跟踪episode统计
        2. 决定是否更新checkpoint
        3. 执行批量训练
        """
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        self.min_return = min(self.min_return, ep_return)

        # End evaluation of current policy early
        if self.min_return < self.best_min_return:
            self.train_and_reset()

        # Update checkpoint
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())
            
            self.train_and_reset()

    def train_and_reset(self):
        """
        TD7的批量训练 - 完全保留原实现
        """
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.hp.steps_before_checkpointing:
                self.best_min_return *= self.hp.reset_weight
                self.max_eps_before_update = self.hp.max_eps_when_checkpointing
            
            self.train_step()

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8

    def save(self, filename, directory):
        """保存所有网络权重"""
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Actor
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_target.state_dict(), f"{directory}/{filename}_actor_target.pth")
        
        # Critic
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth")
        
        # Encoder
        torch.save(self.encoder.state_dict(), f"{directory}/{filename}_encoder.pth")
        torch.save(self.fixed_encoder.state_dict(), f"{directory}/{filename}_fixed_encoder.pth")
        torch.save(self.fixed_encoder_target.state_dict(), f"{directory}/{filename}_fixed_encoder_target.pth")
        
        # Checkpoint
        torch.save(self.checkpoint_actor.state_dict(), f"{directory}/{filename}_checkpoint_actor.pth")
        torch.save(self.checkpoint_encoder.state_dict(), f"{directory}/{filename}_checkpoint_encoder.pth")
        
        # 保存训练状态
        torch.save({
            'training_steps': self.training_steps,
            'eps_since_update': self.eps_since_update,
            'timesteps_since_update': self.timesteps_since_update,
            'max_eps_before_update': self.max_eps_before_update,
            'min_return': self.min_return,
            'best_min_return': self.best_min_return,
            'max': self.max,
            'min': self.min,
            'max_target': self.max_target,
            'min_target': self.min_target,
        }, f"{directory}/{filename}_training_state.pth")

    def load(self, filename, directory):
        """加载所有网络权重"""
        # Actor
        self.actor.load_state_dict(
            torch.load(f"{directory}/{filename}_actor.pth", map_location=self.device)
        )
        self.actor_target.load_state_dict(
            torch.load(f"{directory}/{filename}_actor_target.pth", map_location=self.device)
        )
        
        # Critic
        self.critic.load_state_dict(
            torch.load(f"{directory}/{filename}_critic.pth", map_location=self.device)
        )
        self.critic_target.load_state_dict(
            torch.load(f"{directory}/{filename}_critic_target.pth", map_location=self.device)
        )
        
        # Encoder
        self.encoder.load_state_dict(
            torch.load(f"{directory}/{filename}_encoder.pth", map_location=self.device)
        )
        self.fixed_encoder.load_state_dict(
            torch.load(f"{directory}/{filename}_fixed_encoder.pth", map_location=self.device)
        )
        self.fixed_encoder_target.load_state_dict(
            torch.load(f"{directory}/{filename}_fixed_encoder_target.pth", map_location=self.device)
        )
        
        # Checkpoint
        self.checkpoint_actor.load_state_dict(
            torch.load(f"{directory}/{filename}_checkpoint_actor.pth", map_location=self.device)
        )
        self.checkpoint_encoder.load_state_dict(
            torch.load(f"{directory}/{filename}_checkpoint_encoder.pth", map_location=self.device)
        )
        
        # 加载训练状态（如果存在）
        state_path = f"{directory}/{filename}_training_state.pth"
        if Path(state_path).exists():
            state = torch.load(state_path, map_location=self.device)
            self.training_steps = state['training_steps']
            self.eps_since_update = state['eps_since_update']
            self.timesteps_since_update = state['timesteps_since_update']
            self.max_eps_before_update = state['max_eps_before_update']
            self.min_return = state['min_return']
            self.best_min_return = state['best_min_return']
            self.max = state['max']
            self.min = state['min']
            self.max_target = state['max_target']
            self.min_target = state['min_target']
        
        print(f"✅ Loaded TD7 weights from: {directory} to device: {self.device}")

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """
        准备状态 - 从TD3_lightweight完全移植
        
        将ROS返回的数据转换为神经网络输入格式
        """
        latest_scan = np.array(latest_scan)

        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0

        max_bins = self.state_dim - 5
        bin_size = int(np.ceil(len(latest_scan) / max_bins))

        # Initialize the list to store the minimum values of each bin
        min_values = []

        # Loop through the data and create bins
        for i in range(0, len(latest_scan), bin_size):
            # Get the current bin
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            # Find the minimum value in the current bin and append it to the min_values list
            min_values.append(min(bin))
        state = min_values + [distance, cos, sin] + [action[0], action[1]]

        assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0

        return state, terminal