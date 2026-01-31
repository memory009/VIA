"""
TD3 with Lagrangian Safety Constraint (TD3-Lagrangian)

严格按照SAC-Lagrangian (Ha et al., 2020)的逻辑实现，唯一区别是backbone从SAC改为TD3。

============================================================================
SAC-Lagrangian原论文关键点 (参考WCSAC论文的描述)
============================================================================

1. 两个分离的Critic:
   - Reward Critic Q^r: 估计累积reward (SAC中还包含entropy)
   - Safety Critic Q^c: 估计累积cost

2. Actor Loss (WCSAC论文公式4):
   J_π(θ) = E[β·log π(a|s) - Q^r(s,a) + κ·Q^c(s,a)]
   
   TD3适配（移除entropy项β·log π）:
   J_π(θ) = E[-Q^r(s,a) + κ·Q^c(s,a)]

3. κ (safety weight)更新 (WCSAC论文公式5):
   J_s(κ) = E[κ(d - Q^c(s,a))]
   - 当 d ≥ Q^c 时，κ减小（约束满足，放松惩罚）
   - 当 d < Q^c 时，κ增大（约束违反，加强惩罚）

4. 原论文实现细节 (Ha et al., 2020 Appendix):
   - 网络：两层隐藏层，256神经元，ReLU
   - Adam优化器，学习率0.0003
   - 网络权重随机初始化

============================================================================
TD3适配说明
============================================================================

SAC → TD3 的改变:
1. 移除熵正则化（SAC的α和log π项）
2. 使用确定性策略（TD3）而非随机策略（SAC）
3. Reward Critic使用Twin Q结构（TD3特性，减少过估计）
4. Safety Critic使用单Q结构（与原SAC-Lagrangian一致）
5. 添加目标策略平滑噪声（TD3特性）
6. 延迟策略更新（TD3特性）

保持不变:
1. 分离的Reward Critic和Safety Critic
2. Lagrangian乘子κ的更新逻辑
3. Actor Loss = -Q^r + κ·Q^c
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# 超参数配置
# 基础参数：与你的baseline保持一致
# ============================================================================

# ----- 基础RL参数（与你的baseline一致）-----
BASELINE_GAMMA = 0.99           # 折扣因子
BASELINE_LR = 1e-4              # Actor/Critic学习率（你的baseline用1e-4，原论文用3e-4）
BASELINE_HIDDEN_DIM = 26        # 隐藏层维度（POLAR验证需求，原论文用256）
BASELINE_BATCH_SIZE = 40        # 批次大小
BASELINE_BUFFER_SIZE = 500000   # Buffer大小
BASELINE_TAU = 0.005            # 软更新系数
BASELINE_POLICY_NOISE = 0.2     # TD3目标策略噪声
BASELINE_NOISE_CLIP = 0.5       # TD3噪声裁剪
BASELINE_POLICY_FREQ = 2        # TD3策略更新频率

# ----- Lagrangian特有参数 -----
LAGRANGIAN_KAPPA_LR = 0.001     # κ学习率
LAGRANGIAN_KAPPA_MAX = 100.0    # κ上限（防止过大）
LAGRANGIAN_COST_THRESHOLD = 10.0  # cost约束阈值d


class Actor(nn.Module):
    """Actor网络（与TD3_lightweight保持一致）"""
    def __init__(self, state_dim, action_dim, hidden_dim=BASELINE_HIDDEN_DIM):
        super(Actor, self).__init__()
        
        self.layer_1 = nn.Linear(state_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()
        
        print(f"🔹 Actor: {state_dim} → {hidden_dim} → {hidden_dim} → {action_dim}")

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    """
    Reward Critic - Twin Q结构（TD3标准，与TD3_lightweight完全一致）
    
    估计累积奖励 Q^r(s,a) = E[Σγ^t * r_t]
    使用Twin结构减少过估计（TD3特性）
    """
    def __init__(self, state_dim, action_dim, hidden_dim=BASELINE_HIDDEN_DIM):
        super(Critic, self).__init__()
        
        # Q1网络
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)
        
        # Q2网络
        self.layer_4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_6 = nn.Linear(hidden_dim, 1)
        
        print(f"🔹 Reward Critic (Twin Q): ({state_dim}+{action_dim}) → {hidden_dim} → 1")

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=1)
        
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)
        
        q2 = F.relu(self.layer_4(sa))
        q2 = F.relu(self.layer_5(q2))
        q2 = self.layer_6(q2)
        
        return q1, q2


class SafetyCritic(nn.Module):
    """
    Safety Critic - 单Q网络（与原SAC-Lagrangian一致）
    
    估计累积cost Q^c(s,a) = E[Σγ^t * c_t]
    
    注意：原SAC-Lagrangian使用单一Safety Critic，不是Twin结构
    Twin结构是TD3用于Reward Critic减少过估计的技巧
    Safety Critic不需要Twin，因为：
    1. 原论文没有使用
    2. 我们不需要对cost进行保守估计
    """
    def __init__(self, state_dim, action_dim, hidden_dim=BASELINE_HIDDEN_DIM):
        super(SafetyCritic, self).__init__()
        
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)
        
        print(f"🔹 Safety Critic (单Q): ({state_dim}+{action_dim}) → {hidden_dim} → 1")

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=1)
        
        q = F.relu(self.layer_1(sa))
        q = F.relu(self.layer_2(q))
        q = self.layer_3(q)
        
        return q


class TD3Lagrangian(object):
    """
    TD3 with Lagrangian Safety Constraint
    
    严格按照SAC-Lagrangian逻辑，backbone替换为TD3
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=BASELINE_LR,
        hidden_dim=BASELINE_HIDDEN_DIM,
        kappa_lr=LAGRANGIAN_KAPPA_LR,
        cost_threshold=LAGRANGIAN_COST_THRESHOLD,
        save_every=0,
        load_model=False,
        save_directory=Path("models/TD3_lagrangian"),
        model_name="TD3_lagrangian",
        load_directory=Path("models/TD3_lagrangian"),
        run_id=None,
    ):
        print("\n" + "=" * 80)
        print("🚀 初始化 TD3-Lagrangian (严格按照SAC-Lagrangian逻辑)")
        print("=" * 80)
        
        self.device = device
        self.max_action = max_action
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # ----- Actor -----
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # ----- Reward Critic (Twin Q，TD3特性) -----
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # ----- Safety Critic (单Q，与原SAC-Lagrangian一致) -----
        self.safety_critic = SafetyCritic(state_dim, action_dim, hidden_dim).to(device)
        self.safety_critic_target = SafetyCritic(state_dim, action_dim, hidden_dim).to(device)
        self.safety_critic_target.load_state_dict(self.safety_critic.state_dict())
        self.safety_critic_optimizer = torch.optim.Adam(self.safety_critic.parameters(), lr=lr)
        
        # ----- Lagrangian Multiplier κ -----
        # 原论文：网络权重随机初始化，没有特别说明κ的初始值
        # 通常初始化为小的正值或0
        self.kappa = torch.tensor([0.0], device=device, requires_grad=False)
        self.kappa_lr = kappa_lr
        self.cost_threshold = cost_threshold
        
        # 计算总参数量
        total_params = (
            sum(p.numel() for p in self.actor.parameters()) +
            sum(p.numel() for p in self.critic.parameters()) +
            sum(p.numel() for p in self.safety_critic.parameters())
        )
        
        print(f"\n📋 Lagrangian参数:")
        print(f"   κ学习率: {kappa_lr}")
        print(f"   Cost阈值 d: {cost_threshold}")
        print(f"   初始κ: {self.kappa.item():.4f}")
        print(f"\n✅ 网络总参数量: {total_params:,}")
        print(f"🔧 设备: {device}")
        print(f"🎯 隐藏层维度: {hidden_dim}")
        print("=" * 80 + "\n")
        
        # TensorBoard
        if run_id:
            tensorboard_log_dir = f"runs/{run_id}"
            self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        else:
            self.writer = SummaryWriter()
        
        self.iter_count = 0
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        
        if load_model:
            self.load(filename=model_name, directory=load_directory)

    def get_action(self, obs, add_noise=True):
        """获取动作（带探索噪声）"""
        if add_noise:
            return (
                self.act(obs) + np.random.normal(0, 0.2, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        else:
            return self.act(obs)

    def act(self, state):
        """获取确定性动作（无噪声）"""
        state = torch.Tensor(state).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=BASELINE_BATCH_SIZE,
        discount=BASELINE_GAMMA,
        tau=BASELINE_TAU,
        policy_noise=BASELINE_POLICY_NOISE,
        noise_clip=BASELINE_NOISE_CLIP,
        policy_freq=BASELINE_POLICY_FREQ,
    ):
        """
        训练函数
        
        流程（TD3 + SAC-Lagrangian）：
        1. 更新Reward Critic（与TD3完全一致）
        2. 更新Safety Critic（学习Q^c）
        3. 延迟更新Actor（使用Lagrangian损失：-Q^r + κ·Q^c）
        4. 软更新所有target网络
        """
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        av_safety_loss = 0
        av_safety_Q = 0
        
        for it in range(iterations):
            # 从buffer采样
            batch = replay_buffer.sample_batch(batch_size)
            
            state = torch.Tensor(batch['states']).to(self.device)
            action = torch.Tensor(batch['actions']).to(self.device)
            reward = torch.Tensor(batch['rewards']).to(self.device)
            cost = torch.Tensor(batch['costs']).to(self.device)
            next_state = torch.Tensor(batch['next_states']).to(self.device)
            done = torch.Tensor(batch['dones']).to(self.device)
            
            with torch.no_grad():
                # ===== TD3: 目标动作（带噪声）=====
                noise = (
                    torch.randn_like(action) * policy_noise
                ).clamp(-noise_clip, noise_clip)
                next_action = (
                    self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)
                
                # ===== Reward Critic Target (TD3标准) =====
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)  # TD3: 取min减少过估计
                target_Q = reward + (1 - done) * discount * target_Q
                
                # ===== Safety Critic Target =====
                target_Qc = self.safety_critic_target(next_state, next_action)
                target_Qc = cost + (1 - done) * discount * target_Qc
            
            # ===== 更新Reward Critic（与TD3完全一致）=====
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            av_loss += critic_loss.item()
            av_Q += torch.mean(target_Q).item()
            max_Q = max(max_Q, torch.max(target_Q).item())
            
            # ===== 更新Safety Critic =====
            current_Qc = self.safety_critic(state, action)
            safety_loss = F.mse_loss(current_Qc, target_Qc)
            
            self.safety_critic_optimizer.zero_grad()
            safety_loss.backward()
            self.safety_critic_optimizer.step()
            
            av_safety_loss += safety_loss.item()
            av_safety_Q += torch.mean(current_Qc).item()
            
            # ===== 延迟更新Actor和Target网络（TD3特性）=====
            if it % policy_freq == 0:
                # Actor输出
                actor_action = self.actor(state)
                
                # Reward value（只用Q1，与TD3一致）
                actor_Q, _ = self.critic(state, actor_action)
                
                # Safety value
                actor_Qc = self.safety_critic(state, actor_action)
                
                # ===== SAC-Lagrangian Actor Loss =====
                # 原公式: J_π = E[β·log π - Q^r + κ·Q^c]
                # TD3适配（移除entropy项）: J_π = E[-Q^r + κ·Q^c]
                actor_loss = -actor_Q.mean() + self.kappa.item() * actor_Qc.mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # ===== Soft Update Target Networks =====
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(
                    self.safety_critic.parameters(), self.safety_critic_target.parameters()
                ):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        self.iter_count += 1
        
        # ----- TensorBoard Logging -----
        self.writer.add_scalar("train/loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)
        self.writer.add_scalar("train/safety_loss", av_safety_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_Qc", av_safety_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/kappa", self.kappa.item(), self.iter_count)
        
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def update_kappa(self, avg_Qc):
        """
        更新Lagrangian乘子κ
        
        按照SAC-Lagrangian论文 (WCSAC公式5):
        J_s(κ) = E[κ(d - Q^c)]
        
        梯度: ∂J_s/∂κ = d - Q^c
        更新: κ ← κ - lr * (d - Q^c) = κ + lr * (Q^c - d)
        
        - 当 Q^c > d 时（约束违反），κ增大，加强安全惩罚
        - 当 Q^c < d 时（约束满足），κ减小，放松安全惩罚
        
        Args:
            avg_Qc: 当前batch或epoch的平均Q^c值
        """
        old_kappa = self.kappa.item()
        
        # κ更新：κ + lr * (Q^c - d)
        # 等价于：κ - lr * (d - Q^c)，即最小化 J_s(κ) = κ(d - Q^c)
        constraint_violation = avg_Qc - self.cost_threshold
        new_kappa = old_kappa + self.kappa_lr * constraint_violation
        
        # 投影到非负区间 [0, κ_max]
        new_kappa = np.clip(new_kappa, 0.0, LAGRANGIAN_KAPPA_MAX)
        
        self.kappa = torch.tensor([new_kappa], device=self.device)
        
        # TensorBoard logging
        self.writer.add_scalar("epoch/kappa", self.kappa.item(), self.iter_count)
        self.writer.add_scalar("epoch/constraint_violation", constraint_violation, self.iter_count)
        self.writer.add_scalar("epoch/avg_Qc", avg_Qc, self.iter_count)
        
        # 打印调试信息
        print(f"   📊 Lagrangian κ更新:")
        print(f"      avg Q^c = {avg_Qc:.2f}, threshold d = {self.cost_threshold}")
        print(f"      约束违反 (Q^c - d) = {constraint_violation:.2f}")
        print(f"      κ: {old_kappa:.4f} → {new_kappa:.4f}")
        
        return constraint_violation

    def update_kappa_from_episode_cost(self, avg_episode_cost):
        """
        基于episode累积cost更新κ（替代方案）
        
        如果不想用Critic的Q^c估计，可以直接用真实的episode cost
        
        Args:
            avg_episode_cost: 当前epoch的平均episode累积cost
        """
        old_kappa = self.kappa.item()
        
        constraint_violation = avg_episode_cost - self.cost_threshold
        new_kappa = old_kappa + self.kappa_lr * constraint_violation
        new_kappa = np.clip(new_kappa, 0.0, LAGRANGIAN_KAPPA_MAX)
        
        self.kappa = torch.tensor([new_kappa], device=self.device)
        
        self.writer.add_scalar("epoch/kappa", self.kappa.item(), self.iter_count)
        self.writer.add_scalar("epoch/constraint_violation", constraint_violation, self.iter_count)
        self.writer.add_scalar("epoch/avg_episode_cost", avg_episode_cost, self.iter_count)
        
        print(f"   📊 Lagrangian κ更新 (基于episode cost):")
        print(f"      avg episode cost = {avg_episode_cost:.2f}, threshold d = {self.cost_threshold}")
        print(f"      约束违反 = {constraint_violation:.2f}")
        print(f"      κ: {old_kappa:.4f} → {new_kappa:.4f}")
        
        return constraint_violation

    def save(self, filename, directory):
        """保存模型"""
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_target.state_dict(), f"{directory}/{filename}_actor_target.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth")
        torch.save(self.safety_critic.state_dict(), f"{directory}/{filename}_safety_critic.pth")
        torch.save(self.safety_critic_target.state_dict(), f"{directory}/{filename}_safety_critic_target.pth")
        
        # 保存Lagrangian参数
        torch.save({
            'kappa': self.kappa,
            'cost_threshold': self.cost_threshold,
        }, f"{directory}/{filename}_lagrangian_params.pth")

    def load(self, filename, directory):
        """加载模型"""
        self.actor.load_state_dict(
            torch.load(f"{directory}/{filename}_actor.pth", map_location=self.device)
        )
        self.actor_target.load_state_dict(
            torch.load(f"{directory}/{filename}_actor_target.pth", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f"{directory}/{filename}_critic.pth", map_location=self.device)
        )
        self.critic_target.load_state_dict(
            torch.load(f"{directory}/{filename}_critic_target.pth", map_location=self.device)
        )
        self.safety_critic.load_state_dict(
            torch.load(f"{directory}/{filename}_safety_critic.pth", map_location=self.device)
        )
        self.safety_critic_target.load_state_dict(
            torch.load(f"{directory}/{filename}_safety_critic_target.pth", map_location=self.device)
        )
        
        # 加载Lagrangian参数
        lagrangian_params = torch.load(
            f"{directory}/{filename}_lagrangian_params.pth", map_location=self.device
        )
        self.kappa = lagrangian_params['kappa'].to(self.device)
        self.cost_threshold = lagrangian_params['cost_threshold']
        
        print(f"✅ Loaded model from: {directory}/{filename}")
        print(f"   κ = {self.kappa.item():.4f}")

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """
        准备状态（与baseline TD3_lightweight完全一致）
        """
        latest_scan = np.array(latest_scan)
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0
        
        max_bins = self.state_dim - 5
        bin_size = int(np.ceil(len(latest_scan) / max_bins))
        
        min_values = []
        for i in range(0, len(latest_scan), bin_size):
            bin_data = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            min_values.append(min(bin_data))
        
        state = min_values + [distance, cos, sin] + [action[0], action[1]]
        
        assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0
        
        return state, terminal