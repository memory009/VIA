"""
TD3-WCSAC: TD3 with Worst-Case Safety Critic

基于官方WCSAC实现修正版：
"WCSAC: Worst-Case Soft Actor Critic for Safety-Constrained Reinforcement Learning"
Yang et al., AAAI 2021

GitHub: https://github.com/AlgTUDelft/WCSAC

核心思想：
- 使用分布式Safety Critic估计累积cost的分布（高斯近似）
- 使用CVaR（Conditional Value-at-Risk）作为安全度量
- 通过Lagrangian方法自适应调整安全权重β(论文中用κ表示)

与原始WCSAC(SAC backbone)的区别：
- 移除SAC的熵正则化项α*logπ（TD3使用确定性策略）
- 使用TD3的延迟策略更新和目标平滑
- 保持与baseline一致的基础RL超参数

关键公式（与官方代码完全对齐）：
- CVaR计算: CVaR_α = Q^c + α^{-1} * φ(Φ^{-1}(α)) * sqrt(V^c)  (论文公式9)
- 方差TD目标: V_target = c^2 + 2γcQ^c_target + γ^2V^c_target + γ^2(Q^c_target)^2 - (Q^c)^2  (论文公式8)
- 方差Loss: 2-Wasserstein距离 = V + V_target - 2*sqrt(V*V_target)
- Actor Loss: -Q^r + β * (Q^c + pdf_cdf * sqrt(V^c))
- β Loss: β * (d - Q^c - pdf_cdf * sqrt(V^c))
"""

import math
from pathlib import Path
from scipy.stats import norm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# 超参数配置
# 基础参数：与baseline保持一致
# WCSAC参数：按照论文设置
# ============================================================================

# ----- 基础RL参数（与baseline一致）-----
BASELINE_GAMMA = 0.99           # 折扣因子
BASELINE_LR = 1e-4              # 网络学习率
BASELINE_HIDDEN_DIM = 26        # 隐藏层维度（POLAR验证需求）
BASELINE_BATCH_SIZE = 40        # 批次大小
BASELINE_BUFFER_SIZE = 500000   # Buffer大小
BASELINE_TAU = 0.005            # 软更新系数
BASELINE_POLICY_NOISE = 0.2     # TD3目标策略噪声
BASELINE_NOISE_CLIP = 0.5       # TD3噪声裁剪
BASELINE_POLICY_FREQ = 2        # TD3策略更新频率

# ----- WCSAC特有参数（按照论文设置）-----
WCSAC_CL = 0.1                  # α(论文中用cl表示)，风险水平（关注worst 10%）
WCSAC_COST_LIM = 10.0           # d，cost limit（论文中cost_lim=15），但是ours方法用的是10.如果和cost_threshold是一个意思的话
WCSAC_BETA_LR_SCALE = 50        # β学习率倍数（论文默认lr_scale=50）
WCSAC_DAMP_SCALE = 10           # damp scale（论文默认damp_scale=10）


class Actor(nn.Module):
    """Actor网络（确定性策略，与TD3 baseline一致）"""
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


class RewardCritic(nn.Module):
    """奖励Critic - 双Q网络（TD3标准结构）"""
    def __init__(self, state_dim, action_dim, hidden_dim=BASELINE_HIDDEN_DIM):
        super(RewardCritic, self).__init__()
        
        # Q1
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.layer_4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_6 = nn.Linear(hidden_dim, 1)
        
        print(f"🔹 Reward Critic(双Q): ({state_dim}+{action_dim}) → {hidden_dim} → 1")

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)
        
        q2 = F.relu(self.layer_4(sa))
        q2 = F.relu(self.layer_5(q2))
        q2 = self.layer_6(q2)
        
        return q1, q2


class SafetyCriticMean(nn.Module):
    """
    Safety Critic - 均值网络 Q^c
    
    与官方WCSAC一致：使用单独的网络估计Q^c
    """
    def __init__(self, state_dim, action_dim, hidden_dim=BASELINE_HIDDEN_DIM):
        super(SafetyCriticMean, self).__init__()
        
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)
        
        print(f"🔹 Safety Critic Mean (Q^c): ({state_dim}+{action_dim}) → {hidden_dim} → 1")

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        
        qc = F.relu(self.layer_1(sa))
        qc = F.relu(self.layer_2(qc))
        qc = self.layer_3(qc)
        
        return qc


class SafetyCriticVar(nn.Module):
    """
    Safety Critic - 方差网络 V^c
    
    与官方WCSAC一致：
    - 单独的网络估计V^c
    - 使用softplus确保非负
    """
    def __init__(self, state_dim, action_dim, hidden_dim=BASELINE_HIDDEN_DIM):
        super(SafetyCriticVar, self).__init__()
        
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)
        
        print(f"🔹 Safety Critic Var (V^c): ({state_dim}+{action_dim}) → {hidden_dim} → 1")

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        
        vc = F.relu(self.layer_1(sa))
        vc = F.relu(self.layer_2(vc))
        vc = self.layer_3(vc)
        
        # Softplus确保非负（与官方代码一致）
        vc = F.softplus(vc)
        
        return vc


class TD3_WCSAC(object):
    """
    TD3 with Worst-Case Safety Critic
    
    基于官方WCSAC实现，适配TD3 backbone
    
    核心改动（相比标准TD3）：
    1. 添加分布式Safety Critic（Q^c均值 + V^c方差）
    2. 使用CVaR作为安全度量
    3. Actor loss添加β*(Q^c + pdf_cdf*sqrt(V^c))惩罚项
    4. 自适应更新Lagrangian乘子β
    
    参数设计原则：
    - 基础RL参数: 与baseline一致，确保公平对比
    - WCSAC参数: 按官方代码设置
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        # 基础参数（与baseline一致）
        lr=BASELINE_LR,
        hidden_dim=BASELINE_HIDDEN_DIM,
        gamma=BASELINE_GAMMA,
        # WCSAC参数（按官方代码设置）
        cl=WCSAC_CL,                    # α，risk level
        cost_lim=WCSAC_COST_LIM,        # d，cost limit
        beta_lr_scale=WCSAC_BETA_LR_SCALE,  # β学习率倍数
        damp_scale=WCSAC_DAMP_SCALE,    # damp scale
        max_ep_len=300,                 # 用于计算cost_constraint
        # 其他
        save_every=0,
        load_model=False,
        save_directory=Path("models/TD3_wcsac"),
        model_name="TD3_wcsac",
        load_directory=Path("models/TD3_wcsac"),
        run_id=None,
    ):
        print("\n" + "="*80)
        print("🚀 TD3-WCSAC (Based on Official WCSAC Implementation)")
        print("="*80)
        
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # 保存参数
        self.gamma = gamma
        self.cl = cl  # α，risk level
        self.lr = lr
        self.beta_lr_scale = beta_lr_scale
        self.damp_scale = damp_scale
        
        # 计算cost_constraint（与官方代码一致）
        # cost_constraint = cost_lim * (1 - gamma^T) / (1 - gamma) / T
        self.cost_lim = cost_lim
        self.max_ep_len = max_ep_len
        self.cost_constraint = cost_lim * (1 - gamma ** max_ep_len) / (1 - gamma) / max_ep_len
        
        # 计算CVaR系数 pdf_cdf（与官方代码一致）
        # pdf_cdf = cl^{-1} * φ(Φ^{-1}(cl))
        self.pdf_cdf = cl ** (-1) * norm.pdf(norm.ppf(cl))
        
        # Lagrangian乘子β（使用softplus参数化）
        # 初始化为0，经过softplus后接近0
        self.soft_beta = torch.tensor([0.0], device=device, requires_grad=True)
        self.beta_optimizer = torch.optim.Adam([self.soft_beta], lr=lr * beta_lr_scale)
        
        print(f"\n📊 Ablation Study 参数配置:")
        print(f"   ─── 基础参数（与baseline一致）───")
        print(f"   γ (discount): {gamma}")
        print(f"   lr (network): {lr}")
        print(f"   hidden_dim: {hidden_dim}")
        print(f"   ─── WCSAC参数（按官方代码设置）───")
        print(f"   cl/α (risk level): {cl} (关注worst {cl*100:.0f}%)")
        print(f"   cost_lim (d): {cost_lim}")
        print(f"   cost_constraint: {self.cost_constraint:.6f}")
        print(f"   pdf_cdf (CVaR系数): {self.pdf_cdf:.4f}")
        print(f"   β_lr_scale: {beta_lr_scale}")
        print(f"   damp_scale: {damp_scale}")
        
        # ===== 网络初始化 =====
        
        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # Reward Critic (双Q)
        self.reward_critic = RewardCritic(state_dim, action_dim, hidden_dim).to(device)
        self.reward_critic_target = RewardCritic(state_dim, action_dim, hidden_dim).to(device)
        self.reward_critic_target.load_state_dict(self.reward_critic.state_dict())
        self.reward_critic_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=lr)
        
        # Safety Critic Mean (Q^c)
        self.qc = SafetyCriticMean(state_dim, action_dim, hidden_dim).to(device)
        self.qc_target = SafetyCriticMean(state_dim, action_dim, hidden_dim).to(device)
        self.qc_target.load_state_dict(self.qc.state_dict())
        self.qc_optimizer = torch.optim.Adam(self.qc.parameters(), lr=lr)
        
        # Safety Critic Var (V^c)
        self.qc_var = SafetyCriticVar(state_dim, action_dim, hidden_dim).to(device)
        self.qc_var_target = SafetyCriticVar(state_dim, action_dim, hidden_dim).to(device)
        self.qc_var_target.load_state_dict(self.qc_var.state_dict())
        self.qc_var_optimizer = torch.optim.Adam(self.qc_var.parameters(), lr=lr)
        
        # TensorBoard
        if run_id:
            self.writer = SummaryWriter(log_dir=f"runs/{run_id}")
        else:
            self.writer = SummaryWriter()
        
        self.iter_count = 0
        
        if load_model:
            self.load(filename=model_name, directory=load_directory)
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        
        print("="*80 + "\n")

    def get_beta(self):
        """获取当前β值（经过softplus）"""
        return F.softplus(self.soft_beta).item()

    def get_action(self, state, add_noise):
        """获取动作"""
        state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, BASELINE_POLICY_NOISE, size=self.action_dim)
            action = (action + noise).clip(-self.max_action, self.max_action)
        
        return action

    def act(self, state):
        """确定性动作（评估用）"""
        return self.get_action(state, add_noise=False)

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=BASELINE_BATCH_SIZE,
        discount=None,
        tau=BASELINE_TAU,
        policy_noise=BASELINE_POLICY_NOISE,
        noise_clip=BASELINE_NOISE_CLIP,
        policy_freq=BASELINE_POLICY_FREQ,
    ):
        """
        训练循环（与官方WCSAC逻辑对齐）
        
        更新顺序：
        1. Reward Critics (Q^r1, Q^r2)
        2. Safety Critic Mean (Q^c)
        3. Safety Critic Var (V^c)
        4. Actor (delayed)
        5. β (Lagrangian multiplier)
        6. Target networks (soft update)
        """
        if discount is None:
            discount = self.gamma
        
        av_reward_Q = 0.0
        max_reward_Q = -inf
        av_qr_loss = 0.0
        av_qc_loss = 0.0
        av_qc_var_loss = 0.0
        av_actor_loss = 0.0
        av_beta_loss = 0.0
        av_qc_val = 0.0
        av_qc_var_val = 0.0
        av_cvar = 0.0
        
        for it in range(iterations):
            # 采样
            batch = replay_buffer.sample_batch(batch_size)
            
            # state = torch.FloatTensor(batch['states']).to(self.device)
            # next_state = torch.FloatTensor(batch['next_states']).to(self.device)
            # action = torch.FloatTensor(batch['actions']).to(self.device)
            # reward = torch.FloatTensor(batch['rewards']).to(self.device)
            # cost = torch.FloatTensor(batch['costs']).to(self.device)
            # done = torch.FloatTensor(batch['dones']).to(self.device)

            state = torch.FloatTensor(batch['states']).to(self.device)
            next_state = torch.FloatTensor(batch['next_states']).to(self.device)
            action = torch.FloatTensor(batch['actions']).to(self.device)
            reward = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)  # [batch] -> [batch, 1]
            cost = torch.FloatTensor(batch['costs']).unsqueeze(1).to(self.device)      # [batch] -> [batch, 1]
            done = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)      # [batch] -> [batch, 1]
            
            with torch.no_grad():
                # Target action (TD3风格)
                next_action = self.actor_target(next_state)
                noise = torch.randn_like(next_action) * policy_noise
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                
                # Reward Q target (TD3: min of two Q)
                target_Q1, target_Q2 = self.reward_critic_target(next_state, next_action)
                min_q_pi_targ = torch.min(target_Q1, target_Q2)
                
                # 注意：TD3没有熵项，所以直接用reward + γ*(1-d)*Q_target
                q_backup = reward + (1 - done) * discount * min_q_pi_targ
                
                av_reward_Q += min_q_pi_targ.mean().item()
                max_reward_Q = max(max_reward_Q, min_q_pi_targ.max().item())
                
                # Safety Critic targets
                qc_pi_targ = self.qc_target(next_state, next_action)
                qc_pi_var_targ = self.qc_var_target(next_state, next_action)
                
                # Q^c backup（与官方代码一致）
                qc_backup = cost + (1 - done) * discount * qc_pi_targ
                
            # ===== 更新Reward Critic =====
            current_Q1, current_Q2 = self.reward_critic(state, action)
            qr1_loss = 0.5 * F.mse_loss(current_Q1, q_backup)
            qr2_loss = 0.5 * F.mse_loss(current_Q2, q_backup)
            qr_loss = qr1_loss + qr2_loss
            
            self.reward_critic_optimizer.zero_grad()
            qr_loss.backward()
            self.reward_critic_optimizer.step()
            av_qr_loss += qr_loss.item()
            
            # ===== 更新Safety Critic Mean (Q^c) =====
            current_qc = self.qc(state, action)
            qc_loss = 0.5 * F.mse_loss(current_qc, qc_backup)
            
            self.qc_optimizer.zero_grad()
            qc_loss.backward()
            self.qc_optimizer.step()
            av_qc_loss += qc_loss.item()
            
            # ===== 更新Safety Critic Var (V^c) =====
            # 官方公式（第433行）：
            # qc_var_backup = c^2 + 2γc*qc_pi_targ + γ^2*qc_pi_var_targ + γ^2*(qc_pi_targ)^2 - qc^2
            current_qc_var = self.qc_var(state, action)
            
            # 需要detach current_qc因为它已经更新过了
            with torch.no_grad():
                qc_for_var = self.qc(state, action)
                qc_var_backup = (cost ** 2 
                                + 2 * discount * cost * qc_pi_targ 
                                + (discount ** 2) * qc_pi_var_targ 
                                + (discount ** 2) * (qc_pi_targ ** 2) 
                                - qc_for_var ** 2)
                # Clip确保非负（与官方代码一致）
                qc_var_backup = torch.clamp(qc_var_backup, min=1e-8)
            
            # Clamp current_qc_var（与官方代码一致）
            current_qc_var_clamped = torch.clamp(current_qc_var, min=1e-8)
            
            # 2-Wasserstein Loss（官方公式第447行）
            # qc_var_loss = 0.5 * (V + V_target - 2*sqrt(V*V_target))
            qc_var_loss = 0.5 * torch.mean(
                current_qc_var_clamped + qc_var_backup 
                - 2 * torch.sqrt(current_qc_var_clamped * qc_var_backup)
            )
            
            self.qc_var_optimizer.zero_grad()
            qc_var_loss.backward()
            self.qc_var_optimizer.step()
            av_qc_var_loss += qc_var_loss.item()
            
            # 监控
            with torch.no_grad():
                av_qc_val += current_qc.mean().item()
                av_qc_var_val += current_qc_var.mean().item()
                cvar = current_qc + self.pdf_cdf * torch.sqrt(current_qc_var_clamped)
                av_cvar += cvar.mean().item()
            
            # ===== 更新Actor (延迟更新) =====
            if it % policy_freq == 0:
                actor_action = self.actor(state)
                
                # Reward term
                Q_reward, _ = self.reward_critic(state, actor_action)
                
                # Safety term (Q^c_pi和V^c_pi)
                qc_pi = self.qc(state, actor_action)
                qc_pi_var = self.qc_var(state, actor_action)
                qc_pi_var_clamped = torch.clamp(qc_pi_var, min=1e-8)
                
                # 获取当前β
                beta = F.softplus(self.soft_beta)
                
                # 计算damp（与官方代码一致）
                # damp = damp_scale * mean(cost_constraint - qc - pdf_cdf * sqrt(qc_var))
                with torch.no_grad():
                    qc_for_damp = self.qc(state, action)
                    qc_var_for_damp = torch.clamp(self.qc_var(state, action), min=1e-8)
                    damp = self.damp_scale * torch.mean(
                        self.cost_constraint - qc_for_damp - self.pdf_cdf * torch.sqrt(qc_var_for_damp)
                    )
                
                # Actor Loss（官方公式第442行，移除熵项）
                # pi_loss = -min_q_pi + (beta - damp) * (qc_pi + pdf_cdf * sqrt(qc_pi_var))
                # 注意：官方代码中alpha*logp_pi是SAC特有的，TD3移除
                actor_loss = torch.mean(
                    -Q_reward + (beta - damp) * (qc_pi + self.pdf_cdf * torch.sqrt(qc_pi_var_clamped))
                )
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                av_actor_loss += actor_loss.item()
                
                # ===== 更新β（Lagrangian乘子）=====
                # β Loss（官方公式第467行）
                # beta_loss = beta * (cost_constraint - qc - pdf_cdf * sqrt(qc_var))
                with torch.no_grad():
                    qc_for_beta = self.qc(state, action)
                    qc_var_for_beta = torch.clamp(self.qc_var(state, action), min=1e-8)
                
                beta_for_loss = F.softplus(self.soft_beta)
                beta_loss = torch.mean(
                    beta_for_loss * (self.cost_constraint - qc_for_beta - self.pdf_cdf * torch.sqrt(qc_var_for_beta))
                )
                
                self.beta_optimizer.zero_grad()
                beta_loss.backward()
                self.beta_optimizer.step()
                av_beta_loss += beta_loss.item()
                
                # Soft update target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.reward_critic.parameters(), self.reward_critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.qc.parameters(), self.qc_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.qc_var.parameters(), self.qc_var_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        self.iter_count += 1
        
        # TensorBoard
        self.writer.add_scalar("train/qr_loss", av_qr_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/qc_loss", av_qc_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/qc_var_loss", av_qc_var_loss / iterations, self.iter_count)
        num_actor_updates = iterations // policy_freq
        if num_actor_updates > 0:
            self.writer.add_scalar("train/actor_loss", av_actor_loss / num_actor_updates, self.iter_count)
            self.writer.add_scalar("train/beta_loss", av_beta_loss / num_actor_updates, self.iter_count)
        self.writer.add_scalar("train/avg_reward_Q", av_reward_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_reward_Q", max_reward_Q, self.iter_count)
        self.writer.add_scalar("train/avg_qc", av_qc_val / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_qc_var", av_qc_var_val / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_cvar", av_cvar / iterations, self.iter_count)
        self.writer.add_scalar("train/beta", self.get_beta(), self.iter_count)
        
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def save(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_target.state_dict(), f"{directory}/{filename}_actor_target.pth")
        torch.save(self.reward_critic.state_dict(), f"{directory}/{filename}_reward_critic.pth")
        torch.save(self.reward_critic_target.state_dict(), f"{directory}/{filename}_reward_critic_target.pth")
        torch.save(self.qc.state_dict(), f"{directory}/{filename}_qc.pth")
        torch.save(self.qc_target.state_dict(), f"{directory}/{filename}_qc_target.pth")
        torch.save(self.qc_var.state_dict(), f"{directory}/{filename}_qc_var.pth")
        torch.save(self.qc_var_target.state_dict(), f"{directory}/{filename}_qc_var_target.pth")
        
        torch.save({
            'soft_beta': self.soft_beta,
            'cl': self.cl,
            'cost_lim': self.cost_lim,
            'pdf_cdf': self.pdf_cdf,
            'cost_constraint': self.cost_constraint,
        }, f"{directory}/{filename}_wcsac_params.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth", map_location=self.device))
        self.actor_target.load_state_dict(torch.load(f"{directory}/{filename}_actor_target.pth", map_location=self.device))
        self.reward_critic.load_state_dict(torch.load(f"{directory}/{filename}_reward_critic.pth", map_location=self.device))
        self.reward_critic_target.load_state_dict(torch.load(f"{directory}/{filename}_reward_critic_target.pth", map_location=self.device))
        self.qc.load_state_dict(torch.load(f"{directory}/{filename}_qc.pth", map_location=self.device))
        self.qc_target.load_state_dict(torch.load(f"{directory}/{filename}_qc_target.pth", map_location=self.device))
        self.qc_var.load_state_dict(torch.load(f"{directory}/{filename}_qc_var.pth", map_location=self.device))
        self.qc_var_target.load_state_dict(torch.load(f"{directory}/{filename}_qc_var_target.pth", map_location=self.device))
        
        wcsac_params = torch.load(f"{directory}/{filename}_wcsac_params.pth", map_location=self.device)
        self.soft_beta = wcsac_params['soft_beta'].to(self.device)
        self.soft_beta.requires_grad = True
        self.beta_optimizer = torch.optim.Adam([self.soft_beta], lr=self.lr * self.beta_lr_scale)
        self.cl = wcsac_params['cl']
        self.cost_lim = wcsac_params['cost_lim']
        self.pdf_cdf = wcsac_params['pdf_cdf']
        self.cost_constraint = wcsac_params['cost_constraint']
        
        print(f"✅ Loaded model from: {directory}/{filename}")

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """准备状态（与baseline一致，不需要状态扩展）"""
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