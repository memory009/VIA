"""
TD3 with CVaR-Constrained Policy Optimization

用于Ablation Study：
- 基础RL参数与baseline保持一致（γ=0.99, lr=1e-4, hidden_dim=26, batch_size=40）
- CVaR特有参数按照论文设置（var_lr=0.001, lambda_lr=0.001, α=0.9, M=128）

论文: "CVaR-Constrained Policy Optimization for Safe Reinforcement Learning"
Zhang et al., IEEE TNNLS 2025

核心公式:
- 公式(5): VaR更新 u_{k+1} = u_k + β_u * [P(C >= u_k) - (1-α)]  (修正版)
- 公式(19): CVaR估计 
- 公式(20): Huber Quantile Regression Loss
- 公式(23): Lagrangian更新 w_{k+1} = proj[w_k - β_w * (b - u_k - V_C(s_0))]
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# Ablation Study 超参数配置
# 基础参数：与baseline保持一致
# CVaR参数：按照论文设置
# ============================================================================

# ----- 基础RL参数（与你的baseline一致）-----
BASELINE_GAMMA = 0.99           # 你的baseline使用0.99
BASELINE_LR = 1e-4              # 你的baseline学习率
BASELINE_HIDDEN_DIM = 26        # 你的baseline隐藏层（POLAR验证需求）
BASELINE_BATCH_SIZE = 40        # 你的baseline批次大小
BASELINE_BUFFER_SIZE = 500000   # 你的baseline buffer大小
BASELINE_TAU = 0.005            # TD3软更新系数
BASELINE_POLICY_NOISE = 0.2     # TD3目标策略噪声
BASELINE_NOISE_CLIP = 0.5       # TD3噪声裁剪
BASELINE_POLICY_FREQ = 2        # TD3策略更新频率

# ----- CVaR-CPO特有参数（按照论文设置）-----
CVAR_VAR_LR = 0.1             # β_u，VaR更新步长（论文推荐为0.001）
CVAR_LAMBDA_LR = 0.001          # β_w，Lagrangian更新步长（论文推荐）
CVAR_ALPHA = 0.9                # α，CVaR风险水平（关注worst 10%）
# CVAR_ALPHA = 0.5                # α，CVaR风险水平（关注worst 50%）
# CVAR_ALPHA = 0.1                # α，CVaR风险水平（关注worst 90%）
CVAR_N_QUANTILES = 128          # M，分位数数量（论文第838页）
CVAR_COST_THRESHOLD = 10.0      # b，cost约束阈值（二值cost下：允许的雷达距离低于danger_threshold的次数上限）


class Actor(nn.Module):
    """Actor网络（输入扩展状态 s̄ = (s, e_t)）"""
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


class TaskCritic(nn.Module):
    """任务Critic - 双Q网络（TD3标准结构）"""
    def __init__(self, state_dim, action_dim, hidden_dim=BASELINE_HIDDEN_DIM):
        super(TaskCritic, self).__init__()
        
        # Q1
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.layer_4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_6 = nn.Linear(hidden_dim, 1)
        
        print(f"🔹 Task Critic(双Q): ({state_dim}+{action_dim}) → {hidden_dim} → 1")

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)
        
        q2 = F.relu(self.layer_4(sa))
        q2 = F.relu(self.layer_5(q2))
        q2 = self.layer_6(q2)
        
        return q1, q2


class CVaRCostCritic(nn.Module):
    """
    CVaR Cost Critic - Noncrossing Quantile Network
    
    论文Figure 1: 使用 q_i = k * φ_i + d 保证non-crossing property
    """
    def __init__(self, state_dim, action_dim, 
                 hidden_dim=BASELINE_HIDDEN_DIM,
                 n_quantiles=CVAR_N_QUANTILES):
        super(CVaRCostCritic, self).__init__()
        
        self.n_quantiles = n_quantiles
        
        # τ_k = k/M（论文公式）
        self.register_buffer(
            'tau_hat',
            torch.arange(0, n_quantiles + 1, dtype=torch.float32) / n_quantiles
        )
        # τ̂_i = (τ_{i-1} + τ_i) / 2
        self.register_buffer(
            'tau',
            (self.tau_hat[:-1] + self.tau_hat[1:]) / 2.0
        )
        
        # MLP
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Noncrossing quantile: q_i = k * φ_i + d
        self.fc_k = nn.Linear(hidden_dim, 1)
        self.fc_d = nn.Linear(hidden_dim, 1)
        self.fc_phi = nn.Linear(hidden_dim, n_quantiles)
        
        print(f"🔹 Cost Critic(Quantile): ({state_dim}+{action_dim}) → {hidden_dim} → {n_quantiles} quantiles")

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        
        h = F.relu(self.fc1(sa))
        h = F.relu(self.fc2(h))
        
        # k > 0, d任意
        k = F.softplus(self.fc_k(h))
        d = self.fc_d(h)
        
        # φ: softmax + cumsum 保证单调递增
        phi_logits = self.fc_phi(h)
        phi_weights = F.softmax(phi_logits, dim=-1)
        phi = torch.cumsum(phi_weights, dim=-1)
        
        # q_i = k * φ_i + d
        quantiles = k * phi + d
        
        # Cost非负
        quantiles = F.softplus(quantiles)
        
        return quantiles
    
    def compute_cvar(self, quantiles, e_t):
        """
        计算CVaR（论文公式19）
        
        V̂_C(s̄_t) = Σ (τ_{i+1} - τ_i) * q_i(s) * I(q_i(s) >= e_t)
        
        注意: 这里e_t是动态的累积cost阈值，不是固定的α-VaR
        """
        batch_size = quantiles.shape[0]
        
        # 权重
        weights = (self.tau_hat[1:] - self.tau_hat[:-1]).unsqueeze(0)
        weights = weights.expand(batch_size, -1)
        
        # 指示函数
        indicators = (quantiles >= e_t).float()
        
        # 加权求和
        cvar = torch.sum(weights * quantiles * indicators, dim=1, keepdim=True)
        
        # 归一化
        normalizer = torch.sum(weights * indicators, dim=1, keepdim=True)
        normalizer = torch.clamp(normalizer, min=1e-8)
        cvar = cvar / normalizer
        
        return cvar


class TD3_CVaRCPO(object):
    """
    TD3 with CVaR-CPO (Ablation Study Version)
    
    参数设计原则:
    - 基础RL参数: 与baseline一致，确保公平对比
    - CVaR特有参数: 按论文设置
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
        # CVaR参数（按论文设置）
        n_quantiles=CVAR_N_QUANTILES,
        cvar_alpha=CVAR_ALPHA,
        cost_threshold=CVAR_COST_THRESHOLD,
        var_lr=CVAR_VAR_LR,
        lambda_lr=CVAR_LAMBDA_LR,
        # 其他
        save_every=0,
        load_model=False,
        save_directory=Path("models/TD3_cvar_cpo"),
        model_name="TD3_cvar_cpo",
        load_directory=Path("models/TD3_cvar_cpo"),
        run_id=None,
    ):
        print("\n" + "="*80)
        print("🚀 TD3 with CVaR-CPO (Ablation Study Version)")
        print("="*80)
        
        self.device = device
        self.original_state_dim = state_dim
        self.augmented_state_dim = state_dim + 1  # s̄ = (s, e_t)
        self.action_dim = action_dim
        self.max_action = max_action
        
        # 保存参数
        self.gamma = gamma
        self.cvar_alpha = cvar_alpha
        self.cost_threshold = cost_threshold
        self.n_quantiles = n_quantiles
        self.var_lr = var_lr
        self.lambda_lr = lambda_lr
        
        # VaR参数 u（将由warm-up phase初始化）
        self.var_u = torch.tensor([0.0], device=device)
        
        # Lagrangian乘子 w
        self.lambda_w = torch.tensor([1.0], device=device)
        
        print(f"\n📊 Ablation Study 参数配置:")
        print(f"   ─── 基础参数（与baseline一致）───")
        print(f"   γ (discount): {gamma}")
        print(f"   lr (network): {lr}")
        print(f"   hidden_dim: {hidden_dim}")
        print(f"   ─── CVaR参数（按论文设置）───")
        print(f"   β_u (var_lr): {var_lr}")
        print(f"   β_w (lambda_lr): {lambda_lr}")
        print(f"   α (CVaR level): {cvar_alpha} (关注worst {(1-cvar_alpha)*100:.0f}%)")
        print(f"   M (quantiles): {n_quantiles}")
        print(f"   b (cost threshold): {cost_threshold}")
        
        # 网络初始化
        self.actor = Actor(self.augmented_state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(self.augmented_state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.task_critic = TaskCritic(self.augmented_state_dim, action_dim, hidden_dim).to(device)
        self.task_critic_target = TaskCritic(self.augmented_state_dim, action_dim, hidden_dim).to(device)
        self.task_critic_target.load_state_dict(self.task_critic.state_dict())
        self.task_critic_optimizer = torch.optim.Adam(self.task_critic.parameters(), lr=lr)

        self.cost_critic = CVaRCostCritic(self.augmented_state_dim, action_dim, hidden_dim, n_quantiles).to(device)
        self.cost_critic_target = CVaRCostCritic(self.augmented_state_dim, action_dim, hidden_dim, n_quantiles).to(device)
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=lr)
        
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

    def set_var_u(self, value):
        """
        设置VaR参数u的值（由warm-up phase调用）
        
        Args:
            value: var_u的初始值（通常是warm-up阶段计算的α分位数）
        """
        self.var_u = torch.tensor([value], device=self.device)
        print(f"📊 var_u已设置为: {value:.4f}")

    def augment_state(self, state, e_t):
        """状态扩展: s̄ = (s, e_t)"""
        if isinstance(state, list):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device)

        if isinstance(e_t, (int, float)):
            e_t = torch.FloatTensor([e_t]).to(self.device)
        elif isinstance(e_t, np.ndarray):
            e_t = torch.FloatTensor(e_t).to(self.device)
        elif isinstance(e_t, torch.Tensor):
            e_t = e_t.to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if e_t.dim() == 0:
            e_t = e_t.unsqueeze(0).unsqueeze(0)
        elif e_t.dim() == 1:
            e_t = e_t.unsqueeze(1)
        
        return torch.cat([state, e_t], dim=1)

    def get_action(self, state, e_t, add_noise):
        """获取动作"""
        augmented_state = self.augment_state(state, e_t)
        action = self.actor(augmented_state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, BASELINE_POLICY_NOISE, size=self.action_dim)
            action = (action + noise).clip(-self.max_action, self.max_action)
        
        return action

    def act(self, state, e_t):
        """确定性动作（评估用）"""
        return self.get_action(state, e_t, add_noise=False)

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
        """训练循环"""
        if discount is None:
            discount = self.gamma
        
        av_task_Q = 0.0
        av_cost_cvar = 0.0
        max_task_Q = -inf
        av_task_loss = 0.0
        av_cost_loss = 0.0
        av_actor_loss = 0.0
        av_cost_in_batch = 0.0
        max_cost_in_batch = 0.0
        av_e_t_in_batch = 0.0
        min_e_t_in_batch = inf
        
        for it in range(iterations):
            # 采样（使用buffer中存储的真实e_t）
            batch = replay_buffer.sample_batch_with_augmented_state(batch_size)
            
            state = torch.Tensor(batch['states']).to(self.device)
            next_state = torch.Tensor(batch['next_states']).to(self.device)
            action = torch.Tensor(batch['actions']).to(self.device)
            reward = torch.Tensor(batch['rewards']).to(self.device)
            cost = torch.Tensor(batch['costs']).to(self.device)
            done = torch.Tensor(batch['dones']).to(self.device)
            e_t = torch.Tensor(batch['e_t']).to(self.device)
            
            # 统计
            av_cost_in_batch += cost.mean().item()
            max_cost_in_batch = max(max_cost_in_batch, cost.max().item())
            av_e_t_in_batch += e_t.mean().item()
            min_e_t_in_batch = min(min_e_t_in_batch, e_t.min().item())
            
            # Target计算
            with torch.no_grad():
                next_action = self.actor_target(next_state)
                noise = torch.randn_like(next_action) * policy_noise
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                
                # Task Q target
                target_Q1, target_Q2 = self.task_critic_target(next_state, next_action)
                target_Q_task = torch.min(target_Q1, target_Q2)
                av_task_Q += target_Q_task.mean().item()
                max_task_Q = max(max_task_Q, target_Q_task.max().item())
                target_Q_task = reward + (1 - done) * discount * target_Q_task
                
                # Cost quantiles target
                target_quantiles = self.cost_critic_target(next_state, next_action)
                target_quantiles = cost + (1 - done) * discount * target_quantiles
            
            # 更新Task Critics
            current_Q1, current_Q2 = self.task_critic(state, action)
            task_loss = F.mse_loss(current_Q1, target_Q_task) + F.mse_loss(current_Q2, target_Q_task)
            
            self.task_critic_optimizer.zero_grad()
            task_loss.backward()
            self.task_critic_optimizer.step()
            av_task_loss += task_loss.item()
            
            # 更新Cost Critics（Huber Quantile Loss）
            current_quantiles = self.cost_critic(state, action)
            cost_loss = self.quantile_huber_loss(current_quantiles, target_quantiles)
            
            self.cost_critic_optimizer.zero_grad()
            cost_loss.backward()
            self.cost_critic_optimizer.step()
            av_cost_loss += cost_loss.item()
            
            # CVaR监控
            with torch.no_grad():
                cost_cvar = self.cost_critic.compute_cvar(current_quantiles, e_t)
                av_cost_cvar += cost_cvar.mean().item()
            
            # 更新Actor
            if it % policy_freq == 0:
                actor_action = self.actor(state)
                Q_task, _ = self.task_critic(state, actor_action)
                
                cost_quantiles = self.cost_critic(state, actor_action)
                cost_cvar = self.cost_critic.compute_cvar(cost_quantiles, e_t)
                
                # Actor loss: -Q_task + w * CVaR_cost
                actor_loss = -Q_task.mean() + self.lambda_w.item() * cost_cvar.mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                av_actor_loss += actor_loss.item()
                
                # Soft update
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.task_critic.parameters(), self.task_critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        self.iter_count += 1
        
        # TensorBoard
        self.writer.add_scalar("train/task_loss", av_task_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/cost_loss", av_cost_loss / iterations, self.iter_count)
        num_actor_updates = iterations // policy_freq
        if num_actor_updates > 0:
            self.writer.add_scalar("train/actor_loss", av_actor_loss / num_actor_updates, self.iter_count)
        self.writer.add_scalar("train/avg_task_Q", av_task_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_cost_cvar", av_cost_cvar / iterations, self.iter_count)
        self.writer.add_scalar("train/max_task_Q", max_task_Q, self.iter_count)
        self.writer.add_scalar("train/cost_stats/avg_cost_in_batch", av_cost_in_batch / iterations, self.iter_count)
        self.writer.add_scalar("train/cost_stats/max_cost_in_batch", max_cost_in_batch, self.iter_count)
        self.writer.add_scalar("train/e_t_stats/avg_e_t_in_batch", av_e_t_in_batch / iterations, self.iter_count)
        self.writer.add_scalar("train/e_t_stats/min_e_t_in_batch", min_e_t_in_batch, self.iter_count)
        self.writer.add_scalar("train/cvar_params/var_u", self.var_u.item(), self.iter_count)
        self.writer.add_scalar("train/cvar_params/lambda_w", self.lambda_w.item(), self.iter_count)
        
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)
    
    def quantile_huber_loss(self, current_quantiles, target_quantiles, kappa=1.0):
        """
        Huber Quantile Regression Loss（论文公式20）
        """
        n_quantiles = current_quantiles.shape[1]
        
        current = current_quantiles.unsqueeze(2)
        target = target_quantiles.unsqueeze(1)
        
        td_errors = target - current
        
        abs_errors = torch.abs(td_errors)
        huber = torch.where(
            abs_errors <= kappa,
            0.5 * td_errors ** 2,
            kappa * (abs_errors - 0.5 * kappa)
        )
        
        tau = self.cost_critic.tau.view(1, n_quantiles, 1)
        quantile_weights = torch.abs(tau - (td_errors < 0).float())
        
        loss = (quantile_weights * huber).mean()
        
        return loss
    
    def update_var_and_lambda(self, avg_episode_cost, epoch_costs=None):
        """
        更新VaR和Lagrangian（每个training epoch结束后调用）
        
        公式(5)修正版: u^{k+1} = u^k + β_u * [P(C >= u^k) - (1-α)]
        公式(23): w^{k+1} = proj[w^k - β_w * (b - u^k - V_C(s̄_0))]
        
        注意: var_u的初始化由warm-up phase完成，这里只做更新
        """
        old_var_u = self.var_u.item()
        old_lambda_w = self.lambda_w.item()
        
        # 公式(5)修正版: VaR更新
        if epoch_costs is not None and len(epoch_costs) > 0:
            prob_exceed = np.mean([c >= old_var_u for c in epoch_costs])
            # 修正版公式: 当P(C>=u) > (1-α)时增大u，当P(C>=u) < (1-α)时减小u
            var_update = prob_exceed - (1.0 - self.cvar_alpha)
            new_var_u = old_var_u + self.var_lr * var_update
            
            # 确保u非负
            new_var_u = max(new_var_u, 0.0)
            
            self.var_u = torch.tensor([new_var_u], device=self.device)
        else:
            prob_exceed = 0.0
            var_update = 0.0
        
        # 公式(23): Lagrangian更新
        constraint_slack = self.cost_threshold - self.var_u.item() - avg_episode_cost
        new_lambda_w = old_lambda_w - self.lambda_lr * constraint_slack
        new_lambda_w = np.clip(new_lambda_w, 0.0, 100.0)
        self.lambda_w = torch.tensor([new_lambda_w], device=self.device)
        
        # TensorBoard
        self.writer.add_scalar("epoch/var_u", self.var_u.item(), self.iter_count)
        self.writer.add_scalar("epoch/lambda_w", self.lambda_w.item(), self.iter_count)
        self.writer.add_scalar("epoch/prob_exceed_var", prob_exceed, self.iter_count)
        self.writer.add_scalar("epoch/var_update", var_update, self.iter_count)
        self.writer.add_scalar("epoch/constraint_slack", constraint_slack, self.iter_count)
        
        # 打印调试信息
        print(f"   📊 公式(5)修正版 VaR更新:")
        print(f"      P(C >= u) = {prob_exceed:.3f}, target = {1-self.cvar_alpha:.3f}")
        print(f"      update = {prob_exceed:.3f} - {1-self.cvar_alpha:.3f} = {var_update:.4f}")
        print(f"   📊 公式(23) Lagrangian更新:")
        print(f"      约束余量 = {self.cost_threshold} - {self.var_u.item():.2f} - {avg_episode_cost:.2f} = {constraint_slack:.2f}")

    def save(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_target.state_dict(), f"{directory}/{filename}_actor_target.pth")
        torch.save(self.task_critic.state_dict(), f"{directory}/{filename}_task_critic.pth")
        torch.save(self.task_critic_target.state_dict(), f"{directory}/{filename}_task_critic_target.pth")
        torch.save(self.cost_critic.state_dict(), f"{directory}/{filename}_cost_critic.pth")
        torch.save(self.cost_critic_target.state_dict(), f"{directory}/{filename}_cost_critic_target.pth")
        
        torch.save({
            'var_u': self.var_u,
            'lambda_w': self.lambda_w,
        }, f"{directory}/{filename}_cvar_params.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth", map_location=self.device))
        self.actor_target.load_state_dict(torch.load(f"{directory}/{filename}_actor_target.pth", map_location=self.device))
        self.task_critic.load_state_dict(torch.load(f"{directory}/{filename}_task_critic.pth", map_location=self.device))
        self.task_critic_target.load_state_dict(torch.load(f"{directory}/{filename}_task_critic_target.pth", map_location=self.device))
        self.cost_critic.load_state_dict(torch.load(f"{directory}/{filename}_cost_critic.pth", map_location=self.device))
        self.cost_critic_target.load_state_dict(torch.load(f"{directory}/{filename}_cost_critic_target.pth", map_location=self.device))
        
        cvar_params = torch.load(f"{directory}/{filename}_cvar_params.pth", map_location=self.device)
        self.var_u = cvar_params['var_u'].to(self.device)
        self.lambda_w = cvar_params['lambda_w'].to(self.device)
        
        print(f"✅ Loaded model from: {directory}/{filename}")

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """准备状态（不含e_t扩展，与baseline一致）"""
        latest_scan = np.array(latest_scan)
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0
        
        max_bins = self.original_state_dim - 5
        bin_size = int(np.ceil(len(latest_scan) / max_bins))
        
        min_values = []
        for i in range(0, len(latest_scan), bin_size):
            bin_data = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            min_values.append(min(bin_data))
        
        state = min_values + [distance, cos, sin] + [action[0], action[1]]
        
        assert len(state) == self.original_state_dim
        terminal = 1 if collision or goal else 0
        
        return state, terminal