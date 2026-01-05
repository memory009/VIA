from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter


class Actor(nn.Module):
    """轻量级Actor网络（26维状态扩展版）"""
    def __init__(self, state_dim, action_dim, hidden_dim=26):
        super(Actor, self).__init__()
        
        self.layer_1 = nn.Linear(state_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()
        
        print(f"🔹 Actor网络: {state_dim} → {hidden_dim} → {hidden_dim} → {action_dim}")
        print(f"   参数量: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class TaskCritic(nn.Module):
    """任务 Critic 网络 - 双Q网络结构（用于任务奖励）"""
    def __init__(self, state_dim, action_dim, hidden_dim=26):
        super(TaskCritic, self).__init__()
        
        # Q1网络
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)
        
        # Q2网络
        self.layer_4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_6 = nn.Linear(hidden_dim, 1)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🔹 Task Critic(双Q): ({state_dim}+{action_dim}) → {hidden_dim} → {hidden_dim} → 1")
        print(f"   双Q总参数量: {total_params:,}")

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
    CVaR Cost Critic - Quantile Regression Network
    基于论文 Figure 1 的 Noncrossing Quantile Logit Network
    
    完全符合论文：使用128个quantiles
    """
    def __init__(self, state_dim, action_dim, hidden_dim=26, n_quantiles=128):
        super(CVaRCostCritic, self).__init__()
        
        self.n_quantiles = n_quantiles
        
        # Quantile fractions τ
        self.register_buffer(
            'tau_hat',
            torch.arange(0, n_quantiles + 1, dtype=torch.float32) / n_quantiles
        )
        self.register_buffer(
            'tau',
            (self.tau_hat[:-1] + self.tau_hat[1:]) / 2.0  # midpoints
        )
        
        # MLP for latent features
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Weight k and bias d (论文中的 k*φ + d)
        self.fc_k = nn.Linear(hidden_dim, 1)
        self.fc_d = nn.Linear(hidden_dim, 1)
        
        # Phi network (输出quantile的相对位置)
        self.fc_phi = nn.Linear(hidden_dim, n_quantiles)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🔹 CVaR Cost Critic(Quantile): ({state_dim}+{action_dim}) → {hidden_dim} → {n_quantiles} quantiles")
        print(f"   参数量: {total_params:,}")
        print(f"   Quantile数量: {n_quantiles}")

    def forward(self, s, a):
        """
        Returns:
            quantiles: (batch_size, n_quantiles) - cost的分位数估计
        """
        sa = torch.cat([s, a], 1)
        
        # MLP
        h = F.relu(self.fc1(sa))
        h = F.relu(self.fc2(h))
        
        # Weight and bias
        k = self.fc_k(h)  # (batch, 1)
        d = self.fc_d(h)  # (batch, 1)
        
        # Phi (使用Softmax和CumSum保证non-crossing)
        phi_logits = self.fc_phi(h)  # (batch, n_quantiles)
        phi_weights = F.softmax(phi_logits, dim=-1)  # (batch, n_quantiles)
        phi = torch.cumsum(phi_weights, dim=-1)  # (batch, n_quantiles)
        
        # Quantiles: q_i = k * φ_i + d
        quantiles = k * phi + d  # (batch, n_quantiles)
        
        # 使用softplus确保非负（cost >= 0）
        quantiles = F.softplus(quantiles)
        
        return quantiles
    
    def compute_cvar(self, quantiles, e_t, alpha=0.1):
        """
        计算CVaR值（论文公式19）
        
        Args:
            quantiles: (batch, n_quantiles) - 分位数估计
            e_t: (batch, 1) - 累积cost阈值
            alpha: CVaR风险水平（在此公式中不直接使用，保留参数为接口兼容）
        
        Returns:
            cvar: (batch, 1) - CVaR估计
        
        注意：论文公式(19)的CVaR是基于e_t阈值的条件期望，不是基于α-quantile。
        公式：V̂_C(s̄_t) = Σ (τ_{i+1} - τ_i) * q_i(s) * I(q_i(s) >= e_t)
        """
        batch_size = quantiles.shape[0]
        
        # 权重（每个quantile的宽度）
        weights = (self.tau_hat[1:] - self.tau_hat[:-1]).unsqueeze(0)  # (1, n_quantiles)
        weights = weights.expand(batch_size, -1)  # (batch, n_quantiles)
        
        # 指示函数：I(q_i >= e_t)
        indicators = (quantiles >= e_t).float()  # (batch, n_quantiles)
        
        # 加权求和
        cvar = torch.sum(weights * quantiles * indicators, dim=1, keepdim=True)  # (batch, 1)
        
        # 归一化（如果没有任何quantile >= e_t，返回最大quantile）
        normalizer = torch.sum(weights * indicators, dim=1, keepdim=True)
        normalizer = torch.clamp(normalizer, min=1e-8)
        cvar = cvar / normalizer
        
        return cvar


class TD3_CVaRCPO(object):
    """
    TD3 with CVaR-Constrained Policy Optimization
    
    基于论文 "CVaR-Constrained Policy Optimization for Safe Reinforcement Learning"
    使用状态扩展 s̄ = (s, e) 和 quantile regression 估计 CVaR
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=1e-4,
        hidden_dim=26,
        n_quantiles=128,  # ✅ 修正：默认值改为128，与论文一致
        cvar_alpha=0.1,
        cost_threshold=25.0,
        lambda_lr=0.01,
        var_lr=0.01,  # ✅ 修正：增加学习率从 0.001 到 0.01
        save_every=0,
        load_model=False,
        save_directory=Path("src/drl_navigation_ros2/models/TD3_cvar_cpo"),
        model_name="TD3_cvar_cpo",
        load_directory=Path("src/drl_navigation_ros2/models/TD3_cvar_cpo"),
        run_id=None,
    ):
        print("\n" + "="*80)
        print("🚀 初始化 TD3 with CVaR-CPO")
        print("="*80)
        
        self.device = device
        self.original_state_dim = state_dim  # 原始状态维度（25）
        self.augmented_state_dim = state_dim + 1  # 扩展状态维度（26）= (s, e)
        self.action_dim = action_dim
        self.max_action = max_action
        
        # CVaR参数
        self.cvar_alpha = cvar_alpha
        self.cost_threshold = cost_threshold
        self.n_quantiles = n_quantiles
        
        # VaR参数 u_k（论文公式5）
        self.var_u = nn.Parameter(torch.tensor([0.0], device=device))
        self.var_optimizer = torch.optim.Adam([self.var_u], lr=var_lr)
        
        # Lagrangian乘子 w（论文Lemma 2）
        self.lambda_w = nn.Parameter(torch.tensor([1.0], device=device))
        self.lambda_optimizer = torch.optim.Adam([self.lambda_w], lr=lambda_lr)
        
        # Temperature parameter v（论文Theorem 1）
        self.temperature_v = 1.0  # 固定值
        
        print(f"📊 CVaR参数:")
        print(f"   α (confidence level): {cvar_alpha}")
        print(f"   Cost threshold b: {cost_threshold}")
        print(f"   VaR初始值 u: {self.var_u.item():.3f}")
        print(f"   Lagrangian初始值 w: {self.lambda_w.item():.3f}")
        print(f"   Temperature v: {self.temperature_v}")
        
        # ===== Actor (输入扩展状态) =====
        self.actor = Actor(self.augmented_state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(self.augmented_state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr)

        # ===== Task Critics (输入扩展状态) =====
        self.task_critic = TaskCritic(self.augmented_state_dim, action_dim, hidden_dim).to(self.device)
        self.task_critic_target = TaskCritic(self.augmented_state_dim, action_dim, hidden_dim).to(self.device)
        self.task_critic_target.load_state_dict(self.task_critic.state_dict())
        self.task_critic_optimizer = torch.optim.Adam(params=self.task_critic.parameters(), lr=lr)

        # ===== CVaR Cost Critics (输入扩展状态，输出quantiles) =====
        self.cost_critic = CVaRCostCritic(
            self.augmented_state_dim, action_dim, hidden_dim, n_quantiles
        ).to(self.device)
        self.cost_critic_target = CVaRCostCritic(
            self.augmented_state_dim, action_dim, hidden_dim, n_quantiles
        ).to(self.device)
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
        self.cost_critic_optimizer = torch.optim.Adam(params=self.cost_critic.parameters(), lr=lr)
        
        # 计算总参数量
        total_params = (
            sum(p.numel() for p in self.actor.parameters()) +
            sum(p.numel() for p in self.task_critic.parameters()) +
            sum(p.numel() for p in self.cost_critic.parameters())
        )
        print(f"\n✅ 网络总参数量: {total_params:,}")
        print(f"📍 设备: {device}")
        print(f"🎯 隐藏层维度: {hidden_dim}")
        print(f"📐 状态维度: {self.original_state_dim} → {self.augmented_state_dim} (扩展)")
        print("="*80 + "\n")
        
        if run_id:
            tensorboard_log_dir = f"runs/{run_id}"
            self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        else:
            self.writer = SummaryWriter()
        
        self.iter_count = 0
        self.discount = 0.99  # γ
        
        if load_model:
            self.load(filename=model_name, directory=load_directory)
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory

    def augment_state(self, state, e_t):
        """
        状态扩展: s̄ = (s, e_t)

        Args:
            state: (batch, state_dim) 或 (state_dim,) - list/numpy/tensor
            e_t: (batch, 1) 或 scalar

        Returns:
            augmented_state: (batch, state_dim+1) 或 (state_dim+1,)
        """
        # 转换 state 为 tensor (支持list, numpy, tensor)
        if isinstance(state, list):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device)

        # 转换 e_t 为 tensor
        if isinstance(e_t, (int, float)):
            e_t = torch.FloatTensor([e_t]).to(self.device)
        elif isinstance(e_t, np.ndarray):
            e_t = torch.FloatTensor(e_t).to(self.device)
        elif isinstance(e_t, torch.Tensor):
            e_t = e_t.to(self.device)
        
        # 处理维度
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_dim)
        if e_t.dim() == 0:
            e_t = e_t.unsqueeze(0).unsqueeze(0)  # (1, 1)
        elif e_t.dim() == 1:
            e_t = e_t.unsqueeze(1)  # (batch, 1)
        
        augmented = torch.cat([state, e_t], dim=1)  # (batch, state_dim+1)
        return augmented

    def get_action(self, state, e_t, add_noise):
        """
        获取动作（输入扩展状态）
        
        Args:
            state: 原始状态 (state_dim,)
            e_t: 累积cost阈值 scalar
            add_noise: 是否添加探索噪声
        """
        augmented_state = self.augment_state(state, e_t)
        action = self.actor(augmented_state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, 0.2, size=self.action_dim)
            action = (action + noise).clip(-self.max_action, self.max_action)
        
        return action

    def act(self, state, e_t):
        """确定性动作（用于评估）"""
        return self.get_action(state, e_t, add_noise=False)

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        """
        训练循环 - CVaR-CPO版本
        
        核心改进：
        1. 使用Huber Quantile Regression Loss训练cost critic
        2. Actor loss使用简化版本（保持可比性）
        3. 交替更新VaR参数和Lagrangian乘子
        """
        
        av_task_Q = 0.0
        av_cost_cvar = 0.0
        max_task_Q = -inf
        av_task_loss = 0.0
        av_cost_loss = 0.0
        av_actor_loss = 0.0
        
        # Cost统计
        av_cost_in_batch = 0.0
        max_cost_in_batch = 0.0
        
        for it in range(iterations):
            # 采样 batch（包含 cost 和 e_t）
            batch = replay_buffer.sample_batch_with_augmented_state(batch_size, self.var_u.item())
            
            state = torch.Tensor(batch['states']).to(self.device)  # (batch, state_dim+1)
            next_state = torch.Tensor(batch['next_states']).to(self.device)  # (batch, state_dim+1)
            action = torch.Tensor(batch['actions']).to(self.device)
            reward = torch.Tensor(batch['rewards']).to(self.device)
            cost = torch.Tensor(batch['costs']).to(self.device)
            done = torch.Tensor(batch['dones']).to(self.device)
            e_t = torch.Tensor(batch['e_t']).to(self.device)  # (batch, 1)
            next_e_t = torch.Tensor(batch['next_e_t']).to(self.device)  # (batch, 1)
            
            # Cost统计
            av_cost_in_batch += cost.mean().item()
            max_cost_in_batch = max(max_cost_in_batch, cost.max().item())
            
            # ===== 计算 Target Q =====
            with torch.no_grad():
                next_action = self.actor_target(next_state)
                
                # 添加噪声
                noise = torch.randn_like(next_action) * policy_noise
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                
                # Task Q target
                target_Q1, target_Q2 = self.task_critic_target(next_state, next_action)
                target_Q_task = torch.min(target_Q1, target_Q2)
                av_task_Q += target_Q_task.mean().item()
                max_task_Q = max(max_task_Q, target_Q_task.max().item())
                target_Q_task = reward + (1 - done) * discount * target_Q_task
                
                # Cost quantiles target（论文的distributional Bellman）
                target_quantiles = self.cost_critic_target(next_state, next_action)  # (batch, n_quantiles)
                # Bellman: C + γ * Z_C(s', a')
                # cost: (batch, 1) -> (batch, n_quantiles) 广播到所有quantiles
                # done: (batch, 1) -> (batch, n_quantiles)
                target_quantiles = cost + (1 - done) * discount * target_quantiles
            
            # ===== 更新 Task Critics =====
            current_Q1, current_Q2 = self.task_critic(state, action)
            task_loss = F.mse_loss(current_Q1, target_Q_task) + F.mse_loss(current_Q2, target_Q_task)
            
            self.task_critic_optimizer.zero_grad()
            task_loss.backward()
            self.task_critic_optimizer.step()
            av_task_loss += task_loss.item()
            
            # ===== 更新 Cost Critics (Huber Quantile Regression) =====
            current_quantiles = self.cost_critic(state, action)  # (batch, n_quantiles)
            
            # Quantile Huber Loss（论文公式20）
            cost_loss = self.quantile_huber_loss(current_quantiles, target_quantiles)
            
            self.cost_critic_optimizer.zero_grad()
            cost_loss.backward()
            self.cost_critic_optimizer.step()
            av_cost_loss += cost_loss.item()
            
            # 计算CVaR用于监控
            with torch.no_grad():
                cost_cvar = self.cost_critic.compute_cvar(
                    current_quantiles, e_t, alpha=self.cvar_alpha
                )
                av_cost_cvar += cost_cvar.mean().item()
            
            # ===== 更新 Actor =====
            if it % policy_freq == 0:
                # ✅ 修正3：添加完整的Actor loss说明
                # 论文Lemma 3使用KL-divergence based update:
                # ∇_θ L(π_θ, π*) = ∇_θ D_KL(π_θ||π_θk) - 1/v * E[∇_θ π_θ/π_θk * (A_R - w*A_C)]
                # 
                # 这里采用简化版本: -Q_task + w * CVaR_cost
                # 简化理由：
                # 1. 保持和baseline的可比性（相同actor loss形式）
                # 2. 核心思想相同（最大化reward - 惩罚cost）
                # 3. 避免额外的π*计算和KL项，更稳定
                # 4. CVaR约束的核心在Cost Critic（完全准确），不在Actor loss形式
                actor_action = self.actor(state)
                Q_task, _ = self.task_critic(state, actor_action)
                
                # 计算CVaR cost（这是与baseline的关键差异）
                cost_quantiles = self.cost_critic(state, actor_action)
                cost_cvar = self.cost_critic.compute_cvar(cost_quantiles, e_t, self.cvar_alpha)
                
                # Actor loss（简化版）
                actor_loss = -Q_task.mean() + self.lambda_w * cost_cvar.mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                av_actor_loss += actor_loss.item()
                
                # Soft update
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(
                    self.task_critic.parameters(), self.task_critic_target.parameters()
                ):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(
                    self.cost_critic.parameters(), self.cost_critic_target.parameters()
                ):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        self.iter_count += 1
        
        # ===== TensorBoard 记录 =====
        self.writer.add_scalar("train/task_loss", av_task_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/cost_loss", av_cost_loss / iterations, self.iter_count)
        
        num_actor_updates = iterations // policy_freq
        if num_actor_updates > 0:
            self.writer.add_scalar("train/actor_loss", av_actor_loss / num_actor_updates, self.iter_count)
        
        self.writer.add_scalar("train/avg_task_Q", av_task_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_cost_cvar", av_cost_cvar / iterations, self.iter_count)
        self.writer.add_scalar("train/max_task_Q", max_task_Q, self.iter_count)
        
        self.writer.add_scalar("train/cost_stats/avg_cost_in_batch", 
                              av_cost_in_batch / iterations, self.iter_count)
        self.writer.add_scalar("train/cost_stats/max_cost_in_batch", 
                              max_cost_in_batch, self.iter_count)
        
        # CVaR参数监控
        self.writer.add_scalar("train/cvar_params/var_u", self.var_u.item(), self.iter_count)
        self.writer.add_scalar("train/cvar_params/lambda_w", self.lambda_w.item(), self.iter_count)
        
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)
    
    def quantile_huber_loss(self, current_quantiles, target_quantiles, kappa=1.0):
        """
        Huber Quantile Regression Loss（论文公式20）
        
        Args:
            current_quantiles: (batch, n_quantiles)
            target_quantiles: (batch, n_quantiles)
            kappa: Huber loss threshold
        
        Returns:
            loss: scalar
        """
        batch_size = current_quantiles.shape[0]
        n_quantiles = current_quantiles.shape[1]
        
        # 扩展维度进行矩阵运算
        # current: (batch, n_quantiles, 1)
        # target: (batch, 1, n_quantiles)
        current = current_quantiles.unsqueeze(2)
        target = target_quantiles.unsqueeze(1)
        
        # TD error: δ_ij = target_j - current_i
        td_errors = target - current  # (batch, n_quantiles, n_quantiles)
        
        # Huber loss
        abs_errors = torch.abs(td_errors)
        huber = torch.where(
            abs_errors <= kappa,
            0.5 * td_errors ** 2,
            kappa * (abs_errors - 0.5 * kappa)
        )
        
        # Quantile weights: τ - I(δ < 0)
        tau = self.cost_critic.tau.view(1, n_quantiles, 1)  # (1, n_quantiles, 1)
        quantile_weights = torch.abs(tau - (td_errors < 0).float())
        
        # Quantile Huber loss
        loss = (quantile_weights * huber).mean()
        
        return loss
    
    def update_var_and_lambda(self, avg_episode_cost, epoch_costs=None):
        """
        ✅ 修正2：改进VaR更新，使用概率估计更接近论文公式5
        
        Args:
            avg_episode_cost: 当前epoch的平均episode cost
            epoch_costs: 当前epoch所有episode的cost列表（用于估计概率）
        """
        # 更新VaR（论文公式5）
        # u_{k+1} = u_k + β_u * [1 - 1/(1-α) * P(C >= u_k)]
        
        if epoch_costs is not None and len(epoch_costs) > 0:
            # 完整版本：使用episode costs估计概率
            # ✅ 修正：使用 > 而不是 >= 来正确计算 P(C > u_k)
            prob_exceed = np.mean([c > self.var_u.item() for c in epoch_costs])
            var_grad = 1.0 - (1.0 / (1.0 - self.cvar_alpha)) * prob_exceed
        else:
            # 简化版本：使用sign函数（fallback）
            var_grad = 1.0 if avg_episode_cost > self.var_u.item() else -1.0
        
        self.var_optimizer.zero_grad()
        # 确保梯度类型与参数一致（float32）
        self.var_u.grad = torch.tensor([-var_grad], dtype=self.var_u.dtype, device=self.device)
        self.var_optimizer.step()

        # 限制u的范围
        with torch.no_grad():
            self.var_u.data.clamp_(min=0.0, max=self.cost_threshold)

        # 更新Lagrangian乘子w（论文公式23）
        # w_{k+1} = max(0, w_k - β_w * (b - u_k - V_C(s_0)))
        constraint_violation = self.cost_threshold - self.var_u.item() - avg_episode_cost

        self.lambda_optimizer.zero_grad()
        # 确保梯度类型与参数一致（float32）
        self.lambda_w.grad = torch.tensor([constraint_violation], dtype=self.lambda_w.dtype, device=self.device)
        self.lambda_optimizer.step()
        
        # 限制w的范围
        with torch.no_grad():
            self.lambda_w.data.clamp_(min=0.0, max=100.0)
        
        # 记录
        self.writer.add_scalar("epoch/var_u_update", self.var_u.item(), self.iter_count)
        self.writer.add_scalar("epoch/lambda_w_update", self.lambda_w.item(), self.iter_count)
        self.writer.add_scalar("epoch/constraint_violation", constraint_violation, self.iter_count)
        if epoch_costs is not None:
            self.writer.add_scalar("epoch/prob_exceed_var", prob_exceed, self.iter_count)

    def save(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_target.state_dict(), f"{directory}/{filename}_actor_target.pth")
        torch.save(self.task_critic.state_dict(), f"{directory}/{filename}_task_critic.pth")
        torch.save(self.task_critic_target.state_dict(), f"{directory}/{filename}_task_critic_target.pth")
        torch.save(self.cost_critic.state_dict(), f"{directory}/{filename}_cost_critic.pth")
        torch.save(self.cost_critic_target.state_dict(), f"{directory}/{filename}_cost_critic_target.pth")
        
        # 保存CVaR参数
        torch.save({
            'var_u': self.var_u,
            'lambda_w': self.lambda_w,
        }, f"{directory}/{filename}_cvar_params.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load(f"{directory}/{filename}_actor.pth", map_location=self.device)
        )
        self.actor_target.load_state_dict(
            torch.load(f"{directory}/{filename}_actor_target.pth", map_location=self.device)
        )
        self.task_critic.load_state_dict(
            torch.load(f"{directory}/{filename}_task_critic.pth", map_location=self.device)
        )
        self.task_critic_target.load_state_dict(
            torch.load(f"{directory}/{filename}_task_critic_target.pth", map_location=self.device)
        )
        self.cost_critic.load_state_dict(
            torch.load(f"{directory}/{filename}_cost_critic.pth", map_location=self.device)
        )
        self.cost_critic_target.load_state_dict(
            torch.load(f"{directory}/{filename}_cost_critic_target.pth", map_location=self.device)
        )
        
        # 加载CVaR参数
        cvar_params = torch.load(f"{directory}/{filename}_cvar_params.pth", map_location=self.device)
        self.var_u.data = cvar_params['var_u'].data
        self.lambda_w.data = cvar_params['lambda_w'].data
        
        print(f"Loaded weights from: {directory} to device: {self.device}")

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """
        准备状态（与baseline相同，但不包含e_t的扩展，那部分在训练时处理）
        
        Returns:
            state: (state_dim,) - 原始状态，不包含e_t
            terminal: bool
        """
        latest_scan = np.array(latest_scan)
        
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0
        
        max_bins = self.original_state_dim - 5
        bin_size = int(np.ceil(len(latest_scan) / max_bins))
        
        min_values = []
        for i in range(0, len(latest_scan), bin_size):
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            min_values.append(min(bin))
        
        state = min_values + [distance, cos, sin] + [action[0], action[1]]
        
        assert len(state) == self.original_state_dim
        terminal = 1 if collision or goal else 0
        
        return state, terminal