"""
TD3 with RCPO (Reward Constrained Policy Optimization) - 严格遵循论文版本

论文: "Reward Constrained Policy Optimization" (Tessler et al., ICLR 2019)

============================================================================
RCPO原文 vs 本实现 对比
============================================================================

【RCPO原文】
- Backbone: A2C (Mars Rover) / PPO (Robotics) - 都是on-policy方法
- Critic: 学习 penalized value V̂(λ,s) = V_R(s) - λ·V_C(s)
- 核心: 使用penalized reward r̂ = r - λ·c 直接训练Critic

【本实现的适配】
- Backbone: TD3 (off-policy) - 为了与CVaR-CPO公平对比
- Critic: 学习 penalized Q值 Q̂(λ,s,a)
- 核心: 同样使用penalized reward，但适配到Q-learning框架

【关键公式】(论文公式10-11)
- Penalized reward: r̂(λ,s,a) = r(s,a) - λ·c(s,a)
- Penalized value: V̂^π(λ,s) = E[Σγ^t r̂(λ,s_t,a_t)]

【与SAC-Lagrangian的区别】
- SAC-Lagrangian: 分开学习Q_r和Q_c，然后组合
- RCPO: 直接学习penalized Q值

============================================================================
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# 超参数配置 - 与baseline保持一致
# ============================================================================

BASELINE_GAMMA = 0.99
BASELINE_LR = 1e-4
BASELINE_HIDDEN_DIM = 26
BASELINE_BATCH_SIZE = 40
BASELINE_BUFFER_SIZE = 500000
BASELINE_TAU = 0.005
BASELINE_POLICY_NOISE = 0.2
BASELINE_NOISE_CLIP = 0.5
BASELINE_POLICY_FREQ = 2

# RCPO特有参数
RCPO_LAMBDA_LR = 5e-7
RCPO_LAMBDA_MAX = 100.0
RCPO_COST_THRESHOLD = 10.0


class Actor(nn.Module):
    """Actor网络 - 与TD3_lightweight完全一致"""
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


class PenalizedCritic(nn.Module):
    """
    Penalized Critic - 严格遵循RCPO论文
    
    学习penalized Q值: Q̂(λ,s,a) = E[Σγ^t (r_t - λ·c_t)]
    
    使用TD3的双Q网络结构减少过估计
    """
    def __init__(self, state_dim, action_dim, hidden_dim=BASELINE_HIDDEN_DIM):
        super(PenalizedCritic, self).__init__()
        
        # Q1网络
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)
        
        # Q2网络
        self.layer_4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_6 = nn.Linear(hidden_dim, 1)
        
        print(f"🔹 Penalized Critic(双Q): ({state_dim}+{action_dim}) → {hidden_dim} → 1")

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)
        
        q2 = F.relu(self.layer_4(sa))
        q2 = F.relu(self.layer_5(q2))
        q2 = self.layer_6(q2)
        
        return q1, q2


class TD3_RCPO_Strict(object):
    """
    TD3 with RCPO - 严格遵循论文版本
    
    ============================================================================
    与论文的对应关系
    ============================================================================
    
    【论文Algorithm 1 RCPO Advantage Actor Critic】
    Line 7:  R̂_t = r_t - λ_k·c_t + γV̂(λ,s_{t+1};v_k)    # penalized TD target
    Line 8:  Critic更新                                   # 学习penalized value
    Line 9:  Actor更新: θ ← θ + η∇_θV̂(λ,s)              # 最大化penalized value
    Line 10: λ更新: λ ← λ + η(J_C^π - α)                 # 基于原始约束
    
    【本实现的对应】
    - Line 7-8: 使用penalized_reward = reward - λ·cost 计算TD target
    - Line 9:   Actor最大化penalized Q值
    - Line 10:  每epoch结束后基于episode cost更新λ
    
    ============================================================================
    关键设计决策
    ============================================================================
    
    1. 为什么使用penalized reward而不是分开的Q_r和Q_c？
       → 严格遵循RCPO论文，论文明确使用r̂ = r - λc
    
    2. 为什么不需要单独的Cost Critic？
       → RCPO论文没有单独的Cost Critic，只有一个学习penalized value的Critic
    
    3. λ变化时Critic不会失效吗？
       → 会有一定影响，但论文的三时间尺度设计保证λ变化足够慢
       → λ学习率远小于Critic学习率（论文Assumption 3）
    
    ============================================================================
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        # 基础参数
        lr=BASELINE_LR,
        hidden_dim=BASELINE_HIDDEN_DIM,
        gamma=BASELINE_GAMMA,
        # RCPO参数
        cost_threshold=RCPO_COST_THRESHOLD,
        lambda_lr=RCPO_LAMBDA_LR,
        lambda_max=RCPO_LAMBDA_MAX,
        # 其他
        save_every=0,
        load_model=False,
        save_directory=Path("models/TD3_rcpo_strict"),
        model_name="TD3_rcpo_strict",
        load_directory=Path("models/TD3_rcpo_strict"),
        run_id=None,
    ):
        print("\n" + "="*80)
        print("🚀 TD3 with RCPO (严格遵循论文版本)")
        print("="*80)
        
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        self.gamma = gamma
        self.cost_threshold = cost_threshold
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max
        
        # Lagrange乘子λ（论文Algorithm 1: Initialize λ = 0）
        self.lambda_penalty = torch.tensor([0.0], device=device)
        
        print(f"\n📊 参数配置:")
        print(f"   ─── 基础参数（与baseline一致）───")
        print(f"   γ (discount): {gamma}")
        print(f"   lr (network): {lr}")
        print(f"   hidden_dim: {hidden_dim}")
        print(f"   ─── RCPO参数（论文设置）───")
        print(f"   α (cost threshold): {cost_threshold}")
        print(f"   λ_lr (慢时间尺度): {lambda_lr}")
        print(f"   λ_max: {lambda_max}")
        print(f"\n📖 论文对应:")
        print(f"   Critic学习penalized Q值: Q̂ = E[Σγ^t(r-λc)]")
        print(f"   Actor最大化penalized Q值")
        print(f"   λ基于原始约束更新（每epoch）")
        
        # ===== 网络初始化 =====
        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Penalized Critic（严格RCPO：只有一个Critic学习penalized value）
        self.critic = PenalizedCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = PenalizedCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
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

    def get_action(self, state, add_noise):
        """获取动作"""
        state = torch.Tensor(state).to(self.device)
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
        训练循环 - 严格遵循RCPO论文
        
        ============================================================================
        论文Algorithm 1对应:
        ============================================================================
        Line 7:  R̂_t = r_t - λ·c_t + γV̂(λ,s';v)   # penalized TD target
        Line 8:  v ← v - η·∂(R̂_t - V̂(λ,s;v))²/∂v  # Critic更新
        Line 9:  θ ← θ + η·∇_θV̂(λ,s)              # Actor更新
        ============================================================================
        """
        if discount is None:
            discount = self.gamma
        
        av_Q = 0.0
        max_Q = -inf
        av_loss = 0.0
        av_actor_loss = 0.0
        av_penalized_reward = 0.0
        
        # 获取当前λ
        current_lambda = self.lambda_penalty.item()
        
        for it in range(iterations):
            # 采样
            batch = replay_buffer.sample_batch(batch_size)
            
            state = torch.Tensor(batch['states']).to(self.device)
            next_state = torch.Tensor(batch['next_states']).to(self.device)
            action = torch.Tensor(batch['actions']).to(self.device)
            reward = torch.Tensor(batch['rewards']).to(self.device)
            cost = torch.Tensor(batch['costs']).to(self.device)
            done = torch.Tensor(batch['dones']).to(self.device)
            
            # ===== 论文公式(10): Penalized Reward =====
            # r̂(λ,s,a) = r(s,a) - λ·c(s,a)
            penalized_reward = reward - current_lambda * cost
            av_penalized_reward += penalized_reward.mean().item()
            
            # ===== 论文Line 7: TD Target计算 =====
            with torch.no_grad():
                # 下一状态的动作（TD3目标策略平滑）
                next_action = self.actor_target(next_state)
                noise = torch.randn_like(next_action) * policy_noise
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                
                # Penalized Q target（TD3取min）
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                av_Q += target_Q.mean().item()
                max_Q = max(max_Q, target_Q.max().item())
                
                # TD target: R̂_t = r̂ + γ·Q̂(s',a')
                target_Q = penalized_reward + (1 - done) * discount * target_Q
            
            # ===== 论文Line 8: Critic更新 =====
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            av_loss += critic_loss.item()
            
            # ===== 论文Line 9: Actor更新（延迟更新）=====
            if it % policy_freq == 0:
                # Actor最大化penalized Q值
                actor_action = self.actor(state)
                Q1, _ = self.critic(state, actor_action)
                actor_loss = -Q1.mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                av_actor_loss += actor_loss.item()
                
                # Soft update
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        self.iter_count += 1
        
        # TensorBoard记录
        self.writer.add_scalar("train/critic_loss", av_loss / iterations, self.iter_count)
        num_actor_updates = iterations // policy_freq
        if num_actor_updates > 0:
            self.writer.add_scalar("train/actor_loss", av_actor_loss / num_actor_updates, self.iter_count)
        self.writer.add_scalar("train/avg_Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)
        self.writer.add_scalar("train/avg_penalized_reward", av_penalized_reward / iterations, self.iter_count)
        self.writer.add_scalar("train/lambda", current_lambda, self.iter_count)
        
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)
    
    def update_lambda(self, avg_episode_cost):
        """
        更新Lagrange乘子λ - 论文Line 10
        
        ============================================================================
        论文公式(8): λ_{k+1} = Γ_λ[λ_k + η_1(k)·(E[C(s)] - α)]
        ============================================================================
        
        注意：这里使用原始约束（episode cost的期望），不是penalized cost
        这是RCPO的关键：
        - Critic和Actor使用penalized reward（guiding signal）
        - λ更新使用原始约束（确保约束满足）
        """
        old_lambda = self.lambda_penalty.item()
        
        # 约束违反量 = E[C] - α
        constraint_violation = avg_episode_cost - self.cost_threshold
        
        # 梯度上升更新λ（论文公式8的负梯度形式）
        new_lambda = old_lambda + self.lambda_lr * constraint_violation
        
        # 投影到[0, λ_max]（论文Γ_λ操作）
        new_lambda = np.clip(new_lambda, 0.0, self.lambda_max)
        
        self.lambda_penalty = torch.tensor([new_lambda], device=self.device)
        
        # TensorBoard记录
        self.writer.add_scalar("epoch/lambda", self.lambda_penalty.item(), self.iter_count)
        self.writer.add_scalar("epoch/constraint_violation", constraint_violation, self.iter_count)
        self.writer.add_scalar("epoch/avg_episode_cost", avg_episode_cost, self.iter_count)
        
        print(f"   📊 RCPO λ更新 (论文公式8):")
        print(f"      E[C] = {avg_episode_cost:.2f}, α = {self.cost_threshold}")
        print(f"      约束违反 = {avg_episode_cost:.2f} - {self.cost_threshold} = {constraint_violation:.2f}")
        print(f"      λ: {old_lambda:.6f} → {self.lambda_penalty.item():.6f}")
        
        return constraint_violation

    def save(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_target.state_dict(), f"{directory}/{filename}_actor_target.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth")
        
        torch.save({
            'lambda_penalty': self.lambda_penalty,
        }, f"{directory}/{filename}_rcpo_params.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth", map_location=self.device))
        self.actor_target.load_state_dict(torch.load(f"{directory}/{filename}_actor_target.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth", map_location=self.device))
        self.critic_target.load_state_dict(torch.load(f"{directory}/{filename}_critic_target.pth", map_location=self.device))
        
        rcpo_params = torch.load(f"{directory}/{filename}_rcpo_params.pth", map_location=self.device)
        self.lambda_penalty = rcpo_params['lambda_penalty'].to(self.device)
        
        print(f"✅ Loaded model from: {directory}/{filename}")

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """准备状态（与baseline一致）"""
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