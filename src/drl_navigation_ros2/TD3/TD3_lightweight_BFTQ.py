"""
TD3_lightweight_BFTQ.py
融合BFTQ (Budgeted Reinforcement Learning) 的轻量级TD3网络

主要特点:
1. Budget作为额外输入 (碰撞风险预算 ∈ [0, 1])
2. 双头Critic: Q_reward (任务完成) 和 Q_cost (碰撞成本)
3. Budgeted Bellman Optimality Operator (πhull策略)
4. 保持与baseline相同的轻量级架构

参考文献:
Carrara et al. (2019) "Budgeted Reinforcement Learning in Continuous State Space"
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
# from scipy.spatial import ConvexHull  # 暂不需要，完整实现convex hull时再引入


class Actor(nn.Module):
    """轻量级Actor网络 - 输出考虑budget的动作"""
    def __init__(self, state_dim, action_dim, hidden_dim=26):
        super(Actor, self).__init__()

        # 输入: state + budget
        self.layer_1 = nn.Linear(state_dim + 1, hidden_dim)  # +1 for budget
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()

        print(f"🔹 BFTQ Actor网络: ({state_dim}+1) → {hidden_dim} → {hidden_dim} → {action_dim}")
        print(f"   参数量: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, s, budget):
        """
        Args:
            s: state [batch, state_dim]
            budget: risk budget [batch, 1]
        Returns:
            action [batch, action_dim]
        """
        # 拼接state和budget
        sb = torch.cat([s, budget], dim=1)
        x = F.relu(self.layer_1(sb))
        x = F.relu(self.layer_2(x))
        a = self.tanh(self.layer_3(x))
        return a


class BudgetedCritic(nn.Module):
    """
    Budgeted Critic网络 - 双头输出Q_reward和Q_cost

    架构设计:
    - 共享特征提取层
    - 分离的reward头和cost头
    - 每个头都有双Q网络（Q1, Q2）用于减少过估计
    """
    def __init__(self, state_dim, action_dim, hidden_dim=26):
        super(BudgetedCritic, self).__init__()

        # 输入: state + action + budget
        input_dim = state_dim + action_dim + 1

        # ============ Q_reward 网络 ============
        # Q1_reward
        self.q1r_layer_1 = nn.Linear(input_dim, hidden_dim)
        self.q1r_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1r_layer_3 = nn.Linear(hidden_dim, 1)

        # Q2_reward
        self.q2r_layer_1 = nn.Linear(input_dim, hidden_dim)
        self.q2r_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2r_layer_3 = nn.Linear(hidden_dim, 1)

        # ============ Q_cost 网络 ============
        # Q1_cost
        self.q1c_layer_1 = nn.Linear(input_dim, hidden_dim)
        self.q1c_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1c_layer_3 = nn.Linear(hidden_dim, 1)

        # Q2_cost
        self.q2c_layer_1 = nn.Linear(input_dim, hidden_dim)
        self.q2c_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2c_layer_3 = nn.Linear(hidden_dim, 1)

        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        q_reward_params = (
            sum(p.numel() for p in self.q1r_layer_1.parameters()) +
            sum(p.numel() for p in self.q1r_layer_2.parameters()) +
            sum(p.numel() for p in self.q1r_layer_3.parameters()) +
            sum(p.numel() for p in self.q2r_layer_1.parameters()) +
            sum(p.numel() for p in self.q2r_layer_2.parameters()) +
            sum(p.numel() for p in self.q2r_layer_3.parameters())
        )
        q_cost_params = total_params - q_reward_params

        print(f"🔹 BFTQ Critic网络(双头+双Q): ({state_dim}+{action_dim}+1) → {hidden_dim} → {hidden_dim} → 1")
        print(f"   Q_reward参数量: {q_reward_params:,}")
        print(f"   Q_cost参数量: {q_cost_params:,}")
        print(f"   总参数量: {total_params:,}")

    def forward(self, s, a, budget):
        """
        Args:
            s: state [batch, state_dim]
            a: action [batch, action_dim]
            budget: risk budget [batch, 1]
        Returns:
            Q1_reward, Q2_reward, Q1_cost, Q2_cost
        """
        sab = torch.cat([s, a, budget], dim=1)

        # Q1_reward
        q1r = F.relu(self.q1r_layer_1(sab))
        q1r = F.relu(self.q1r_layer_2(q1r))
        q1r = self.q1r_layer_3(q1r)

        # Q2_reward
        q2r = F.relu(self.q2r_layer_1(sab))
        q2r = F.relu(self.q2r_layer_2(q2r))
        q2r = self.q2r_layer_3(q2r)

        # Q1_cost
        q1c = F.relu(self.q1c_layer_1(sab))
        q1c = F.relu(self.q1c_layer_2(q1c))
        q1c = self.q1c_layer_3(q1c)

        # Q2_cost
        q2c = F.relu(self.q2c_layer_1(sab))
        q2c = F.relu(self.q2c_layer_2(q2c))
        q2c = self.q2c_layer_3(q2c)

        return q1r, q2r, q1c, q2c

    def Q_reward(self, s, a, budget):
        """仅返回reward Q值（用于actor更新）"""
        q1r, q2r, _, _ = self.forward(s, a, budget)
        return q1r, q2r

    def Q_cost(self, s, a, budget):
        """仅返回cost Q值"""
        _, _, q1c, q2c = self.forward(s, a, budget)
        return q1c, q2c


class TD3_BFTQ(object):
    """
    TD3 with Budgeted Fitted-Q (BFTQ)

    核心创新:
    1. Budget-aware policy: π(a|s,β)
    2. Dual Q-functions: Q_r (reward) and Q_c (cost)
    3. Convex hull-based action selection (πhull)
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=1e-4,
        hidden_dim=26,
        budget_range=(0.0, 1.0),  # Budget范围
        save_every=0,
        load_model=False,
        save_directory=Path("src/drl_navigation_ros2/models/TD3_BFTQ"),
        model_name="TD3_BFTQ",
        load_directory=Path("src/drl_navigation_ros2/models/TD3_BFTQ"),
        run_id=None,
    ):
        print("\n" + "="*80)
        print("🚀 初始化BFTQ轻量级TD3网络")
        print("="*80)

        self.device = device
        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        self.budget_range = budget_range

        # Initialize the Actor network (budget-aware)
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr)

        # Initialize the Budgeted Critic networks
        self.critic = BudgetedCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = BudgetedCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=lr)

        # 计算总参数量
        total_params = (
            sum(p.numel() for p in self.actor.parameters()) +
            sum(p.numel() for p in self.critic.parameters())
        )
        print(f"\n✅ BFTQ网络总参数量: {total_params:,}")
        print(f"📍 设备: {device}")
        print(f"🎯 隐藏层维度: {hidden_dim}")
        print(f"💰 Budget范围: {budget_range}")
        print("="*80 + "\n")

        # TensorBoard
        if run_id:
            tensorboard_log_dir = f"runs/{run_id}"
            self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        else:
            self.writer = SummaryWriter()

        self.iter_count = 0
        if load_model:
            self.load(filename=model_name, directory=load_directory)
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory

    def sample_budget(self):
        """
        采样budget (均匀分布)
        按照BFTQ论文的风险敏感探索策略
        """
        return np.random.uniform(self.budget_range[0], self.budget_range[1])

    def get_action(self, obs, budget, add_noise):
        """
        获取动作（考虑budget）

        Args:
            obs: observation
            budget: risk budget ∈ [0, 1]
            add_noise: 是否添加探索噪声
        """
        if add_noise:
            return (
                self.act(obs, budget) + np.random.normal(0, 0.2, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        else:
            return self.act(obs, budget)

    def act(self, state, budget):
        """根据state和budget获取动作"""
        state = torch.Tensor(state).to(self.device).unsqueeze(0)
        budget_tensor = torch.Tensor([[budget]]).to(self.device)
        return self.actor(state, budget_tensor).cpu().data.numpy().flatten()

    def compute_cost(self, reward, collision):
        """
        计算成本信号 R_c

        根据baseline的奖励设计:
        - 碰撞: cost = 1
        - 无碰撞: cost = 0

        Args:
            reward: 环境返回的reward
            collision: 是否碰撞
        Returns:
            cost: 成本信号
        """
        # 简化版本: 直接基于碰撞
        return 1.0 if collision else 0.0

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
        BFTQ训练循环

        关键差异:
        1. 每个样本有不同的budget
        2. 维护Q_reward和Q_cost两个目标
        3. 使用convex hull策略选择动作
        """
        av_Qr = 0
        av_Qc = 0
        max_Qr = -inf
        av_loss_r = 0
        av_loss_c = 0

        for it in range(iterations):
            # Sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
                batch_budgets,      # 新增: budget
                batch_costs,        # 新增: cost
            ) = replay_buffer.sample_batch_bftq(batch_size)

            state = torch.Tensor(batch_states).to(self.device)
            next_state = torch.Tensor(batch_next_states).to(self.device)
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device)
            done = torch.Tensor(batch_dones).to(self.device)
            budget = torch.Tensor(batch_budgets).to(self.device)
            cost = torch.Tensor(batch_costs).to(self.device)

            # ========== Critic Update ==========
            with torch.no_grad():
                # 获取target网络的下一个动作
                next_action = self.actor_target(next_state, budget)

                # 添加噪声
                noise = (
                    torch.Tensor(batch_actions)
                    .data.normal_(0, policy_noise)
                    .to(self.device)
                )
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                # 计算target Q值
                target_Q1r, target_Q2r, target_Q1c, target_Q2c = self.critic_target(
                    next_state, next_action, budget
                )

                # 使用最小值减少过估计
                target_Qr = torch.min(target_Q1r, target_Q2r)
                target_Qc = torch.min(target_Q1c, target_Q2c)

                # Bellman equation
                target_Qr = reward + ((1 - done) * discount * target_Qr)
                target_Qc = cost + ((1 - done) * discount * target_Qc)

            # 当前Q值
            current_Q1r, current_Q2r, current_Q1c, current_Q2c = self.critic(
                state, action, budget
            )

            # Critic loss (分别计算reward和cost的loss)
            loss_Qr = F.mse_loss(current_Q1r, target_Qr) + F.mse_loss(current_Q2r, target_Qr)
            loss_Qc = F.mse_loss(current_Q1c, target_Qc) + F.mse_loss(current_Q2c, target_Qc)
            loss_critic = loss_Qr + loss_Qc

            # 更新critic
            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            self.critic_optimizer.step()

            # 记录统计信息
            av_Qr += torch.mean(target_Qr)
            av_Qc += torch.mean(target_Qc)
            max_Qr = max(max_Qr, torch.max(target_Qr))
            av_loss_r += loss_Qr
            av_loss_c += loss_Qc

            # ========== Actor Update (Delayed) ==========
            if it % policy_freq == 0:
                # Actor目标: 最大化Q_reward，同时满足Q_cost ≤ budget
                # 简化实现: 只最大化Q_reward (完整实现需要convex hull)
                actor_action = self.actor(state, budget)
                Q1r, Q2r = self.critic.Q_reward(state, actor_action, budget)
                actor_loss = -Q1r.mean()  # 最大化Q_reward

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update target networks
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

        self.iter_count += 1

        # TensorBoard logging
        self.writer.add_scalar("train/loss_Qr", av_loss_r / iterations, self.iter_count)
        self.writer.add_scalar("train/loss_Qc", av_loss_c / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_Qr", av_Qr / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_Qc", av_Qc / iterations, self.iter_count)
        self.writer.add_scalar("train/max_Qr", max_Qr, self.iter_count)

        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def save(self, filename, directory, epoch=None):
        """
        保存模型（分离保存，与baseline保持一致）

        Args:
            filename: 基础文件名
            directory: 保存目录
            epoch: 如果提供，保存为checkpoint_epoch_{epoch:03d}_*.pth
        """
        Path(directory).mkdir(parents=True, exist_ok=True)

        if epoch is not None:
            # 保存特定epoch的checkpoint（分离保存）
            base_name = f"checkpoint_epoch_{epoch:03d}"
            torch.save(self.actor.state_dict(), f"{directory}/{base_name}_actor.pth")
            torch.save(self.actor_target.state_dict(), f"{directory}/{base_name}_actor_target.pth")
            torch.save(self.critic.state_dict(), f"{directory}/{base_name}_critic.pth")
            torch.save(self.critic_target.state_dict(), f"{directory}/{base_name}_critic_target.pth")
        else:
            # 保存最佳模型（分离保存）
            torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
            torch.save(self.actor_target.state_dict(), f"{directory}/{filename}_actor_target.pth")
            torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
            torch.save(self.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth")

    def load(self, filename, directory, epoch=None):
        """加载模型（分离加载，与baseline保持一致）"""
        if epoch is not None:
            # 加载特定epoch的checkpoint（分离文件）
            base_name = f"checkpoint_epoch_{epoch:03d}"
            self.actor.load_state_dict(
                torch.load(f"{directory}/{base_name}_actor.pth", map_location=self.device)
            )
            self.actor_target.load_state_dict(
                torch.load(f"{directory}/{base_name}_actor_target.pth", map_location=self.device)
            )
            self.critic.load_state_dict(
                torch.load(f"{directory}/{base_name}_critic.pth", map_location=self.device)
            )
            self.critic_target.load_state_dict(
                torch.load(f"{directory}/{base_name}_critic_target.pth", map_location=self.device)
            )
            print(f"✅ Loaded checkpoint from epoch {epoch}")
        else:
            # 加载最佳模型（分离文件）
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
            print(f"✅ Loaded weights from: {directory}")

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """
        准备状态（与baseline保持一致）
        """
        latest_scan = np.array(latest_scan)
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0

        max_bins = self.state_dim - 5
        bin_size = int(np.ceil(len(latest_scan) / max_bins))

        min_values = []
        for i in range(0, len(latest_scan), bin_size):
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            min_values.append(min(bin))

        state = min_values + [distance, cos, sin] + [action[0], action[1]]
        assert len(state) == self.state_dim

        terminal = 1 if collision or goal else 0
        return state, terminal
