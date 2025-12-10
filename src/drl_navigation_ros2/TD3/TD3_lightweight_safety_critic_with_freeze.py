#!/usr/bin/env python3
"""
TD3 with Safety Critics - 支持加载baseline权重并冻结Task Critic

关键修改：
1. 添加 load_baseline_task_critic() 方法
2. 添加 freeze_task_critic() 方法
3. 修改训练循环，跳过Task Critic的梯度更新
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter


class Actor(nn.Module):
    """轻量级Actor网络"""
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


class SafetyCritic(nn.Module):
    """安全 Critic 网络 - 双Q网络结构（用于安全代价）"""
    def __init__(self, state_dim, action_dim, hidden_dim=26):
        super(SafetyCritic, self).__init__()
        
        # Q1_safe网络
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)
        
        # Q2_safe网络
        self.layer_4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_6 = nn.Linear(hidden_dim, 1)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🔹 Safety Critic(双Q): ({state_dim}+{action_dim}) → {hidden_dim} → {hidden_dim} → 1")
        print(f"   双Q总参数量: {total_params:,}")
        print(f"   ⚡ 输出激活: Softplus (保证非负)")

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        
        q1_safe = F.relu(self.layer_1(sa))
        q1_safe = F.relu(self.layer_2(q1_safe))
        q1_safe = self.layer_3(q1_safe)
        q1_safe = F.softplus(q1_safe)
        
        q2_safe = F.relu(self.layer_4(sa))
        q2_safe = F.relu(self.layer_5(q2_safe))
        q2_safe = self.layer_6(q2_safe)
        q2_safe = F.softplus(q2_safe)
        
        return q1_safe, q2_safe


class TD3_SafetyCritic(object):
    """TD3 with Safety Critics (2+2 架构) + Baseline加载功能"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=1e-4,
        hidden_dim=26,
        lambda_safe=50.0,
        save_every=0,
        load_model=False,
        save_directory=Path("src/drl_navigation_ros2/models/TD3_safety"),
        model_name="TD3_safety",
        load_directory=Path("src/drl_navigation_ros2/models/TD3_safety"),
        run_id=None,
        # ✅ 新增参数
        load_baseline=False,
        baseline_path=None,
        freeze_task_critic=False,
    ):
        print("\n" + "="*80)
        print("🚀 初始化 TD3 with Safety Critics (2+2 架构)")
        if load_baseline:
            print("📦 将从baseline加载Task Critic权重")
        if freeze_task_critic:
            print("❄️  Task Critic将被冻结（不更新）")
        print("="*80)
        
        self.device = device
        self.lambda_safe = lambda_safe
        self.freeze_task_critic = freeze_task_critic
        
        # ===== Actor =====
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=lr)

        # ===== Task Critics =====
        self.task_critic = TaskCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.task_critic_target = TaskCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.task_critic_target.load_state_dict(self.task_critic.state_dict())
        self.task_critic_optimizer = torch.optim.Adam(params=self.task_critic.parameters(), lr=lr)

        # ===== Safety Critics (新增) =====
        self.safety_critic = SafetyCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.safety_critic_target = SafetyCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.safety_critic_target.load_state_dict(self.safety_critic.state_dict())
        self.safety_critic_optimizer = torch.optim.Adam(params=self.safety_critic.parameters(), lr=lr)

        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        
        # ✅ 加载baseline权重（如果指定）
        if load_baseline and baseline_path:
            self.load_baseline_task_critic(baseline_path)
            if freeze_task_critic:
                self.freeze_task_critic_params()
        
        # 计算总参数量
        total_params = (
            sum(p.numel() for p in self.actor.parameters()) +
            sum(p.numel() for p in self.task_critic.parameters()) +
            sum(p.numel() for p in self.safety_critic.parameters())
        )
        print(f"\n✅ 网络总参数量: {total_params:,}")
        print(f"📍 设备: {device}")
        print(f"🎯 隐藏层维度: {hidden_dim}")
        print(f"🛡️  安全权重 λ: {lambda_safe}")
        if freeze_task_critic:
            frozen_params = sum(p.numel() for p in self.task_critic.parameters())
            print(f"❄️  冻结参数量: {frozen_params:,} (Task Critic)")
        print("="*80 + "\n")
        
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

    def load_baseline_task_critic(self, baseline_path):
        """
        从baseline模型加载Task Critic和Actor的权重
        
        Args:
            baseline_path: baseline模型的目录路径
        """
        print("\n" + "="*80)
        print("📦 加载Baseline权重...")
        print("="*80)
        
        baseline_path = Path(baseline_path)
        
        # ✅ 加载Task Critic (从baseline的critic)
        try:
            # Baseline使用的是普通的Critic（没有分Task/Safety）
            # 我们需要加载到Task Critic
            # critic_path = baseline_path / "TD3_lightweight_critic.pth"
            critic_path = baseline_path / "TD3_lightweight_best_critic.pth"
            if not critic_path.exists():
                print(f"❌ 找不到baseline critic: {critic_path}")
                return False
            
            baseline_critic_state = torch.load(critic_path, map_location=self.device)
            
            # 加载到Task Critic
            self.task_critic.load_state_dict(baseline_critic_state)
            self.task_critic_target.load_state_dict(baseline_critic_state)
            
            print(f"✅ Task Critic权重已加载: {critic_path.name}")
            
        except Exception as e:
            print(f"❌ 加载Task Critic失败: {e}")
            return False
        
        # ✅ 加载Actor（可选，但推荐）
        try:
            actor_path = baseline_path / "TD3_lightweight_actor.pth"
            if actor_path.exists():
                baseline_actor_state = torch.load(actor_path, map_location=self.device)
                self.actor.load_state_dict(baseline_actor_state)
                self.actor_target.load_state_dict(baseline_actor_state)
                print(f"✅ Actor权重已加载: {actor_path.name}")
            else:
                print(f"⚠️  Actor权重未找到，使用随机初始化")
        except Exception as e:
            print(f"⚠️  加载Actor失败: {e}, 使用随机初始化")
        
        print("="*80 + "\n")
        return True
    
    def freeze_task_critic_params(self):
        """冻结Task Critic的所有参数"""
        print("❄️  冻结Task Critic参数...")
        
        # 冻结Task Critic
        for param in self.task_critic.parameters():
            param.requires_grad = False
        
        # 冻结Task Critic Target
        for param in self.task_critic_target.parameters():
            param.requires_grad = False
        
        # 验证
        frozen_count = sum(1 for p in self.task_critic.parameters() if not p.requires_grad)
        total_count = sum(1 for p in self.task_critic.parameters())
        
        print(f"   ✅ 已冻结 {frozen_count}/{total_count} 个参数张量")
        print(f"   ✅ Task Critic将不再更新\n")

    def get_action(self, obs, add_noise):
        if add_noise:
            return (
                self.act(obs) + np.random.normal(0, 0.3, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        else:
            return self.act(obs)

    def act(self, state):
        state = torch.Tensor(state).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

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
        """训练循环 - 支持冻结Task Critic"""
        
        av_task_Q = 0.0
        av_safe_Q = 0.0
        max_task_Q = -inf
        av_task_loss = 0.0
        av_safe_loss = 0.0
        av_actor_loss = 0.0
        av_Q_task_component = 0.0
        av_Q_safe_component = 0.0
        
        av_cost_in_batch = 0.0
        max_cost_in_batch = 0.0
        
        for it in range(iterations):
            # 采样 batch（包含 cost）
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_costs,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch_with_cost(batch_size)
            
            state = torch.Tensor(batch_states).to(self.device)
            next_state = torch.Tensor(batch_next_states).to(self.device)
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device)
            cost = torch.Tensor(batch_costs).to(self.device)
            done = torch.Tensor(batch_dones).to(self.device)

            av_cost_in_batch += cost.mean().item()
            max_cost_in_batch = max(max_cost_in_batch, cost.max().item())

            # ===== 计算 Target Q =====
            next_action = self.actor_target(next_state)

            noise = (
                torch.Tensor(batch_actions)
                .data.normal_(0, policy_noise)
                .to(self.device)
            )
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Task Critics Target
            target_Q1_task, target_Q2_task = self.task_critic_target(next_state, next_action)
            target_Q_task = torch.min(target_Q1_task, target_Q2_task)
            av_task_Q += torch.mean(target_Q_task).item()
            max_task_Q = max(max_task_Q, torch.max(target_Q_task).item())
            target_Q_task = reward + ((1 - done) * discount * target_Q_task).detach()

            # Safety Critics Target
            target_Q1_safe, target_Q2_safe = self.safety_critic_target(next_state, next_action)
            target_Q_safe = torch.min(target_Q1_safe, target_Q2_safe)
            av_safe_Q += torch.mean(target_Q_safe).item()
            target_Q_safe = cost + ((1 - done) * discount * target_Q_safe).detach()

            # ===== 更新 Task Critics =====
            # ✅ 只有在未冻结时才更新
            if not self.freeze_task_critic:
                current_Q1_task, current_Q2_task = self.task_critic(state, action)
                task_loss = F.mse_loss(current_Q1_task, target_Q_task) + F.mse_loss(current_Q2_task, target_Q_task)

                self.task_critic_optimizer.zero_grad()
                task_loss.backward()
                self.task_critic_optimizer.step()
                av_task_loss += task_loss.item()
            else:
                # 冻结状态下，仍然计算loss用于记录，但不更新
                with torch.no_grad():
                    current_Q1_task, current_Q2_task = self.task_critic(state, action)
                    task_loss = F.mse_loss(current_Q1_task, target_Q_task) + F.mse_loss(current_Q2_task, target_Q_task)
                    av_task_loss += task_loss.item()

            # ===== 更新 Safety Critics =====
            current_Q1_safe, current_Q2_safe = self.safety_critic(state, action)
            safe_loss = F.mse_loss(current_Q1_safe, target_Q_safe) + F.mse_loss(current_Q2_safe, target_Q_safe)

            self.safety_critic_optimizer.zero_grad()
            safe_loss.backward()
            self.safety_critic_optimizer.step()
            av_safe_loss += safe_loss.item()

            # ===== 更新 Actor =====
            if it % policy_freq == 0:
                actor_action = self.actor(state)
                Q_task, _ = self.task_critic(state, actor_action)
                Q_safe, _ = self.safety_critic(state, actor_action)
                
                av_Q_task_component += Q_task.mean().item()
                av_Q_safe_component += (self.lambda_safe * Q_safe).mean().item()
                
                actor_loss = -Q_task.mean() + self.lambda_safe * Q_safe.mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                av_actor_loss += actor_loss.item()

                # Soft update - Actor
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                
                # Soft update - Task Critics
                # ✅ 只有在未冻结时才soft update
                if not self.freeze_task_critic:
                    for param, target_param in zip(
                        self.task_critic.parameters(), self.task_critic_target.parameters()
                    ):
                        target_param.data.copy_(
                            tau * param.data + (1 - tau) * target_param.data
                        )
                
                # Soft update - Safety Critics
                for param, target_param in zip(
                    self.safety_critic.parameters(), self.safety_critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

        self.iter_count += 1
        
        # ===== TensorBoard 记录 =====
        self.writer.add_scalar("train/task_loss", av_task_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/safe_loss", av_safe_loss / iterations, self.iter_count)
        
        num_actor_updates = iterations // policy_freq
        self.writer.add_scalar("train/actor_loss", av_actor_loss / num_actor_updates, self.iter_count)
        
        self.writer.add_scalar("train/avg_task_Q", av_task_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_safe_Q", av_safe_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_task_Q", max_task_Q, self.iter_count)
        
        self.writer.add_scalar("train/actor_components/Q_task", 
                              av_Q_task_component / num_actor_updates, self.iter_count)
        self.writer.add_scalar("train/actor_components/lambda_Q_safe", 
                              av_Q_safe_component / num_actor_updates, self.iter_count)
        
        self.writer.add_scalar("train/cost_stats/avg_cost_in_batch", 
                              av_cost_in_batch / iterations, self.iter_count)
        self.writer.add_scalar("train/cost_stats/max_cost_in_batch", 
                              max_cost_in_batch, self.iter_count)
        
        # ✅ 记录是否冻结
        self.writer.add_scalar("train/task_critic_frozen", 
                              1.0 if self.freeze_task_critic else 0.0, self.iter_count)
        
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def save(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_target.state_dict(), f"{directory}/{filename}_actor_target.pth")
        torch.save(self.task_critic.state_dict(), f"{directory}/{filename}_task_critic.pth")
        torch.save(self.task_critic_target.state_dict(), f"{directory}/{filename}_task_critic_target.pth")
        torch.save(self.safety_critic.state_dict(), f"{directory}/{filename}_safety_critic.pth")
        torch.save(self.safety_critic_target.state_dict(), f"{directory}/{filename}_safety_critic_target.pth")

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
        self.safety_critic.load_state_dict(
            torch.load(f"{directory}/{filename}_safety_critic.pth", map_location=self.device)
        )
        self.safety_critic_target.load_state_dict(
            torch.load(f"{directory}/{filename}_safety_critic_target.pth", map_location=self.device)
        )
        print(f"Loaded weights from: {directory} to device: {self.device}")

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """与原版相同"""
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