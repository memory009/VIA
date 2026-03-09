from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter


# Baseline hyperparameters — kept identical to TD3_lightweight for fair comparison
BASELINE_GAMMA = 0.99
BASELINE_LR = 1e-4
BASELINE_HIDDEN_DIM = 26        # required for POLAR reachability verification
BASELINE_BATCH_SIZE = 40
BASELINE_BUFFER_SIZE = 500000
BASELINE_TAU = 0.005
BASELINE_POLICY_NOISE = 0.2
BASELINE_NOISE_CLIP = 0.5
BASELINE_POLICY_FREQ = 2

# VIA-specific hyperparameters (per paper settings)
VIA_VAR_LR = 0.1               # beta_u: VaR update step size
VIA_LAMBDA_LR = 0.001          # beta_w: Lagrangian update step size
CVAR_ALPHA = 0.9               # alpha: risk level (focus on worst 10%)
VIA_N_QUANTILES = 128          # M: number of quantiles
VIA_COST_THRESHOLD = 10.0      # b: cost constraint threshold


class Actor(nn.Module):
    """Actor network operating on augmented state s̄ = (s, e_t)."""
    def __init__(self, state_dim, action_dim, hidden_dim=BASELINE_HIDDEN_DIM):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class TaskCritic(nn.Module):
    """Twin-Q task critic (standard TD3 structure)."""
    def __init__(self, state_dim, action_dim, hidden_dim=BASELINE_HIDDEN_DIM):
        super(TaskCritic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)

        self.layer_4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_6 = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)
        q2 = F.relu(self.layer_4(sa))
        q2 = F.relu(self.layer_5(q2))
        q2 = self.layer_6(q2)
        return q1, q2


class VIACostCritic(nn.Module):
    """
    Noncrossing quantile network for cost.
    Enforces q_i = k * phi_i + d to guarantee the non-crossing property.
    """
    def __init__(self, state_dim, action_dim,
                 hidden_dim=BASELINE_HIDDEN_DIM,
                 n_quantiles=VIA_N_QUANTILES):
        super(VIACostCritic, self).__init__()
        self.n_quantiles = n_quantiles

        # tau_k = k/M
        self.register_buffer(
            'tau_hat',
            torch.arange(0, n_quantiles + 1, dtype=torch.float32) / n_quantiles
        )
        # tau_hat_i = (tau_{i-1} + tau_i) / 2
        self.register_buffer(
            'tau',
            (self.tau_hat[:-1] + self.tau_hat[1:]) / 2.0
        )

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, 1)
        self.fc_d = nn.Linear(hidden_dim, 1)
        self.fc_phi = nn.Linear(hidden_dim, n_quantiles)

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        h = F.relu(self.fc1(sa))
        h = F.relu(self.fc2(h))

        k = F.softplus(self.fc_k(h))   # k > 0
        d = self.fc_d(h)

        # phi: softmax + cumsum enforces monotone ordering
        phi_weights = F.softmax(self.fc_phi(h), dim=-1)
        phi = torch.cumsum(phi_weights, dim=-1)

        quantiles = k * phi + d
        quantiles = F.softplus(quantiles)   # cost non-negative
        return quantiles

    def compute_via(self, quantiles, e_t):
        """
        Compute CVaR estimate .
        V̂_C(s̄_t) = sum_i (tau_{i+1} - tau_i) * q_i(s) * I(q_i(s) >= e_t)
        """
        batch_size = quantiles.shape[0]
        weights = (self.tau_hat[1:] - self.tau_hat[:-1]).unsqueeze(0).expand(batch_size, -1)
        indicators = (quantiles >= e_t).float()
        cvar = torch.sum(weights * quantiles * indicators, dim=1, keepdim=True)
        normalizer = torch.clamp(torch.sum(weights * indicators, dim=1, keepdim=True), min=1e-8)
        return cvar / normalizer


class TD3_VIA(object):
    """
    TD3 with VIA (Ablation Study).
    Baseline RL hyperparameters are kept identical to TD3_lightweight;
    VIA-specific parameters follow the paper settings.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=BASELINE_LR,
        hidden_dim=BASELINE_HIDDEN_DIM,
        gamma=BASELINE_GAMMA,
        n_quantiles=VIA_N_QUANTILES,
        via_alpha=CVAR_ALPHA,
        cost_threshold=VIA_COST_THRESHOLD,
        var_lr=VIA_VAR_LR,
        lambda_lr=VIA_LAMBDA_LR,
        save_every=0,
        load_model=False,
        save_directory=Path("models/TD3_VIA"),
        model_name="TD3_VIA",
        load_directory=Path("models/TD3_VIA"),
        run_id=None,
    ):
        self.device = device
        self.original_state_dim = state_dim
        self.augmented_state_dim = state_dim + 1   # s̄ = (s, e_t)
        self.action_dim = action_dim
        self.max_action = max_action

        self.gamma = gamma
        self.via_alpha = via_alpha
        self.cost_threshold = cost_threshold
        self.n_quantiles = n_quantiles
        self.var_lr = var_lr
        self.lambda_lr = lambda_lr

        # VaR parameter u — initialized by the warm-up phase
        self.var_u = torch.tensor([0.0], device=device)
        # Lagrangian multiplier w
        self.lambda_w = torch.tensor([1.0], device=device)

        self.actor = Actor(self.augmented_state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(self.augmented_state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.task_critic = TaskCritic(self.augmented_state_dim, action_dim, hidden_dim).to(device)
        self.task_critic_target = TaskCritic(self.augmented_state_dim, action_dim, hidden_dim).to(device)
        self.task_critic_target.load_state_dict(self.task_critic.state_dict())
        self.task_critic_optimizer = torch.optim.Adam(self.task_critic.parameters(), lr=lr)

        self.cost_critic = VIACostCritic(self.augmented_state_dim, action_dim, hidden_dim, n_quantiles).to(device)
        self.cost_critic_target = VIACostCritic(self.augmented_state_dim, action_dim, hidden_dim, n_quantiles).to(device)
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=lr)

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

    def set_var_u(self, value):
        """Set VaR parameter u (called by warm-up phase with the alpha-quantile of episode costs)."""
        self.var_u = torch.tensor([value], device=self.device)

    def augment_state(self, state, e_t):
        """Construct augmented state s̄ = (s, e_t)."""
        if isinstance(state, (list, np.ndarray)):
            state = torch.FloatTensor(state).to(self.device)
        else:
            state = state.to(self.device)

        if isinstance(e_t, (int, float)):
            e_t = torch.FloatTensor([e_t]).to(self.device)
        elif isinstance(e_t, np.ndarray):
            e_t = torch.FloatTensor(e_t).to(self.device)
        else:
            e_t = e_t.to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)
        if e_t.dim() == 0:
            e_t = e_t.unsqueeze(0).unsqueeze(0)
        elif e_t.dim() == 1:
            e_t = e_t.unsqueeze(1)

        return torch.cat([state, e_t], dim=1)

    def get_action(self, state, e_t, add_noise):
        augmented_state = self.augment_state(state, e_t)
        action = self.actor(augmented_state).cpu().data.numpy().flatten()
        if add_noise:
            noise = np.random.normal(0, BASELINE_POLICY_NOISE, size=self.action_dim)
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action

    def act(self, state, e_t):
        """Deterministic action (evaluation)."""
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
        if discount is None:
            discount = self.gamma

        av_task_Q = 0.0
        av_cost_via = 0.0
        max_task_Q = -inf
        av_task_loss = 0.0
        av_cost_loss = 0.0
        av_actor_loss = 0.0
        av_cost_in_batch = 0.0
        max_cost_in_batch = 0.0
        av_e_t_in_batch = 0.0
        min_e_t_in_batch = inf

        for it in range(iterations):
            batch = replay_buffer.sample_batch_with_augmented_state(batch_size)

            state      = torch.Tensor(batch['states']).to(self.device)
            next_state = torch.Tensor(batch['next_states']).to(self.device)
            action     = torch.Tensor(batch['actions']).to(self.device)
            reward     = torch.Tensor(batch['rewards']).to(self.device)
            cost       = torch.Tensor(batch['costs']).to(self.device)
            done       = torch.Tensor(batch['dones']).to(self.device)
            e_t        = torch.Tensor(batch['e_t']).to(self.device)

            av_cost_in_batch += cost.mean().item()
            max_cost_in_batch = max(max_cost_in_batch, cost.max().item())
            av_e_t_in_batch += e_t.mean().item()
            min_e_t_in_batch = min(min_e_t_in_batch, e_t.min().item())

            with torch.no_grad():
                next_action = self.actor_target(next_state)
                noise = torch.randn_like(next_action) * policy_noise
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                target_Q1, target_Q2 = self.task_critic_target(next_state, next_action)
                target_Q_task = torch.min(target_Q1, target_Q2)
                av_task_Q += target_Q_task.mean().item()
                max_task_Q = max(max_task_Q, target_Q_task.max().item())
                target_Q_task = reward + (1 - done) * discount * target_Q_task

                target_quantiles = self.cost_critic_target(next_state, next_action)
                target_quantiles = cost + (1 - done) * discount * target_quantiles

            # Update task critics
            current_Q1, current_Q2 = self.task_critic(state, action)
            task_loss = F.mse_loss(current_Q1, target_Q_task) + F.mse_loss(current_Q2, target_Q_task)
            self.task_critic_optimizer.zero_grad()
            task_loss.backward()
            self.task_critic_optimizer.step()
            av_task_loss += task_loss.item()

            # Update cost critic (Huber quantile loss)
            current_quantiles = self.cost_critic(state, action)
            cost_loss = self.quantile_huber_loss(current_quantiles, target_quantiles)
            self.cost_critic_optimizer.zero_grad()
            cost_loss.backward()
            self.cost_critic_optimizer.step()
            av_cost_loss += cost_loss.item()

            with torch.no_grad():
                cost_via = self.cost_critic.compute_via(current_quantiles, e_t)
                av_cost_via += cost_via.mean().item()

            if it % policy_freq == 0:
                actor_action = self.actor(state)
                Q_task, _ = self.task_critic(state, actor_action)
                cost_quantiles = self.cost_critic(state, actor_action)
                cost_via = self.cost_critic.compute_via(cost_quantiles, e_t)

                # Actor loss: maximize task Q subject to CVaR cost constraint
                actor_loss = -Q_task.mean() + self.lambda_w.item() * cost_via.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                av_actor_loss += actor_loss.item()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.task_critic.parameters(), self.task_critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.iter_count += 1

        self.writer.add_scalar("train/task_loss", av_task_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/cost_loss", av_cost_loss / iterations, self.iter_count)
        num_actor_updates = iterations // policy_freq
        if num_actor_updates > 0:
            self.writer.add_scalar("train/actor_loss", av_actor_loss / num_actor_updates, self.iter_count)
        self.writer.add_scalar("train/avg_task_Q", av_task_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_cost_via", av_cost_via / iterations, self.iter_count)
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
        """Huber quantile regression loss ."""
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
        return (quantile_weights * huber).mean()

    def update_var_and_lambda(self, avg_episode_cost, epoch_costs=None):
        """
        Update CVaR parameters after each training epoch.

        VaR update :  u^{k+1} = u^k + beta_u * [P(C >= u^k) - (1 - alpha)]
        Lagrangian : w^{k+1} = proj[w^k - beta_w * (b - u^k - V_C(s̄_0))]
        """
        old_var_u = self.var_u.item()
        old_lambda_w = self.lambda_w.item()

        # [Phase 1] VaR update
        if epoch_costs is not None and len(epoch_costs) > 0:
            prob_exceed = np.mean([c >= old_var_u for c in epoch_costs])
            var_update = prob_exceed - (1.0 - self.via_alpha)
            new_var_u = max(old_var_u + self.var_lr * var_update, 0.0)
            self.var_u = torch.tensor([new_var_u], device=self.device)
        else:
            prob_exceed = 0.0
            var_update = 0.0

        # [Phase 2] Update CVaR parameters
        constraint_slack = self.cost_threshold - self.var_u.item() - avg_episode_cost
        new_lambda_w = np.clip(old_lambda_w - self.lambda_lr * constraint_slack, 0.0, 100.0)
        self.lambda_w = torch.tensor([new_lambda_w], device=self.device)

        self.writer.add_scalar("epoch/var_u", self.var_u.item(), self.iter_count)
        self.writer.add_scalar("epoch/lambda_w", self.lambda_w.item(), self.iter_count)
        self.writer.add_scalar("epoch/prob_exceed_var", prob_exceed, self.iter_count)
        self.writer.add_scalar("epoch/var_update", var_update, self.iter_count)
        self.writer.add_scalar("epoch/constraint_slack", constraint_slack, self.iter_count)

        print(f"   [VaR eq.5]  P(C>=u)={prob_exceed:.3f}, target={1-self.via_alpha:.3f}, "
              f"update={var_update:.4f}, u: {old_var_u:.4f} -> {self.var_u.item():.4f}")
        print(f"   [Lambda eq.23]  slack={constraint_slack:.2f}, "
              f"w: {old_lambda_w:.4f} -> {self.lambda_w.item():.4f}")

    def save(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(),               f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_target.state_dict(),        f"{directory}/{filename}_actor_target.pth")
        torch.save(self.task_critic.state_dict(),         f"{directory}/{filename}_task_critic.pth")
        torch.save(self.task_critic_target.state_dict(),  f"{directory}/{filename}_task_critic_target.pth")
        torch.save(self.cost_critic.state_dict(),         f"{directory}/{filename}_cost_critic.pth")
        torch.save(self.cost_critic_target.state_dict(),  f"{directory}/{filename}_cost_critic_target.pth")
        torch.save({'var_u': self.var_u, 'lambda_w': self.lambda_w},
                   f"{directory}/{filename}_via_params.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load(f"{directory}/{filename}_actor.pth", map_location=self.device))
        self.actor_target.load_state_dict(
            torch.load(f"{directory}/{filename}_actor_target.pth", map_location=self.device))
        self.task_critic.load_state_dict(
            torch.load(f"{directory}/{filename}_task_critic.pth", map_location=self.device))
        self.task_critic_target.load_state_dict(
            torch.load(f"{directory}/{filename}_task_critic_target.pth", map_location=self.device))
        self.cost_critic.load_state_dict(
            torch.load(f"{directory}/{filename}_cost_critic.pth", map_location=self.device))
        self.cost_critic_target.load_state_dict(
            torch.load(f"{directory}/{filename}_cost_critic_target.pth", map_location=self.device))

        via_params = torch.load(f"{directory}/{filename}_via_params.pth", map_location=self.device)
        self.var_u = via_params['var_u'].to(self.device)
        self.lambda_w = via_params['lambda_w'].to(self.device)
        print(f"Loaded model from: {directory}/{filename}")

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """Preprocess raw sensor data into a fixed-length state vector (without e_t augmentation)."""
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
