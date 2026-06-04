import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


# ============================================================================
# Loss / Penalty Functions
# ============================================================================

def zero_mean_loss(actions: torch.Tensor) -> torch.Tensor:
    """Penalize deviations from zero mean across the action vector."""
    return torch.mean(actions.mean(dim=-1) ** 2)


def spatial_smoothness_loss_2d(actions: torch.Tensor,
                               agent_grid_nx: int,
                               agent_grid_ny: int) -> torch.Tensor:
    """
    Penalize large differences between adjacent agents on a 2D grid.
    actions: (B, n_agents) scalar actions.
    Reshapes to (B, gx, gy) and computes finite differences in both
    directions with periodic wrapping.
    """
    grid = actions.view(-1, agent_grid_nx, agent_grid_ny)
    dx = grid - torch.roll(grid, 1, dims=1)
    dy = grid - torch.roll(grid, 1, dims=2)
    return torch.mean(dx ** 2 + dy ** 2)


def temporal_smoothness_loss(actions: torch.Tensor,
                             prev_actions: torch.Tensor) -> torch.Tensor:
    """Penalize rapid changes from previous actions."""
    return torch.mean((actions - prev_actions) ** 2)


def action_energy_loss(actions: torch.Tensor) -> torch.Tensor:
    """Penalize large action magnitudes."""
    return torch.mean(actions ** 2)


# ============================================================================
# Recurrent Actor (GRU-based)
# ============================================================================

class RecurrentPatchCNNActor(nn.Module):
    """
    GRU-augmented per-agent actor for channel flow patch control.
    Observation: (C, obs_extent, obs_extent) from 3x3 ring of patches.
    Output: scalar action in [-1, 1].
    """
    def __init__(self, in_channels=3, obs_extent=12, conv_channels=None, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        if conv_channels is None:
            conv_channels = [16, 32]

        layers = []
        in_ch = in_channels
        for out_ch in conv_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(4, out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # After 2 pools: obs_extent/4 in each dim
        n_pools = len(conv_channels)
        spatial = obs_extent // (2 ** n_pools)
        flat_size = conv_channels[-1] * spatial * spatial
        self.feature_proj = nn.Linear(flat_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            gain = 0.01 if isinstance(module, nn.Linear) and module.out_features == 1 else 1.0
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _encode(self, obs):
        """(B, C, H, W) -> (B, hidden_size)."""
        features = self.conv(obs)
        return self.feature_proj(features.reshape(features.size(0), -1))

    def forward(self, obs, hidden=None):
        """
        obs: (B, C, H, W) single step or (B, T, C, H, W) sequence.
        Returns: (actions, new_hidden).
        """
        self.gru.flatten_parameters()
        if obs.dim() == 4:
            proj = self._encode(obs).unsqueeze(1)
            gru_out, hidden = self.gru(proj, hidden)
            return torch.tanh(self.fc(self.gru_norm(gru_out.squeeze(1)))), hidden
        else:
            B, T = obs.shape[:2]
            proj = self._encode(obs.reshape(B * T, *obs.shape[2:])).view(B, T, -1)
            gru_out, hidden = self.gru(proj, hidden)
            return torch.tanh(self.fc(self.gru_norm(gru_out))), hidden


# ============================================================================
# Feedforward Actor (fallback)
# ============================================================================

class PatchCNNActor(nn.Module):
    """Feedforward per-agent actor. Same CNN encoder, no GRU."""
    def __init__(self, in_channels=3, obs_extent=12, conv_channels=None):
        super().__init__()
        if conv_channels is None:
            conv_channels = [16, 32]

        layers = []
        in_ch = in_channels
        for out_ch in conv_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(4, out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        n_pools = len(conv_channels)
        spatial = obs_extent // (2 ** n_pools)
        flat_size = conv_channels[-1] * spatial * spatial
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            gain = 0.01 if isinstance(module, nn.Linear) and module.out_features == 1 else 1.0
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs):
        x = self.conv(obs)
        x = x.reshape(x.size(0), -1)
        return torch.tanh(self.fc(x))


# ============================================================================
# Centralized Critic
# ============================================================================

class SharedCritic(nn.Module):
    """
    Centralized critic for MADDPG.
    Input: (n_obs_ch, nx, ny) full observation fields + (1, nx, ny) action field.
    Auto-scales pooling for different grid sizes (64 or 192).
    """
    def __init__(self, state_channels, n_agents, agent_grid_nx, agent_grid_ny,
                 patch_size, nx, ny, conv_channels=None, mlp_layers=None):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 32]
        if mlp_layers is None:
            mlp_layers = [256, 128]

        self.n_agents = n_agents
        self.agent_grid_nx = agent_grid_nx
        self.agent_grid_ny = agent_grid_ny
        self.patch_size = patch_size
        self.nx = nx
        self.ny = ny

        in_ch = state_channels + 1  # obs channels + action field
        layers = []
        for out_ch in conv_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(4, out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        final_h = nx // (2 ** len(conv_channels))
        final_w = ny // (2 ** len(conv_channels))
        conv_flat = conv_channels[-1] * final_h * final_w

        mlp = [nn.LayerNorm(conv_flat)]
        prev = conv_flat
        for h in mlp_layers:
            mlp.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()])
            prev = h
        self.mlp = nn.Sequential(*mlp)
        self.output = nn.Linear(prev, 1)

    def _action_to_field(self, actions):
        """Convert (B, n_agents) scalar actions to (B, 1, nx, ny) field."""
        # (B, n_agents) -> (B, 1, gx, gy) -> repeat each value over patch_size x patch_size
        grid = actions.view(-1, 1, self.agent_grid_nx, self.agent_grid_ny)
        return grid.repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3)

    def forward(self, state_fields, actions):
        """
        state_fields: (B, state_channels, nx, ny)
        actions: (B, n_agents) scalar actions
        """
        act_field = self._action_to_field(actions)
        x = torch.cat([state_fields, act_field], dim=1)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return self.output(x)


# ============================================================================
# MADDPG Policy with GRU support
# ============================================================================

class SharedPolicyMADDPG:
    """
    Multi-Agent DDPG with shared policy for turbulent channel flow control.
    Supports both feedforward and GRU-recurrent actor.
    """
    def __init__(self, config, device='cpu'):
        self.device = device
        self.config = config
        mc = config['maddpg']
        gc = config['grid']
        cc = config.get('control', {})
        oc = config.get('observation', {})

        self.nx = gc['nx']
        self.ny = gc['ny']
        self.patch_size = cc.get('patch_size', 4)
        self.agent_grid_nx = self.nx // self.patch_size
        self.agent_grid_ny = self.ny // self.patch_size
        self.n_agents = self.agent_grid_nx * self.agent_grid_ny

        ring_radius = cc.get('ring_radius', 1)
        self.obs_extent = (2 * ring_radius + 1) * self.patch_size
        self.include_prev_action = oc.get('include_prev_action', True)
        self.in_channels = 3 if self.include_prev_action else 2

        rc = config.get('recurrence', {})
        self.recurrent = rc.get('enabled', False)
        self.hidden_size = rc.get('hidden_size', 64)
        self.seq_len = rc.get('sequence_length', 16)
        self.burn_in = rc.get('burn_in_length', 4)

        self.gamma = mc.get('gamma', 0.995)
        self.tau = mc.get('tau', 0.005)
        self.gradient_clip = mc.get('gradient_clip', 0.5)
        self.lambda_temporal = mc.get('lambda_temporal', 0.025)
        self.lambda_spatial = mc.get('lambda_spatial', 0.05)
        self.lambda_zero = mc.get('lambda_zero', 2.0)
        self.lambda_energy = mc.get('lambda_energy', 0.1)
        self.reg_target_ratio = mc.get('reg_target_ratio', 0.05)
        self.diff_zero_mean = mc.get('differentiable_zero_mean', True)

        actor_ch = mc.get('actor_channels', [16, 32])
        critic_conv = mc.get('critic_conv', [32, 64, 32])
        critic_mlp = mc.get('critic_mlp', [256, 128])

        # Actor
        if self.recurrent:
            self.actor = RecurrentPatchCNNActor(
                self.in_channels, self.obs_extent, actor_ch, self.hidden_size
            ).to(device)
        else:
            self.actor = PatchCNNActor(
                self.in_channels, self.obs_extent, actor_ch
            ).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        # Critic
        self.critic = SharedCritic(
            self.in_channels, self.n_agents,
            self.agent_grid_nx, self.agent_grid_ny,
            self.patch_size, self.nx, self.ny,
            critic_conv, critic_mlp
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(mc.get('actor_lr', 0.0005))
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(mc.get('critic_lr', 0.0001))
        )

        self.use_amp = (device != 'cpu')
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

    def reset_hidden(self, batch_size=None):
        if not self.recurrent:
            return None
        n = batch_size if batch_size is not None else self.n_agents
        return torch.zeros(1, n, self.hidden_size, device=self.device)

    def select_actions(self, all_obs, hidden=None):
        """
        all_obs: (n_agents, C, H, W) tensor.
        Returns: (n_agents,) actions array, new_hidden.
        """
        with torch.no_grad():
            if self.recurrent:
                actions, hidden = self.actor(all_obs, hidden)
                return actions.squeeze(-1).cpu().numpy(), hidden
            else:
                actions = self.actor(all_obs)
                return actions.squeeze(-1).cpu().numpy(), None

    def update(self, batch):
        if self.recurrent:
            return self._update_recurrent(batch)
        return self._update_feedforward(batch)

    def _update_feedforward(self, batch):
        obs = batch['obs']        # (B, n_agents, C, H, W)
        acts = batch['acts'].squeeze(-1)      # (B, n_agents)
        prev_acts = batch['prev_acts'].squeeze(-1)  # (B, n_agents)
        rews = batch['rews']      # (B, n_agents)
        next_obs = batch['next_obs']
        done = batch['done']      # (B, n_agents)
        B = obs.size(0)
        amp_dtype = torch.float16 if self.use_amp else torch.float32

        # --- Critic update ---
        with torch.no_grad():
            next_obs_flat = next_obs.reshape(B * self.n_agents, *next_obs.shape[2:])
            next_acts_flat = self.actor_target(next_obs_flat)
            next_acts = next_acts_flat.view(B, self.n_agents)

            next_state = self._obs_to_full_state(next_obs)
            target_q = self.critic_target(next_state, next_acts)
            mean_rew = rews.mean(dim=1, keepdim=True)
            done_flag = done[:, 0:1]
            target_val = mean_rew + self.gamma * (1.0 - done_flag) * target_q

        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=amp_dtype):
            state = self._obs_to_full_state(obs)
            current_q = self.critic(state, acts)
            critic_loss = F.mse_loss(current_q, target_val)

        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.scaler.step(self.critic_optimizer)

        # --- Actor update ---
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=amp_dtype):
            obs_flat = obs.reshape(B * self.n_agents, *obs.shape[2:])
            new_acts_raw = self.actor(obs_flat).view(B, self.n_agents)

            if self.diff_zero_mean:
                new_acts = new_acts_raw - new_acts_raw.mean(dim=1, keepdim=True)
                z_loss = zero_mean_loss(new_acts_raw)
            else:
                new_acts = new_acts_raw
                z_loss = zero_mean_loss(new_acts)

            state_new = self._obs_to_full_state(obs)
            q_loss = -self.critic(state_new, new_acts).mean()
            t_loss = temporal_smoothness_loss(new_acts, prev_acts)
            s_loss = spatial_smoothness_loss_2d(new_acts, self.agent_grid_nx, self.agent_grid_ny)
            e_loss = action_energy_loss(new_acts)

            actor_loss = (q_loss
                          + self.lambda_temporal * t_loss
                          + self.lambda_spatial * s_loss
                          + self.lambda_zero * z_loss
                          + self.lambda_energy * e_loss)

        self.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        self.scaler.step(self.actor_optimizer)

        self.scaler.update()
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item(), {
            'q_loss': q_loss.item(), 'temporal': t_loss.item(),
            'spatial': s_loss.item(), 'zero_mean': z_loss.item(),
            'energy': e_loss.item(),
        }

    def _update_recurrent(self, batch):
        obs = batch['obs']          # (B, T, n_agents, C, H, W)
        acts = batch['acts'].squeeze(-1)        # (B, T, n_agents)
        prev_acts = batch['prev_acts'].squeeze(-1)
        rews = batch['rews']        # (B, T, n_agents)
        next_obs = batch['next_obs']
        done = batch['done']        # (B, T, n_agents)
        mask = batch['mask']        # (B, T)
        B, T = obs.shape[:2]
        amp_dtype = torch.float16 if self.use_amp else torch.float32

        # Collapse B*T for critic
        obs_cr = obs.reshape(B * T, self.n_agents, *obs.shape[3:])
        acts_cr = acts.reshape(B * T, self.n_agents)
        next_obs_cr = next_obs.reshape(B * T, self.n_agents, *next_obs.shape[3:])
        rews_cr = rews.reshape(B * T, self.n_agents)
        done_cr = done.reshape(B * T, self.n_agents)
        mask_flat = mask.reshape(B * T)

        # --- Critic update ---
        with torch.no_grad():
            nobs_flat = next_obs_cr.reshape(B * T * self.n_agents, *next_obs.shape[3:])
            nacts_flat, _ = self.actor_target(nobs_flat)
            next_acts = nacts_flat.view(B * T, self.n_agents)

            next_state = self._obs_to_full_state(next_obs_cr)
            target_q = self.critic_target(next_state, next_acts)
            mean_rew = rews_cr.mean(dim=1, keepdim=True)
            done_flag = done_cr[:, 0:1]
            target_val = mean_rew + self.gamma * (1.0 - done_flag) * target_q

        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=amp_dtype):
            state = self._obs_to_full_state(obs_cr)
            current_q = self.critic(state, acts_cr)
            critic_loss = ((current_q - target_val) ** 2 * mask_flat.unsqueeze(1)).sum() / mask_flat.sum().clamp(min=1)

        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.scaler.step(self.critic_optimizer)

        # --- Actor update with GRU ---
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=amp_dtype):
            # Reshape for per-agent GRU: (B*n_agents, T, C, H, W)
            obs_ag = obs.permute(0, 2, 1, 3, 4, 5).reshape(
                B * self.n_agents, T, *obs.shape[3:]
            )
            new_acts_ag, _ = self.actor(obs_ag, None)
            # new_acts_ag: (B*n_agents, T, 1) -> (B, n_agents, T) -> (B, T, n_agents)
            new_acts_raw = new_acts_ag.squeeze(-1).view(B, self.n_agents, T).permute(0, 2, 1)

            # Burn-in masking
            actor_mask = mask.clone()
            actor_mask[:, :self.burn_in] = 0.0
            valid = actor_mask.sum().clamp(min=1)

            if self.diff_zero_mean:
                new_acts = new_acts_raw - new_acts_raw.mean(dim=2, keepdim=True)
                z_loss = zero_mean_loss(new_acts_raw.reshape(B * T, self.n_agents))
            else:
                new_acts = new_acts_raw
                z_loss = zero_mean_loss(new_acts.reshape(B * T, self.n_agents))

            new_acts_bt = new_acts.reshape(B * T, self.n_agents)
            state_new = self._obs_to_full_state(obs_cr).detach()
            q_per_step = -self.critic(state_new, new_acts_bt).reshape(B, T)
            q_loss = (q_per_step * actor_mask).sum() / valid

            # Temporal loss along the sequence axis: penalizes the new policy's
            # own consecutive changes, not changes vs. old buffer actions.
            t_loss = torch.mean((new_acts[:, 1:, :] - new_acts[:, :-1, :]) ** 2)
            s_loss = spatial_smoothness_loss_2d(
                new_acts.reshape(B * T, self.n_agents),
                self.agent_grid_nx, self.agent_grid_ny
            )
            e_loss = action_energy_loss(new_acts.reshape(B * T, self.n_agents))

            actor_loss = (q_loss
                          + self.lambda_temporal * t_loss
                          + self.lambda_spatial * s_loss
                          + self.lambda_zero * z_loss
                          + self.lambda_energy * e_loss)

        self.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        self.scaler.step(self.actor_optimizer)

        self.scaler.update()
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item(), {
            'q_loss': q_loss.item(), 'temporal': t_loss.item(),
            'spatial': s_loss.item(), 'zero_mean': z_loss.item(),
            'energy': e_loss.item(),
        }

    def _obs_to_full_state(self, obs):
        """
        Reconstruct full (B, C, nx, ny) state from per-agent ring observations.
        obs: (B, n_agents, C, H, W) — extract center patch from each agent's ring.
        """
        ps = self.patch_size
        r = (self.obs_extent - ps) // 2  # halo size = ring_radius * patch_size

        # Extract all center patches at once: (B, n_agents, C, ps, ps)
        patches = obs[:, :, :, r:r + ps, r:r + ps]
        # Reshape to grid: (B, gx, gy, C, ps, ps) -> (B, C, gx, ps, gy, ps) -> (B, C, nx, ny)
        B, _, C = patches.shape[:3]
        return (patches
                .view(B, self.agent_grid_nx, self.agent_grid_ny, C, ps, ps)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(B, C, self.nx, self.ny))

    def _soft_update(self, source, target):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd['actor'])
        self.critic.load_state_dict(sd['critic'])
        self.actor_target.load_state_dict(sd['actor_target'])
        self.critic_target.load_state_dict(sd['critic_target'])
        self.actor_optimizer.load_state_dict(sd['actor_optimizer'])
        self.critic_optimizer.load_state_dict(sd['critic_optimizer'])
