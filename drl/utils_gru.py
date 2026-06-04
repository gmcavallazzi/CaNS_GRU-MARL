import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _fig_to_numpy(fig):
    """Convert matplotlib figure to numpy RGB array (HWC format for TensorBoard)."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    image = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return image


def plot_snapshot_grid(snapshots, episode=0):
    """
    3-row x N-col grid: rows = (u, w, wall actuation), columns = time snapshots.
    Each snapshot dict has keys: 'u', 'w', 'action_field', 'step'.
    """
    n_cols = len(snapshots)
    if n_cols == 0:
        return None

    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 12))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    row_labels = ['u-velocity', 'w-velocity', 'Wall actuation']
    field_keys = ['u', 'w', 'action_field']
    cmaps = ['RdBu_r', 'RdBu_r', 'seismic']

    # Compute shared color limits per row across all snapshots
    row_limits = []
    for key in field_keys:
        all_vals = np.concatenate([snap[key].ravel() for snap in snapshots])
        vmax = max(np.abs(all_vals).max(), 1e-6)
        row_limits.append((-vmax, vmax))

    for col, snap in enumerate(snapshots):
        for row, (key, cmap) in enumerate(zip(field_keys, cmaps)):
            ax = axes[row, col]
            vmin, vmax = row_limits[row]
            im = ax.imshow(snap[key].T, origin='lower', cmap=cmap,
                           vmin=vmin, vmax=vmax, aspect='equal')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(f"step {snap['step']}", fontsize=11)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f'Episode {episode}', fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return _fig_to_numpy(fig)


def plot_action_heatmap(action_history, agent_grid_nx, agent_grid_ny, episode=0):
    """
    Heatmap of actions over time within an episode.
    action_history: list of (n_agents,) arrays, one per step.
    Shows the 2D action field at 3 time points + the time-averaged field.
    """
    if len(action_history) < 3:
        return None

    data = np.array(action_history)  # (T, n_agents)
    T = data.shape[0]
    indices = [0, T // 2, T - 1]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    vmax = max(np.abs(data).max(), 1e-6)

    for i, idx in enumerate(indices):
        field = data[idx].reshape(agent_grid_nx, agent_grid_ny)
        im = axes[i].imshow(field.T, origin='lower', cmap='seismic',
                            vmin=-vmax, vmax=vmax, aspect='equal')
        axes[i].set_title(f'step {idx}')
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    # Time-averaged
    avg_field = data.mean(axis=0).reshape(agent_grid_nx, agent_grid_ny)
    vmax_avg = max(np.abs(avg_field).max(), 1e-6)
    im = axes[3].imshow(avg_field.T, origin='lower', cmap='seismic',
                        vmin=-vmax_avg, vmax=vmax_avg, aspect='equal')
    axes[3].set_title('time-averaged')
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    fig.suptitle(f'Action fields — Episode {episode}', fontsize=12)
    plt.tight_layout()
    return _fig_to_numpy(fig)


def plot_agent_action_history(action_history, agent_grid_nx, agent_grid_ny,
                              episode=0, agent_idx=None):
    """
    1D time series of a random agent's action, plus its 4 cardinal neighbors.
    action_history: list of (n_agents,) arrays.
    """
    if len(action_history) < 2:
        return None

    data = np.array(action_history)  # (T, n_agents)
    T, n_agents = data.shape
    gx, gy = agent_grid_nx, agent_grid_ny

    # Pick a random agent near the center (avoid edges for cleaner neighbor display)
    if agent_idx is None:
        agent_idx = np.random.randint(gx * gy)
    ix = agent_idx // gy
    iy = agent_idx % gy

    # Collect center + 4 cardinal neighbors (periodic)
    neighbors = {
        f'agent ({ix},{iy}) [center]': agent_idx,
        f'agent ({(ix-1)%gx},{iy}) [up]': ((ix - 1) % gx) * gy + iy,
        f'agent ({(ix+1)%gx},{iy}) [down]': ((ix + 1) % gx) * gy + iy,
        f'agent ({ix},{(iy-1)%gy}) [left]': ix * gy + (iy - 1) % gy,
        f'agent ({ix},{(iy+1)%gy}) [right]': ix * gy + (iy + 1) % gy,
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})

    steps = np.arange(T)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for (label, idx), color in zip(neighbors.items(), colors):
        lw = 2.0 if 'center' in label else 0.8
        alpha = 1.0 if 'center' in label else 0.6
        axes[0].plot(steps, data[:, idx], label=label, color=color,
                     linewidth=lw, alpha=alpha)

    axes[0].set_ylabel('Raw action [-1, 1]')
    axes[0].set_title(f'Agent action history — Episode {episode}')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].set_xlim(0, T - 1)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: temporal change |a(t) - a(t-1)| for center agent
    center_data = data[:, agent_idx]
    temporal_change = np.abs(np.diff(center_data))
    axes[1].fill_between(steps[1:], 0, temporal_change, alpha=0.4, color='#1f77b4')
    axes[1].plot(steps[1:], temporal_change, color='#1f77b4', linewidth=0.8)
    axes[1].set_xlabel('DRL step')
    axes[1].set_ylabel('|Δa|')
    axes[1].set_title('Temporal change (center agent)')
    axes[1].set_xlim(0, T - 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return _fig_to_numpy(fig)


def log_episode_to_tensorboard(writer, episode_num, episode_data, config=None):
    """Centralized TensorBoard logging for one episode."""
    d = episode_data

    # -- Scalars: Reward & Physics --
    writer.add_scalar('Reward/episode', d['reward'], episode_num)
    writer.add_scalar('Physics/drag_reduction_pct', d.get('drag_reduction_pct', 0), episode_num)
    writer.add_scalar('Physics/avg_dpdx', d.get('avg_dpdx', 0), episode_num)

    # -- Scalars: Losses --
    tc = d.get('train_count', 0)
    if tc > 0:
        writer.add_scalar('Loss/critic', d['critic_loss'] / tc, episode_num)
        writer.add_scalar('Loss/actor_total', d['actor_loss'] / tc, episode_num)
        breakdown = d.get('loss_breakdown', {})
        for key, val in breakdown.items():
            writer.add_scalar(f'Loss/{key}', val / tc, episode_num)

    # -- Scalars: Action statistics --
    action_history = d.get('action_history', [])
    if action_history:
        last_actions = action_history[-1]
        writer.add_scalar('Action/mean', float(np.mean(last_actions)), episode_num)
        writer.add_scalar('Action/std', float(np.std(last_actions)), episode_num)
        writer.add_scalar('Action/abs_max', float(np.abs(last_actions).max()), episode_num)
        writer.add_scalar('Control/actuation_energy',
                          float(np.mean(np.array(action_history) ** 2)), episode_num)

        if len(action_history) > 1:
            diffs = np.diff(np.array(action_history), axis=0)
            writer.add_scalar('Action/temporal_change',
                              float(np.mean(diffs ** 2)), episode_num)

    # -- Scalars: Exploration & GRU --
    if 'noise_sigma' in d:
        writer.add_scalar('Exploration/noise_sigma', d['noise_sigma'], episode_num)
    if 'hidden_norm' in d:
        writer.add_scalar('GRU/hidden_norm', d['hidden_norm'], episode_num)

    # -- Images: Snapshots (u, w, action at 3 time points) --
    snapshots = d.get('snapshots', [])
    if len(snapshots) >= 2:
        try:
            img = plot_snapshot_grid(snapshots, episode_num)
            if img is not None:
                writer.add_image('Snapshots/fields', img, episode_num, dataformats='HWC')
        except Exception as e:
            print(f"Warning: snapshot viz failed: {e}")

    # -- Images: Action field heatmaps --
    agent_grid_nx = d.get('agent_grid_nx', 16)
    agent_grid_ny = d.get('agent_grid_ny', 16)

    if action_history and len(action_history) >= 3:
        try:
            img = plot_action_heatmap(action_history, agent_grid_nx, agent_grid_ny, episode_num)
            if img is not None:
                writer.add_image('Control/action_heatmap', img, episode_num, dataformats='HWC')
        except Exception as e:
            print(f"Warning: action heatmap failed: {e}")

    # -- Images: Random agent action history (1D plot) --
    if action_history and len(action_history) >= 2:
        try:
            img = plot_agent_action_history(
                action_history, agent_grid_nx, agent_grid_ny, episode_num)
            if img is not None:
                writer.add_image('Control/agent_action_trace', img, episode_num, dataformats='HWC')
        except Exception as e:
            print(f"Warning: agent trace plot failed: {e}")
