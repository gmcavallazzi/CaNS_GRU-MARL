import numpy as np
from mpi4py import MPI
from gymnasium import spaces
from pettingzoo import ParallelEnv
from scipy.ndimage import gaussian_filter
import os


class STWGRUParallelEnv(ParallelEnv):
    """
    GRU-based multi-agent environment for turbulent channel flow control.

    Each agent controls a patch_size x patch_size patch of the wall with a
    scalar blowing/suction velocity. Observations are 3x3 neighborhoods
    of patches (ring observations) with periodic wrapping in x and y.
    Actions are smoothed with a Gaussian filter before being sent to CaNS.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "name": "stw_gru_v0"}

    def __init__(self, config, render_mode=None):
        self.config = config
        self.render_mode = render_mode

        # Grid dimensions from config
        self.full_nx = config['grid']['nx']
        self.full_ny = config['grid']['ny']

        # Patch and agent grid
        cc = config.get('control', {})
        self.patch_size = cc.get('patch_size', 4)
        self.ring_radius = cc.get('ring_radius', 1)
        self.action_scaling = cc.get('action_scaling', 0.064)
        self.smooth_sigma = cc.get('smooth_sigma', 1.5)
        self.scaling_factor = cc.get('scaling_factor', 2.0)

        self.agent_grid_nx = self.full_nx // self.patch_size
        self.agent_grid_ny = self.full_ny // self.patch_size
        self.n_agents = self.agent_grid_nx * self.agent_grid_ny

        # Observation spatial extent: (2*ring_radius+1) * patch_size
        self.obs_extent = (2 * self.ring_radius + 1) * self.patch_size

        # Observation config
        oc = config.get('observation', {})
        self.include_prev_action = oc.get('include_prev_action', True)
        self.normalize = oc.get('normalize', True)
        self.om_max = config.get('action', {}).get('om_max', 0.064)

        self.n_obs_channels = 3 if self.include_prev_action else 2

        # Agent setup
        self.possible_agents = [
            f"agent_{i}_{j}"
            for i in range(self.agent_grid_nx)
            for j in range(self.agent_grid_ny)
        ]
        self.agents = self.possible_agents[:]

        # Spaces
        self.observation_spaces = {
            agent: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_obs_channels, self.obs_extent, self.obs_extent),
                dtype=np.float32
            ) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # State
        self.episode_length = config['training']['start_episode_length']
        self.current_step = 0
        self.total_steps = 0
        self.prev_actions_scalar = np.zeros(self.n_agents, dtype=np.float32)
        self.prev_action_field = np.zeros((self.full_nx, self.full_ny), dtype=np.float32)

        # Initial obs cache
        self.initial_obs_captured = False
        self.initial_u_obs_field = None
        self.initial_w_obs_field = None

        # Field shift
        if config.get('field_shift', {}).get('enable', False):
            Lx = config['field_shift']['domain_size']['Lx']
            Ly = config['field_shift']['domain_size']['Ly']
            self.shift_x = np.linspace(0, Lx, self.full_nx, endpoint=False)
            self.shift_y = np.linspace(0, Ly, self.full_ny, endpoint=False)

        # MPI
        self.setup_mpi()
        self.rank = self.intracomm.Get_rank()

        self.dpdx = 0.0

    def setup_mpi(self):
        print("Python: Setting up MPI communication (MPMD mode)")
        comm = MPI.COMM_WORLD
        self.intracomm = comm.Dup()
        self.python_comm = comm.Split(color=0, key=comm.Get_rank())
        print("Python: MPMD communicators created")

        sync_flag = np.array([1], dtype=np.int32)
        self.intracomm.Bcast([sync_flag, MPI.INT], root=0)
        print("Python: Initial sync complete")

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.prev_actions_scalar = np.zeros(self.n_agents, dtype=np.float32)
        self.prev_action_field = np.zeros((self.full_nx, self.full_ny), dtype=np.float32)
        self.agents = self.possible_agents[:]

        if self.initial_obs_captured:
            u_field = self.initial_u_obs_field.copy()
            w_field = self.initial_w_obs_field.copy()
        else:
            u_field = np.zeros((self.full_nx, self.full_ny), dtype=np.float32)
            w_field = np.zeros((self.full_nx, self.full_ny), dtype=np.float32)

        observations = self._get_all_observations(u_field, w_field)
        infos = {agent: {"step": 0, "total_steps": self.total_steps}
                 for agent in self.agents}
        return observations, infos

    def step(self, actions):
        # Send control message
        control_msg = b'START' if self.current_step == 0 else b'CONTN'
        self.intracomm.Bcast([control_msg, MPI.CHAR], root=0)

        # Collect scalar actions from all agents
        scalar_actions = np.zeros(self.n_agents, dtype=np.float32)
        for idx, agent in enumerate(self.possible_agents):
            a = actions[agent]
            scalar_actions[idx] = a[0] if isinstance(a, np.ndarray) else float(a)

        # Build smoothed action field
        action_field = self._scalar_actions_to_field(scalar_actions)

        # Store for prev_action observations
        self.prev_actions_scalar = scalar_actions.copy()
        self.prev_action_field = action_field.copy()

        # Scale and send to CaNS
        amp_send = action_field.astype(np.float64) * self.om_max * self.scaling_factor
        amp_send -= amp_send.mean()
        self.intracomm.Send([amp_send, MPI.DOUBLE], dest=1, tag=1)

        # Receive observations from CaNS
        u_obs_all = np.zeros((self.full_nx, self.full_ny), dtype=np.float64)
        w_obs_all = np.zeros((self.full_nx, self.full_ny), dtype=np.float64)
        p_obs_all = np.zeros((self.full_nx, self.full_ny), dtype=np.float64)
        t_obs_all = np.zeros((self.full_nx, self.full_ny), dtype=np.float64)
        self.dpdx = np.array(0.0, dtype=np.float64)

        self.intracomm.Recv([u_obs_all, MPI.DOUBLE], source=1, tag=5)
        self.intracomm.Recv([w_obs_all, MPI.DOUBLE], source=1, tag=9)
        self.intracomm.Recv([p_obs_all, MPI.DOUBLE], source=1, tag=8)
        self.intracomm.Recv([t_obs_all, MPI.DOUBLE], source=1, tag=7)
        self.intracomm.Recv([self.dpdx, MPI.DOUBLE], source=1, tag=4)

        # Normalize and store for snapshot access
        if self.normalize:
            u_field = ((u_obs_all - np.mean(u_obs_all)) / self.om_max).astype(np.float32)
            w_field = ((w_obs_all - np.mean(w_obs_all)) / self.om_max).astype(np.float32)
        else:
            u_field = u_obs_all.astype(np.float32)
            w_field = w_obs_all.astype(np.float32)
        self.u_obs_field = u_field
        self.w_obs_field = w_field

        # Cache initial obs
        if not self.initial_obs_captured and self.total_steps == 0:
            self.initial_u_obs_field = u_field.copy()
            self.initial_w_obs_field = w_field.copy()
            self.initial_obs_captured = True

        # Field shift
        if self.config.get('field_shift', {}).get('enable', False):
            u_avg = np.mean(u_field)
            dx_shift = u_avg * self.config['field_shift']['dt']
            u_field = self._shift_field(u_field, dx_shift)
            w_field = self._shift_field(w_field, dx_shift)

        # Compute reward from pressure gradient
        dpdx_uncontrolled = self.config['reward']['dpdx']['uncontrolled']
        reduction = 1.0 - (float(self.dpdx) / dpdx_uncontrolled)
        global_reward = self.config['reward']['global_weight'] * reduction

        # Build outputs
        observations = self._get_all_observations(u_field, w_field)
        rewards = {agent: global_reward for agent in self.agents}
        terminated = self.current_step >= self.episode_length
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {
            'dpdx': float(self.dpdx),
            'step': self.current_step,
            'total_steps': self.total_steps,
            'global_reward': global_reward,
        } for agent in self.agents}

        self.current_step += 1
        self.total_steps += 1

        self._check_simulation_end()

        return observations, rewards, terminations, truncations, infos

    def _get_all_observations(self, u_field, w_field):
        observations = {}
        for idx, agent in enumerate(self.possible_agents):
            ix = idx // self.agent_grid_ny
            iy = idx % self.agent_grid_ny
            observations[agent] = self._get_ring_observation(u_field, w_field, ix, iy)
        return observations

    def _get_ring_observation(self, u_field, w_field, agent_ix, agent_iy):
        """
        Extract 3x3 neighborhood of patches around (agent_ix, agent_iy).
        Periodic wrapping in both x and y directions.

        Returns: (n_channels, obs_extent, obs_extent) array
        """
        r = self.ring_radius
        ps = self.patch_size
        extent = self.obs_extent

        channels = []
        for field in [u_field, w_field]:
            obs = np.zeros((extent, extent), dtype=np.float32)
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    ni = (agent_ix + di) % self.agent_grid_nx
                    nj = (agent_iy + dj) % self.agent_grid_ny
                    x0 = ni * ps
                    y0 = nj * ps
                    ox = (di + r) * ps
                    oy = (dj + r) * ps
                    obs[ox:ox + ps, oy:oy + ps] = field[x0:x0 + ps, y0:y0 + ps]
            channels.append(obs)

        if self.include_prev_action:
            prev_obs = np.zeros((extent, extent), dtype=np.float32)
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    ni = (agent_ix + di) % self.agent_grid_nx
                    nj = (agent_iy + dj) % self.agent_grid_ny
                    idx = ni * self.agent_grid_ny + nj
                    ox = (di + r) * ps
                    oy = (dj + r) * ps
                    prev_obs[ox:ox + ps, oy:oy + ps] = self.prev_actions_scalar[idx]
            channels.append(prev_obs)

        return np.stack(channels, axis=0)

    def _scalar_actions_to_field(self, scalar_actions):
        """
        Convert n_agents scalar actions to a full (nx, ny) action field with smoothing.

        1. Zero-mean subtraction (mass conservation)
        2. Build piecewise-constant field
        3. Gaussian smoothing with periodic padding
        4. Re-apply zero-mean after smoothing (numerical precision)
        """
        actions = scalar_actions.copy()
        actions -= actions.mean()

        # Build piecewise-constant field
        field = np.zeros((self.full_nx, self.full_ny), dtype=np.float32)
        for idx in range(self.n_agents):
            ix = idx // self.agent_grid_ny
            iy = idx % self.agent_grid_ny
            x0 = ix * self.patch_size
            y0 = iy * self.patch_size
            field[x0:x0 + self.patch_size, y0:y0 + self.patch_size] = actions[idx]

        # Gaussian smoothing with periodic boundary handling
        if self.smooth_sigma > 0:
            pad = max(1, int(3 * self.smooth_sigma))
            padded = np.pad(field, pad, mode='wrap')
            smoothed = gaussian_filter(padded, sigma=self.smooth_sigma, mode='nearest')
            field = smoothed[pad:-pad, pad:-pad]
            field -= field.mean()  # restore zero-mean after smoothing

        return field

    def _shift_field(self, field, dx_shift):
        nx, ny = field.shape
        dx = self.shift_x[1] - self.shift_x[0]
        shift_units = dx_shift / dx
        result = np.zeros_like(field)
        for i in range(nx):
            for j in range(ny):
                src = (i - shift_units) % nx
                i1 = int(np.floor(src)) % nx
                i2 = (i1 + 1) % nx
                w = src - np.floor(src)
                result[i, j] = (1 - w) * field[i1, j] + w * field[i2, j]
        return result

    def _check_simulation_end(self):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

        if self.total_steps >= self.config['total_timesteps']:
            print("Python: Sending ENDED - simulation complete")
            self.intracomm.Bcast([b'ENDED', MPI.CHAR], root=0)
            self.intracomm.Free()
        elif self.current_step >= self.episode_length:
            print("Python: Sending CONTR - episode complete")
            file_num = np.random.randint(1, 10)
            src_file = os.path.join(data_dir, f"fld_{file_num:04d}.bin")
            dst_file = os.path.join(data_dir, "fld.bin")
            if self.rank == 0:
                print(f"Copying file {file_num:04d} to fld.bin")
                os.system(f"cp {src_file} {dst_file}")
            self.intracomm.Bcast([b'CONTR', MPI.CHAR], root=0)
        else:
            self.intracomm.Bcast([b'CONTN', MPI.CHAR], root=0)

    def close(self):
        try:
            self.intracomm.Bcast([b'ENDED', MPI.CHAR], root=0)
            self.intracomm.Free()
        except Exception:
            pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def set_episode_length(self, length):
        self.episode_length = length
