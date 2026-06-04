import numpy as np
import torch
from typing import Dict, List, Optional


class ReplayBuffer:
    """Replay buffer for single-agent TD3. Stores on CPU (NumPy arrays)."""

    def __init__(self, state_shape, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind],
        )

    def save(self, path):
        np.savez_compressed(path,
                            state=self.state[:self.size],
                            action=self.action[:self.size],
                            next_state=self.next_state[:self.size],
                            reward=self.reward[:self.size],
                            not_done=self.not_done[:self.size],
                            ptr=self.ptr, size=self.size)

    def load(self, path):
        data = np.load(path)
        n = int(data['size'])
        self.state[:n] = data['state']
        self.action[:n] = data['action']
        self.next_state[:n] = data['next_state']
        self.reward[:n] = data['reward']
        self.not_done[:n] = data['not_done']
        self.ptr = int(data['ptr'])
        self.size = n


class BatchedReplayBuffer:
    """
    Replay buffer for multi-agent MADDPG with previous actions.
    Adapted from run0/models_pettingzoo.py BatchedReplayBuffer.
    """

    def __init__(self, capacity, n_agents, obs_shape_per_agent, act_dim_per_agent=1):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_shape = obs_shape_per_agent  # e.g., (3, ny, obs_width)
        self.act_dim = act_dim_per_agent      # scalar per agent

        self.obs_buf = np.zeros((capacity, n_agents, *obs_shape_per_agent), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, n_agents, *obs_shape_per_agent), dtype=np.float32)
        self.acts_buf = np.zeros((capacity, n_agents, act_dim_per_agent), dtype=np.float32)
        self.prev_acts_buf = np.zeros((capacity, n_agents, act_dim_per_agent), dtype=np.float32)
        self.rews_buf = np.zeros((capacity, n_agents), dtype=np.float32)
        self.done_buf = np.zeros((capacity, n_agents), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add_batch(self, obs_batch, acts_batch, prev_acts_batch,
                  rews_batch, next_obs_batch, dones_batch):
        self.obs_buf[self.ptr] = obs_batch
        self.next_obs_buf[self.ptr] = next_obs_batch
        self.acts_buf[self.ptr] = acts_batch
        self.prev_acts_buf[self.ptr] = prev_acts_batch
        self.rews_buf[self.ptr] = rews_batch
        self.done_buf[self.ptr] = dones_batch

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device='cpu'):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs': torch.as_tensor(self.obs_buf[idxs], device=device),
            'next_obs': torch.as_tensor(self.next_obs_buf[idxs], device=device),
            'acts': torch.as_tensor(self.acts_buf[idxs], device=device),
            'prev_acts': torch.as_tensor(self.prev_acts_buf[idxs], device=device),
            'rews': torch.as_tensor(self.rews_buf[idxs], device=device),
            'done': torch.as_tensor(self.done_buf[idxs], device=device),
        }

    def save(self, path):
        np.savez(path,
                 obs=self.obs_buf[:self.size],
                 next_obs=self.next_obs_buf[:self.size],
                 acts=self.acts_buf[:self.size],
                 prev_acts=self.prev_acts_buf[:self.size],
                 rews=self.rews_buf[:self.size],
                 done=self.done_buf[:self.size],
                 ptr=self.ptr, size=self.size)

    def load(self, path):
        data = np.load(path)
        n = int(data['size'])
        self.obs_buf[:n] = data['obs']
        self.next_obs_buf[:n] = data['next_obs']
        self.acts_buf[:n] = data['acts']
        if 'prev_acts' in data:
            self.prev_acts_buf[:n] = data['prev_acts']
        self.rews_buf[:n] = data['rews']
        self.done_buf[:n] = data['done']
        self.ptr = int(data['ptr'])
        self.size = n


class SequenceReplayBuffer:
    """
    Episode-based replay buffer for recurrent single-agent training.
    Stores complete episodes, samples contiguous sequences.
    """

    def __init__(self, state_shape, action_dim, max_episodes=200):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.max_episodes = max_episodes

        self.episodes = []
        self._current_ep = None

    def add(self, state, action, next_state, reward, done):
        if self._current_ep is None:
            self._current_ep = {
                'states': [], 'actions': [], 'next_states': [],
                'rewards': [], 'not_done': [],
            }
        self._current_ep['states'].append(state.copy())
        self._current_ep['actions'].append(action.copy())
        self._current_ep['next_states'].append(next_state.copy())
        self._current_ep['rewards'].append(np.array([reward], dtype=np.float32))
        self._current_ep['not_done'].append(np.array([1.0 - done], dtype=np.float32))

        if done > 0.5:
            self._finalize_episode()

    def _finalize_episode(self):
        ep = {k: np.stack(v) for k, v in self._current_ep.items()}
        self.episodes.append(ep)
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
        self._current_ep = None

    @property
    def size(self):
        return sum(len(ep['states']) for ep in self.episodes)

    def sample_sequences(self, batch_size, seq_len):
        """Sample random contiguous chunks of length seq_len from stored episodes."""
        states = np.zeros((batch_size, seq_len, *self.state_shape), dtype=np.float32)
        actions = np.zeros((batch_size, seq_len, self.action_dim), dtype=np.float32)
        next_states = np.zeros((batch_size, seq_len, *self.state_shape), dtype=np.float32)
        rewards = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        not_done = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        mask = np.zeros((batch_size, seq_len), dtype=np.float32)

        for i in range(batch_size):
            ep = self.episodes[np.random.randint(len(self.episodes))]
            ep_len = len(ep['states'])
            if ep_len <= seq_len:
                start = 0
                length = ep_len
            else:
                start = np.random.randint(0, ep_len - seq_len + 1)
                length = seq_len

            states[i, :length] = ep['states'][start:start + length]
            actions[i, :length] = ep['actions'][start:start + length]
            next_states[i, :length] = ep['next_states'][start:start + length]
            rewards[i, :length] = ep['rewards'][start:start + length]
            not_done[i, :length] = ep['not_done'][start:start + length]
            mask[i, :length] = 1.0

        return {
            'states': states, 'actions': actions, 'next_states': next_states,
            'rewards': rewards, 'not_done': not_done, 'mask': mask,
        }

    def sample(self, batch_size):
        """Fallback: sample individual transitions (for compatibility)."""
        indices = []
        for _ in range(batch_size):
            ep_idx = np.random.randint(len(self.episodes))
            ep = self.episodes[ep_idx]
            t_idx = np.random.randint(len(ep['states']))
            indices.append((ep_idx, t_idx))

        states = np.stack([self.episodes[e]['states'][t] for e, t in indices])
        actions = np.stack([self.episodes[e]['actions'][t] for e, t in indices])
        next_states = np.stack([self.episodes[e]['next_states'][t] for e, t in indices])
        rewards = np.stack([self.episodes[e]['rewards'][t] for e, t in indices])
        not_done = np.stack([self.episodes[e]['not_done'][t] for e, t in indices])
        return states, actions, next_states, rewards, not_done

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'episodes': self.episodes}, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.episodes = data['episodes']


class BatchedSequenceReplayBuffer:
    """
    Episode-based replay buffer for recurrent multi-agent training.
    Stores complete episodes with (n_agents, ...) shaped data per step.
    """

    def __init__(self, capacity_episodes, n_agents, obs_shape_per_agent, act_dim_per_agent=1):
        self.max_episodes = capacity_episodes
        self.n_agents = n_agents
        self.obs_shape = obs_shape_per_agent
        self.act_dim = act_dim_per_agent

        self.episodes = []
        self._current_ep = None

    def add_batch(self, obs_batch, acts_batch, prev_acts_batch,
                  rews_batch, next_obs_batch, dones_batch):
        """Add one timestep for all agents."""
        if self._current_ep is None:
            self._current_ep = {
                'obs': [], 'acts': [], 'prev_acts': [],
                'rews': [], 'next_obs': [], 'done': [],
            }
        self._current_ep['obs'].append(obs_batch.copy())
        self._current_ep['acts'].append(acts_batch.copy())
        self._current_ep['prev_acts'].append(prev_acts_batch.copy())
        self._current_ep['rews'].append(rews_batch.copy())
        self._current_ep['next_obs'].append(next_obs_batch.copy())
        self._current_ep['done'].append(dones_batch.copy())

        if dones_batch[0] > 0.5:
            self._finalize_episode()

    def _finalize_episode(self):
        ep = {k: np.stack(v) for k, v in self._current_ep.items()}
        self.episodes.append(ep)
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
        self._current_ep = None

    @property
    def size(self):
        return sum(len(ep['obs']) for ep in self.episodes)

    def sample(self, batch_size, device='cpu'):
        """Sample individual transitions (fallback for non-recurrent training)."""
        indices = []
        for _ in range(batch_size):
            ep_idx = np.random.randint(len(self.episodes))
            ep = self.episodes[ep_idx]
            t_idx = np.random.randint(len(ep['obs']))
            indices.append((ep_idx, t_idx))

        obs = np.stack([self.episodes[e]['obs'][t] for e, t in indices])
        next_obs = np.stack([self.episodes[e]['next_obs'][t] for e, t in indices])
        acts = np.stack([self.episodes[e]['acts'][t] for e, t in indices])
        prev_acts = np.stack([self.episodes[e]['prev_acts'][t] for e, t in indices])
        rews = np.stack([self.episodes[e]['rews'][t] for e, t in indices])
        done = np.stack([self.episodes[e]['done'][t] for e, t in indices])

        return {
            'obs': torch.as_tensor(obs, device=device),
            'next_obs': torch.as_tensor(next_obs, device=device),
            'acts': torch.as_tensor(acts, device=device),
            'prev_acts': torch.as_tensor(prev_acts, device=device),
            'rews': torch.as_tensor(rews, device=device),
            'done': torch.as_tensor(done, device=device),
        }

    def sample_sequences(self, batch_size, seq_len, device='cpu'):
        """Sample contiguous sequences for recurrent training."""
        obs = np.zeros((batch_size, seq_len, self.n_agents, *self.obs_shape), dtype=np.float32)
        acts = np.zeros((batch_size, seq_len, self.n_agents, self.act_dim), dtype=np.float32)
        prev_acts = np.zeros((batch_size, seq_len, self.n_agents, self.act_dim), dtype=np.float32)
        rews = np.zeros((batch_size, seq_len, self.n_agents), dtype=np.float32)
        next_obs = np.zeros((batch_size, seq_len, self.n_agents, *self.obs_shape), dtype=np.float32)
        done = np.zeros((batch_size, seq_len, self.n_agents), dtype=np.float32)
        mask = np.zeros((batch_size, seq_len), dtype=np.float32)

        for i in range(batch_size):
            ep = self.episodes[np.random.randint(len(self.episodes))]
            ep_len = len(ep['obs'])
            if ep_len <= seq_len:
                start = 0
                length = ep_len
            else:
                start = np.random.randint(0, ep_len - seq_len + 1)
                length = seq_len

            obs[i, :length] = ep['obs'][start:start + length]
            acts[i, :length] = ep['acts'][start:start + length]
            prev_acts[i, :length] = ep['prev_acts'][start:start + length]
            rews[i, :length] = ep['rews'][start:start + length]
            next_obs[i, :length] = ep['next_obs'][start:start + length]
            done[i, :length] = ep['done'][start:start + length]
            mask[i, :length] = 1.0

        return {
            'obs': torch.as_tensor(obs, device=device),
            'next_obs': torch.as_tensor(next_obs, device=device),
            'acts': torch.as_tensor(acts, device=device),
            'prev_acts': torch.as_tensor(prev_acts, device=device),
            'rews': torch.as_tensor(rews, device=device),
            'done': torch.as_tensor(done, device=device),
            'mask': torch.as_tensor(mask, device=device),
        }

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'episodes': self.episodes}, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.episodes = data['episodes']
