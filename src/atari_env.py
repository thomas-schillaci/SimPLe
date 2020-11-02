import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym import Env, spaces

from utils import one_hot_encode


class AtariEnv(Env):

    def __init__(self, config, base_env_name, recording=True):
        super(AtariEnv, self).__init__()
        self.config = config
        self.base_env = gym.make(base_env_name)
        self.action_space = self.base_env.action_space
        self.observation_space = spaces.Box(low=0, high=255, shape=(105, 80, 3), dtype=np.uint8)
        self.recording = recording

        self.buffer = []
        self.last_obs = None
        self.values = 0

        self.sequence = None
        self.actions = None
        self.rewards = None
        self.targets = None
        self.values = None
        self.reset_sartv()

        self.current_rewards = []

        self.initial_frames = None
        self.initial_actions = None

    def new_epoch(self):
        self.initial_frames = None
        self.initial_actions = None

    def get_first_small_rollout(self):
        return self.initial_frames, self.initial_actions

    def reset_sartv(self):
        self.sequence = torch.empty((0, *self.config.frame_shape), dtype=torch.uint8)
        self.actions = torch.empty((0, self.base_env.action_space.n), dtype=torch.uint8)
        self.rewards = torch.empty((0,), dtype=torch.uint8)
        self.targets = torch.empty((0, *self.config.frame_shape), dtype=torch.uint8)
        self.values = torch.empty((0,), dtype=torch.float32)

    def add_interaction(self, last_obs, action, obs, reward):
        last_state = torch.tensor(last_obs, dtype=torch.uint8).permute((2, 0, 1))
        self.sequence = torch.cat((self.sequence, last_state.unsqueeze(0)))
        action = one_hot_encode(action, self.base_env.action_space.n)
        self.actions = torch.cat((self.actions, action.unsqueeze(0)))

        n = len(self.sequence)

        if n == self.config.stacking and self.initial_frames is None:
            self.initial_frames = self.sequence
            self.initial_actions = self.actions

        if n >= self.config.stacking:
            self.current_rewards.append(reward)

            # reward = (reward > 0) - (reward < 0)  # Convert to {-1; 0; 1}
            reward += 1  # Convert to {0; 1; 2}
            reward = torch.tensor(reward, dtype=torch.uint8)
            self.rewards = torch.cat((self.rewards, reward.unsqueeze(0)))

            target = torch.tensor(obs, dtype=torch.uint8).permute((2, 0, 1))
            self.targets = torch.cat((self.targets, target.unsqueeze(0)))

        if n == self.config.stacking + self.config.rollout_length - 1:
            self.buffer.append([self.sequence, self.actions, self.rewards, self.targets, None])
            self.reset_sartv()

    def step(self, action, reduce_rewards=True):
        obs, reward, done, info = self.base_env.step(action)

        if self.config.render_training:
            self.render()

        obs = self.downscale_obs(obs)
        if reduce_rewards:
            reward = (reward > 0) - (reward < 0)

        if self.recording:
            self.add_interaction(self.last_obs, action, obs, reward)

            if done:
                for _ in range(len(self.rewards)):
                    self.current_rewards.pop()
                self.reset_sartv()
                value = 0
                for i in range(len(self.buffer) - 1, -1, -1):
                    if self.buffer[i][-1] is not None:
                        break
                    self.buffer[i][-1] = torch.empty((self.config.rollout_length,), dtype=torch.float32)
                    for j in range(self.config.rollout_length - 1, -1, -1):
                        value = self.current_rewards.pop() + self.config.ppo_gamma * value
                        self.buffer[i][-1][j] = value

                assert not self.current_rewards

        self.last_obs = obs

        return obs, reward, done, info

    def reset(self):
        self.last_obs = self.base_env.reset()
        self.last_obs = self.downscale_obs(self.last_obs)
        return self.last_obs

    def render(self, mode='human'):
        self.base_env.render(mode)

    def close(self):
        self.base_env.close()

    def downscale_obs(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.permute((2, 0, 1))
        obs = F.interpolate(obs.unsqueeze(0), self.config.frame_shape[1:]).squeeze()
        obs = obs.permute((1, 2, 0))
        obs = obs.numpy()
        return obs

    def sample_buffer(self, batch_size=-1):
        if not self.buffer:
            return None
        if self.buffer[0][-1] is None:
            return None

        if batch_size != -1:
            sequences = []
            actions = []
            rewards = []
            targets = []
            values = []
            for _ in range(batch_size):
                sequence, action, reward, target, value = self.sample_buffer()
                sequences.append(sequence)
                actions.append(action)
                rewards.append(reward)
                targets.append(target)
                values.append(value)
            sequences = torch.stack(sequences, dim=1)
            actions = torch.stack(actions, dim=1)
            rewards = torch.stack(rewards, dim=1)
            targets = torch.stack(targets, dim=1)
            values = torch.stack(values, dim=1)
            return sequences, actions, rewards, targets, values

        index = int(torch.randint(len(self.buffer), size=(1,)))

        if self.buffer[index][-1] is None:
            return self.sample_buffer()

        return self.buffer[index]

    def set_recording(self, recording):
        self.recording = recording
