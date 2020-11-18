import gym
import torch
import torch.nn.functional as F
from baselines.common import vec_env
from baselines.common import atari_wrappers
from baselines.common.vec_env import DummyVecEnv
import numpy as np

from utils import one_hot_encode


class ClipRewardEnv(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.reduce_rewards = True

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.reduce_rewards:
            reward = (reward > 0) - (reward < 0)
        return obs, reward, done, info

    def set_reduce_rewards(self, reduce_rewards):
        self.reduce_rewards = reduce_rewards


class RecorderEnv(gym.Wrapper):

    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.recording = True
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

    def set_recording(self, recording):
        self.recording = recording

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

    def new_epoch(self):
        self.initial_frames = None
        self.initial_actions = None

    def get_first_small_rollout(self):
        return self.initial_frames, self.initial_actions

    def reset_sartv(self):
        self.sequence = torch.empty((0, *self.config.frame_shape), dtype=torch.uint8)
        self.actions = torch.empty((0, self.action_space.n), dtype=torch.uint8)
        self.rewards = torch.empty((0,), dtype=torch.uint8)
        self.targets = torch.empty((0, *self.config.frame_shape), dtype=torch.uint8)
        self.values = torch.empty((0,), dtype=torch.float32)

    def add_interaction(self, last_obs, action, obs, reward):
        last_state = last_obs.clone().detach().byte()
        self.sequence = torch.cat((self.sequence, last_state.unsqueeze(0)))
        action = one_hot_encode(action, self.action_space.n)
        self.actions = torch.cat((self.actions, action))

        n = len(self.sequence)

        if n == self.config.stacking and self.initial_frames is None:
            self.initial_frames = self.sequence
            self.initial_actions = self.actions

        if n >= self.config.stacking:
            self.current_rewards.append(reward)

            reward += 1  # Convert to {0; 1; 2}
            reward = torch.tensor(reward, dtype=torch.uint8)
            self.rewards = torch.cat((self.rewards, reward.unsqueeze(0)))

            target = obs.clone().detach().byte()
            self.targets = torch.cat((self.targets, target.unsqueeze(0)))

        if n == self.config.stacking + self.config.rollout_length - 1:
            self.buffer.append([self.sequence, self.actions, self.rewards, self.targets, None])
            self.reset_sartv()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.config.render_training:
            self.render()

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
        obs = super().reset()
        self.last_obs = obs
        return obs


class VecPytorchWrapper(vec_env.VecEnvWrapper):

    def reset(self):
        obs = self.venv.reset()
        return torch.tensor(obs)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.tensor(obs)
        reward = torch.tensor(reward).unsqueeze(1)
        return obs, reward, done, info


class WarpFrame(gym.ObservationWrapper):

    def __init__(self, env, config):
        super().__init__(env)
        self.config = config

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.config.frame_shape,
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.permute((2, 0, 1))
        obs = F.interpolate(obs.unsqueeze(0), self.config.frame_shape[1:]).squeeze()
        obs = obs.byte()
        return obs


def make_env(config):
    env = atari_wrappers.make_atari(f'{config.env_name}NoFrameskip-v0', max_episode_steps=10000)
    env = atari_wrappers.EpisodicLifeEnv(env)
    env = WarpFrame(env, config)
    env = ClipRewardEnv(env)
    env = RecorderEnv(env, config)
    env = DummyVecEnv([lambda: env])
    env = VecPytorchWrapper(env)
    return env
