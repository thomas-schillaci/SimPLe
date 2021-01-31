import cv2

# See https://stackoverflow.com/questions/54013846/pytorch-dataloader-stucked-if-using-opencv-resize-method
# See https://github.com/pytorch/pytorch/issues/1355
from atari_utils.buffer import Buffer

cv2.setNumThreads(0)
import gym
import torch
from baselines.common import atari_wrappers
from baselines.common.atari_wrappers import NoopResetEnv
from baselines.common.vec_env import ShmemVecEnv, VecEnvWrapper
import numpy as np
from gym.wrappers import TimeLimit

from atari_utils.utils import one_hot_encode, DummyVecEnv


class WarpFrame(gym.ObservationWrapper):

    def __init__(self, env, width=84, height=84, grayscale=True, inter_area=False):
        super().__init__(env)

        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.inter_area = inter_area

        channels = 1 if grayscale else self.env.observation_space.shape[-1]

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(channels, self.height, self.width),
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32)
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        obs = cv2.resize(
            obs,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA if self.inter_area else cv2.INTER_NEAREST
        )

        obs = torch.tensor(obs, dtype=torch.uint8)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-1)
        obs = obs.permute((2, 0, 1))

        return obs


class RenderingEnv(gym.ObservationWrapper):

    def observation(self, observation):
        self.render()
        return observation


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPytorchWrapper(VecEnvWrapper):
    def __init__(self, venv, device, nstack=4):
        self.venv = venv
        self.device = device
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        obs = torch.tensor(obs).to(self.device)
        rews = torch.tensor(rews).unsqueeze(1)
        self.stacked_obs[:, :-self.shape_dim0] = self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        obs = torch.tensor(obs).to(self.device)
        self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs


class ClipRewardEnv(gym.RewardWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.cum_reward = 0

    def reset(self, **kwargs):
        self.cum_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.cum_reward += reward
        if done:
            info['r'] = self.cum_reward
            self.cum_reward = 0
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        return (reward > 0) - (reward < 0)


class RecorderEnv(gym.Wrapper):

    def __init__(self, env, frame_shape, rollout_length, gamma, frame_stacking=4):
        super().__init__(env)
        self.frame_shape = frame_shape
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.frame_stacking = frame_stacking

        self.temp_buffer = []
        self.buffer = Buffer()

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

    def update_priority(self, indices, deltas):
        self.buffer.update_deltas(indices, deltas)

    def sample_buffer(self, n):
        return self.buffer.sample(n)

    def new_epoch(self):
        self.initial_frames = None

    def get_first_small_rollout(self):
        assert len(self.initial_frames) == self.frame_stacking
        return self.initial_frames

    def reset_sartv(self):
        self.sequence = torch.empty((0, *self.frame_shape), dtype=torch.uint8)
        self.actions = torch.empty((0, self.action_space.n), dtype=torch.uint8)
        self.rewards = torch.empty((0,), dtype=torch.uint8)
        self.targets = torch.empty((0, *self.frame_shape), dtype=torch.uint8)
        self.values = torch.empty((0,), dtype=torch.float32)

    def add_interaction(self, last_obs, action, obs, reward):
        last_state = last_obs.clone().detach().byte()
        self.sequence = torch.cat((self.sequence, last_state.unsqueeze(0)))

        n = len(self.sequence)

        if n == self.frame_stacking and self.initial_frames is None:
            self.initial_frames = self.sequence

        if n >= self.frame_stacking:
            action = one_hot_encode(action, self.action_space.n)
            self.actions = torch.cat((self.actions, action))

            self.current_rewards.append(reward)
            reward = reward + 1  # Convert to {0; 1; 2}
            reward = torch.tensor(reward, dtype=torch.uint8)
            self.rewards = torch.cat((self.rewards, reward.unsqueeze(0)))

            target = obs.clone().detach().byte()
            self.targets = torch.cat((self.targets, target.unsqueeze(0)))

        if n == self.frame_stacking + self.rollout_length - 1:
            self.temp_buffer.append([self.sequence, self.actions, self.rewards, self.targets, None])
            self.reset_sartv()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        self.add_interaction(self.last_obs, action, obs, reward)

        if done:
            for _ in range(len(self.rewards)):
                self.current_rewards.pop()
            self.reset_sartv()
            value = 0
            for i in range(len(self.temp_buffer) - 1, -1, -1):
                self.temp_buffer[i][-1] = torch.empty((self.rollout_length,), dtype=torch.float32)
                for j in range(self.rollout_length - 1, -1, -1):
                    r = self.current_rewards.pop()
                    assert int(r) == -1 or int(r) == 0 or int(r) == 1
                    value = r + self.gamma * value
                    self.temp_buffer[i][-1][j] = value
            for data in self.temp_buffer:
                self.buffer.insert(data)
            self.temp_buffer = []

            assert not self.current_rewards

        self.last_obs = obs

        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        self.last_obs = obs
        return obs


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        obs = None
        total_reward = 0.0
        done = None
        info = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def _make_env(
        env_name,
        render=False,
        max_episode_steps=18000,
        frame_shape=(1, 84, 84),
        inter_area=False,
        record=False,
        rollout_length=-1,
        gamma=0.99,
        noop_max=30
):
    env = gym.make(f'{env_name}NoFrameskip-v4')
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=noop_max)
    env = SkipEnv(env, skip=4)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = atari_wrappers.FireResetEnv(env)
    grayscale = frame_shape[0] == 1
    height, width = frame_shape[1:]
    env = WarpFrame(env, width=width, height=height, grayscale=grayscale, inter_area=inter_area)
    env = ClipRewardEnv(env)
    if render:
        env = RenderingEnv(env)
    if record:
        assert rollout_length > 0
        env = RecorderEnv(env, frame_shape, rollout_length, gamma)
    return env


def make_envs(env_name, num, device, **kwargs):
    env_fns = [lambda: _make_env(env_name, **kwargs)]
    kwargs_no_render = kwargs.copy()
    kwargs_no_render['render'] = False
    env_fns += [lambda: _make_env(env_name, **kwargs_no_render)] * (num - 1)
    if num == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = ShmemVecEnv(env_fns)
    env = VecPytorchWrapper(env, device)
    return env


def make_env(env_name, device, **kwargs):
    return make_envs(env_name, 1, device, **kwargs)
