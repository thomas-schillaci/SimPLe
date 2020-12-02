import cv2
import gym
import torch
from baselines.common import atari_wrappers
from baselines.common.vec_env import ShmemVecEnv, VecEnvWrapper
import numpy as np

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
        self.sequence = torch.empty((0, *self.frame_shape), dtype=torch.uint8)
        self.actions = torch.empty((0, self.action_space.n), dtype=torch.uint8)
        self.rewards = torch.empty((0,), dtype=torch.uint8)
        self.targets = torch.empty((0, *self.frame_shape), dtype=torch.uint8)
        self.values = torch.empty((0,), dtype=torch.float32)

    def add_interaction(self, last_obs, action, obs, reward):
        last_state = last_obs.clone().detach().byte()
        self.sequence = torch.cat((self.sequence, last_state.unsqueeze(0)))
        action = one_hot_encode(action, self.action_space.n)
        self.actions = torch.cat((self.actions, action))

        n = len(self.sequence)

        if n == self.frame_stacking and self.initial_frames is None:
            self.initial_frames = self.sequence
            self.initial_actions = self.actions

        if n >= self.frame_stacking:
            self.current_rewards.append(reward)

            reward += 1  # Convert to {0; 1; 2}
            reward = torch.tensor(reward, dtype=torch.uint8)
            self.rewards = torch.cat((self.rewards, reward.unsqueeze(0)))

            target = obs.clone().detach().byte()
            self.targets = torch.cat((self.targets, target.unsqueeze(0)))

        if n == self.frame_stacking + self.rollout_length - 1:
            self.buffer.append([self.sequence, self.actions, self.rewards, self.targets, None])
            self.reset_sartv()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        self.add_interaction(self.last_obs, action, obs, reward)

        if done:
            for _ in range(len(self.rewards)):
                self.current_rewards.pop()
            self.reset_sartv()
            value = 0
            for i in range(len(self.buffer) - 1, -1, -1):
                if self.buffer[i][-1] is not None:
                    break
                self.buffer[i][-1] = torch.empty((self.rollout_length,), dtype=torch.float32)
                for j in range(self.rollout_length - 1, -1, -1):
                    value = self.current_rewards.pop() + self.gamma * value
                    self.buffer[i][-1][j] = value

            assert not self.current_rewards

        self.last_obs = obs

        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        self.last_obs = obs
        return obs


class CropVecEnv(VecEnvWrapper):

    def __init__(self, env, agents):
        self.agents = agents
        self.w1 = None
        self.h1 = None
        self.reset_params()

        shape = list(env.observation_space.low.shape)
        shape[-1] = 84
        shape[-2] = 84

        low = np.zeros(shape)
        high = np.full(shape, 255)

        observation_space = gym.spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)
        super().__init__(env, observation_space, env.action_space)

    def crop(self, obs):
        assert len(obs.shape) == 4
        assert obs.shape[-1] == 100

        res = torch.empty((len(obs),) + self.observation_space.shape).to(obs.device)

        for i in range(len(obs)):
            res[i] = obs[i, :, self.h1[i]:self.h1[i] + 84, self.w1[i]:self.w1[i] + 84]

        return res

    def reset_params(self):
        self.w1 = torch.randint(high=17, size=(self.agents,))
        self.h1 = torch.randint(high=17, size=(self.agents,))

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        obs = self.crop(obs)
        return obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        obs = self.crop(obs)
        return obs


def _make_env(
        env_name,
        render=False,
        max_episode_steps=18000,
        frame_shape=(1, 84, 84),
        inter_area=False,
        record=False,
        rollout_length=-1,
        gamma=0.99
):
    env = atari_wrappers.make_atari(f'{env_name}NoFrameskip-v4', max_episode_steps=max_episode_steps)
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


def make_envs(env_name, num, device, crop=False, **kwargs):
    if crop:  # FIXME
        kwargs['frame_shape'] = (kwargs['frame_shape'][0], 100, 100)
    env_fns = [lambda: _make_env(env_name, **kwargs)]
    kwargs_no_render = kwargs.copy()
    kwargs_no_render['render'] = False
    env_fns += [lambda: _make_env(env_name, **kwargs_no_render)] * (num - 1)
    if num == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = ShmemVecEnv(env_fns)
    env = VecPytorchWrapper(env, device)
    if crop:
        env = CropVecEnv(env, num)
    return env


def make_env(env_name, device, **kwargs):
    return make_envs(env_name, 1, device, **kwargs)
