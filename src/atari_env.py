import gym
import torch
from baselines.common.atari_wrappers import make_atari, EpisodicLifeEnv, WarpFrame

from a2c_ppo_acktr.envs import TransposeImage
from utils import one_hot_encode


class ClipRewardEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action, reduce_rewards=False):
        obs, reward, done, info = super().step(action)
        if reduce_rewards:
            reward = (reward > 0) - (reward < 0)
        return obs, reward, done, info


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
        last_state = torch.tensor(last_obs, dtype=torch.uint8)
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

            target = torch.tensor(obs, dtype=torch.uint8)
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


def make_env(config):
    env = make_atari(f'{config.env_name}NoFrameskip-v0', max_episode_steps=10000)
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env, width=config.frame_shape[2], height=config.frame_shape[1], grayscale=False)
    env = TransposeImage(env)
    env = ClipRewardEnv(env)
    env = RecorderEnv(env, config)
    return env
