import cv2
import gym
import torch
from gym import Env, spaces
import numpy as np
from stable_baselines.common.env_checker import check_env


class SimulatedEnv(Env):

    def __init__(self, config, action_space, main=False):
        super(SimulatedEnv, self).__init__()
        self.config = config
        self.action_space = action_space
        frame_shape = self.config.frame_shape
        frame_shape = (frame_shape[1], frame_shape[2], frame_shape[0])
        self.observation_space = spaces.Box(low=0, high=255, shape=frame_shape, dtype=np.uint8)
        self.frame_stack = torch.empty(
            (0, *self.config.frame_shape[1:]), dtype=torch.uint8
        ).to(self.config.device)
        self.main = main

    def get_frame_stack_(self):
        return self.frame_stack

    def step(self, args):
        state, reward = args

        self.frame_stack = torch.cat((self.frame_stack, state))
        self.frame_stack = self.frame_stack[self.config.frame_shape[0]:]

        if self.main:
            self.render()

        state = state.permute((1, 2, 0)).cpu().detach().numpy()
        return state, reward, False, {}

    def reset(self):
        if len(self.frame_stack) == 0:
            raise ValueError('frame_stack has not been initialized. Call the restart method.')

        state = self.frame_stack[-self.config.frame_shape[0]:]
        state = state.permute((1, 2, 0)).cpu().detach().numpy()
        return state

    def render(self, mode='human'):
        frame = self.frame_stack[-self.config.frame_shape[0]:]
        frame = frame.permute((1, 2, 0)).cpu().detach().numpy()
        frame = frame[:, :, [2, 1, 0]]
        cv2.namedWindow('World model', cv2.WINDOW_NORMAL)
        cv2.imshow('World model', frame)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow('World model')

    def restart(self, frame_stack):
        self.frame_stack = frame_stack.to(self.config.device)

    @classmethod
    def construct(cls, *args, **kwargs):
        def wrapper():
            return SimulatedEnv(*args, **kwargs)

        return wrapper


if __name__ == '__main__':
    minimal_config = {'frame_shape': (3, 105, 80), 'stacking': 4, 'hidden_size': 64, 'compress_steps': 5,
                      'hidden_layers': 2, 'device': 'cpu', 'dropout': 0.15}
    env = SimulatedEnv(minimal_config, gym.spaces.Discrete(4))
    env.restart(torch.randint(255, (12, 105, 80), dtype=torch.uint8))
    check_env(env)
