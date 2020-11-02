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
        self.initial_frames = None
        self.initial_actions = None
        self.last_state = None
        self.main = main

    def get_initial_short_rollouts(self):
        return self.initial_frames, self.initial_actions

    def step(self, args):
        state, reward = args

        state = state.permute((1, 2, 0)).cpu().detach().numpy()
        self.last_state = state

        if self.main and self.config.render_training:
            self.render()

        return state, reward, False, {}

    def reset(self):
        return self.initial_frames[-1].permute((1, 2, 0)).cpu().detach().numpy()

    def render(self, mode='human'):
        if self.last_state is None:
            return

        frame = self.last_state
        frame = frame[:, :, [2, 1, 0]]
        cv2.namedWindow('World model', cv2.WINDOW_NORMAL)
        cv2.imshow('World model', frame)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow('World model')

    def restart(self, initial_frames, initial_actions):
        self.initial_frames = initial_frames
        self.initial_actions = initial_actions

    @classmethod
    def construct(cls, *args, **kwargs):
        def wrapper():
            return SimulatedEnv(*args, **kwargs)

        return wrapper
