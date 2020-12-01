import cv2
from gym import Env, spaces
import numpy as np


class SimulatedEnv(Env):

    def __init__(self, config, action_space, main=False):
        super(SimulatedEnv, self).__init__()
        self.config = config
        self.action_space = action_space
        self.observation_space = spaces.Box(low=0, high=255, shape=self.config.frame_shape, dtype=np.uint8)
        self.initial_frames = None
        self.initial_actions = None
        self.last_state = None
        self.main = main

    def get_initial_short_rollouts(self):
        return self.initial_frames, self.initial_actions

    def step(self, args):
        state, reward, done = args

        # state = state.cpu().detach()
        self.last_state = state

        if self.main and self.config.render_training:
            self.render()

        return state, reward, done, {}

    def reset(self):
        return self.initial_frames[-1].cpu().detach()

    def render(self, mode='human'):
        if self.last_state is None:
            return

        frame = self.last_state
        frame = frame.permute((1, 2, 0))[:, :, [2, 1, 0]]
        frame = frame.detach().cpu().numpy()
        cv2.namedWindow('World model', cv2.WINDOW_NORMAL)
        cv2.imshow('World model', frame)
        cv2.waitKey(1)

    def close(self):
        super().close()
        cv2.destroyWindow('World model')

    def restart(self, initial_frames, initial_actions):
        self.initial_frames = initial_frames
        self.initial_actions = initial_actions


def _make_simulated_env(config, action_space, main=False):
    return SimulatedEnv(config, action_space, main)
