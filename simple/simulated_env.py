import cv2
# See https://stackoverflow.com/questions/54013846/pytorch-dataloader-stucked-if-using-opencv-resize-method
# See https://github.com/pytorch/pytorch/issues/1355
cv2.setNumThreads(0)
from gym import Env, spaces
import numpy as np


class SimulatedEnv(Env):

    def __init__(self, config, action_space, main=False):
        super(SimulatedEnv, self).__init__()
        self.config = config
        self.action_space = action_space
        shape = list(self.config.frame_shape)
        shape[0] *= self.config.stacking
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.initial_frames = None
        self.last_state = None
        self.main = main

    def get_initial_frames(self):
        return self.initial_frames.cpu()

    def step(self, args):
        state, reward, done = args

        self.last_state = state

        if self.main and self.config.render_training:
            self.render()

        return state, reward, done, {}

    def reset(self):
        if self.initial_frames is None:
            raise ValueError('This environment has not been initialized. Call the restart function.')
        return self.initial_frames.cpu()

    def render(self, mode='human'):
        if self.last_state is None:
            return

        frame = self.last_state[-3:]
        frame = frame.permute((1, 2, 0))
        frame = frame[:, :, [2, 1, 0]]
        frame = frame.detach().cpu().numpy()
        cv2.namedWindow('World model', cv2.WINDOW_NORMAL)
        cv2.imshow('World model', frame)
        cv2.waitKey(1)

    def close(self):
        super().close()
        try:
            cv2.destroyWindow('World model')
        except:
            pass

    def restart(self, initial_frames):
        self.initial_frames = initial_frames


def _make_simulated_env(config, action_space, main=False):
    return SimulatedEnv(config, action_space, main)
