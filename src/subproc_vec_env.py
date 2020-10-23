import torch
from stable_baselines.common.vec_env import subproc_vec_env
from utils import one_hot_encode


class SubprocVecEnv(subproc_vec_env.SubprocVecEnv):

    def __init__(self, env_fns, model, n_action, config):
        super().__init__(env_fns)
        self.model = model
        self.n_action = n_action
        self.config = config
        self.step_count = 0

    def step_async(self, actions):
        self.step_count += 1

        x = torch.stack(self.env_method('get_frame_stack_'))
        x = x.to(self.config.device).float() / 255
        actions = one_hot_encode(actions, self.n_action, dtype=torch.float32).to(self.config.device)

        self.model.eval()
        with torch.no_grad():
            states, rewards, values = self.model(x, actions)

        states = torch.argmax(states, dim=1).byte()
        rewards = (torch.argmax(rewards, dim=1).cpu() - 1).numpy().astype('float')

        if self.step_count == self.config.rollout_length:
            self.step_count = 0
            rewards += values.cpu().numpy().astype('float')

        for remote, arg in zip(self.remotes, zip(states, list(rewards))):
            remote.send(('step', arg))
        self.waiting = True
