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
        self.frames = None
        self.actions = None

    def step_async(self, actions):
        if self.step_count == 0:
            res = self.env_method('get_initial_short_rollouts')

            initial_frames = torch.stack([entry[0] for entry in res], dim=1)
            initial_frames = initial_frames.float() / 255
            initial_frames = initial_frames.to(self.config.device)

            initial_actions = torch.stack([entry[1] for entry in res], dim=1)
            initial_actions = initial_actions.float()
            initial_actions = initial_actions.to(self.config.device)

            self.model.warmup(initial_frames[:self.config.stacking], initial_actions[:self.config.stacking - 1])

            self.frames = initial_frames[-1]

        self.step_count += 1

        self.model.eval()
        with torch.no_grad():
            actions = one_hot_encode(actions, self.n_action, dtype=torch.float32)
            actions = actions.to(self.config.device)
            states, rewards, values = self.model(self.frames, actions)

            states = torch.argmax(states, dim=1)
            self.frames = states.float() / 255
            states = states.detach().cpu().byte()
            rewards = (torch.argmax(rewards, dim=1).detach().cpu() - 1).numpy().astype('float')

            # if int(rewards[0]) == 2:
            #     print('reward')

            if self.step_count == self.config.rollout_length:
                self.step_count = 0
                rewards += values.detach().cpu().numpy().astype('float')

            # print(rewards[0])

            for remote, arg in zip(self.remotes, zip(states, list(rewards))):
                remote.send(('step', arg))
            self.waiting = True
