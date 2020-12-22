import torch

from atari_utils.utils import sample_with_temperature


class PolicyWrapper:

    def __init__(self, agent):
        self.agent = agent

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.agent, name)

    def act(self, obs):
        value, action, action_log_prob = self.agent.act(obs)
        return value, self.wrap(action), action_log_prob

    def wrap(self, action):
        return action


class EpsilonGreedy(PolicyWrapper):

    def __init__(self, agent, epsilon=0.05):
        super().__init__(agent)
        self.epsilon = epsilon

    def wrap(self, action):
        if float(torch.rand((1,))) < self.epsilon:
            return torch.randint(high=self.agent.env.action_space.n, size=(len(action), 1))
        return action


class SampleWithTemperature(PolicyWrapper):

    def __init__(self, agent, temperature=0.5):
        super().__init__(agent)
        self.temperature = temperature

    def act(self, obs):
        value, real_action, action_log_prob = self.agent.act(obs, full_log_prob=True)
        action = sample_with_temperature(action_log_prob, self.temperature)
        action = action.view(real_action.shape)
        return value, action, action_log_prob
