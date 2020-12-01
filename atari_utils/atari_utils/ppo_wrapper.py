from collections import deque
from math import ceil

import torch
import numpy as np
from tqdm import trange

from a2c_ppo_acktr import ppo
from a2c_ppo_acktr.policy import Policy
from a2c_ppo_acktr.rollout_storage import RolloutStorage
from a2c_ppo_acktr.utils import update_linear_schedule, Augmentation
from atari_utils.evaluation import evaluate
from atari_utils.policy_wrappers import PolicyWrapper


class PPO:

    def __init__(self,
                 env,
                 device,
                 augmentation=Augmentation(),
                 num_steps=128,
                 gamma=0.99,
                 lr=2.5e-4,
                 clip_param=0.1,
                 value_loss_coef=0.5,
                 num_mini_batch=4,
                 entropy_coef=0.01,
                 eps=1e-5,
                 max_grad_norm=0.5,
                 ppo_epoch=4,
                 use_wandb=False
                 ):

        self.env = env
        self.device = device
        self.gamma = gamma
        self.lr = lr
        self.num_steps = num_steps
        self.num_processes = self.env.num_envs
        self.use_wandb = use_wandb

        self.actor_critic = Policy(self.env.observation_space.shape, self.env.action_space)
        self.actor_critic.to(self.device)

        self.agent = ppo.PPO(
            self.actor_critic,
            clip_param,
            ppo_epoch,
            num_mini_batch,
            value_loss_coef,
            entropy_coef,
            lr=lr,
            eps=eps,
            max_grad_norm=max_grad_norm,
            augmentation=augmentation
        )

    def learn(
            self,
            steps,
            verbose=True,
            eval_env_name=None,
            eval_policy_wrapper=PolicyWrapper,
            eval_episodes=30,
            eval_agents=None,
            evaluations=10,
            graph=False,
            score_training=True
    ):
        if eval_agents is None:
            eval_agents = self.env.num_envs

        num_updates = steps // self.num_steps // self.num_processes

        rollouts = RolloutStorage(
            self.num_steps,
            self.num_processes,
            self.env.observation_space.shape,
            self.env.action_space
        )

        obs = self.env.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(self.device)
        score_queue = deque(maxlen=10)
        moving_average_score = []
        eval_mean_scores = []
        eval_mean_scores_std = []
        if verbose:
            iterator = trange(num_updates, desc='Training agent', unit_scale=self.num_steps * self.num_processes)
        else:
            iterator = range(num_updates)

        for j in iterator:
            update_linear_schedule(self.agent.optimizer, j, num_updates, self.lr)
            for step in range(self.num_steps):
                # Sample actions
                with torch.no_grad():
                    # value, action, action_log_prob = self.actor_critic.act(rollouts.obs[step])
                    value, action, action_log_prob = self.actor_critic.act(self.agent.augmentation(rollouts.obs[step]))

                # Obser reward and next obs
                obs, reward, done, infos = self.env.step(action)

                for info in infos:
                    if 'r' in info.keys():
                        score_queue.append(info['r'])
                        moving_average_score.append(np.mean(score_queue))

                # If done then clean the history of observations.
                masks = (~torch.tensor(done)).float().unsqueeze(1)
                rollouts.insert(obs, action, action_log_prob, value, reward, masks)

            with torch.no_grad():
                next_value = self.actor_critic.get_value(rollouts.obs[-1]).detach()

            rollouts.compute_returns(next_value, True, self.gamma, 0.95)
            value_loss, action_loss, dist_entropy = self.agent.update(rollouts)
            rollouts.after_update()

            if (j % ceil(num_updates / evaluations) == 0 or j == num_updates - 1) and eval_env_name is not None:
                scores = evaluate(
                    eval_policy_wrapper(self),
                    eval_env_name,
                    self.device,
                    agents=eval_agents,
                    episodes=eval_episodes,
                    verbose=False
                )
                eval_mean_scores.append(np.mean(scores))
                eval_mean_scores_std.append(np.std(scores))

            metrics = {
                'ppo_value_loss': float(value_loss),
                'ppo_action_loss': float(action_loss)
            }
            if score_queue and score_training:
                metrics.update({'mean_score': moving_average_score[-1]})
            if eval_mean_scores:
                metrics.update({'mean_eval_score': eval_mean_scores[-1]})
            if verbose:
                iterator.set_postfix(metrics)
            if self.use_wandb:
                import wandb
                wandb.log(metrics)

        if graph:
            import matplotlib.pyplot as plot
            eval_mean_scores = np.array(eval_mean_scores)
            eval_mean_scores_std = np.array(eval_mean_scores_std)
            plot.style.use('seaborn-darkgrid')
            x = np.linspace(0, steps, len(moving_average_score))
            plot.plot(x, moving_average_score)
            if len(eval_mean_scores) > 0:
                x = np.linspace(0, steps, len(eval_mean_scores))
                plot.fill_between(
                    x,
                    eval_mean_scores - eval_mean_scores_std,
                    eval_mean_scores + eval_mean_scores_std,
                    color='tab:orange',
                    alpha=0.5
                )
                plot.plot(x, eval_mean_scores, c='tab:orange')
            plot.title('Training curves')
            plot.xlabel('Step')
            plot.ylabel('Score')
            legend = ['Training']
            if len(eval_mean_scores) > 0:
                legend.append('Evaluation')
            plot.legend(legend, loc='upper left')
            plot.show()

        return eval_mean_scores

    def set_env(self, env):
        self.env = env
        self.num_processes = env.num_envs

    def save(self, path):
        torch.save(self.actor_critic, path)

    def load(self, path):
        self.actor_critic = torch.load(path)

    def act(self, obs, full_log_prob=False):
        return self.actor_critic.act(obs, deterministic=True, full_log_prob=full_log_prob)
