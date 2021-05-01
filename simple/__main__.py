import os
import argparse
from time import strftime
from warnings import warn

import torch
from tqdm import trange

from atari_utils.envs import make_env
from atari_utils.evaluation import evaluate
from atari_utils.logger import WandBLogger
from atari_utils.policy_wrappers import SampleWithTemperature
from atari_utils.ppo_wrapper import PPO
from atari_utils.utils import print_config, disable_baselines_logging
from simple.subproc_vec_env import make_simulated_env
from simple.trainer import Trainer
from simple.next_frame_predictor import NextFramePredictor


class SimPLe:

    def __init__(self, config):
        self.config = config
        self.logger = None
        if self.config.use_wandb:
            self.logger = WandBLogger()
        self.real_env = make_env(
            config.env_name,
            config.device,
            render=config.render_training,
            frame_shape=config.frame_shape,
            record=True,
            gamma=config.ppo_gamma,
            noop_max=config.noop_max
        )
        self.model = NextFramePredictor(config, self.real_env.action_space.n).to(config.device)
        self.trainer = Trainer(self.model, config)
        self.simulated_env = make_simulated_env(config, self.model, self.real_env.action_space)
        self.agent = PPO(
            self.simulated_env,
            config.device,
            gamma=config.ppo_gamma,
            num_steps=self.config.rollout_length,
            num_mini_batch=5,
            lr=config.ppo_lr
        )

        if self.config.use_wandb:
            import wandb
            wandb.init(project='SimPLe', name=self.config.experiment_name, config=config)
            wandb.watch(self.model)

    def random_search(self):
        self.real_env.reset()
        for _ in trange(6400, desc='Random exploration'):
            done = self.real_env.step(torch.randint(high=self.real_env.action_space.n, size=(1, 1)))[2]
            if done:
                self.real_env.reset()

    def collect_interactions(self):
        self.real_env.new_epoch()
        self.agent.set_env(self.real_env)
        agent = SampleWithTemperature(self.agent, temperature=1.0)
        obs = self.real_env.reset()
        for _ in trange(6400, desc='Collecting interactions'):
            with torch.no_grad():
                action = agent.act(obs)[1]
            obs, _, done, _ = self.real_env.step(action)
            if done[0]:
                obs = self.real_env.reset()

    def train_agent_sim_env(self, epoch, eval_period=100):
        z = 1
        if epoch == 7 or epoch == 11:
            z = 2
        if epoch == 14:
            z = 3
        n = 1000
        self.agent.set_env(self.simulated_env)

        postfix = {}

        with trange(n * z, desc='Training agent in simulated env') as t:
            for i in t:
                for j in range(self.config.agents):
                    if j == self.config.agents - 1 and self.config.simulation_flip_first_random_for_beginning:
                        initial_frames = self.real_env.get_first_small_rollout()
                    else:
                        initial_frames = self.real_env.sample_buffer(1)[0][0]

                    self.simulated_env.env_method('restart', initial_frames, indices=j)

                losses = self.agent.learn(
                    self.config.rollout_length * self.config.agents,
                    verbose=False,
                    score_training=False,
                    logger=self.logger,
                    use_ppo_lr_decay=self.config.use_ppo_lr_decay
                )
                postfix.update(losses)

                if eval_period > 0 and (i + 1) % eval_period == 0:
                    eval_metrics = self.evaluate_agent()
                    postfix.update(eval_metrics)

                t.set_postfix(postfix)

        if self.config.save_models:
            self.agent.save(os.path.join('models', 'ppo.pt'))

    def evaluate_agent(self):
        metrics = evaluate(
            SampleWithTemperature(self.agent),
            self.config.env_name,
            self.config.device,
            render=self.config.render_training,
            frame_shape=config.frame_shape,
            agents=15,
            noop_max=config.noop_max
        )

        if self.logger is not None:
            self.logger.log(metrics)

        return metrics

    def load_models(self):
        self.model.load_state_dict(torch.load(os.path.join('models', 'model.pt')))
        self.agent = PPO(self.simulated_env, config, num_steps=self.config.rollout_length, num_mini_batch=5)
        self.agent.load(os.path.join('models', 'ppo.pt'))

    def train(self):
        self.random_search()

        if not self.real_env.buffer:
            self.__init__(self.config)
            warn('The agent was not able to collect even one full rollout in the real environment.\n'
                 'Restarting the training.\n'
                 'If this happens continuously, consider improving the agent, reducing the rollout length,'
                 'or changing the environment.')
            return self.train()

        for epoch in trange(self.config.epochs, desc='Epoch'):
            self.collect_interactions()
            self.trainer.train(epoch, self.real_env)
            self.train_agent_sim_env(epoch)

        self.real_env.close()
        self.simulated_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--bottleneck-bits', type=int, default=128)
    parser.add_argument('--bottleneck-noise', type=float, default=0.1)
    parser.add_argument('--clip-grad-norm', type=float, default=1.0)
    parser.add_argument('--compress-steps', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--done-on-last-rollout-step', default=True, action='store_false')
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--env-name', type=str, default='Freeway')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--experiment-name', type=str, default=strftime('%d-%m-%y-%H:%M:%S'))
    parser.add_argument('--filter-double-steps', type=int, default=3)
    parser.add_argument('--frame-shape', type=int, nargs=3, default=(3, 105, 80))
    parser.add_argument('--hidden-layers', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=96)
    parser.add_argument('--input-noise', type=float, default=0.05)
    parser.add_argument('--latent-rnn-max-sampling', type=float, default=0.5)
    parser.add_argument('--latent-state-size', type=int, default=128)
    parser.add_argument('--latent-use-max-probability', type=float, default=0.8)
    parser.add_argument('--load-models', default=False, action='store_true')
    parser.add_argument('--noop-max', type=int, default=8)
    parser.add_argument('--ppo-gamma', type=float, default=0.99)
    parser.add_argument('--ppo-lr', type=float, default=1e-4)
    parser.add_argument('--recurrent-state-size', type=int, default=64)
    parser.add_argument('--render-evaluation', default=False, action='store_true')
    parser.add_argument('--render-training', default=False, action='store_true')
    parser.add_argument('--residual-dropout', type=float, default=0.5)
    parser.add_argument('--rollout-length', type=int, default=50)
    parser.add_argument('--save-models', default=False, action='store_true')
    parser.add_argument('--scheduled-sampling-decay-steps', type=int, default=22250)
    parser.add_argument('--simulation-flip-first-random-for-beginning', default=True, action='store_false')
    parser.add_argument('--stacking', type=float, default=4)
    parser.add_argument('--stack-internal-states', default=True, action='store_false')
    parser.add_argument('--target-loss-clipping', type=float, default=0.03)
    parser.add_argument('--use-ppo-lr-decay', default=False, action='store_true')
    parser.add_argument('--use-stochastic-model', default=True, action='store_false')
    parser.add_argument('--use-wandb', default=False, action='store_true')
    config = parser.parse_args()

    print_config(config)
    disable_baselines_logging()
    torch.multiprocessing.set_sharing_strategy('file_system')

    if config.save_models and not os.path.isdir('models'):
        os.mkdir('models')

    simple = SimPLe(config)
    if config.load_models:
        simple.load_models()
    else:
        simple.train()
