import os
import argparse
from time import strftime
from warnings import warn

import torch
import numpy as np
from tqdm import trange

from atari_utils.envs import make_env
from atari_utils.evaluation import evaluate
from atari_utils.policy_wrappers import SampleWithTemperature
from atari_utils.ppo_wrapper import PPO
from atari_utils.utils import print_config, disable_baselines_logging, fix_ulimit
from simple.subproc_vec_env import make_simulated_env
from simple.trainer import Trainer
from simple.next_frame_predictor import NextFramePredictor


class SimPLe:

    def __init__(self, config):
        self.config = config
        self.real_env = make_env(
            config.env_name,
            config.device,
            render=config.render_training,
            frame_shape=config.frame_shape,
            record=True,
            rollout_length=50,
            gamma=config.ppo_gamma
        )
        self.model = NextFramePredictor(config, self.real_env.action_space.n).to(config.device)
        self.trainer = Trainer(self.model, config)
        self.simulated_env = make_simulated_env(config, self.model, self.real_env.action_space)
        self.agent = PPO(
            self.simulated_env,
            config.device,
            gamma=config.ppo_gamma,
            use_wandb=config.use_wandb,
            num_steps=self.config.rollout_length,
            num_mini_batch=5,
            lr=config.ppo_lr
        )

        if self.config.use_wandb:
            import wandb
            wandb.init(project='msth', name=self.config.experiment_name, config=config)
            wandb.watch(self.model)

    def train_agent_real_env(self):
        self.real_env.envs[0].new_epoch()
        self.agent.set_env(self.real_env)

        self.agent.learn(6400, score_training=False)

        if self.config.save_models:
            self.agent.save(os.path.join('models', 'ppo.pt'))

    def train_agent_sim_env(self, epoch):
        z = 1
        if epoch == 7 or epoch == 11:
            z = 2
        if epoch == 14:
            z = 3
        n = 1000
        self.agent.set_env(self.simulated_env)

        with trange(n * z, desc='Training agent in simulated env') as t:
            for _ in t:
                for i in range(self.config.agents):
                    if i == self.config.agents - 1:
                        initial_frames, initial_actions = self.real_env.envs[0].get_first_small_rollout()
                    else:
                        sequence, actions, _, _, _ = self.real_env.envs[0].sample_buffer()
                        index = int(torch.randint(len(sequence) - self.config.stacking, (1,)))
                        initial_frames = sequence[index:index + self.config.stacking]
                        initial_actions = actions[index:index + self.config.stacking]

                    self.simulated_env.env_method('restart', initial_frames, initial_actions, indices=i)

                losses = self.agent.learn(
                    self.config.rollout_length * self.config.agents,
                    verbose=False,
                    score_training=False
                )
                t.set_postfix(losses)

        if self.config.save_models:
            self.agent.save(os.path.join('models', 'ppo.pt'))

    def evaluate_agent(self):
        scores = evaluate(
            SampleWithTemperature(self.agent),
            self.config.env_name,
            self.config.device,
            render=self.config.render_training,
            frame_shape=config.frame_shape,
            agents=self.config.agents
        )

        if self.config.use_wandb:
            import wandb
            wandb.log({'eval_score': np.mean(scores), 'eval_score_std': np.std(scores)})

    def load_models(self):
        self.model.load_state_dict(torch.load(os.path.join('models', 'model.pt')))
        if self.model.decouple_optimizers:  # FIXME
            self.model.stochastic_model.bits_predictor.load_state_dict(
                torch.load(os.path.join('models', 'bits_predictor.pt'))
            )
            self.model.reward_estimator.load_state_dict(torch.load(os.path.join('models', 'reward_model.pt')))
            self.model.value_estimator.load_state_dict(torch.load(os.path.join('models', 'value_model.pt')))
        self.agent = PPO(self.simulated_env, config, num_steps=self.config.rollout_length, num_mini_batch=5)
        self.agent.load(os.path.join('models', 'ppo.pt'))

    def train(self):
        self.train_agent_real_env()
        if not self.real_env.envs[0].buffer:
            self.__init__(self.config)
            warn('The agent was not able to collect even one full rollout in the real environment.\n'
                 'Restarting the training.\n'
                 'If this happens continuously, consider improving the agent, reducing the rollout length,'
                 'or changing the environment.')
            return self.train()
        self.evaluate_agent()

        for epoch in trange(15, desc='Epoch'):
            self.trainer.train(epoch, self.real_env.envs[0])
            self.train_agent_sim_env(epoch)
            self.evaluate_agent()
            self.train_agent_real_env()
            self.evaluate_agent()

        self.real_env.close()
        self.simulated_env.close()

    def test(self):
        evaluate(
            SampleWithTemperature(self.agent),
            self.config.env_name,
            self.config.device,
            render=self.config.render_evaluation,
            frame_shape=config.frame_shape,
            agents=self.config.agents
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=16)
    parser.add_argument('--agent-evaluation-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--bottleneck-bits', type=int, default=128)
    parser.add_argument('--bottleneck-noise', type=float, default=0.1)
    parser.add_argument('--clip-grad-norm', type=float, default=1.0)
    parser.add_argument('--compress-steps', type=int, default=5)
    parser.add_argument('--decouple-optimizers', default=False, action='store_true')  # TODO benchmark me
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--done-on-final-rollout-step', default=True, action='store_false')  # TODO benchmark me
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--env-name', type=str, default='Freeway')
    parser.add_argument('--experiment-name', type=str, default=strftime('%d-%m-%y-%H:%M:%S'))
    parser.add_argument('--filter-double-steps', type=int, default=3)
    parser.add_argument('--frame-shape', type=int, nargs=3, default=(3, 105, 80))
    parser.add_argument('--hidden-layers', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=96)
    parser.add_argument('--latent-rnn-max-sampling', type=float, default=0.5)
    parser.add_argument('--latent-state-size', type=int, default=128)
    parser.add_argument('--latent-use-max-probability', type=float, default=0.8)
    parser.add_argument('--load-models', default=False, action='store_true')
    parser.add_argument('--ppo-gamma', type=float, default=0.99)
    parser.add_argument('--ppo-lr', type=float, default=1e-4)
    parser.add_argument('--recurrent-state-size', type=int, default=64)
    parser.add_argument('--render-evaluation', default=False, action='store_true')
    parser.add_argument('--render-training', default=False, action='store_true')  # FIXME doesn't work
    parser.add_argument('--residual-dropout', type=float, default=0.5)
    parser.add_argument('--reward-model-batch-size', type=int, default=16)
    parser.add_argument('--rollout-length', type=int, default=50)
    parser.add_argument('--save-models', default=False, action='store_true')
    parser.add_argument('--scheduled-sampling-decay-steps', type=int, default=22250)  # TODO benchmark me vs 40000
    parser.add_argument('--stacking', type=float, default=4)
    parser.add_argument('--target-loss-clipping', type=float, default=0.03)
    parser.add_argument('--use-wandb', default=False, action='store_true')
    parser.add_argument('--value-model-batch-size', type=int, default=16)
    config = parser.parse_args()

    print_config(config)
    disable_baselines_logging()
    fix_ulimit()

    if config.save_models and not os.path.isdir('models'):
        os.mkdir('models')

    simple = SimPLe(config)
    if config.load_models:
        simple.load_models()
    else:
        simple.train()
    simple.test()
