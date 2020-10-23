import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse
from stable_baselines.common.vec_env import DummyVecEnv
from subproc_vec_env import SubprocVecEnv
import sys
import time
from next_frame_predictor import NextFramePredictor
from trainer import Trainer
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy
from atari_env import AtariEnv
from simulated_env import SimulatedEnv
from time import strftime
import torch
import wandb
import numpy as np
from stable_baselines import PPO2


class SimPLe:

    def __init__(self, config):
        self.config = config
        self.real_env = AtariEnv(config, base_env_name=config.env_name)
        self.model = NextFramePredictor(config, self.real_env.action_space.n).to(config.device)
        self.trainer = Trainer(self.model, config)
        self.simulated_env = [
            SimulatedEnv.construct(config, self.real_env.action_space, i == 0) for i in range(config.agents)
        ]
        self.simulated_env = SubprocVecEnv(self.simulated_env, self.model, self.real_env.action_space.n, config)
        self.agent = PPO2(
            CnnPolicy,
            self.simulated_env,
            gamma=self.config.ppo_gamma,
            n_steps=self.config.rollout_length,
            nminibatches=5  # TODO hyper-parameter search
        )

        if self.config.use_wandb:
            wandb.init(project='msth', name=self.config.experiment_name, config=config)
            wandb.watch(self.model)

    def set_agent_env(self, env):
        assert env == self.real_env or env == self.simulated_env
        if env == self.real_env:
            self.agent.set_env(DummyVecEnv((lambda: self.real_env,)))
        else:
            self.agent.set_env(self.simulated_env)
        # PPO2 is not updated properly with set_env, this fixes the issue
        self.agent.n_batch = self.agent.n_envs * self.agent.n_steps

    def train_agent_real_env(self):
        if self.config.verbose:
            sys.stdout.write('Training agent in real env\n')
            sys.stdout.flush()

        self.set_agent_env(self.real_env)
        self.agent.learn(total_timesteps=6400)

    def train_world_model(self, epoch):
        steps = [15, 1500, 15000][self.config.complexity]

        if epoch == 0:
            steps *= 3

        for j in range(0, steps, self.config.inference_batch_size // self.config.backprop_batch_size):
            # Scheduled sampling
            if epoch == 0:
                epsilon = max(0., (config.scheduled_sampling_decay_steps - j) / config.scheduled_sampling_decay_steps)
            else:
                epsilon = 0

            losses = self.trainer.train(*self.real_env.sample_buffer(), epsilon)

            if self.config.verbose:
                progress = format(100 * j / steps, '.2f')
                losses = [format(loss, '.4e') for loss in losses]

                sys.stdout.write(
                    f'Training world model: {progress}% - loss_reconstruct: {losses[0]} - loss_lstm: {losses[1]}\r'
                )
                sys.stdout.flush()

        if self.config.verbose:
            sys.stdout.write(
                f'Training world model: 100.00% - loss_reconstruct: {losses[0]} - loss_lstm: {losses[1]}\n'
            )
            sys.stdout.flush()

    def train_reward_model(self, epoch):
        steps = [15, 1500, 1500][self.config.complexity]

        if epoch == 0:
            steps *= 3

        r_epochs = [10, 10, 100][self.config.complexity]

        for r_epoch in range(r_epochs):
            reward_counts = [0] * 3
            for _, _, rewards, _, _ in self.real_env.buffer:
                for reward in rewards:
                    reward_counts[int(reward)] += 1

            frame_stacks = []
            actions = []
            rewards = []

            non_zeros = 3
            for reward_count in reward_counts:
                if reward_count == 0:
                    non_zeros -= 1

            for i, reward_count in enumerate(reward_counts):
                if reward_count == 0:
                    continue

                if reward_count < max(reward_counts) and reward_count < steps / non_zeros:
                    multiplier = int(steps / non_zeros / reward_count)
                    for sequence, b_actions, b_rewards, _, _ in self.real_env.buffer:
                        for j, reward in enumerate(b_rewards):
                            if int(reward) == i:
                                frame_stack = sequence[j * self.config.frame_shape[0]:(j + self.config.stacking) *
                                                                                      self.config.frame_shape[0]]
                                action = b_actions[j]
                                frame_stacks.extend([frame_stack] * multiplier)
                                actions.extend([action] * multiplier)
                                rewards.extend([reward] * multiplier)
                else:
                    count = 0
                    while count < steps / non_zeros:
                        sequence, b_actions, b_rewards, _, _ = self.real_env.sample_buffer()
                        for j, reward in enumerate(b_rewards):
                            if int(reward) == i:
                                frame_stack = sequence[j * self.config.frame_shape[0]:(j + self.config.stacking) *
                                                                                      self.config.frame_shape[0]]
                                action = b_actions[j]
                                frame_stacks.append(frame_stack)
                                actions.append(action)
                                rewards.append(reward)
                                count += 1

            reward_counts = [0] * 3
            for reward in rewards:
                reward_counts[int(reward)] += 1

            frame_stacks = torch.stack(frame_stacks)
            actions = torch.stack(actions)
            rewards = torch.stack(rewards)

            permutation = torch.randperm(len(frame_stacks))
            frame_stacks = frame_stacks[permutation]
            actions = actions[permutation]
            rewards = rewards[permutation]

            frame_stacks = frame_stacks.view((-1, *frame_stacks.shape[1:]))
            rewards = rewards.view((-1,))

            loss = self.trainer.train_reward_model(frame_stacks, actions, rewards)

            if self.config.verbose:
                progress = format(r_epoch * 100 / r_epochs, '.2f')
                loss = format(loss, '.4e')
                sys.stdout.write(f'Training reward model: {progress}% - loss_reward: {loss}\r')
                sys.stdout.flush()

        if self.config.verbose:
            sys.stdout.write(f'Training reward model: 100.00% - loss_reward: {loss}\n')
            sys.stdout.flush()

    def train_value_model(self, epoch):
        steps = [15, 1500, 1500][self.config.complexity]

        if epoch == 0:
            steps *= 3

        v_epochs = [10, 10, 100][self.config.complexity]

        for v_epoch in range(v_epochs):
            frame_stacks = []
            actions = []
            values = []

            for _ in range(steps // self.config.inference_batch_size):
                sequence, b_actions, _, _, b_values = self.real_env.sample_buffer()
                for j, value in enumerate(b_values):
                    frame_stack = sequence[j * self.config.frame_shape[0]:(j + self.config.stacking) *
                                                                          self.config.frame_shape[0]]
                    action = b_actions[j]
                    frame_stacks.append(frame_stack)
                    actions.append(action)
                    values.append(value)

            frame_stacks = torch.stack(frame_stacks)
            actions = torch.stack(actions)
            values = torch.stack(values)

            permutation = torch.randperm(len(frame_stacks))
            frame_stacks = frame_stacks[permutation]
            actions = actions[permutation]
            values = values[permutation]

            frame_stacks = frame_stacks.view((-1, *frame_stacks.shape[1:]))
            values = values.view((-1,))

            loss = self.trainer.train_value_model(frame_stacks, actions, values)

            if self.config.verbose:
                progress = format(v_epoch * 100 / v_epochs, '.2f')
                loss = format(loss, '.4e')
                sys.stdout.write(f'Training value model: {progress}% - loss_value: {loss}\r')
                sys.stdout.flush()

        if self.config.verbose:
            sys.stdout.write(f'Training value model: 100.00% - loss_value: {loss}\n')
            sys.stdout.flush()

    def train_agent_sim_env(self, epoch):
        z = 1
        if epoch == 7 or epoch == 11:
            z = 2
        if epoch == 14:
            z = 3
        n = [10, 100, 1000][self.config.complexity]
        self.set_agent_env(self.simulated_env)
        for ppo_epoch in range(n * z):
            if self.config.verbose:
                progress = format(100 * ppo_epoch / n / z, '.2f')
                sys.stdout.write(f'Training agent in simulated env: {progress}%\r')
                sys.stdout.flush()

            for i in range(self.config.agents):
                sequence = self.real_env.sample_buffer()[0]
                index = int(torch.randint(len(sequence) // self.config.frame_shape[0] - self.config.stacking, (1,)))
                index *= self.config.frame_shape[0]
                frame_stack = sequence[index:index + self.config.frame_shape[0] * self.config.stacking]

                self.simulated_env.env_method('restart', frame_stack, indices=i)

            self.agent.learn(total_timesteps=self.config.rollout_length * self.config.agents)

        if self.config.verbose:
            sys.stdout.write(f'Training agent in simulated env: 100.00%\n')
            sys.stdout.flush()

    def evaluate_agent(self):
        if self.config.agent_evaluation_epochs < 1:
            return

        self.real_env.set_recording(False)
        self.set_agent_env(self.real_env)
        cum_rewards = []
        for i in range(self.config.agent_evaluation_epochs):
            if self.config.verbose:
                progress = format(100 * i / self.config.agent_evaluation_epochs, '.2f')
                sys.stdout.write(f'Evaluating agent in real env: {progress}%\r')
                sys.stdout.flush()

            obs = self.real_env.reset()
            cum_reward = 0
            for _ in range(1000):
                action = self.agent.predict(obs)[0]
                obs, reward, done, _ = self.real_env.step(action)
                self.real_env.render()
                cum_reward += reward
                if done:
                    break
            cum_rewards.append(cum_reward)
        self.real_env.set_recording(True)

        cum_reward = np.mean(cum_rewards)

        if self.config.verbose:
            sys.stdout.write(f'Evaluating agent in real env: 100.00%\n')
            sys.stdout.flush()
            sys.stdout.write(f'Average cumulative reward: {cum_reward}\n')
            sys.stdout.flush()

        if self.config.use_wandb:
            wandb.log({'cum_reward': cum_reward})

    def train(self):
        self.train_agent_real_env()

        for epoch in range(15):
            if self.config.verbose:
                sys.stdout.write(f'Epoch: {epoch + 1}/{15}\n')
                sys.stdout.flush()

            self.train_world_model(epoch)
            self.train_reward_model(epoch)
            self.train_value_model(epoch)
            self.train_agent_sim_env(epoch)
            self.train_agent_real_env()
            self.evaluate_agent()

        self.real_env.close()
        self.simulated_env.close()

    def test(self):
        while True:
            observation = self.real_env.reset()
            while True:
                action = self.agent.predict(observation)[0]
                observation, _, done, _ = self.real_env.step(action)
                self.real_env.render()
                time.sleep(1 / 60)
                if done:
                    break


if __name__ == '__main__':
    # import os
    #
    # os.environ['WANDB_MODE'] = 'dryrun'

    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=4)
    parser.add_argument('--agent-evaluation-epochs', type=int, default=10)
    parser.add_argument('--backprop-batch-size', type=int, default=2)
    parser.add_argument('--bottleneck-bits', type=int, default=128)
    parser.add_argument('--bottleneck-noise', type=float, default=0.1)
    parser.add_argument('--clip-grad-norm', type=float, default=1.0)
    parser.add_argument('--complexity', type=int, default=2)
    parser.add_argument('--compress-steps', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--env-name', type=str, default='KungFuMasterDeterministic-v4')
    parser.add_argument('--experiment-name', type=str, default=strftime('%d-%m-%y-%H:%M:%S'))
    parser.add_argument('--frame-shape', type=int, nargs=3, default=(3, 105, 80))
    parser.add_argument('--hidden-layers', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=96)
    parser.add_argument('--inference-batch-size', type=int, default=16)
    parser.add_argument('--latent-rnn-max-sampling', type=float, default=0.5)
    parser.add_argument('--latent-state-size', type=int, default=128)
    parser.add_argument('--latent-use-max-probability', type=float, default=0.8)
    parser.add_argument('--ppo-gamma', type=float, default=0.95)
    parser.add_argument('--residual-dropout', type=float, default=0.5)
    parser.add_argument('--reward-model-batch-size', type=int, default=16)
    parser.add_argument('--rollout-length', type=int, default=50)
    parser.add_argument('--scheduled-sampling-decay-steps', type=int, default=40000)
    parser.add_argument('--stacking', type=float, default=4)
    parser.add_argument('--target-loss-clipping', type=float, default=0.03)
    parser.add_argument('--use-stochastic-model', type=bool, default=True)
    parser.add_argument('--use-wandb', type=bool, default=True)
    parser.add_argument('--value-model-batch-size', type=int, default=16)
    parser.add_argument('--verbose', type=bool, default=True)

    for use_stochastic_model in [False]:
        config = parser.parse_args()
        config.use_stochastic_model = use_stochastic_model
        config.experiment_name = strftime('%d-%m-%y-%H:%M:%S')

        if config.verbose:
            args = vars(config)
            max_len = 0
            for arg in args:
                max_len = max(max_len, len(arg))
            for arg in args:
                value = str(getattr(config, arg))
                display = '{:<%i}: {}' % (max_len + 1)
                print(display.format(arg, value))

        simple = SimPLe(config)
        simple.train()
        # input('Press enter to take over the world...')
        # simple.test()
