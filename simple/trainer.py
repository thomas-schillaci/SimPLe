import os

import torch
import torch.nn as nn
from torch.cuda import empty_cache
from torch.nn.utils import clip_grad_norm_
from tqdm import trange

from atari_utils.logger import WandBLogger
from simple.adafactor import Adafactor


class Trainer:

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.logger = None
        if self.config.use_wandb:
            self.logger = WandBLogger()

        self.optimizer = Adafactor(self.model.parameters())
        self.reward_optimizer = Adafactor(self.model.reward_estimator.parameters())

        self.model_step = 1
        self.reward_step = 1

    def train(self, epoch, real_env):
        self.train_world_model(epoch, real_env)
        self.train_reward_model(real_env.buffer, real_env.action_space.n)

    def train_world_model(self, epoch, env):
        self.model.train()

        steps = 15000

        if epoch == 0:
            steps *= 3

        iterator = trange(
            0,
            steps,
            self.config.rollout_length,
            desc='Training world model',
            unit_scale=self.config.rollout_length
        )
        for j in iterator:
            # Scheduled sampling
            if epoch == 0:
                decay_steps = self.config.scheduled_sampling_decay_steps
                inv_base = torch.exp(torch.log(torch.tensor(0.01)) / (decay_steps // 4))
                epsilon = inv_base ** max(decay_steps // 4 - j, 0)
                progress = min(j / decay_steps, 1)
                progress = progress * (1 - 0.01) + 0.01
                epsilon *= progress
                epsilon = 1 - epsilon
            else:
                epsilon = 0

            data = env.sample_buffer(self.config.batch_size)
            losses = self.train_world_model_impl(*data, epsilon)

            losses = {key: format(losses[key], '.4e') for key in losses.keys()}
            iterator.set_postfix(losses)

        empty_cache()
        if self.config.save_models:
            torch.save(self.model.state_dict(), os.path.join('models', 'model.pt'))

    def train_world_model_impl(self, sequences, actions, rewards, targets, values, epsilon=0.0):
        del rewards
        assert sequences.dtype == torch.uint8
        assert actions.dtype == torch.uint8
        assert targets.dtype == torch.uint8
        assert values.dtype == torch.float32

        sequences = sequences.to(self.config.device)
        actions = actions.to(self.config.device)
        targets = targets.to(self.config.device)
        values = values.to(self.config.device)

        sequences = sequences.float() / 255
        noise_prob = torch.tensor([[self.config.input_noise, 1 - self.config.input_noise]])
        noise_prob = torch.softmax(torch.log(noise_prob), dim=-1)
        noise_mask = torch.multinomial(noise_prob, sequences.numel(), replacement=True).view(sequences.shape)
        noise_mask = noise_mask.to(sequences)
        sequences = sequences * noise_mask + torch.median(sequences) * (1 - noise_mask)

        actions = actions.float()
        targets = targets.long()

        self.model.train()
        if self.config.stack_internal_states:
            self.model.init_internal_states(self.config.batch_size)

        frames = sequences[:, :self.config.stacking]
        n_losses = 4 if self.config.use_stochastic_model else 3
        losses = torch.empty((self.config.rollout_length, n_losses))

        for i in range(self.config.rollout_length):
            action = actions[:, i]
            target = targets[:, i]
            target_input = target.float() / 255
            value = values[:, i]

            input_shape = (
                self.config.batch_size,
                self.config.stacking * self.config.frame_shape[0],
                *self.config.frame_shape[1:]
            )
            frames_pred, _, values_pred = self.model(frames.view(input_shape), action, target_input, epsilon)

            if i < self.config.rollout_length - 1:
                for j in range(self.config.batch_size):
                    if float(torch.rand((1,))) < epsilon:
                        frame = sequences[j, i + self.config.stacking]
                    else:
                        frame = torch.argmax(frames_pred[j], dim=0).float() / 255
                    frames[j] = torch.cat((frames[j, 1:], frame.unsqueeze(0)), dim=0)

            loss_reconstruct = nn.CrossEntropyLoss(reduction='none')(frames_pred, target)
            clip = torch.tensor(self.config.target_loss_clipping).to(self.config.device)
            loss_reconstruct = torch.max(loss_reconstruct, clip)
            loss_reconstruct = loss_reconstruct.mean() - self.config.target_loss_clipping

            loss_value = nn.MSELoss()(values_pred, value)
            loss = loss_reconstruct + loss_value
            if self.config.use_stochastic_model:
                loss_lstm = self.model.stochastic_model.get_lstm_loss()
                loss = loss + loss_lstm

            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()

            tab = [float(loss), float(loss_reconstruct), float(loss_value)]
            if self.config.use_stochastic_model:
                tab.append(float(loss_lstm))
            losses[i] = torch.tensor(tab)

        losses = torch.mean(losses, dim=0)
        dict_losses = {
            'loss': float(losses[0]),
            'loss_reconstruct': float(losses[1]),
            'loss_value': float(losses[2]),
        }
        if self.config.use_stochastic_model:
            dict_losses.update({'loss_lstm': float(losses[3])})
        losses = dict_losses

        if self.logger is not None:
            d = {'model_step': self.model_step, 'epsilon': epsilon}
            d.update(losses)
            self.logger.log(d)
            self.model_step += self.config.rollout_length

        del sequences, targets, actions

        return losses

    def train_reward_model(self, buffer, n_action):
        self.model.eval()
        self.model.reward_estimator.train()

        steps = 500
        generator = self.reward_model_generator(buffer, steps, n_action)

        losses = []
        t = trange(0, steps, desc='Training reward model')
        for frames, actions, rewards in generator:
            if self.config.stack_internal_states:
                self.model.init_internal_states(self.config.reward_model_batch_size)

            input_shape = (
                self.config.reward_model_batch_size,
                self.config.stacking * self.config.frame_shape[0],
                *self.config.frame_shape[1:]
            )
            reward_pred = self.model(frames.view(input_shape), actions[-1])[1]
            loss = nn.CrossEntropyLoss()(reward_pred, rewards)

            self.reward_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.reward_estimator.parameters(), self.config.clip_grad_norm)
            self.reward_optimizer.step()

            losses.append(float(loss))

            loss = format(float(loss), '.4e')
            t.set_postfix({'loss_reward': loss})
            t.update()

            if self.logger is not None:
                self.logger.log({'loss_reward': float(loss), 'reward_step': self.reward_step})
                self.reward_step += 1

            del frames, actions, rewards

        empty_cache()
        if self.config.save_models:
            torch.save(self.model.reward_estimator.state_dict(), os.path.join('models', 'reward_model.pt'))

    def reward_model_generator(self, buffer, steps, n_action):
        unpacked_rewards = torch.empty((len(buffer.data), self.config.rollout_length), dtype=torch.long)
        for i, (_, _, rewards, _, _) in enumerate(buffer.data.values()):
            unpacked_rewards[i] = rewards

        reward_count = torch.tensor([(unpacked_rewards == i).sum() for i in range(3)], dtype=torch.float)
        weights = 1 - reward_count / reward_count.sum()
        epsilon = 1e-4
        weights += epsilon
        indices_weights = torch.zeros_like(unpacked_rewards, dtype=torch.float)
        for i in range(3):
            indices_weights[unpacked_rewards == i] = weights[i]
        indices = torch.multinomial(
            indices_weights.view((-1)),
            steps * self.config.reward_model_batch_size,
            replacement=True
        )

        def get_far():
            return \
                torch.empty(
                    (self.config.reward_model_batch_size, self.config.stacking, *self.config.frame_shape),
                    dtype=torch.uint8
                ), \
                torch.empty((self.config.reward_model_batch_size, n_action), dtype=torch.uint8), \
                torch.empty((self.config.reward_model_batch_size,), dtype=torch.uint8)

        frames, actions, rewards = get_far()
        for i, index in enumerate(indices):
            x = index // self.config.rollout_length
            y = index % self.config.rollout_length

            sequences, b_actions, b_rewards, _, _ = list(buffer.data.values())[x]
            frames[i % self.config.reward_model_batch_size] = sequences[y:y + self.config.stacking]
            actions[i % self.config.reward_model_batch_size] = b_actions[y]
            rewards[i % self.config.reward_model_batch_size] = b_rewards[y]

            if (i + 1) % self.config.reward_model_batch_size == 0:
                assert frames.dtype == torch.uint8
                assert actions.dtype == torch.uint8
                assert rewards.dtype == torch.uint8
                frames = frames.float() / 255
                actions = actions.float()
                rewards = rewards.long()

                frames = frames.to(self.config.device)
                actions = actions.to(self.config.device)
                rewards = rewards.to(self.config.device)

                yield frames, actions, rewards

                frames, actions, rewards = get_far()
