import os

import torch
import torch.nn as nn
from torch.cuda import empty_cache
from torch.nn.utils import clip_grad_norm_
from tqdm import trange

from simple.adafactor import Adafactor


class Trainer:

    def __init__(self, model, config):
        self.config = config
        self.model = model

        self.optimizer = Adafactor(self.model.parameters())
        self.reward_optimizer = Adafactor(self.model.reward_estimator.parameters())
        if self.config.decouple_optimizers:
            self.lstm_optimizer = Adafactor(self.model.stochastic_model.bits_predictor.parameters())
            self.value_optimizer = Adafactor(self.model.value_estimator.parameters())

        self.model_step = 1
        self.reward_step = 1
        self.value_step = 1

    def train(self, epoch, real_env):
        self.train_world_model(epoch, real_env.sample_buffer)
        self.train_reward_model(real_env.buffer, real_env.action_space.n)

    def train_world_model(self, epoch, sample_buffer):
        self.model.train()
        if self.config.decouple_optimizers:
            self.model.reward_estimator.eval()
            self.model.value_estimator.eval()

        steps = 15000

        if epoch == 0:
            steps *= 3

        with trange(
                0,
                steps,
                self.config.rollout_length,
                desc='Training world model',
                unit_scale=self.config.rollout_length
        ) as t:
            for j in t:
                # Scheduled sampling
                if epoch == 0:
                    inv_base = torch.exp(torch.log(torch.tensor(0.01)) / 10000)
                    epsilon = inv_base ** max(self.config.scheduled_sampling_decay_steps // 4 - j, 0)
                    progress = min(j / self.config.scheduled_sampling_decay_steps, 1)
                    progress = progress * (1 - 0.01) + 0.01
                    epsilon *= progress
                    epsilon = 1 - epsilon
                else:
                    epsilon = 0

                sequences, actions, _, targets, values = sample_buffer(self.config.batch_size)
                losses = self.train_world_model_impl(sequences, actions, targets, values, epsilon)
                losses = {key: format(losses[key], '.4e') for key in losses.keys()}
                t.set_postfix(losses)

        empty_cache()
        if self.config.save_models:
            torch.save(self.model.state_dict(), os.path.join('models', 'model.pt'))
            if self.config.decouple_optimizers:
                torch.save(
                    self.model.stochastic_model.bits_predictor.state_dict(),
                    os.path.join('models', 'bits_predictor.pt')
                )
                torch.save(
                    self.model.value_estimator.state_dict(),
                    os.path.join('models', 'value_model.pt')
                )

    def train_world_model_impl(self, sequences, actions, targets, values, epsilon=0.0):
        assert sequences.dtype == torch.uint8
        assert actions.dtype == torch.uint8
        assert targets.dtype == torch.uint8
        assert values.dtype == torch.float32

        sequences = sequences.to(self.config.device)
        actions = actions.to(self.config.device)
        targets = targets.to(self.config.device)
        values = values.to(self.config.device)

        sequences = sequences.float() / 255
        actions = actions.float()
        targets = targets.long()

        initial_frames = sequences[:self.config.stacking]
        initial_frames = initial_frames.view(
            (self.config.stacking, self.config.batch_size, *self.config.frame_shape))
        initial_actions = actions[:self.config.stacking - 1]
        initial_actions = initial_actions.view((self.config.stacking - 1, self.config.batch_size, -1))
        warmup_loss = self.model.warmup(initial_frames, initial_actions)

        if self.config.decouple_optimizers:
            self.optimizer.zero_grad()
            warmup_loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()

        sequences = sequences[self.config.stacking - 1:]
        actions = actions[self.config.stacking - 1:]

        frames_pred = None
        losses = torch.empty((self.config.rollout_length, 3))

        for i in range(self.config.rollout_length):
            frames = torch.empty_like(sequences[0]).to(self.config.device)
            for j in range(len(frames)):
                if i == 0 or float(torch.rand((1,))) < epsilon:
                    frames[j] = sequences[i, j]
                else:
                    frames[j] = torch.argmax(frames_pred[j], dim=0).float() / 255

            action = actions[i]
            target = targets[i]
            target_input = target.float() / 255
            value = values[i]

            frames_pred, _, values_pred = self.model(frames, action, target_input, epsilon)

            loss = nn.CrossEntropyLoss(reduction='none')(frames_pred, target)
            clip = torch.tensor(self.config.target_loss_clipping).to(self.config.device)
            loss = torch.max(loss, clip)
            offset = self.config.target_loss_clipping * self.config.frame_shape[0] * self.config.frame_shape[1] \
                     * self.config.frame_shape[2]
            loss = loss.sum() / self.config.batch_size - offset

            if self.config.decouple_optimizers:
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                self.optimizer.step()

                lstm_loss = self.model.stochastic_model.get_lstm_loss()
                self.lstm_optimizer.zero_grad()
                lstm_loss.backward(retain_graph=True)
                clip_grad_norm_(self.model.stochastic_model.bits_predictor.parameters(), self.config.clip_grad_norm)
                self.lstm_optimizer.step()

                value_loss = nn.MSELoss()(values_pred, value)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                clip_grad_norm_(self.model.value_estimator.parameters(), self.config.clip_grad_norm)
                self.value_optimizer.step()
            else:
                lstm_loss = self.model.stochastic_model.get_lstm_loss()
                value_loss = nn.MSELoss()(values_pred, value)
                loss = loss + lstm_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                self.optimizer.step()

            losses[i] = torch.tensor([float(loss), float(lstm_loss), float(value_loss)])

        losses = torch.mean(losses, dim=0)
        losses = {
            'loss_warmup': float(warmup_loss),
            'loss_reconstruct': float(losses[0]),
            'loss_lstm': float(losses[1]),
            'loss_value': float(losses[2]),
        }

        if self.config.use_wandb:
            import wandb
            d = {'model_step': self.model_step, 'epsilon': epsilon}
            d.update(losses)
            wandb.log(d)
            self.model_step += self.config.rollout_length

        del sequences, initial_frames, initial_actions, targets, actions

        return losses

    def train_reward_model(self, buffer, n_action):
        self.model.eval()
        self.model.reward_estimator.train()

        steps = 300
        generator = self.reward_model_generator(buffer, steps, n_action)

        t = trange(0, steps, desc='Training reward model')
        for frames, actions, rewards in generator:
            assert frames.dtype == torch.uint8
            assert actions.dtype == torch.uint8
            assert rewards.dtype == torch.uint8
            frames = frames.float() / 255
            actions = actions.float()
            rewards = rewards.long()

            frame = frames.to(self.config.device)
            action = actions.to(self.config.device)
            reward = rewards.to(self.config.device)

            self.model.warmup(frame[:self.config.stacking], action[:self.config.stacking - 1])

            reward_pred = self.model(frame[-1], action[-1])[1]
            loss = nn.CrossEntropyLoss()(reward_pred, reward)

            if float(torch.rand((1,))) < 0.01:
                import matplotlib.pyplot as plot
                for j in range(4):
                    plot.subplot(1, 4, j + 1)
                    plot.imshow(frame[j, 0, 0].cpu().detach().numpy(), cmap='gray')
                plot.suptitle(f'{int(reward[0])} {reward_pred[0].cpu().detach().numpy()} {float(loss)}')
                plot.show()

            self.reward_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.reward_estimator.parameters(), self.config.clip_grad_norm)
            self.reward_optimizer.step()

            loss = format(loss, '.4e')
            t.set_postfix({'loss_reward': loss})
            t.update()

            if self.config.use_wandb:
                import wandb
                wandb.log({'loss_reward': float(loss), 'reward_step': self.reward_step})
                self.reward_step += 1

            del frames, actions, rewards

        empty_cache()
        if self.config.save_models:
            torch.save(self.model.reward_estimator.state_dict(), os.path.join('models', 'reward_model.pt'))

    def reward_model_generator(self, buffer, steps, n_action):
        unpacked_rewards = torch.empty((len(buffer), self.config.rollout_length), dtype=torch.long)
        for i, (_, _, rewards, _, _) in enumerate(buffer):
            unpacked_rewards[i] = rewards

        reward_count = torch.tensor([(unpacked_rewards == i).sum() for i in range(3)])
        weights = 1 - reward_count / reward_count.sum()
        indices_weights = torch.zeros_like(unpacked_rewards, dtype=torch.float)
        for i in range(3):
            indices_weights[unpacked_rewards == i] = weights[i]
        indices = torch.multinomial(indices_weights.view((-1)), steps * self.config.batch_size, replacement=True)

        def get_far():
            return \
                torch.empty(
                    (self.config.stacking, self.config.batch_size, *self.config.frame_shape),
                    dtype=torch.uint8
                ), \
                torch.empty((self.config.stacking, self.config.batch_size, n_action), dtype=torch.uint8), \
                torch.empty((self.config.batch_size,), dtype=torch.uint8)

        frames, actions, rewards = get_far()
        for i, index in enumerate(indices):
            x = index // self.config.rollout_length
            y = index % self.config.rollout_length

            sequences, b_actions, b_rewards, _, _ = buffer[x]
            frames[:, i % self.config.batch_size] = sequences[y:y + self.config.stacking]
            actions[:, i % self.config.batch_size] = b_actions[y:y + self.config.stacking]
            rewards[i % self.config.batch_size] = b_rewards[y]

            if (i + 1) % self.config.batch_size == 0:
                yield frames, actions, rewards
                frames, actions, rewards = get_far()
