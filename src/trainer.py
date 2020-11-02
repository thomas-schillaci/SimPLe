import os

import torch
import torch.nn as nn
from torch.cuda import empty_cache
from torch.nn.utils import clip_grad_norm_
from tqdm import trange

from adafactor import Adafactor

import matplotlib.pyplot as plot

plot.rcParams["figure.figsize"] = (20, 10)  # FIXME


class Trainer:

    def __init__(self, model, config):
        self.config = config
        self.model = model

        self.optimizers = [
            Adafactor(self.model.parameters())
        ]
        self.optimizers.append(Adafactor(self.model.stochastic_model.bits_predictor.parameters()))
        self.reward_optimizer = Adafactor(self.model.reward_estimator.parameters())
        self.value_optimizer = Adafactor(self.model.value_estimator.parameters())

        self.model_step = 1
        self.reward_step = 1
        self.value_step = 1

    def train(self, epoch, real_env):
        self.train_world_model(epoch, real_env.sample_buffer)
        self.train_reward_model(epoch, real_env)
        self.train_value_model(epoch, real_env.sample_buffer)

    def train_world_model(self, epoch, sample_buffer):
        self.model.train()
        self.model.reward_estimator.eval()
        self.model.value_estimator.eval()

        steps = [15, 1500, 15000][self.config.complexity]  # FIXME

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

                sequences, actions, _, targets, _ = sample_buffer(self.config.batch_size)
                losses = self.train_world_model_impl(sequences, actions, targets, epsilon)

                losses = [format(loss, '.4e') for loss in losses]
                t.set_postfix({'loss_reconstruct': losses[0], 'loss_lstm': losses[1], 'loss_warmup': losses[2]})

        empty_cache()
        if self.config.save_models:
            torch.save(self.model.state_dict(), os.path.join('models', 'model.pt'))
            torch.save(
                self.model.stochastic_model.bits_predictor.state_dict(),
                os.path.join('models', 'bits_predictor.pt')
            )

    def train_world_model_impl(self, sequences, actions, targets, epsilon=0.0):
        assert sequences.dtype == torch.uint8
        assert actions.dtype == torch.uint8
        assert targets.dtype == torch.uint8
        sequences = sequences.to(self.config.device)
        actions = actions.to(self.config.device)
        targets = targets.to(self.config.device)
        sequences = sequences.float() / 255
        actions = actions.float()
        targets = targets.long()

        initial_frames = sequences[:self.config.stacking]
        initial_frames = initial_frames.view(
            (self.config.stacking, self.config.batch_size, *self.config.frame_shape))
        initial_actions = actions[:self.config.stacking - 1]
        initial_actions = initial_actions.view((self.config.stacking - 1, self.config.batch_size, -1))
        warmup_loss = self.model.warmup(initial_frames, initial_actions)

        self.optimizers[0].zero_grad()
        warmup_loss.backward()
        clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
        self.optimizers[0].step()

        sequences = sequences[self.config.stacking - 1:]
        actions = actions[self.config.stacking - 1:]
        frames_pred = None

        float_losses = torch.empty(self.config.rollout_length, 2)

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

            frames_pred, rewards_pred, values_pred = self.model(frames, action, target_input, epsilon)

            loss = nn.CrossEntropyLoss(reduction='none')(frames_pred, target)
            clip = torch.tensor(self.config.target_loss_clipping).to(self.config.device)
            loss = torch.max(loss, clip)
            offset = self.config.target_loss_clipping * self.config.frame_shape[0] \
                     * self.config.frame_shape[1] * self.config.frame_shape[2]
            loss = loss.sum() / self.config.batch_size - offset

            self.optimizers[0].zero_grad()
            loss.backward(retain_graph=True)
            clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.optimizers[0].step()

            float_losses[i, 0] = float(loss)

            # if float(torch.rand((1,))) > 0.9995 and i > 2:
            #     for k in range(4):
            #         plot.subplot(2, 5, k + 1)
            #         plot.imshow(sequences[i - 3 + k, 0].permute((1, 2, 0)).detach().cpu().numpy())
            #         plot.axis('off')
            #     plot.subplot(2, 5, 5)
            #     plot.imshow((target[0].float() / 255).permute((1, 2, 0)).detach().cpu().numpy())
            #     plot.axis('off')
            #     plot.subplot(2, 5, 10)
            #     plot.imshow((torch.argmax(frames_pred[0], 0).float() / 255).permute((1, 2, 0))
            #                 .detach().cpu().numpy())
            #     plot.axis('off')
            #     plot.suptitle(f'action: {torch.argmax(action[0], dim=0).detach().cpu().numpy()}'
            #                   f' - loss: {float_losses[i, 0]}')
            #     plot.show()

            loss = self.model.stochastic_model.get_lstm_loss()

            self.optimizers[1].zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.stochastic_model.bits_predictor.parameters(), self.config.clip_grad_norm)
            self.optimizers[1].step()

            float_losses[i, 1] = float(loss)

        float_losses = torch.mean(float_losses, dim=0)
        float_losses = torch.cat((float_losses, warmup_loss.detach().cpu().view((1,))))

        if self.config.use_wandb:
            import wandb
            d = {'model_step': self.model_step, 'epsilon': epsilon, 'loss_warmup': float(warmup_loss)}
            for j, name in enumerate(['reconstruct', 'lstm']):
                d[f'loss_{name}'] = float_losses[j]
            wandb.log(d)
            self.model_step += self.config.rollout_length

        del sequences, initial_frames, initial_actions, targets, actions

        return float_losses

    def train_reward_model(self, epoch, real_env):
        self.model.eval()
        self.model.reward_estimator.train()

        steps = [1, 15, 150][self.config.complexity]  # FIXME
        steps *= self.config.reward_model_batch_size

        epochs = [1, 1, 7][self.config.complexity]  # FIXME

        if epoch == 0:
            epochs *= 3

        with trange(epochs, desc='Training reward model') as t:
            for _ in t:
                reward_counts = [0] * 3
                for _, _, rewards, _, _ in real_env.buffer:
                    for reward in rewards:
                        reward_counts[int(reward)] += 1

                frames = torch.empty((self.config.stacking, 0, *self.config.frame_shape), dtype=torch.uint8)
                actions = torch.empty((self.config.stacking, 0, real_env.action_space.n), dtype=torch.uint8)
                rewards = torch.empty((0,), dtype=torch.uint8)

                non_zeros = 3
                for reward_count in reward_counts:
                    if reward_count == 0:
                        non_zeros -= 1

                for i, reward_count in enumerate(reward_counts):
                    if reward_count == 0:
                        continue

                    if reward_count < max(reward_counts) and reward_count < steps / non_zeros:
                        multiplier = int(steps / non_zeros / reward_count)
                        for sequence, b_actions, b_rewards, _, _ in real_env.buffer:
                            additional_samples = 0
                            for j, reward in enumerate(b_rewards):
                                if int(reward) == i or additional_samples > 0:
                                    if int(reward) == i:
                                        additional_samples = 2
                                    else:
                                        additional_samples -= 1
                                    frames = torch.cat(
                                        [frames] + [sequence[j:j + self.config.stacking].unsqueeze(1)] * multiplier,
                                        dim=1
                                    )
                                    actions = torch.cat(
                                        [actions] + [b_actions[j:j + self.config.stacking].unsqueeze(1)] * multiplier,
                                        dim=1
                                    )
                                    rewards = torch.cat([rewards] + [reward.unsqueeze(0)] * multiplier)
                    else:
                        count = 0
                        while count < steps / non_zeros:
                            sequence, b_actions, b_rewards, _, _ = real_env.sample_buffer()
                            j = int(torch.randint(high=len(b_rewards), size=(1,)))
                            reward = b_rewards[j]
                            if int(reward) == i:
                                frames = torch.cat(
                                    (frames, sequence[j:j + self.config.stacking].unsqueeze(1))
                                    , dim=1
                                )
                                actions = torch.cat(
                                    (actions, b_actions[j:j + self.config.stacking].unsqueeze(1)),
                                    dim=1
                                )
                                rewards = torch.cat((rewards, reward.unsqueeze(0)))
                                count += 1

                # reward_counts = [0] * 3
                # for reward in rewards:
                #     reward_counts[int(reward)] += 1
                # print(reward_counts)

                permutation = torch.randperm(len(rewards))
                frames = frames[:, permutation]
                actions = actions[:, permutation]
                rewards = rewards[permutation]

                loss = self.train_reward_model_impl(frames, actions, rewards)
                loss = format(loss, '.4e')
                t.set_postfix({'loss_reward': loss})

        empty_cache()
        if self.config.save_models:
            torch.save(self.model.reward_estimator.state_dict(), os.path.join('models', 'reward_model.pt'))

    def train_reward_model_impl(self, frames, actions, rewards):
        assert frames.dtype == torch.uint8
        assert actions.dtype == torch.uint8
        assert rewards.dtype == torch.uint8
        frames = frames.float() / 255
        actions = actions.float()
        rewards = rewards.long()

        batch_size = self.config.reward_model_batch_size
        mean_loss = 0
        for i in range(len(rewards) // batch_size):
            frame = frames[:, i * batch_size:(i + 1) * batch_size].to(self.config.device)
            action = actions[:, i * batch_size:(i + 1) * batch_size].to(self.config.device)
            reward = rewards[i * batch_size:(i + 1) * batch_size].to(self.config.device)

            self.model.warmup(frame[:self.config.stacking], action[:self.config.stacking - 1])

            reward_pred = self.model(frame[-1], action[-1])[1]
            # frame_pred, reward_pred, value_pred = self.model(frame[-1], action[-1])
            #
            # if float(torch.rand((1,))) > 0.9 or int(torch.argmax(reward_pred[0],dim=0))==2:
            #     plot.subplot(1, 2, 1)
            #     plot.imshow(frame[-1, 0].permute((1, 2, 0)).detach().cpu().numpy())
            #     plot.axis('off')
            #     plot.subplot(1, 2, 2)
            #     plot.imshow((torch.argmax(frame_pred[0], dim=0).float() / 255).permute((1, 2, 0)).detach().cpu().numpy())
            #     plot.axis('off')
            #     plot.suptitle(f'reward: {reward[0]} - reward_pred: {reward_pred[0].detach().cpu().numpy()}')
            #     plot.show()

            loss = nn.CrossEntropyLoss()(reward_pred, reward)
            mean_loss += float(loss)

            self.reward_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.reward_estimator.parameters(), self.config.clip_grad_norm)
            self.reward_optimizer.step()

        mean_loss = mean_loss / (len(rewards) // batch_size)

        if self.config.use_wandb:
            import wandb
            wandb.log({'loss_reward': mean_loss, 'reward_step': self.reward_step})
            self.reward_step += len(rewards) // batch_size

        del frames, actions, rewards

        return mean_loss

    def train_value_model(self, epoch, sample_buffer):
        self.model.eval()
        self.model.value_estimator.train()

        steps = [15, 1500, 4000][self.config.complexity]  # FIXME

        if epoch == 0:
            steps *= 3

        with trange(
                0,
                steps,
                self.config.rollout_length,
                desc='Training value model',
                unit_scale=self.config.rollout_length
        ) as t:
            for _ in t:
                sequences, actions, _, _, values = sample_buffer(self.config.value_model_batch_size)
                loss = self.train_value_model_impl(sequences, actions, values)
                loss = format(loss, '.4e')
                t.set_postfix({'loss_value': loss})

        empty_cache()
        if self.config.save_models:
            torch.save(self.model.value_estimator.state_dict(), os.path.join('models', 'value_model.pt'))

    def train_value_model_impl(self, sequences, actions, values):
        assert sequences.dtype == torch.uint8
        assert actions.dtype == torch.uint8
        assert values.dtype == torch.float32
        sequences = sequences.to(self.config.device)
        actions = actions.to(self.config.device)
        values = values.to(self.config.device)

        sequences = sequences.float() / 255
        actions = actions.float()

        initial_frames = sequences[:self.config.stacking]
        initial_frames = initial_frames.view(
            (self.config.stacking, self.config.value_model_batch_size, *self.config.frame_shape))
        initial_actions = actions[:self.config.stacking - 1]
        initial_actions = initial_actions.view((self.config.stacking - 1, self.config.value_model_batch_size, -1))
        self.model.warmup(initial_frames, initial_actions)

        sequences = sequences[self.config.stacking - 1:]
        actions = actions[self.config.stacking - 1:]

        mean_loss = 0

        for i in range(self.config.rollout_length):
            frames = sequences[i]
            action = actions[i]

            values_pred = self.model(frames, action)[2]

            loss = nn.MSELoss()(values_pred, values[i])

            self.value_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.value_estimator.parameters(), self.config.clip_grad_norm)
            self.value_optimizer.step()

            mean_loss += float(loss)

        mean_loss = mean_loss / self.config.rollout_length

        if self.config.use_wandb:
            import wandb
            wandb.log({'value_step': self.value_step, 'loss_value': mean_loss})
            self.value_step += self.config.rollout_length

        del sequences, initial_frames, initial_actions, actions, values

        return mean_loss
