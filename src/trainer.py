import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.nn.utils import clip_grad_norm_

from adafactor import Adafactor
from utils import mix

import matplotlib.pyplot as plot

plot.rcParams["figure.figsize"] = (16, 5)


class Trainer:

    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.optimizers = [
            Adafactor(self.model.parameters())
        ]
        if config.use_stochastic_model:
            self.optimizers.append(Adafactor(self.model.stochastic_model.bits_predictor.parameters()))
        self.reward_optimizer = Adafactor(self.model.reward_estimator.parameters())
        self.value_optimizer = Adafactor(self.model.value_estimator.parameters())

    def train(self, sequence, actions, _, targets, values, epsilon=0.0):
        assert sequence.dtype == torch.uint8
        assert actions.dtype == torch.uint8
        assert targets.dtype == torch.uint8
        sequence = sequence.to(self.config.device)
        actions = actions.to(self.config.device)
        targets = targets.to(self.config.device)
        sequence = sequence.float() / 255
        actions = actions.float()
        targets = targets.long()

        frame_stack = sequence[:self.config.stacking * self.config.frame_shape[0]]

        float_losses = torch.empty(self.config.inference_batch_size // self.config.backprop_batch_size, 3)

        for i in range(self.config.inference_batch_size // self.config.backprop_batch_size):
            losses = torch.zeros((3,)).to('cuda')
            for j in range(self.config.backprop_batch_size):
                index = i * self.config.backprop_batch_size + j
                a = actions[index].unsqueeze(0)
                target = targets[index]
                target_input = target.float().unsqueeze(0) / 255

                frame_pred, reward_pred, value_pred = self.model(frame_stack.unsqueeze(0), a, target_input, epsilon)

                loss = nn.CrossEntropyLoss(reduction='none')(frame_pred, target.unsqueeze(0))
                clip = torch.tensor(self.config.target_loss_clipping).to(self.config.device)
                offset = self.config.target_loss_clipping * self.config.frame_shape[0] \
                         * self.config.frame_shape[1] * self.config.frame_shape[2]
                loss = torch.max(loss, clip)
                losses[0] += loss.sum() - offset

                # if float(torch.rand((1,))) > 0.995:
                #     for k in range(4):
                #         plot.subplot(1, 4, k + 1)
                #         plot.imshow(frame_stack[3 * k:3 * (k + 1)].permute((1, 2, 0)).detach().cpu().numpy())
                #         plot.axis('off')
                #     # plot.subplot(2, 5, 5)
                #     # plot.imshow((target.float() / 255).permute((1, 2, 0)).detach().cpu().numpy())
                #     # plot.axis('off')
                #     # plot.subplot(2, 5, 10)
                #     # plot.imshow((torch.argmax(frame_pred[0], 0).float() / 255).permute((1, 2, 0))
                #     #             .detach().cpu().numpy())
                #     # plot.axis('off')
                #     v0 = format(float(value_pred[0].detach().cpu()), '.1f')
                #     v1 = format(float(value.detach().cpu()), '.1f')
                #     plot.suptitle(f'{v0}/{v1}')
                #     plot.show()

                if self.config.use_stochastic_model:
                    losses[1] += self.model.stochastic_model.get_lstm_loss()

                frame_stack = frame_stack[self.config.frame_shape[0]:]
                if index < self.config.inference_batch_size - 1:
                    x1 = torch.argmax(frame_pred.squeeze(), dim=0).float() / 255
                    x1 = x1.to(self.config.device)
                    start = (self.config.stacking + index) * self.config.frame_shape[0]
                    x2 = sequence[start:start + self.config.frame_shape[0]]
                    frame = mix(x1, x2, epsilon)
                    frame_stack = torch.cat((frame_stack, frame))

            # total_loss = losses.sum()
            #
            # self.optimizers[0].zero_grad()
            # total_loss.backward()
            # clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            # self.optimizers[0].step()

            self.optimizers[0].zero_grad()
            losses[0].backward(retain_graph=True)
            clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.optimizers[0].step()

            if self.config.use_stochastic_model:
                self.optimizers[1].zero_grad()
                losses[1].backward()
                clip_grad_norm_(self.model.stochastic_model.bits_predictor.parameters(), self.config.clip_grad_norm)
                self.optimizers[1].step()

            float_losses[i] = losses.detach().cpu() / self.config.backprop_batch_size

            if self.config.use_wandb:
                d = {}
                for j, name in enumerate(['reconstruct', 'lstm']):
                    d[f'loss_{name}'] = float_losses[i, j]
                wandb.log(d)

        return torch.mean(float_losses, dim=0)

    def train_reward_model(self, frame_stacks, actions, rewards):
        assert frame_stacks.dtype == torch.uint8
        assert actions.dtype == torch.uint8
        assert rewards.dtype == torch.uint8
        frame_stacks = frame_stacks.float() / 255
        actions = actions.float()
        rewards = rewards.long()

        batch_size = self.config.reward_model_batch_size
        for i in range(0, len(rewards) // batch_size, batch_size):
            frame_stack = frame_stacks[i * batch_size:(i + 1) * batch_size].to(self.config.device)
            action = actions[i * batch_size:(i + 1) * batch_size].to(self.config.device)
            reward = rewards[i * batch_size:(i + 1) * batch_size].to(self.config.device)

            reward_pred = self.model(frame_stack, action)[1]

            loss = nn.CrossEntropyLoss()(reward_pred, reward)

            self.reward_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.reward_estimator.parameters(), self.config.clip_grad_norm)
            self.reward_optimizer.step()

            if self.config.use_wandb:
                wandb.log({'loss_reward': float(loss)})

        return float(loss)

    def train_value_model(self, frame_stacks, actions, values):
        assert frame_stacks.dtype == torch.uint8
        assert actions.dtype == torch.uint8
        assert values.dtype == torch.float32

        frame_stacks = frame_stacks.float() / 255
        actions = actions.float()

        batch_size = self.config.value_model_batch_size
        for i in range(0, len(values) // batch_size, batch_size):
            frame_stack = frame_stacks[i * batch_size:(i + 1) * batch_size].to(self.config.device)
            action = actions[i * batch_size:(i + 1) * batch_size].to(self.config.device)
            value = values[i * batch_size:(i + 1) * batch_size].to(self.config.device)

            value_pred = self.model(frame_stack, action)[2]

            loss = nn.MSELoss()(value_pred, value)

            self.value_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            clip_grad_norm_(self.model.value_estimator.parameters(), self.config.clip_grad_norm)
            self.value_optimizer.step()

            if self.config.use_wandb:
                wandb.log({'loss_value': float(loss)})

        return float(loss)
