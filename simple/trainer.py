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

        self.model_step = 1
        self.reward_step = 1

    def train(self, epoch, env, steps=15000):
        if epoch == 0:
            steps *= 3

        c, h, w = self.config.frame_shape
        rollout_len = self.config.rollout_length
        states, actions, rewards, new_states, dones, values = env.buffer[0]

        if env.buffer[0][5] is None:
            raise BufferError('Can\'t train the world model, the buffer does not contain one full episode.')

        assert states.dtype == torch.uint8
        assert actions.dtype == torch.uint8
        assert rewards.dtype == torch.uint8
        assert new_states.dtype == torch.uint8
        assert values.dtype == torch.float32

        def get_index():
            index = -1
            while index == -1:
                index = int(torch.randint(len(env.buffer) - rollout_len, size=(1,)))
                for i in range(rollout_len):
                    done, value = env.buffer[index + i][4:6]
                    if done or value is None:
                        index = -1
                        break
            return index

        def get_indices():
            return [get_index() for _ in range(self.config.batch_size)]

        def preprocess_state(state):
            state = state.float() / 255
            noise_prob = torch.tensor([[self.config.input_noise, 1 - self.config.input_noise]])
            noise_prob = torch.softmax(torch.log(noise_prob), dim=-1)
            noise_mask = torch.multinomial(noise_prob, state.numel(), replacement=True).view(state.shape)
            noise_mask = noise_mask.to(state)
            state = state * noise_mask + torch.median(state) * (1 - noise_mask)
            return state

        reward_criterion = nn.CrossEntropyLoss()

        iterator = trange(
            0,
            steps,
            rollout_len,
            desc='Training world model',
            unit_scale=rollout_len
        )
        for i in iterator:
            # Scheduled sampling
            if epoch == 0:
                decay_steps = self.config.scheduled_sampling_decay_steps
                inv_base = torch.exp(torch.log(torch.tensor(0.01)) / (decay_steps // 4))
                epsilon = inv_base ** max(decay_steps // 4 - i, 0)
                progress = min(i / decay_steps, 1)
                progress = progress * (1 - 0.01) + 0.01
                epsilon *= progress
                epsilon = 1 - epsilon
            else:
                epsilon = 0

            indices = get_indices()
            frames = torch.empty((self.config.batch_size, c * self.config.stacking, h, w))
            frames = frames.to(self.config.device)

            for j in range(self.config.batch_size):
                frames[j] = env.buffer[indices[j]][0].clone()

            frames = preprocess_state(frames)

            n_losses = 5 if self.config.use_stochastic_model else 4
            losses = torch.empty((rollout_len, n_losses))

            if self.config.stack_internal_states:
                self.model.init_internal_states(self.config.batch_size)

            for j in range(rollout_len):
                _, actions, rewards, new_states, _, values = env.buffer[0]
                actions = torch.empty((self.config.batch_size, *actions.shape))
                actions = actions.to(self.config.device)
                rewards = torch.empty((self.config.batch_size, *rewards.shape), dtype=torch.long)
                rewards = rewards.to(self.config.device)
                new_states = torch.empty((self.config.batch_size, *new_states.shape), dtype=torch.long)
                new_states = new_states.to(self.config.device)
                values = torch.empty((self.config.batch_size, *values.shape))
                values = values.to(self.config.device)
                for k in range(self.config.batch_size):
                    actions[k] = env.buffer[indices[k] + j][1]
                    rewards[k] = env.buffer[indices[k] + j][2]
                    new_states[k] = env.buffer[indices[k] + j][3]
                    values[k] = env.buffer[indices[k] + j][5]

                new_states_input = new_states.float() / 255

                self.model.train()
                frames_pred, reward_pred, values_pred = self.model(frames, actions, new_states_input, epsilon)

                if j < rollout_len - 1:
                    for k in range(self.config.batch_size):
                        if float(torch.rand((1,))) < epsilon:
                            frame = new_states[k]
                        else:
                            frame = torch.argmax(frames_pred[k], dim=0)
                        frame = preprocess_state(frame)
                        frames[k] = torch.cat((frames[k, c:], frame), dim=0)

                loss_reconstruct = nn.CrossEntropyLoss(reduction='none')(frames_pred, new_states)
                clip = torch.tensor(self.config.target_loss_clipping).to(self.config.device)
                loss_reconstruct = torch.max(loss_reconstruct, clip)
                loss_reconstruct = loss_reconstruct.mean() - self.config.target_loss_clipping

                loss_value = nn.MSELoss()(values_pred, values)
                loss_reward = reward_criterion(reward_pred, rewards)
                loss = loss_reconstruct + loss_value + loss_reward
                if self.config.use_stochastic_model:
                    loss_lstm = self.model.stochastic_model.get_lstm_loss()
                    loss = loss + loss_lstm

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                self.optimizer.step()

                tab = [float(loss), float(loss_reconstruct), float(loss_value), float(loss_reward)]
                if self.config.use_stochastic_model:
                    tab.append(float(loss_lstm))
                losses[j] = torch.tensor(tab)

            losses = torch.mean(losses, dim=0)
            metrics = {
                'loss': float(losses[0]),
                'loss_reconstruct': float(losses[1]),
                'loss_value': float(losses[2]),
                'loss_reward': float(losses[3])
            }
            if self.config.use_stochastic_model:
                metrics.update({'loss_lstm': float(losses[4])})

            if self.logger is not None:
                d = {'model_step': self.model_step, 'epsilon': epsilon}
                d.update(metrics)
                self.logger.log(d)
                self.model_step += rollout_len

            iterator.set_postfix(metrics)

        empty_cache()
        if self.config.save_models:
            torch.save(self.model.state_dict(), os.path.join('models', 'model.pt'))
