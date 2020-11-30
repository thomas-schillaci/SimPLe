import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm

from atari_utils.utils import one_hot_encode
from simple.utils import MeanAttention, ActionInjector, standardize_frame, get_timing_signal_nd, mix, Container, bit_to_int, \
    sample_with_temperature, int_to_bit, ParameterSealer


class RewardEstimator(nn.Module):

    def __init__(self, config, input_size):
        super().__init__()
        self.config = config

        self.dense1 = nn.Linear(input_size, 128)
        self.dense2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x


class ValueEstimator(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.dense = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.dense(x)


class MiddleNetwork(nn.Module):

    def __init__(self, config, filters):
        super().__init__()
        self.config = config

        self.middle_network = []
        for i in range(self.config.hidden_layers):
            self.middle_network.append(nn.Conv2d(filters, filters, 3, padding=1))
            if i == 0:
                self.middle_network.append(None)
            else:
                self.middle_network.append(nn.InstanceNorm2d(filters, affine=True, eps=1e-6))
        self.middle_network = nn.ModuleList(self.middle_network)

    def forward(self, x):
        for i in range(self.config.hidden_layers):
            y = F.dropout(x, self.config.residual_dropout)
            y = self.middle_network[2 * i](y)  # Conv
            y = F.relu(y)

            if i == 0:
                x = y
            else:
                x = self.middle_network[2 * i + 1](x + y)  # LayerNorm

        return x


class BitsPredictor(nn.Module):

    def __init__(self, config, input_size, state_size, total_number_bits, bits_at_once=8):
        super().__init__()
        self.config = config
        self.total_number_bits = total_number_bits
        self.bits_at_once = bits_at_once
        self.dense1 = nn.Linear(input_size, state_size)
        self.dense2 = nn.Linear(input_size, state_size)
        self.dense3 = nn.Linear(input_size, state_size)
        self.dense4 = nn.Linear(2 ** bits_at_once, state_size)
        self.dense5 = nn.Linear(state_size, 2 ** bits_at_once)
        self.lstm = nn.LSTMCell(state_size, state_size)

    def forward(self, x, temperature, target_bits=None):
        x = torch.flatten(x, start_dim=1)

        first_lstm_input = self.dense1(x)
        h_state = self.dense2(x)
        c_state = self.dense3(x)

        if target_bits is not None:
            target_bits = target_bits.view((-1, self.total_number_bits // self.bits_at_once, self.bits_at_once))
            target_bits = torch.max(target_bits, torch.tensor(0.).to(self.config.device))
            target_ints = bit_to_int(target_bits, self.bits_at_once).long()
            target_hot = one_hot_encode(target_ints, 2 ** self.bits_at_once, dtype=torch.float32)
            target_embedded = self.dense4(target_hot)
            target_embedded = F.dropout(target_embedded, 0.1)
            teacher_input = torch.cat((first_lstm_input.unsqueeze(1), target_embedded), dim=1)

            outputs = []
            for i in range(self.total_number_bits // self.bits_at_once):
                lstm_input = teacher_input[:, i, :]
                h_state, c_state = self.lstm(lstm_input, (h_state, c_state))
                outputs.append(h_state)
            outputs = torch.stack(outputs, dim=1)
            outputs = F.dropout(outputs, 0.1)
            pred = self.dense5(outputs)

            loss = nn.CrossEntropyLoss()(pred.permute((0, 2, 1)), target_ints)

            return pred, loss / self.config.rollout_length

        outputs = []
        lstm_input = first_lstm_input
        for i in range(self.total_number_bits // self.bits_at_once):
            h_state, c_state = self.lstm(lstm_input, (h_state, c_state))
            discrete_logits = self.dense5(h_state)
            discrete_samples = sample_with_temperature(discrete_logits, temperature)
            outputs.append(discrete_samples)
            lstm_input = self.dense4(one_hot_encode(discrete_samples, 256, dtype=torch.float32))
        outputs = torch.stack(outputs, dim=1)
        outputs = int_to_bit(outputs, self.bits_at_once)
        outputs = outputs.view((-1, self.total_number_bits))

        return 2 * outputs - 1, 0.0


class StochasticModel(Container):

    def __init__(self, config, layer_shape, n_action):
        super().__init__()
        self.config = config
        channels = 2 * self.config.frame_shape[0]
        filters = [128, 512]
        self.lstm_loss = None
        self.get_lstm_loss()

        self.input_embedding = nn.Conv2d(channels, self.config.hidden_size, 1)
        self.conv1 = nn.Conv2d(self.config.hidden_size, filters[0], 8, 4, padding=2)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 8, 4, padding=2)
        self.dense1 = nn.Linear(2 * 2 * channels, self.config.bottleneck_bits)
        self.dense2 = nn.Linear(self.config.bottleneck_bits, layer_shape[0])
        self.dense3 = nn.Linear(self.config.bottleneck_bits, layer_shape[0])

        self.action_injector = ActionInjector(n_action, self.config.hidden_size)
        self.mean_attentions = nn.ModuleList([MeanAttention(n_filter, 2 * channels) for n_filter in filters])
        bits_predictor = BitsPredictor(
            config,
            layer_shape[0] * layer_shape[1] * layer_shape[2],
            self.config.latent_state_size,
            self.config.bottleneck_bits
        )
        if self.config.decouple_optimizers:
            self.bits_predictor = ParameterSealer(bits_predictor)
        else:
            self.bits_predictor = bits_predictor

    def add_bits(self, layer, bits):
        z_mul = self.dense2(bits)
        z_mul = torch.sigmoid(z_mul)
        z_add = self.dense3(bits)
        z_mul = z_mul.unsqueeze(-1).unsqueeze(-1)
        z_add = z_add.unsqueeze(-1).unsqueeze(-1)
        return layer * z_mul + z_add

    def forward(self, layer, inputs, action, target, epsilon):
        if self.training and target is not None:
            x = torch.cat((inputs, target), dim=1)
            x = self.input_embedding(x)
            x = x + get_timing_signal_nd(x.shape).to(self.config.device)

            x = self.action_injector(x, action)

            x = self.conv1(x)
            x1 = self.mean_attentions[0](x)
            x = F.relu(x)
            x = self.conv2(x)
            x2 = self.mean_attentions[1](x)

            x = torch.cat((x1, x2), dim=-1)
            x = self.dense1(x)

            bits_clean = (2 * (0 < x).float()).detach() - 1
            truncated_normal = truncnorm.rvs(-2, 2, size=x.shape, scale=0.2)
            truncated_normal = torch.tensor(truncated_normal, dtype=torch.float32).to(self.config.device)
            x = x + truncated_normal
            x = torch.tanh(x)
            bits = x + (2 * (0 < x).float() - 1 - x).detach()
            noise = torch.rand_like(x)
            noise = 2 * (self.config.bottleneck_noise < noise).float() - 1
            bits = bits * noise
            bits = mix(bits, x, 1 - epsilon)

            _, lstm_loss = self.bits_predictor(layer, 1.0, bits_clean)
            self.lstm_loss = self.lstm_loss + lstm_loss

            bits_pred, _ = self.bits_predictor(layer, 1.0)
            bits_pred = bits_clean + (bits_pred - bits_clean).detach()
            bits = mix(bits_pred, bits, 1 - (1 - epsilon) * self.config.latent_rnn_max_sampling)

            res = self.add_bits(layer, bits)
            return mix(res, layer, 1 - (1 - epsilon) * self.config.latent_use_max_probability)

        bits, _ = self.bits_predictor(layer, 1.0)
        return self.add_bits(layer, bits)

    def get_lstm_loss(self, reset=True):
        res = self.lstm_loss
        if reset:
            self.lstm_loss = torch.tensor(0.).to(self.config.device)
        return res


class NextFramePredictor(Container):

    def __init__(self, config, n_action):
        super().__init__()
        self.config = config
        filters = self.config.hidden_size

        # Internal states

        self.internal_states = None
        self.gate = nn.Conv2d(
            self.config.frame_shape[0] + self.config.recurrent_state_size,
            2 * self.config.recurrent_state_size,
            3,
            padding=1
        )

        # Model
        self.input_embedding = nn.Conv2d(
            self.config.frame_shape[0] + self.config.recurrent_state_size,
            self.config.hidden_size,
            1
        )

        self.downscale_layers = []
        shape = list(self.config.frame_shape)
        shapes = [shape]
        for i in range(self.config.compress_steps):
            in_filters = filters
            if i < self.config.filter_double_steps:
                filters *= 2

            shape = [filters, shape[1] // 2, shape[2] // 2]
            shapes.append(shape)

            self.downscale_layers.append(
                nn.Conv2d(in_filters, filters, 4, stride=2, padding=1))
            self.downscale_layers.append(nn.InstanceNorm2d(filters, affine=True, eps=1e-6))

        self.downscale_layers = nn.ModuleList(self.downscale_layers)

        middle_shape = shape

        self.upscale_layers = []
        self.action_injectors = [ActionInjector(n_action, filters)]
        for i in range(self.config.compress_steps):
            self.action_injectors.append(ActionInjector(n_action, filters))

            in_filters = filters
            if i >= self.config.compress_steps - self.config.filter_double_steps:
                filters //= 2

            shape = [filters, shape[1] * 2, shape[2] * 2]
            output_padding = (0 if shape[1] == shapes[-i - 2][1] else 1, 0 if shape[2] == shapes[-i - 2][2] else 1)
            shape = [filters, shape[1] + output_padding[0], shape[2] + output_padding[1]]

            self.upscale_layers.append(nn.ConvTranspose2d(
                in_filters, filters, 4, stride=2, padding=1, output_padding=output_padding
            ))
            self.upscale_layers.append(nn.InstanceNorm2d(filters, affine=True, eps=1e-6))

        self.upscale_layers = nn.ModuleList(self.upscale_layers)
        self.action_injectors = nn.ModuleList(self.action_injectors)

        self.logits = nn.Conv2d(self.config.hidden_size, 256 * self.config.frame_shape[0], 1)

        # Sub-models
        self.middle_network = MiddleNetwork(self.config, middle_shape[0])
        reward_estimator = RewardEstimator(self.config, middle_shape[0] + filters)
        value_estimator = ValueEstimator(middle_shape[0] * middle_shape[1] * middle_shape[2])
        self.stochastic_model = StochasticModel(self.config, middle_shape, n_action)

        if self.config.decouple_optimizers:
            self.reward_estimator = ParameterSealer(reward_estimator)
            self.value_estimator = ParameterSealer(value_estimator)
        else:
            self.reward_estimator = reward_estimator
            self.value_estimator = value_estimator

    def init_internal_states(self, batch_size):
        self.internal_states = torch.zeros(
            (batch_size, self.config.recurrent_state_size, *self.config.frame_shape[1:])
        ).to(self.config.device)

    def update_internal_states_early(self, frames):
        internal_state = self.internal_states.detach()
        state_activation = torch.cat((internal_state, frames), dim=1)
        state_gate_candidate = self.gate(state_activation)
        state_gate, state_candidate = torch.split(state_gate_candidate, self.config.recurrent_state_size, dim=1)
        state_gate = torch.sigmoid(state_gate)
        state_candidate = torch.tanh(state_candidate)
        internal_state = internal_state * state_gate
        internal_state = internal_state + state_candidate * (1 - state_gate)
        self.internal_states = internal_state

    def warmup(self, frames, actions):
        assert len(frames) == self.config.stacking
        assert len(actions) == self.config.stacking - 1

        batch_size = frames.shape[1]
        self.init_internal_states(batch_size)

        loss = torch.zeros((1,)).to(frames.device)

        for i in range(self.config.stacking - 1):
            frame = frames[i]
            action = actions[i]
            frame_pred = self(frame, action)[0]
            target = (frames[i + 1].detach() * 255).long()
            loss = loss + nn.CrossEntropyLoss()(frame_pred, target)

        return loss

    def forward(self, x, action, target=None, epsilon=0.0):
        x_start = torch.stack([standardize_frame(frame) for frame in x])

        x = torch.cat((x_start, self.internal_states), dim=1)
        self.update_internal_states_early(x_start)

        x = self.input_embedding(x)
        x = x + get_timing_signal_nd(x.shape).to(self.config.device)

        inputs = []
        for i in range(self.config.compress_steps):
            inputs.append(x)
            x = F.dropout(x, self.config.dropout)
            x = x + get_timing_signal_nd(x.shape).to(self.config.device)
            x = self.downscale_layers[2 * i](x)  # Conv
            x = F.relu(x)
            x = self.downscale_layers[2 * i + 1](x)  # LayerNorm

        value_pred = self.value_estimator(torch.flatten(x, start_dim=1)).squeeze(-1)

        x = self.action_injectors[0](x, action)

        if target is not None:
            for batch_index in range(len(target)):
                target[batch_index] = standardize_frame(target[batch_index])

        x = self.stochastic_model(x, x_start, action, target, epsilon)

        x_mid = torch.mean(x, dim=(2, 3))

        x = self.middle_network(x)

        inputs = list(reversed(inputs))
        for i in range(self.config.compress_steps):
            x = F.dropout(x, self.config.dropout)
            x = self.action_injectors[i + 1](x, action)
            x = self.upscale_layers[2 * i](x)  # ConvTranspose
            x = F.relu(x)
            x = x + inputs[i]
            x = self.upscale_layers[2 * i + 1](x)  # LayerNorm
            x = x + get_timing_signal_nd(x.shape).to(self.config.device)

        x_fin = torch.mean(x, dim=(2, 3))

        reward_pred = self.reward_estimator(torch.cat((x_mid, x_fin), dim=1))

        x = self.logits(x)
        x = x.view((-1, 256, *self.config.frame_shape))

        return x, reward_pred, value_pred
