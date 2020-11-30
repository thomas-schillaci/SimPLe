import torch
from torch import nn as nn
from torch.nn import functional as F


class Container(nn.Module):

    def __init__(self):
        super().__init__()

    def sealed_models_iterator(self):
        for attr in dir(self):
            attr = getattr(self, attr)
            if isinstance(attr, ParameterSealer) or isinstance(attr, Container):
                yield attr

    def to(self, device):
        super().to(device)

        for sealed_model in self.sealed_models_iterator():
            sealed_model.to(device)

        return self

    def train(self, mode=True):
        super().train(mode)

        for sealed_model in self.sealed_models_iterator():
            sealed_model.train(mode)

        return self

    def eval(self):
        super().eval()

        for sealed_model in self.sealed_models_iterator():
            sealed_model.eval()

        return self


class MeanAttention(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.input_embedding = nn.Conv2d(in_size, 4, 1)
        self.dense = nn.Linear(5 * in_size, out_size)

    def forward(self, x):
        m = torch.mean(x, dim=(2, 3))
        a = self.input_embedding(x)
        s = a.view((x.shape[0], -1, 4))
        s = F.softmax(s, dim=1)
        s = s.view((x.shape[0], 1, 4, *x.shape[2:]))
        am = torch.mean(x.unsqueeze(2) * s, dim=(3, 4))
        l = torch.cat((am, m.unsqueeze(-1)), dim=-1)
        l = l.view((x.shape[0], 5 * x.shape[1]))
        return self.dense(l)


class ParameterSealer:
    '''
    Used to hide sub-module's parameters
    '''

    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def parameters(self, recurse=True):
        return self.module.parameters(recurse)

    def to(self, device):
        self.module.to(device)

    def train(self, mode=True):
        self.module.train(mode)

    def eval(self):
        self.module.eval()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)


class ActionInjector(nn.Module):

    def __init__(self, n_action, size):
        super().__init__()
        self.dense1 = nn.Linear(n_action, size)
        self.dense2 = nn.Linear(n_action, size)

    def forward(self, x, action):
        mask = self.dense1(action)
        mask = mask.view((-1, x.shape[1], 1, 1))
        x = x * torch.sigmoid(mask)
        mask = self.dense2(action)
        mask = mask.view((-1, x.shape[1], 1, 1))
        x = x + mask
        return x


def sample_with_temperature(logits, temperature):
    reshaped_logits = logits.view((-1, logits.shape[-1])) / temperature
    reshaped_logits = torch.exp(reshaped_logits)
    choices = torch.multinomial(reshaped_logits, 1)
    choices = choices.view((logits.shape[:len(logits.shape) - 1]))
    return choices


def bit_to_int(x_bit, num_bits):
    x_l = x_bit.view((-1, num_bits)).int().detach().cpu()
    x_labels = [x_l[:, i] * torch.pow(torch.tensor(2.0), i) for i in range(num_bits)]
    res = sum(x_labels)
    return res.view(x_bit.shape[:-1]).int().to(x_bit.device)


def int_to_bit(x_int, num_bits):
    x_l = x_int.unsqueeze(-1).int()
    x_labels = [torch.remainder(x_l // 2 ** i, 2) for i in range(num_bits)]
    res = torch.cat(x_labels, -1)
    return res.float().to(x_int.device)


def standardize_frame(x):
    x_mean = torch.mean(x, dim=(-1, -2)).view((-1, 1, 1))
    x_var = torch.var(x, dim=(-1, -2)).view((-1, 1, 1))
    num_pixels = torch.tensor(x.shape[-1] * x.shape[-2], dtype=torch.float32).to(x.device)
    return (x - x_mean) / torch.max(torch.sqrt(x_var), torch.rsqrt(num_pixels))


def get_timing_signal_nd(shape, min_timescale=1.0, max_timescale=1.0e4):
    channels = shape[1]
    num_timescales = channels // 4
    log_timescale_increment = (torch.log(torch.tensor(float(max_timescale) / float(min_timescale)))
                               / (torch.tensor(num_timescales, dtype=torch.float32) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
    )

    res = torch.zeros(shape).cpu()

    for dim in range(2):
        length = shape[dim + 2]
        position = torch.arange(length, dtype=torch.float32)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), dim=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = F.pad(signal, [prepad, postpad])
        signal = signal.permute((1, 0))
        new_dim = [1, channels, 1, 1]
        new_dim[dim + 2] = -1
        signal = signal.view(new_dim)
        res = res + signal

    return res


def mix(x1, x2, epsilon):
    '''
    Returns ~ x1 * (1 - epsilon) + x2 * epsilon
    '''
    mask = torch.rand_like(x1)
    mask = (mask < epsilon).float()
    return (1 - mask) * x1 + mask * x2
