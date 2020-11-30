import torch
from baselines import logger


def sample_with_temperature(logits, temperature):
    assert temperature > 0
    reshaped_logits = logits.view((-1, logits.shape[-1])) / temperature
    reshaped_logits = torch.exp(reshaped_logits)
    choices = torch.multinomial(reshaped_logits, 1)
    choices = choices.view((logits.shape[:len(logits.shape) - 1]))
    return choices


def print_config(config):
    args = vars(config)
    max_len = 0
    for arg in args:
        max_len = max(max_len, len(arg))
    for arg in args:
        value = str(getattr(config, arg))
        display = '{:<%i}: {}' % (max_len + 1)
        print(display.format(arg, value))


def disable_baselines_logging():
    logger.configure(format_strs='')


def one_hot_encode(action, n, dtype=torch.uint8):
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)

    res = action.long().view((-1, 1))
    res = torch.zeros((len(res), n)).to(res.device).scatter(1, res, 1).type(dtype).to(res.device)
    res = res.view((*action.shape, n))

    return res
