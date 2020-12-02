import kornia  # FIXME
import torch
from torch import nn as nn


class Augmentation(nn.Module):

    def forward(self, x):
        return self.augment(x)

    def augment(self, x):
        return x


class RandomCrop(Augmentation):

    def augment(self, x):
        return nn.Sequential(
            nn.ReplicationPad2d(4),
            kornia.augmentation.RandomCrop((84, 84)),
            Intensity()
        )(x)


class Intensity(Augmentation):

    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise