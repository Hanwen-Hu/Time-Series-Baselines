import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_len, stride_len):
        super().__init__()
        self.kernel_len = kernel_len
        self.window = nn.AvgPool1d(kernel_size=kernel_len, stride=stride_len, padding=0)

    def forward(self, x):
        left = x[:, :1].repeat(1, (self.kernel_len - 1) // 2, 1)
        right = x[:, -1:].repeat(1, (self.kernel_len - 1) // 2, 1)
        x = self.window(torch.cat([left, x, right], dim=1).transpose(1, 2))
        return x.transpose(1, 2)


class Decomposition(nn.Module):
    def __init__(self, kernel_len):
        super().__init__()
        self.window = MovingAverage(kernel_len, stride_len=1)

    def forward(self, x):
        avg = self.window(x)
        res = x - avg
        return res, avg


class Normalization(nn.Module):
    def __init__(self, channel_dim, affine=True, subtract_last=False):
        super().__init__()
        self.affine = affine
        self.subtract_last = subtract_last
        if affine:
            self.weight = nn.Parameter(torch.ones(channel_dim))
            self.bias = nn.Parameter(torch.zeros(channel_dim))

    def _get_statistics(self, x):
        if self.subtract_last:
            self.avg = x[:, -1:, :]
        else:
            self.avg = torch.mean(x, dim=1, keepdim=True).detach()
        self.std = torch.std(x, dim=1, keepdim=True).detach()

    def _normalize(self, x):
        self._get_statistics(x)
        x = (x - self.avg) / (self.std + 1e-5)
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.bias) / (self.weight + 1e-5)
        return x * self.std + self.avg

    def forward(self, x, mode):
        if mode == 'norm':
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            print('Error Normalization Mode')
        return x


def activation_func(func_name):
    if callable(func_name):
        return func_name()
    elif func_name.lower() == 'relu':
        return nn.ReLU()
    elif func_name.lower() == 'gelu':
        return nn.GELU()
    raise ValueError('Activation function is not available. You can use "relu", "gelu" or a callable name.')
