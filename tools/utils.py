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
