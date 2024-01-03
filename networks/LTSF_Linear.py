# (2023 AAAI) Are Transformers Effective for Time Series Forecasting?
# https://github.com/cure-lab/LTSF-Linear

import torch
import torch.nn as nn

from tools import Decomposition

# Default Hyper Parameters
settings = {
    'individual': True, # True表示多元预测时每个维度用各自不同的参数，False表示共用参数
    'kernel_len': 25  # 成分分解时滤波窗口的长度
}

class Linear(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channel_dim = args.channel_dim
        if settings['individual']:
            self.layers = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channel_dim)])
        else:
            self.layers = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        if settings['individual']:
            y = torch.zeros([x.shape[0], x.shape[1], self.pred_len], device=x.device)
            for i in range(self.channel_dim):
                y[:, i] = self.layers[i](x[:, i])
            x = y
        else:
            x = self.layers(x)
        return x.transpose(1, 2)

class DLinear(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.decomposition = Decomposition(settings['kernel_len'])
        self.season_model = Linear(args).to(args.device)
        self.trend_model = Linear(args).to(args.device)

    def forward(self, x):
        season_x, trend_x = self.decomposition(x)
        season_y, trend_y = self.season_model(season_x), self.trend_model(trend_x)
        return season_y + trend_y

class NLinear(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = Linear(args).to(args.device)

    def forward(self, x):
        last_x = x[:, -1:].detach()
        x = x - last_x
        x = self.model(x)
        return x + last_x