import torch
import torch.nn as nn

from tools import Decomposition, Normalization

settings = {
    'decomposition': True,  # 是否需要成分分解
    'individual': True,  # 是否各个维度独立
    'kernel_len': 25,  # 成分分解时滤波窗口的长度
    'normalization': True,  # 是否需要归一化
    'affine': True,  # 是否需要可学习的归一化参数
    'subtract_last': False,  # 是否用输入序列的最后一个值替代均值来归一化
    'patch_len': 16,  # 每个时间片段的长度
    'stride_len': 8,  # 每两个时间片段之间的步长
    'end_padding': True, # 是否在尾部填充，若否则不填充
    'model_dim':128,
}

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch * dim * patch_num * patch_len
        channel_dim = x.shape[1]
        return x

class FlattenHead(nn.Module):
    def __init__(self, channel_dim, input_len, pred_len, dropout=0):
        super().__init__()
        self.channel_dim = channel_dim
        if settings['individual']:
            self.flattens = nn.ModuleList([nn.Flatten(start_dim=-2) for _ in range(channel_dim)])
            self.layers = nn.ModuleList([nn.Linear(input_len, pred_len) for _ in range(channel_dim)])
            self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(channel_dim)])
        else:
            self.flattens = nn.Flatten(start_dim=-2)
            self.layers = nn .Linear(input_len, pred_len)
            self.dropouts = nn.Dropout(dropout)

    def forward(self, x):
        if settings['individual']:
            y = []
            for i in range(self.channel_dim):
                tmp = self.flattens[i](x[:, i])
                tmp = self.layers[i](tmp)
                tmp = self.dropouts[i](tmp)
                y.append(tmp)
            x = torch.stack(y, dim=1)
        else:
            x = self.flattens(x)
            x = self.layers(x)
            x = self.dropouts(x)
        return x


class Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        if settings['normalization']:
            self.norm_layer = Normalization(args.channel_dim, settings['affine'], settings['subtract_last'])
        patch_num = int((args.seq_len - settings['patch_len']) / settings['stride_len']) + 1
        if settings['end_padding']:
            self.padding_layer = nn.ReflectionPad1d((0, settings['stride_len']))
            patch_num += 1
        self.encoder = Encoder()

    def forward(self, x):
        if settings['normalization']:
            x = self.norm_layer(x, norm=True)
        x = x.transpose(1, 2)  # batch * dim * length
        if settings['end_padding']:
            x = self.padding_layer(x)
        x = x.unfold(-1, settings['patch_len'], settings['stride_len'])  # batch * dim * patch_num * patch_len
        x = self.encoder(x)


class PatchTST(nn.Module):
    def __init__(self, args):
        super().__init__()
        if settings['decomposition']:
            self.decomposition = Decomposition(settings['kernel_len'])
            self.season_model = Backbone(args)
            self.trend_model = Backbone(args)
        else:
            self.model = Backbone(args)

    def forward(self, x):
        if settings['decomposition']:
            season_x, trend_x = self.decomposition(x)
            season_x, trend_x = self.season_model(season_x), self.trend_model(trend_x)
            return season_x + trend_x
        else:
            return self.model(x)