# (2023 ICLR) A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
# https://github.com/yuqinie98/PatchTST

import torch
import torch.nn as nn

from tools import Decomposition, Normalization, activation_func
from units import positional_embedding, MultiHeadAttention

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
    'embed_dim':128,  # 将每个序列片段映射成的向量的长度
    'layer_num': 2,  # 编码器中attention的层数
    'pos_embed_mode': 'zeros',  # Transformer中位置编码选择的方式
    'head_num': 8,  # 多头注意力机制头的数目
    'dropout': 0.05,  # Dropout的概率
    'activation_func': 'relu',  # 前向传播的激活函数
    'norm_mode': 'batch',  # attention层之间的归一化方式，默认为batch，否则为layer
}

# 编码器中的每一层
class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        assert settings['embed_dim'] % settings['head_num'] == 0
        self.attn = MultiHeadAttention(settings['embed_dim'], settings['head_num'], settings['dropout'])
        self.ff_layers = nn.Sequential(nn.Linear(settings['embed_dim'], settings['embed_dim']),
                                       activation_func(settings['activation_func']),
                                       nn.Dropout(settings['dropout']),
                                       nn.Linear(settings['embed_dim'], settings['embed_dim']))
        self.dropout = nn.Dropout(settings['dropout'])
        if settings['norm_mode'] == 'batch':
            self.norm_layer = nn.BatchNorm1d(settings['embed_dim'])
        else:
            self.norm_layer = nn.LayerNorm(settings['embed_dim'])

    def normalize(self, x):
        if settings['norm_mode'] == 'batch':
            return self.norm_layer(x.transpose(1, 2)).transpose(1, 2)
        else:
            return self.norm_layer(x)
    def forward(self, x):
        x = self.dropout(self.attn(x, x, x)) + x
        x = self.normalize(x)
        x = self.dropout(self.ff_layers(x)) + x
        return self.normalize(x)

# 中间的编码器，包含Attention层
class Encoder(nn.Module):
    def __init__(self, patch_num):
        super().__init__()
        self.input_layer = nn.Linear(settings['patch_len'], settings['embed_dim'])
        self.pos = positional_embedding(patch_num, settings['embed_dim'], settings['pos_embed_mode'], learnable=True)
        self.dropout = nn.Dropout(settings['dropout'])
        self.encoder_layers = nn.ModuleList([Layer() for _ in range(settings['layer_num'])])

    def forward(self, x):
        # batch * dim * patch_num * patch_len
        channel_dim = x.shape[1]
        x = self.input_layer(x)  # batch * channel * patch_num * embed_dim
        x = self.dropout(x.reshape(-1, x.shape[2], x.shape[3]) + self.pos)
        for layer in self.encoder_layers:
            x = layer(x)
        return x.reshape(-1, channel_dim, x.shape[-2], x.shape[-1])  # batch * channel * patch_num * embed_dim

# 最终输出层
class FlattenHead(nn.Module):
    def __init__(self, channel_dim, input_len, pred_len):
        super().__init__()
        self.channel_dim = channel_dim
        if settings['individual']:
            self.flattens = nn.ModuleList([nn.Flatten(start_dim=-2) for _ in range(channel_dim)])
            self.layers = nn.ModuleList([nn.Linear(input_len, pred_len) for _ in range(channel_dim)])
            self.dropouts = nn.ModuleList([nn.Dropout(settings['dropout']) for _ in range(channel_dim)])
        else:
            self.flattens = nn.Flatten(start_dim=-2)
            self.layers = nn.Linear(input_len, pred_len)
            self.dropouts = nn.Dropout(settings['dropout'])

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
        self.encoder = Encoder(patch_num)
        self.out_layer = FlattenHead(args.channel_dim, settings['embed_dim'] * patch_num, args.pred_len)

    def forward(self, x):
        if settings['normalization']:
            x = self.norm_layer(x, norm=True)
        x = x.transpose(1, 2)  # batch * dim * length
        if settings['end_padding']:
            x = self.padding_layer(x)
        x = x.unfold(-1, settings['patch_len'], settings['stride_len'])  # batch * dim * patch_num * patch_len
        x = self.encoder(x)
        x = self.out_layer(x).transpose(1, 2)
        if settings['normalization']:
            x = self.norm_layer(x, norm=False)
        return x


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