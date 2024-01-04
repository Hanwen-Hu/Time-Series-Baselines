import math

import torch
import torch.nn as nn

def line_1d_pos_embed(pos_num, exp=False, norm=True):
    pos = 2 * (torch.linspace(0, 1, pos_num).reshape(-1, 1) ** (0.5 if exp else 1)) - 1
    if norm:
        pos = pos - torch.mean(pos)
        pos = pos / (torch.std(pos) * 10)
    return pos

def line_2d_pos_embed(pos_num, embed_dim, exp=False, norm=True, eps=1e-3):
    x = 0.5 if exp else 1
    pos = 2 * (torch.linspace(0, 1, pos_num).reshape(-1, 1) ** x) * (torch.linspace(0, 1, embed_dim).reshape(1, -1) ** x) - 1
    for _ in range(100):
        if abs(torch.mean(pos)) <= eps:
            break
        elif torch.mean(pos) > eps:
            x += 0.001
        else:
            x -= 0.001
        pos = 2 * (torch.linspace(0, 1, pos_num).reshape(-1, 1) ** x) * (torch.linspace(0, 1, embed_dim).reshape(1, -1) ** x) - 1
    if norm:
        pos = pos - torch.mean(pos)
        pos = pos / (torch.std(pos) * 10)
    return pos

def sin_cos_pos_embed(pos_num, embed_dim, norm=True):
    pos = torch.zeros(pos_num, embed_dim)
    positions = torch.arange(pos_num).unsqueeze(1)
    term = torch.exp(torch.arange(0, embed_dim, 2) / embed_dim * -math.log(10000.0))
    pos[:, 0::2] = torch.sin(term * positions)
    pos[:, 1::2] = torch.cos(term * positions)
    if norm:
        pos = pos - torch.mean(pos)
        pos = pos / (torch.std(pos) * 10)
    return pos

def positional_embedding(pos_num, embed_dim, mode=None, learnable=False):
    if mode == 'zero':
        pos = torch.empty((pos_num, 1))
        nn.init.uniform_(pos, -0.02, 0.02)
    elif mode == 'zeros':
        pos = torch.empty((pos_num, embed_dim))
        nn.init.uniform_(pos, -0.02, 0.02)
    elif mode == 'normal':
        pos = torch.zeros((pos_num, 1))
        torch.nn.init.normal_(pos, mean=0, std=0.1)
    elif mode == 'uniform':
        pos = torch.zeros((pos_num, 1))
        torch.nn.init.uniform_(pos, a=0, b=0.1)
    elif mode == 'line1d':
        pos = line_1d_pos_embed(pos_num, exp=False, norm=True)
    elif mode == 'exp1d':
        pos = line_1d_pos_embed(pos_num, exp=True, norm=True)
    elif mode == 'line2d':
        pos = line_2d_pos_embed(pos_num, embed_dim, exp=False, norm=True)
    elif mode == 'exp2d':
        pos = line_2d_pos_embed(pos_num, embed_dim, exp=True, norm=True)
    elif mode == 'sin_cos':
        pos = sin_cos_pos_embed(pos_num, embed_dim, norm=True)
    else:
        assert mode is None
        pos = torch.empty((pos_num, embed_dim))
        nn.init.uniform_(pos, -0.02, 0.02)
        learnable = False
    return nn.Parameter(pos, requires_grad=learnable)
