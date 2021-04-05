import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import math



"""Wadain使用"""
class Wadain(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size=3, spk_emb_dim=128):
        super().__init__()

        self.style_linear = EqualLinear(spk_emb_dim, dim_in, bias_init=1)
        self.style_linear_beta = EqualLinear(spk_emb_dim, dim_in, bias_init=1)
        self.weight = nn.Parameter(torch.randn(1, dim_out, dim_in, kernel_size), requires_grad=True)

        fan_in = dim_in * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)

        self.padding = kernel_size // 2
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.kernel_size = kernel_size

    def forward(self, x, c_trg):
        batch_size, in_channel, t = x.size()

        # c = torch.cat([c_src, c_trg], dim = -1)

        s = self.style_linear(c_trg).view(batch_size, 1, in_channel, 1)
        b = self.style_linear(c_trg).view(batch_size, 1, in_channel, 1)
        # scale weights —— modulate
        weight = self.scale * (self.weight * s + b)  # b out in ks

        # demodulate
        demod = torch.rsqrt(weight.pow(2).sum([2, 3]) + 1e-8)
        demod_mean = torch.mean(weight.view(batch_size, self.dim_out, -1), dim=2)
        weight = (weight - demod_mean.view(batch_size, self.dim_out, 1, 1)) * demod.view(batch_size, self.dim_out, 1, 1)

        weight = weight.view(batch_size * self.dim_out, self.dim_in, self.kernel_size)

        x = x.reshape(1, batch_size * in_channel, t)

        out = F.conv1d(x, weight, padding=self.padding, groups=batch_size)

        _, _, new_t = out.size()

        out = out.reshape(batch_size, self.dim_out, new_t)

        return out


    
    
"""Wadain用到"""
class EqualLinear(nn.Module):

    def __init__(self, dim_in, dim_out, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(dim_out, dim_in).div_(lr_mul), requires_grad=True)

        self.bias = nn.Parameter(torch.zeros(dim_out).fill_(bias_init), requires_grad=True)

        self.activation = activation
        self.scale = (1 / math.sqrt(dim_in)) * lr_mul

        # self.relu = nn.LeakyReLU(0.2)
        self.lr_mul = lr_mul

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul)
        # out = self.relu(out)
        return out
