import torch
import torch.nn as nn
import numpy as np
import argparse
from data_loader_vctk import get_loader, to_categorical
import torch.nn.functional as F
from modules import Style2ResidualBlock1DBeta
from torch.nn.utils import weight_norm
import math


# SPEncoder使用
class SPEncoder(nn.Module):
    '''speaker encoder for adaptive instance normalization'''

    def __init__(self, num_speakers=4, num_ft_speakers=None):

        super().__init__()
        self.num_speakers = num_speakers
        self.num_ft_speakers = num_ft_speakers
        self.down_sample_1 = nn.Sequential(
            nn.Conv1d(in_channels=80, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.unshared = nn.ModuleList()

        for _ in range(num_speakers):
            self.unshared += [nn.Linear(512, 128)]
        self.ft_layers = nn.ModuleList()

    def init_ft_params(self):
        if self.num_ft_speakers is None:
            raise Exception
        for _ in range(self.num_ft_speakers):
            self.ft_layers.append(nn.Linear(512, 128))

    def forward(self, x, trg_c, cls_out=False):

        x = x.squeeze(1)

        out = self.down_sample_1(x)

        out = self.down_sample_2(out)

        out = self.down_sample_3(out)

        out = self.down_sample_4(out)

        out = self.down_sample_5(out)

        out_mean = torch.mean(out, dim=2)
        out_std = torch.std(out, dim=2)

        out = torch.cat([out_mean, out_std], dim=1)

        res = []
        for layer in self.unshared:
            res += [layer(out)]
        if not len(self.ft_layers) == 0:
            for layer in self.ft_layers:
                res += [layer(out)]
        res = torch.stack(res, dim=1)

        idx = torch.LongTensor(range(x.size(0))).to(x.device)
        s = res[idx, trg_c.long()]

        return s


# Generator使用
class GeneratorPlain(nn.Module):
    """Generator network."""

    def __init__(self, num_speakers=4, res_block_name='Style2ResidualBlock1DBeta', num_res_blocks=6, num_heads=128,
                 kernel=9, use_kconv=True, spk_emb_dim=128, hidden_size=512):
        super(GeneratorPlain, self).__init__()
        # Down-sampling layers
        self.res_block_name = res_block_name
        conv1_kernels = [1, 1, 3, 3, 5, 5, 7, 7]
        conv1_layers = [nn.Conv1d(80, 32, kernel_size=k, stride=1, padding=k // 2) for k in conv1_kernels]
        self.conv1 = nn.ModuleList(conv1_layers)

        def norm(dim_in, mode='in'):
            if mode == 'in':
                return nn.InstanceNorm1d(dim_in, affine=True)
            # elif mode == 'tfan':
            #     return TFAN1d(dim_in)

        self.down_sample_2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=hidden_size * 2, kernel_size=9, stride=1, padding=4),
            norm(hidden_size * 2),
            nn.GLU(dim=1),
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size * 2, kernel_size=9, stride=1, padding=4),
            norm(hidden_size * 2),
            nn.GLU(dim=1),
            nn.Conv1d(hidden_size, hidden_size, 1, 1, 0, bias=False),
            norm(hidden_size)
        )

        # Bottleneck layers.
        # 先用 固定kernel size，由main参数表传入
        res_blocks = [eval(self.res_block_name)(hidden_size, hidden_size, num_heads=num_heads, kernel_size=kernel,
                                                use_kconv=use_kconv, spk_emb_dim=spk_emb_dim, wada_kernel=3) for i in
                      range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        # Up-sampling layers.

        self.up_sample_1 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1, 1, 0, bias=False),
            norm(hidden_size, 'in'),
            # nn.ConvTranspose1d(hidden_size, hidden_size*2, kernel_size = 4, stride = 2, padding = 1),
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=9, stride=1, padding=4),
            norm(hidden_size * 2, 'in'),
            nn.GLU(dim=1),
        )
        self.up_sample_2 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=9, stride=1, padding=4),
            # nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size = 4, stride = 2,padding = 1),
            norm(hidden_size, 'in'),
            nn.GLU(dim=1),
        )

        # Out.
        self.out = nn.Conv1d(in_channels=hidden_size // 2, out_channels=80, kernel_size=5, stride=1, padding=2,
                             bias=False)

    def forward(self, x, c_src, c_trg):

        # convert to 1d
        if len(x.size()) == 4:
            x = x.squeeze(1)
        conv1_outs = []

        for layer in self.conv1:
            layer_out = layer(x)
            conv1_outs.append(layer_out)

        conv1_out = torch.cat(conv1_outs, 1)

        down1_out = self.down_sample_2(conv1_out)

        down2_out = self.down_sample_3(down1_out)

        b_out = down2_out

        b_out, _, _ = self.res_blocks((b_out, c_src, c_trg))

        up1_out = self.up_sample_1(b_out)
        up2_out = self.up_sample_2(up1_out)
        out = self.out(up2_out)
        out = out.unsqueeze(1)
        return out


# Discriminator使用
class PatchDiscriminator1(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, num_speakers=4, num_ft_speakers=None):
        super(PatchDiscriminator1, self).__init__()

        self.num_speakers = num_speakers
        self.num_ft_speakers = num_ft_speakers
        # Initial layers.

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        self.down_sample_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        # self.dis_conv = nn.Conv2d(256, num_speakers, kernel_size = (2,8), stride = 1, padding = 0, bias = False )
        self.dis_conv = nn.ModuleList()
        self.ft_layers = nn.ModuleList()
        for _ in range(num_speakers):
            self.dis_conv.append(nn.Conv2d(256, 1, kernel_size=(2, 8), stride=1, padding=0, bias=False))

    def init_ft_params(self):
        if self.num_ft_speakers is None:
            raise Exception
        for _ in range(self.num_ft_speakers):
            self.ft_layers.append(nn.Conv2d(256, 1, kernel_size=(2, 8), stride=1, padding=0, bias=False))
        # self.dis_conv.extend(self.ft_layers)

    def forward(self, x, c, c_, trg_cond=None):

        x = self.conv_layer_1(x)

        x = self.down_sample_1(x)

        x = self.down_sample_2(x)

        x = self.down_sample_3(x)

        x = self.down_sample_4(x)
        res = []
        for layer in self.dis_conv:
            res += [layer(x)]
        if not len(self.ft_layers) == 0:
            for layer in self.ft_layers:
                res += [layer(x)]
        x = torch.cat(res, dim=1)
        b, c, h, w = x.size()
        x = x.view(b, c)

        idx = torch.LongTensor(range(x.size(0))).to(x.device)

        x = x[idx, c_.long()]

        return x



class GLU(nn.Module):
    ''' Test GLU block in new way, which do not split channels dimension'''

    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class AdaptiveInstanceNormalisation(nn.Module):
    """Test AdaIN Block."""

    def __init__(self, dim_in, dim_c):
        super(AdaptiveInstanceNormalisation, self).__init__()
        self.dim_in = dim_in
        self.gamma_t = nn.Linear(2 * dim_c, dim_in)
        self.beta_t = nn.Linear(2 * dim_c, dim_in)

    def forward(self, x, c_src, c_trg):
        u = torch.mean(x, dim=2, keepdim=True)
        var = torch.mean((x - u) * (x - u), dim=2, keepdim=True)
        std = torch.sqrt(var + 1e-8)

        c = torch.cat([c_src, c_trg], dim=-1)
        gamma = self.gamma_t(c)
        gamma = gamma.view(-1, self.dim_in, 1)
        beta = self.beta_t(c)
        beta = beta.view(-1, self.dim_in, 1)

        h = (x - u) / std
        h = h * gamma + beta
        return h


class TFAN1d(nn.Module):
    '''
    This is a time frequency attention normalization layer from CycleGAN-VC3 paper.
    Doesn't perform well
    '''

    def __init__(self, in_channels, hidden_size=128, n_layers=3):
        super().__init__()

        self.norm = nn.InstanceNorm1d(in_channels, affine=False)
        layers = [nn.Conv1d(in_channels, hidden_size, 5, 1, 2), nn.Sigmoid()]
        for _ in range(1, n_layers):
            layers.append(nn.Conv1d(hidden_size, hidden_size, 5, 1, 2))
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

        self.gamma = nn.Conv1d(hidden_size, in_channels, 5, 1, 2)
        self.beta = nn.Conv1d(hidden_size, in_channels, 5, 1, 2)

    def forward(self, x):
        # x: [B,C,T]

        normed = self.norm(x)
        h = self.layers(x)
        gamma = self.gamma(h)
        beta = self.beta(h)
        out = normed * gamma + beta

        return out
