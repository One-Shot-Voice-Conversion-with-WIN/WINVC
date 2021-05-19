import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import math


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


"""Wadain-resblock"""
class Style2ResidualBlock1DBeta(nn.Module):
    '''a stylegan2 module'''

    def __init__(self, dim_in, dim_out, kernel_size=3, num_speakers=4, spk_emb_dim=128, **kwargs):
        super().__init__()

        self.dim_out = dim_out * 2
        self.style_linear = EqualLinear(spk_emb_dim, dim_in, bias_init=1)
        self.style_linear_beta = EqualLinear(spk_emb_dim, dim_in, bias_init=1)
        self.weight = nn.Parameter(torch.randn(1, self.dim_out, dim_in, kernel_size), requires_grad=True)

        fan_in = dim_in * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)

        self.padding = kernel_size // 2
        self.dim_in = dim_in
        self.kernel_size = kernel_size
        self.glu = nn.GLU(dim=1)
        # self.relu = nn.LeakyReLU(0.2)

    def forward(self, inputs):
        x, c_src, c_trg = inputs
        """
            x.shape =  torch.Size([8, 256, 256])  # 一维卷积之后的结果
            c_src.shape =  torch.Size([8, 128])   
            c_trg.shape =  torch.Size([8, 128])   # # 128维 speaker embedding
        """
        batch_size, in_channel, t = x.size()  # 8, 256, 256

        s = self.style_linear(c_trg).view(batch_size, 1, in_channel, 1)
        beta = self.style_linear_beta(c_trg).view(batch_size, 1, in_channel, 1)
        """
        self.style_linear: 
            torch.autograd.Function.linear: 全联接层，
        
        input:  
            speaker embedding with dimention = 128
        output: 
            s = gamma = torch.Size([8, 1, 256, 1])
            beta = torch.Size([8, 1, 256, 1])
        """

        # modulate: scale weights
        weight = self.scale * (self.weight * s + beta)  # b out in ks
        """
        self.weight.shape = torch.Size([1, 512, 256, 3])
        s.shape = torch.Size([8, 1, 256, 1])
        beta.shape = torch.Size([8, 1, 256, 1])

        weight.shape =  torch.Size([8, 512, 256, 3])
        """

        # demodulate
        """
        Related work refered to image style transfer paper:
            T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila, 
            “Analyzing and improving the image quality of stylegan,” 
            in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 8110–8119.
        
        Before torch.rsqrt()平方根倒数:
            weight.shape =  torch.Size([8, 512, 256, 3])

        After torch.rsqrt():
            demod.shape =  torch.Size([8, 512])
        """
        demod = torch.rsqrt(weight.pow(2).sum([2, 3]) + 1e-8)  # demod.shape =  torch.Size([8, 512])

        demod_mean = torch.mean(weight.view(batch_size, self.dim_out, -1), dim=2)  # demod_mean.shape =  torch.Size([8, 512])

        weight = (weight - demod_mean.view(batch_size, self.dim_out, 1, 1)) * demod.view(batch_size, self.dim_out, 1, 1)
        """
        Input:
            weight.shape =  torch.Size([8, 512, 256, 3])
            demod_mean.view(batch_size, self.dim_out, 1, 1).shape =  torch.Size([8, 512, 1, 1])
            demod.view(batch_size, self.dim_out, 1, 1).shape =       torch.Size([8, 512, 1, 1])
        
        Output:
            weight.shape =  torch.Size([8, 512, 256, 3])
        """

        weight = weight.view(batch_size * self.dim_out, self.dim_in, self.kernel_size)  # weight.shape =  torch.Size([4096, 256, 3])

        x = x.view(1, batch_size * in_channel, t)  # x.shape =  torch.Size([1, 2048, 256])

        out = F.conv1d(x, weight, padding=self.padding, groups=batch_size)  # out.shape =  torch.Size([1, 4096, 256])

        _, _, new_t = out.size()

        out = out.view(batch_size, self.dim_out, new_t)  # out.shape =  torch.Size([8, 512, 256])
        out = self.glu(out)  # out.shape =  torch.Size([8, 256, 256])

        # out = self.relu(out)
        return (out, c_src, c_trg)
