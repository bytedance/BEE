# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”). All Bytedance Modifications are Copyright 2022 Bytedance Inc.

from typing import Any

import torch
import torch.nn as nn

from torch import Tensor
from torch import nn
import torch.nn.functional as F

from Common.utils.paraops import NonNegativeParametrizer

__all__ = [
    "AttentionBlock",
    "MaskedConv2d",
    "ResidualBlock",
    "ResidualBlockUpsample",
    "ResidualBlockWithStride",
    "conv3x3",
    "subpel_conv3x3",
    'conv',
    'deconv'
]


class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
        device='cpu'
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min,device=device)
        beta = torch.ones(in_channels,device=device)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer(device=device)
        gamma = gamma_init * torch.eye(in_channels,device=device)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x ** 2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out


class GDN1(GDN):
    r"""Simplified GDN layer.

    Introduced in `"Computationally Efficient Neural Image Compression"
    <http://arxiv.org/abs/1912.08771>`_, by Johnston Nick, Elad Eban, Ariel
    Gordon, and Johannes Ballé, (2019).

    .. math::

        y[i] = \frac{x[i]}{\beta[i] + \sum_j(\gamma[j, i] * |x[j]|}

    """

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(torch.abs(x), gamma, beta)

        if not self.inverse:
            norm = 1.0 / norm

        out = x * norm

        return out


def conv(in_channels, out_channels, kernel_size=5, stride=2,device = 'cpu'):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        device=device
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2,device = 'cpu'):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
        device=device
    )

class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", device='cpu',**kwargs: Any):
        super().__init__(*args, device=device, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data,device=device))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B"):] = 0
        self.mask[:, :, h // 2 + 1:] = 0
        self.mask[:, :, h // 2 - 1, w - 1] = 0
        self.mask[:, :, h // 2 - 1, w - 2] = 0
        self.mask[:, :, h // 2 - 2, w - 1] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1,device='cpu') -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1,device=device)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1,device='cpu') -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1,device=device), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1,device='cpu') -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride,device=device)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2,device='cpu'):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride,device=device)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch,device=device)
        self.gdn = GDN(out_ch,device=device)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride,device=device)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2,device='cpu'):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample,device=device)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch,device=device)
        self.igdn = GDN(out_ch, inverse=True,device=device)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample,device=device)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int,device='cpu'):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch,device=device)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch,device=device)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch,device=device)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    args:
        N (int): Number of channels)
    """

    def __init__(self, N: int,device='cpu'):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2,device=device),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2,device=device),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N,device=device),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N,device=device),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out
