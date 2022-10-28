# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”). All Bytedance Modifications are Copyright 2022 Bytedance Inc.

# Code for Color space conversion 
from typing import Tuple, Union

import torch
import torch.nn.functional as F

from torch import Tensor

__all__ = [
    "RGB2YCbCr",
    "YCbCr2RGB",
    "YUV444To420",
    "YUV420To444",
]

YCBCR_WEIGHTS = {
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}


def _check_input_tensor(tensor: Tensor) -> None:
    if (
        not isinstance(tensor, Tensor)
        or not tensor.is_floating_point()
        or not len(tensor.size()) in (3, 4)
        or not tensor.size(-3) == 3
    ):
        raise ValueError(
            "Expected a 3D or 4D tensor with shape (Nx3xHxW) or (3xHxW) as input"
        )


def rgb2ycbcr(rgb: Tensor) -> Tensor:
    """RGB to YCbCr conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    args:
        rgb (torch.Tensor): 3D or 4D floating point RGB tensor

    Returns:
        ycbcr (torch.Tensor): converted tensor
    """
    _check_input_tensor(rgb)

    r, g, b = rgb.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = torch.cat((y, cb, cr), dim=-3)
    return ycbcr


def ycbcr2rgb(ycbcr: Tensor) -> Tensor:
    """YCbCr to RGB conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    args:
        ycbcr (torch.Tensor): 3D or 4D floating point RGB tensor

    Returns:
        rgb (torch.Tensor): converted tensor
    """
    _check_input_tensor(ycbcr)

    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = torch.cat((r, g, b), dim=-3)
    return rgb


def yuv_444_to_420(
    yuv: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
    mode: str = "avg_pool",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert a 444 tensor to a 420 representation.

    args:
        yuv (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): 444
            input to be downsampled. Takes either a (Nx3xHxW) tensor or a tuple
            of 3 (Nx1xHxW) tensors.
        mode (str): algorithm used for downsampling: ``'avg_pool'``. Default
            ``'avg_pool'``

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): Converted 420
    """
    if mode not in ("avg_pool",):
        raise ValueError(f'Invalid downsampling mode "{mode}".')

    if mode == "avg_pool":
        def _downsample(tensor):
            return F.avg_pool2d(tensor, kernel_size=2, stride=2)

    if isinstance(yuv, torch.Tensor):
        y, u, v = yuv.chunk(3, 1)
    else:
        y, u, v = yuv

    return (y, _downsample(u), _downsample(v))


def yuv_420_to_444(
    yuv: Tuple[Tensor, Tensor, Tensor],
    mode: str = "bilinear",
    return_tuple: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Convert a 420 input to a 444 representation.

    args:
        yuv (torch.Tensor, torch.Tensor, torch.Tensor): 420 input frames in
            (Nx1xHxW) format
        mode (str): algorithm used for upsampling: ``'bilinear'`` |
            ``'nearest'`` Default ``'bilinear'``
        return_tuple (bool): return input as tuple of tensors instead of a
            concatenated tensor, 3 (Nx1xHxW) tensors instead of one (Nx3xHxW)
            tensor (default: False)

    Returns:
        (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): Converted
            444
    """
    if len(yuv) != 3 or any(not isinstance(c, torch.Tensor) for c in yuv):
        raise ValueError("Expected a tuple of 3 torch tensors")

    if mode not in ("bilinear", "nearest"):
        raise ValueError(f'Invalid upsampling mode "{mode}".')

    if mode in ("bilinear", "nearest"):

        def _upsample(tensor):
            return F.interpolate(tensor, scale_factor=2, mode=mode, align_corners=False)

    y, u, v = yuv
    u, v = _upsample(u), _upsample(v)
    if return_tuple:
        return y, u, v
    return torch.cat((y, u, v), dim=1)


class RGB2YCbCr:
    """Convert a RGB tensor to YCbCr.
    The tensor is expected to be in the [0, 1] floating point range, with a
    shape of (3xHxW) or (Nx3xHxW).
    """

    def __call__(self, rgb):
        """
        args:
            rgb (torch.Tensor): 3D or 4D floating point RGB tensor

        Returns:
            ycbcr(torch.Tensor): converted tensor
        """
        return rgb2ycbcr(rgb)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class YCbCr2RGB:
    """Convert a YCbCr tensor to RGB.
    The tensor is expected to be in the [0, 1] floating point range, with a
    shape of (3xHxW) or (Nx3xHxW).
    """

    def __call__(self, ycbcr):
        """
        args:
            ycbcr(torch.Tensor): 3D or 4D floating point RGB tensor

        Returns:
            rgb(torch.Tensor): converted tensor
        """
        return ycbcr2rgb(ycbcr)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class YUV444To420:
    """Convert a YUV 444 tensor to a 420 representation.

    args:
        mode (str): algorithm used for downsampling: ``'avg_pool'``. Default
            ``'avg_pool'``

    Example:
        >>> x = torch.rand(1, 3, 32, 32)
        >>> y, u, v = YUV444To420()(x)
        >>> y.size()  # 1, 1, 32, 32
        >>> u.size()  # 1, 1, 16, 16
    """

    def __init__(self, mode: str = "avg_pool"):
        self.mode = str(mode)

    def __call__(self, yuv):
        """
        args:
            yuv (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)):
                444 input to be downsampled. Takes either a (Nx3xHxW) tensor or
                a tuple of 3 (Nx1xHxW) tensors.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Converted 420
        """
        return yuv_444_to_420(yuv, mode=self.mode)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class YUV420To444:
    """Convert a YUV 420 input to a 444 representation.

    args:
        mode (str): algorithm used for upsampling: ``'bilinear'`` | ``'nearest'``.
            Default ``'bilinear'``
        return_tuple (bool): return input as tuple of tensors instead of a
            concatenated tensor, 3 (Nx1xHxW) tensors instead of one (Nx3xHxW)
            tensor (default: False)

    Example:
        >>> y = torch.rand(1, 1, 32, 32)
        >>> u, v = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16)
        >>> x = YUV420To444()((y, u, v))
        >>> x.size()  # 1, 3, 32, 32
    """

    def __init__(self, mode: str = "bilinear", return_tuple: bool = False):
        self.mode = str(mode)
        self.return_tuple = bool(return_tuple)

    def __call__(self, yuv):
        """
        args:
            yuv (torch.Tensor, torch.Tensor, torch.Tensor): 420 input frames in
                (Nx1xHxW) format

        Returns:
            (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): Converted
                444
        """
        return yuv_420_to_444(yuv, return_tuple=self.return_tuple)

    def __repr__(self):
        return f"{self.__class__.__name__}(return_tuple={self.return_tuple})"
