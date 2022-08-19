#    Copyright (C) 2017-2019, Arraiy, Inc., all rights reserved.
#    Copyright (C) 2019-    , Kornia authors, all rights reserved.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

#This file may have been modified by Bytedance Inc. (“Bytedance Modifications”). All Bytedance Modifications are Copyright 2022 Bytedance Inc.

import math

import torch
from Metric.metric_tool.ssim import ms_ssim
from IQA_pytorch import VIFs, NLPD, FSIM
import os
from Metric.metric_tool import IW_SSIM
from Metric.metric_tool import iwssimgpu
from Metric.metric_tool import VMAF
from psnr_hvsm import psnr_hvs_hvsm
import numpy as np
import torch.nn.functional as F


def color_conv_matrix(color_conv="709"):
    if color_conv == "601":
        # BT.601
        a = 0.299
        b = 0.587
        c = 0.114
        d = 1.772
        e = 1.402
    elif color_conv == "709":
        # BT.709
        a = 0.2126
        b = 0.7152
        c = 0.0722
        d = 1.8556
        e = 1.5748
    elif color_conv == "2020":
        # BT.2020
        a = 0.2627
        b = 0.6780
        c = 0.0593
        d = 1.8814
        e = 1.4747
    else:
        raise NotImplementedError

    return a, b, c, d, e


def yuv_to_rgb(image: torch.Tensor, color_conv="709") -> torch.Tensor:
    r"""Convert an YUV image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = yuv_to_rgb(input)  # 2x3x4x5

    Took from https://kornia.readthedocs.io/en/latest/_modules/kornia/color/yuv.html#rgb_to_yuv
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    y: torch.Tensor = image[..., 0, :, :]
    u: torch.Tensor = image[..., 1, :, :] - 0.5
    v: torch.Tensor = image[..., 2, :, :] - 0.5

    # r: torch.Tensor = y + 1.14 * v  # coefficient for g is 0
    # g: torch.Tensor = y + -0.396 * u - 0.581 * v
    # b: torch.Tensor = y + 2.029 * u  # coefficient for b is 0

    a, b, c, d, e = color_conv_matrix(color_conv)

    r: torch.Tensor = y + e * v  # coefficient for g is 0
    g: torch.Tensor = y - (c * d / b) * u - (a * e / b) * v
    b: torch.Tensor = y + d * u  # coefficient for b is 0

    out: torch.Tensor = torch.stack([r, g, b], -3)

    return out


def rgb_to_yuv(image: torch.Tensor, color_conv="709") -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    a1, b1, c1, d1, e1 = color_conv_matrix(color_conv)

    # y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    # u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    # v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b
    y: torch.Tensor = a1 * r + b1 * g + c1 * b
    u: torch.Tensor = (b - y) / d1 + 0.5
    v: torch.Tensor = (r - y) / e1 + 0.5

    out: torch.Tensor = torch.stack([y, u, v], -3)

    return out


def calPSNR(mse, data_range=255.0):
    return 20 * math.log10(data_range) - 10 * math.log10(mse) if mse != 0.0 else 100


def yuvPSNR(x_yuv, org_yuv, data_range=255.0):
    _, __, h, w = org_yuv.shape
    mse = ((x_yuv - org_yuv) ** 2) / (h * w)
    mse = mse.squeeze()
    mse = torch.sum(torch.sum(mse, dim = 1), dim = 1)
    psnrY = calPSNR(mse[0], data_range=data_range)
    psnrU = calPSNR(mse[1], data_range=data_range)
    psnrV = calPSNR(mse[2], data_range=data_range)
    return psnrY, psnrU, psnrV

def rgbPSNR(x, org, data_range=255.0):
    _, __, h, w = org.shape
    mse = ((x - org) ** 2) / (h * w * 3)
    mse = mse.squeeze()
    mse = torch.sum(torch.sum(torch.sum(mse, dim = 1), dim = 1))
    psnr = calPSNR(mse, data_range=data_range)
    return psnr

def rfft_fn(im, dim, onesided=False):
    if onesided:
        raise NotImplemented
    else:
        if dim == 2:
            output_fft_new = torch.fft.fft2(im, dim = (-2, -1))
            return torch.stack((output_fft_new.real, output_fft_new.imag), -1)
        else:
            raise NotImplemented


def irfft_fn(im, dim):
    if dim == 2:
        output_ifft_new = torch.fft.ifft2(torch.complex(im[..., 0], im[..., 1]), dim = (-2, -1))
        return torch.stack((output_ifft_new.real, output_ifft_new.imag), -1)
    else:
        raise NotImplemented


def calculateMetrics(x, x_org, bindir, q, image, timeinfo=None,fast_iwssim=True,skip = []):
    # the input must be [0,3,height,width] in shape, value range [0 to 1] and in RGB format. It is designed to take
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, h, w = x.shape
    pixelnum = h * w
    binsize = os.path.getsize(bindir) * 8
    bpp = binsize / pixelnum
    bits = 10
    data_range = (1 << bits)-1

    rgb_psnr = rgbPSNR(x.mul(data_range).round(), x_org.mul(data_range).round(), data_range=data_range)

    if "cuda" in str(x.device):
        gpu_x_yuv = rgb_to_yuv(x).mul(data_range).round()
        x_yuv = gpu_x_yuv.to('cpu')
    else:
        x_yuv = rgb_to_yuv(x).mul(data_range).round()
        gpu_x_yuv = x_yuv.to(device)

    if "cuda" in str(x_org.device):
        gpu_org_yuv = rgb_to_yuv(x_org).mul(data_range).round()
        org_yuv = gpu_org_yuv.to('cpu')
    else:
        org_yuv = rgb_to_yuv(x_org).mul(data_range).round()
        gpu_org_yuv = org_yuv.to(device)
    # value range is determined by bit depth
    psnrY, psnrU, psnrV = yuvPSNR(gpu_x_yuv, gpu_org_yuv, data_range=data_range)
    if 'y_msssim' in skip:
        y_msssim = 100
    else:
        y_msssim = ms_ssim(gpu_org_yuv[0:1, 0:1, :, :], gpu_x_yuv[0:1, 0:1, :, :], data_range = data_range).item()
    if 'iw_ssim' in skip:
        iw_ssim = 100
    else:
        if fast_iwssim:
            iwssim_gpu = iwssimgpu()
            iw_ssim = iwssim_gpu(gpu_org_yuv[0:1, 0:1, :, :].div(data_range).mul(255), gpu_x_yuv[0:1, 0:1, :, :].div(data_range).mul(255)).item()
        else:
            iwssim = IW_SSIM()
            iw_ssim = iwssim.test(org_yuv[0, 0, :, :].div(data_range).mul(255).numpy(), x_yuv[0, 0, :, :].div(data_range).mul(255).numpy()).item()
    _, _, h, w = x_org.shape
    # value range must in [0 to 1]
    if 'psnr_hvsm' in skip:
        hvsm = 100
    else:
        w_pad = (1 + (w-1)//8)*8 - w
        h_pad = (1 + (h-1)//8)*8 - h
        x_yuv_pad = F.pad(x_yuv, (0, w_pad, 0, h_pad),mode='replicate')
        org_yuv_pad = F.pad(org_yuv, (0, w_pad, 0, h_pad),mode='replicate')
        hvsm, _ = psnr_hvs_hvsm(x_yuv_pad[0, 0, :, :].div(data_range).cpu().numpy().astype(np.float64),
                            org_yuv_pad[0, 0, :, :].div(data_range).cpu().numpy().astype(np.float64))
    # Input of FSIM is rgb, value range is [0 to 1]
    if torch.__version__.split('+')[0] > '1.7.1' or torch.__version__ > '1.7.1':
        torch.rfft = rfft_fn
        torch.ifft = irfft_fn
    if 'fsim' in skip:
        fsim_result = 100
    else:
        fsim = FSIM(channels = 3).to(device)
        fsim_result = fsim(x.to(device), x_org.to(device), as_loss = False).item()
    if 'vmaf' in skip:
        vmaf_results = 100
    else:
        vmaf = VMAF(bits=bits, max_val=data_range)
        vmaf_results = vmaf.calc(org_yuv, x_yuv)
    if 'nlpd' in skip:
        nlpd_result = 100
    else:
        nlpd = NLPD(channels = 1).to(device)
        nlpd_result = nlpd(gpu_x_yuv[0, 0, :, :].unsqueeze(0).unsqueeze(0).div(data_range),
                        gpu_org_yuv[0, 0, :, :].unsqueeze(0).unsqueeze(0).div(data_range), as_loss = False).item()
    if 'vif' in skip:
        vif_metric = 100
    else:
        vif = VIFs(channels = 1).to(device)
        vif_metric = vif(gpu_x_yuv[0, 0, :, :].unsqueeze(0).unsqueeze(0).div(data_range),
                        gpu_org_yuv[0, 0, :, :].unsqueeze(0).unsqueeze(0).div(data_range), as_loss = False).item()
    
    if timeinfo:
        enc, dec = 0, 0
        for line in open(timeinfo):
            if 'Enc' in line:
                enc = float(line.split(':')[-1])
            if 'Dec' in line:
                dec = float(line.split(':')[-1])
    else:
        enc, dec = 0, 0

    out = {"y_msssim": y_msssim,
           "psnr": {"y": psnrY, "u": psnrU, "v": psnrV},
           "vif": vif_metric,
           "fsim": fsim_result,
           "nlpd": nlpd_result,
           "iw_ssim": iw_ssim,
           "vmaf": vmaf_results,
           "psnr_hvsm": hvsm,
           "psnr_rgb": rgb_psnr,
           "bpp": bpp,
           "enc": enc,
           "dec": dec,
           "q": q,
           "image": image}
    return out
