# Copyright 2022 Bytedance Inc.

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from Common.ans import BufferedRansEncoder, RansDecoder
from Common.models.entropy_models import EntropyBottleneck, GaussianConditional
from Common.models.quantmodule import quantHSS
import Common.models.online as on
from Common.models.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    MaskedConv2d,
    conv, 
    deconv, 
)

from Common.utils.csc import RGB2YCbCr, YCbCr2RGB
from Common.utils.tensorops import update_registered_buffers, crop

__all__ = [
    "QuantYUV444Decoupled",
    "YUV444DecoupledV0",
]

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

MeanAndResidualScale = True

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class QuantYUV444Decoupled(nn.Module):
    def __init__(self, N=192, M=192, refine=4, init_weights=True, Quantized=False, oldversion=False, **kwargs):
        super().__init__()
        device = 'cpu' if kwargs.get("device") == None else kwargs.get("device")
        if init_weights:
            self._initialize_weights()
        self.N = int(N)
        self.M = int(M)
        self.Quantized = Quantized
        self.refine = refine
        self.noise = 0
        self.numthreads_min = 50
        self.numthreads_max = 100
        self.waveShift = 1
        self.numfilters = [0, 0, 0]
        self.filterCoeffs1 = []
        self.filterCoeffs2 = []
        self.filterCoeffs3 = []
        self.decSplit1 = 1
        self.decSplit2 = 1
        self.likely = None
        self.channelOffsetsTool = False
        self.encChannelOffsets = []
        self.decChannelOffsets = []
        self.offsetSplit_w = 0
        self.offsetSplit_h = 0
        self.DeterminismSpeedup = True
        self.DoublePrecisionProcessing = False
        self.encParams = None
        self.encSkip = False
        self.ACSkip = False
        self.y_q_full = None
        self.encOutput = None
        self.encCompleteSkip = False
        self.decCompleteSkip = False
        self.oldversion = oldversion
        self.decParameters = None
        self.kernel_height = 3
        self.kernel_width = 4
        self.yuv2rgb = YCbCr2RGB()
        self.rgb2yuv = RGB2YCbCr()

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size = 5, padding = 2, stride = 1, device = device
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1, device = device),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1, device = device),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1, device = device),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(M * 6 // 3, M * 5 // 3, 1, device = device),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(M * 5 // 3, M * 4 // 3, 1, device = device),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(M * 4 // 3, M * 3 // 3, 1, device = device),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.entropy_bottleneck = EntropyBottleneck(self.N)

        if not kwargs.get("decoderOnly"):
            self.g_a = nn.Sequential(
                ResidualBlockWithStride(3, N, stride = 2, device=device),
                ResidualBlock(N, N, device=device),
                ResidualBlockWithStride(N, N, stride = 2, device=device),
                AttentionBlock(N, device=device),
                ResidualBlock(N, N, device=device),
                ResidualBlockWithStride(N, N, stride = 2, device=device),
                ResidualBlock(N, N, device=device),
                conv3x3(N, N, stride = 2, device=device),
            )
            self.h_a = nn.Sequential(
                conv(M, N, stride = 1, kernel_size = 3),
                nn.LeakyReLU(inplace = True),
                conv(N, N, stride = 2, kernel_size = 5),
                nn.LeakyReLU(inplace = True),
                conv(N, N, stride = 2, kernel_size = 5),
            )
        self.g_s = nn.Sequential(
            ResidualBlock(N, N, device = device),
            ResidualBlockUpsample(N, N, 2, device = device),
            ResidualBlock(N, N, device = device),
            ResidualBlockUpsample(N, N, 2, device = device),
            AttentionBlock(N, device = device),
            ResidualBlock(N, N, device = device),
        )
        self.g_s_extension = nn.Sequential(
            ResidualBlockUpsample(N, N, 2, device = device),
            ResidualBlock(N, N, device = device),
            subpel_conv3x3(N, 3, 2, device = device),
        )
        self.h_s = nn.Sequential(
            deconv(N, M, stride = 2, kernel_size = 5, device = device),
            nn.LeakyReLU(inplace = True),
            deconv(M, M * 3 // 2, stride = 2, kernel_size = 5, device = device),
            nn.LeakyReLU(inplace = True),
            conv(M * 3 // 2, M * 2, stride = 1, kernel_size = 3, device = device),
        )
        if self.Quantized:
            self.h_s_scale = quantHSS(self.N, self.M, vbit = 16, cbit = 16, use_float = False, device = device)
            self.sbit = 12
        else:
            self.h_s_scale = nn.Sequential(
                deconv(N, M, stride = 2, kernel_size = 5,device=device),
                nn.LeakyReLU(inplace = True),
                deconv(M, M, stride = 2, kernel_size = 5,device=device),
                nn.LeakyReLU(inplace = True),
                conv(M, M, stride = 1, kernel_size = 3,device=device),
                nn.LeakyReLU(inplace = True),
                conv(M, M, stride = 1, kernel_size = 3,device=device),
            )

    def aux_loss(self):
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        u1 = self.gaussian_conditional.update_scale_table(scale_table, force = force)

        u2 = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force = force)
            u2 |= rv
        u1 |= u2
        return u1

    def load_state_dict(self, state_dict):
        state_dict_new = state_dict.copy()
        exclude = ["g_s.6", "g_s.7", "g_s.8"]
        include = ["g_s_extension.0", "g_s_extension.1", "g_s_extension.2"]
        for k in state_dict:
            for idx, j in enumerate(exclude):
                if j in k:
                    k_new = k.replace(j, include[idx])
                    state_dict_new[k_new] = state_dict_new[k]
                    del state_dict_new[k]
        state_dict = state_dict_new
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
        state_dict)
        super().load_state_dict(state_dict, strict=False)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    @staticmethod
    def splitFunc(x, func, splitNum, pad, crop):
        _, _, _, w = x.shape
        w_step = ((w + splitNum - 1) // splitNum)
        pp = []
        for i in range(splitNum):
            start_offset = pad if i > 0 else 0
            start_crop = crop if i > 0 else 0
            end_offset = pad if i < (splitNum - 1) else 0
            end_crop = crop if i < (splitNum - 1) else 0
            dummy = func(x[:, :, :, (w_step * i) - start_offset:w_step * (i + 1) + end_offset])
            dummy = dummy[:, :, :, start_crop:]
            if end_crop > 0:
                dummy = dummy[:, :, :, :-end_crop]
            pp.append(dummy)
        x_hat = torch.cat(pp, dim = 3)
        return x_hat

    def get_mask(self, scales_hat, filtercoeffs):
        mode = filtercoeffs["mode"]
        thr = filtercoeffs["thr"]
        block_size = filtercoeffs["block_size"]
        greater = filtercoeffs["greater"]
        scale = filtercoeffs["scale"]
        channels = filtercoeffs["channels"]
        mask = None
        self.likely = torch.abs(scales_hat)

        _, _, h, w = scales_hat.shape
        h_pad = ((h + block_size - 1) // block_size) * block_size - h
        w_pad = ((w + block_size - 1) // block_size) * block_size - w
        likely = F.pad(self.likely, (0, w_pad, 0, h_pad), value = 1)
        if mode == 1:  # minpool
            maxpool = torch.nn.MaxPool2d((block_size, block_size), (block_size, block_size), 0)
            likely = -(maxpool(-likely))
        if mode == 2:  # avgpool
            avgpool = torch.nn.AvgPool2d((block_size, block_size), (block_size, block_size), 0)
            likely = avgpool(likely)
        if mode == 3:  # maxpool
            maxpool = torch.nn.MaxPool2d((block_size, block_size), (block_size, block_size), 0)
            likely = (maxpool(likely))

        if greater:
            mask = (likely > thr)
        else:
            mask = (likely < thr)
        if mode == 4:  # maxpool
            for i in range(192):
                if i in channels:
                    pass
                else:
                    mask[:, i, :, :] = False
        mask = einops.repeat(mask, 'a b c d -> a b (c repeat1) (d repeat2)', repeat1 = block_size, repeat2 = block_size)
        mask = F.pad(mask, (0, -w_pad, 0, -h_pad), value = 1)
        return mask, scale

    def proxyfunc(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, _ = self.entropy_bottleneck(z)
        return z_hat

    def evalEstimate(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)
        _,_,h,w = y.shape
        y_hat = self._compress_ar_scale(y,params,h,w,padding = 2,scales_hat = y,forward = True)
        ctx_params = self.context_prediction(y_hat)

        means_hat = self.entropy_parameters(torch.cat((params, ctx_params), dim = 1))
        scales_hat = self.h_s_scale(z_hat)

        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means = means_hat)
        
        x_hat = self.g_s_extension(self.g_s(y_hat)).clamp_(0, 1)
        measurements = {"x_hat": x_hat, "y_hat":y_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}
        return measurements

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        if self.noise == 0:
            y_hat = self.gaussian_conditional.quantize(y, "noise")
        elif self.noise == 1:
            y_hat = y + y.round().detach() - y.detach()
        elif self.noise == 2:
            with torch.no_grad():
                c = self.context_prediction(y.round())
                means = self.entropy_parameters(torch.cat((params, c), dim = 1))
            y_hat = y + (y - means).round().detach() - y.detach() + means.detach()
        ctx_params = self.context_prediction(y_hat)

        means_hat = self.entropy_parameters(torch.cat((params, ctx_params), dim = 1))
        scales_hat = self.h_s_scale(z_hat)

        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means = means_hat)
        x_hat = self.g_s(y_hat)

        x_hat = self.g_s_extension(x_hat)

        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }   

    def forward_fineTune(self, x):
        with torch.no_grad():
            y = self.g_a(x)
            z = self.h_a(y)
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
            params = self.h_s(z_hat)
            _,_,h,w = y.shape
            y_hat = self._compress_ar_scale(y,params,h,w,padding = 2,scales_hat = y,forward = True)
            
        params = self.h_s(z_hat)  
        ctx_params = self.context_prediction(y_hat)
        means_hat = self.entropy_parameters(torch.cat((params, ctx_params), dim = 1))
        scales_hat = self.h_s_scale(z_hat)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means = means_hat)
        x_hat = self.g_s(y_hat)
        x_hat = self.g_s_extension(x_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods},
        }

    def refiner_init(self, y, z, quality, toUpdate):
        criterion = on.RateDistortionLoss(quality)
        updater = on.latentUpdater(toUpdate)
        optimizer = torch.optim.Adagrad([{'params': updater.parameters(), 'lr': 0.04}])
        params_est = self.h_s(z.round())
        ctx_params = self.context_prediction(y.round())
        means = self.entropy_parameters(torch.cat((params_est, ctx_params), dim = 1))
        means = means.detach()
        return criterion, updater, optimizer, means

    def y_refiner(self, y, z, x_true_org, quality, totalIte):
        criterion = on.RateDistortionLoss(quality)
        updater = on.latentUpdater(y)
        optimizer = torch.optim.Adagrad([{'params': updater.parameters(), 'lr': 0.03}])
        _, _, h, w = x_true_org.shape
        y = updater()
        y_best = y.clone()
        with torch.set_grad_enabled(True):
            for i in range(totalIte):
                x_hat = self.g_s(y)
                x_hat = self.g_s_extension(x_hat)
                x_hat = crop(x_hat, (h, w))
                measurements = {"x_hat": x_hat, "likelihoods": {"y": [], "z": []}}
                loss = criterion(measurements, x_true_org)
                optimizer.zero_grad()
                loss["loss"].backward()
                optimizer.step()
                y_best = y.clone()
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 3
        return y_best, z

    def _compress_ar_scale_old(self, y_hat, params, height, width, padding, scales_hat):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()

        count = ((height + self.numthreads_min - 1) // self.numthreads_min)
        numThreads = self.numthreads_min
        for n in range(self.numthreads_min, self.numthreads_max + 1):
            if ((height + n - 1) // n) < count:
                numThreads = n
                count = ((height + n - 1) // n)

        threadNum = range(numThreads)
        xtra = (numThreads - 1) * self.waveShift
        padDown = ((height + numThreads - 1) // numThreads) * numThreads - height
        y_q_full = torch.zeros_like(scales_hat)
        params = F.pad(params, (xtra, xtra, 0, padDown))
        y_org = y_hat.clone()
        y_hat = F.pad(y_hat, (padding + xtra, padding + xtra, padding, padding + padDown))

        _, params_channel_size, _, _ = params.shape
        ctx_p_channel_shape = self.context_prediction.bias.shape
        entropy_input = torch.zeros((1, int(params_channel_size) + int(ctx_p_channel_shape[0]), 1, numThreads),
                                    device = params.device)
        mask1 = []
        scale1 = []
        for i in range(self.numfilters[0]):
            mask, scale = self.get_mask(scales_hat, dict(self.filterCoeffs1[i]))
            mask = F.pad(mask, (padding + xtra, padding + xtra, padding, padding + padDown))
            mask1.append(mask)
            scale1.append(torch.tensor(scale[0]))
        torch.backends.cudnn.deterministic = False if self.DeterminismSpeedup else True
        for h in range(0, height + padDown, numThreads):
            for w in range(width + xtra):

                params_crop = params[:, :, h:h + numThreads, w:w + numThreads]
                entropy_input[:, :params_channel_size, :, :] = einops.rearrange(params_crop,
                                                                                'a b c d ->a b (c d)').unsqueeze(2)[:,
                                                               :, :, numThreads - 1:-1:numThreads - 1]

                y_hat_crop = y_hat[:, :, h: h + numThreads + self.kernel_height - 1,
                             w: w + numThreads + self.kernel_width - 1]
                y_hat_crop = F.unfold(y_hat_crop, kernel_size = (self.kernel_height, self.kernel_width))
                y_hat_crop = einops.rearrange(y_hat_crop[:, :, numThreads - 1:-(numThreads - 1):numThreads - 1],
                                              'a (C r e) d->a C r (d e)', C = self.N, e = self.kernel_width,
                                              r = self.kernel_height)
                entropy_input[:, params_channel_size:, :, :] = F.conv2d(
                    y_hat_crop,
                    self.context_prediction.weight[:, :, 0:self.kernel_height, 0:self.kernel_width] *
                    self.context_prediction.mask[:, :, 0:self.kernel_height, 0:self.kernel_width],
                    bias = self.context_prediction.bias,
                    stride = self.kernel_width,
                )

                means_hat = self.entropy_parameters(entropy_input)

                y_crop = y_hat[:, :, h + padding: h + numThreads + padding, w + padding: w + numThreads + padding]
                y_crop = einops.rearrange(y_crop, 'a b c d ->a b (c d)').unsqueeze(2)[:, :, :,
                         numThreads - 1:-1:numThreads - 1]
                y_q = (y_crop - means_hat)
                mask1_crop = []
                for i in range(self.numfilters[0]):
                    mask1_crop.append(
                        mask1[i][:, :, h + padding: h + numThreads + padding, w + padding: w + numThreads + padding])
                    mask1_crop[i] = einops.rearrange(mask1_crop[i], 'a b c d ->a b (c d)').unsqueeze(2)[:, :, :,
                                    numThreads - 1:-1:numThreads - 1]
                    y_q = torch.where(mask1_crop[i], y_q * scale1[i], y_q)

                y_q = y_q.round()
                for thr in threadNum:
                    if w < (width + thr * self.waveShift) and w >= (thr * self.waveShift):
                        y_q_full[:, :, h + thr: h + 1 + thr,
                        w - thr * self.waveShift: w - thr * self.waveShift + 1] = y_q[:, :, :, thr:thr + 1]

                for i in range(self.numfilters[0] - 1, -1, -1):
                    y_q = torch.where(mask1_crop[i], y_q / scale1[i], y_q)

                wmin = max(w + padding, padding + xtra)
                wdiff = max(0, wmin - w - padding)
                wmax = min(w + padding + numThreads, width + padding + xtra)
                wdiff2 = max(w + padding + numThreads - wmax, 0)

                y_hat_crop = y_hat[:, :, h + padding: h + padding + numThreads,
                             w + padding: w + padding + numThreads].reshape(1, self.N, 1, -1)
                y_hat_crop[:, :, :, numThreads - 1:-1:numThreads - 1] = means_hat + y_q
                y_hat[:, :, h + padding: h + padding + numThreads,
                w + padding + wdiff: w + padding + numThreads - wdiff2] = y_hat_crop.view(1, self.N, numThreads,
                                                                                          numThreads)[:, :, :,
                                                                          wdiff:numThreads - wdiff2]
        y_hat = F.pad(y_hat, (-padding - xtra, - padding - xtra, -padding, -padding - padDown))
        # Start of channel offsetting
        if self.channelOffsetsTool:
            wpad = ((width + self.offsetSplit_w - 1) // self.offsetSplit_w) * self.offsetSplit_w - width
            hpad = ((height + self.offsetSplit_h - 1) // self.offsetSplit_h) * self.offsetSplit_h - height
            kernel_w = (width + wpad) // self.offsetSplit_w
            kernel_h = (height + hpad) // self.offsetSplit_h
            diff = F.pad(y_org - y_hat, (0, wpad, 0, hpad))
            avg = torch.nn.AvgPool2d(kernel_size = (kernel_h, kernel_w), stride = (kernel_h, kernel_w), padding = 0)
            self.encChannelOffsets = avg(diff).squeeze()
            if self.offsetSplit_h == 1:
                self.encChannelOffsets.unsqueeze_(1)
            if self.offsetSplit_w == 1:
                self.encChannelOffsets.unsqueeze_(2)
            self.encChannelOffsets = self.encChannelOffsets.permute(1, 2, 0)
        # End of channel offsetting
        torch.backends.cudnn.deterministic = True

        mask2 = torch.ones_like(scales_hat, dtype = torch.bool)
        for i in range(self.numfilters[1]):
            mask, _ = self.get_mask(scales_hat, dict(self.filterCoeffs2[i]))
            mask2 = mask2 * mask
        mask1 = []
        scale1 = []
        for i in range(self.numfilters[0]):
            mask, scale = self.get_mask(scales_hat, dict(self.filterCoeffs1[i]))
            mask1.append(mask)
            scale1.append(scale)
        for i in range(self.numfilters[0]):
            scales_hat = scales_hat.double()
            scales_hat[mask1[i]] *= self.h_s_scale.h_s_scale[-1].scale_v
            scales_hat[mask1[i]].round().clamp(-(2 ** (self.h_s_scale.h_s_scale[-1].abit - 1)),
                                               2 ** (self.h_s_scale.h_s_scale[-1].abit - 1)-1)
            scalesfactor = round(scale1[i][0] * 2**self.sbit)
            scales_hat[mask1[i]] *= scalesfactor
            scales_hat[mask1[i]] /= (self.h_s_scale.h_s_scale[-1].scale_v * (2**self.sbit))
            scales_hat = scales_hat.float()

        y_q_full = y_q_full[mask2]
        scales_hat = scales_hat[mask2]

        symbols_list = y_q_full.to('cpu').numpy()
        indexes_full = self.gaussian_conditional.build_indexes(scales_hat)
        indexes_list = indexes_full.to('cpu').numpy()

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        string = encoder.flush()
        return string

    def _decompress_ar_scale_old(self, y_string, params, height, width, padding, scales_hat, device):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_string)

        mask2 = torch.ones_like(scales_hat, dtype = torch.bool)
        for i in range(self.numfilters[1]):
            mask, _ = self.get_mask(scales_hat, dict(self.filterCoeffs2[i]))
            mask2 = mask2 * mask
        mask1 = []
        scale1 = []
        for i in range(self.numfilters[0]):
            mask, scale = self.get_mask(scales_hat, dict(self.filterCoeffs1[i]))
            mask1.append(mask)
            scale1.append(scale)
        for i in range(self.numfilters[0]):
            # Quantized scale operation
            scales_hat = scales_hat.double()
            scales_hat[mask1[i]] *= self.h_s_scale.h_s_scale[-1].scale_v
            scales_hat[mask1[i]].round().clamp(-(2 ** (self.h_s_scale.h_s_scale[-1].abit - 1)),
                                               2 ** (self.h_s_scale.h_s_scale[-1].abit - 1) - 1)
            scalesfactor = round(scale1[i][0] * 2 ** self.sbit)
            scales_hat[mask1[i]] *= scalesfactor
            scales_hat[mask1[i]] /= (self.h_s_scale.h_s_scale[-1].scale_v * (2 ** self.sbit))
            scales_hat = scales_hat.float()
            # scales_hat[mask1[i]] *= scale1[i][0]

        indexes_full = self.gaussian_conditional.build_indexes(scales_hat[mask2])
        indexes = indexes_full.to('cpu').numpy()

        rv = decoder.decode_stream(
            indexes, cdf, cdf_lengths, offsets
        )

        rv_full = torch.zeros_like(scales_hat)
        rv_full[mask2] = torch.Tensor(rv).to(device)

        for i in range(self.numfilters[0] - 1, -1, -1):
            rv_full = torch.where(mask1[i], rv_full / scale1[i][0], rv_full)

        rv_add = rv_full.clone()

        count = ((height + self.numthreads_min - 1) // self.numthreads_min)
        numThreads = self.numthreads_min
        for n in range(self.numthreads_min, self.numthreads_max + 1):
            if ((height + n - 1) // n) < count:
                numThreads = n
                count = ((height + n - 1) // n)
        xtra = (numThreads - 1) * self.waveShift
        padDown = ((height + numThreads - 1) // numThreads) * numThreads - height
        params = F.pad(params, (xtra, xtra, 0, padDown))
        rv_full = F.pad(rv_full, (xtra, xtra, 0, padDown))

        y_hat = torch.zeros(
            (scales_hat.size(0), self.M, height + 2 * padding + padDown, width + 2 * padding + xtra * 2),
            device = scales_hat.device,
        )

        torch.backends.cudnn.deterministic = False if self.DeterminismSpeedup else True

        _, params_channel_size, _, _ = params.shape
        ctx_p_channel_shape = self.context_prediction.bias.shape
        entropy_input = torch.zeros((1, int(params_channel_size) + int(ctx_p_channel_shape[0]), 1, numThreads),
                                    device = device)

        cout, cin, _, _ = self.context_prediction.weight.shape
        ctxlayer = nn.Conv2d(cin, cout, (self.kernel_height, self.kernel_width), stride = self.kernel_width)
        ws = self.context_prediction.weight[:, :, 0:self.kernel_height,
             0:self.kernel_width] * self.context_prediction.mask[:, :, 0:self.kernel_height, 0:self.kernel_width]
        ctxlayer.weight = nn.Parameter(ws)
        ctxlayer.bias = nn.Parameter(self.context_prediction.bias)
        if self.DoublePrecisionProcessing:
            ctxlayer = ctxlayer.double()
            self.entropy_parameters.double()

        for h in range(0, height + padDown, numThreads):
            for w in range(width + xtra):
                params_crop = params[:, :, h:h + numThreads, w:w + numThreads]
                entropy_input[:, :params_channel_size, :, :] = einops.rearrange(params_crop, 'a b c d ->a b (c d)') \
                                                                   .unsqueeze(2)[:, :, :,
                                                               numThreads - 1:-1:numThreads - 1]

                y_hat_crop = y_hat[:, :, h: h + numThreads + self.kernel_height - 1,
                             w: w + numThreads + self.kernel_width - 1]
                y_hat_crop = F.unfold(y_hat_crop, kernel_size = (self.kernel_height, self.kernel_width))
                y_hat_crop = einops.rearrange(y_hat_crop[:, :, numThreads - 1:-(numThreads - 1):numThreads - 1] \
                                              , 'a (C r e) d->a C r (d e)', C = self.N, e = self.kernel_width,
                                              r = self.kernel_height)

                if self.DoublePrecisionProcessing:
                    y_hat_crop = y_hat_crop.double()
                    entropy_input = entropy_input.double()
                entropy_input[:, params_channel_size:, :, :] = ctxlayer(y_hat_crop)

                means_hat = self.entropy_parameters(entropy_input)
                means_hat = means_hat.float()

                rv_crop = rv_full[:, :, h:h + numThreads, w:w + numThreads]
                rv_crop = einops.rearrange(rv_crop, 'a b c d ->a b (c d)') \
                              .unsqueeze(2)[:, :, :, numThreads - 1:-1:numThreads - 1]
                y_hat2 = rv_crop + means_hat

                wmin = max(w + padding,
                           padding + xtra)  # y_hat is padded padding + xtra on the left and padding + xtra on the right. these padded areas must not be modified.
                wdiff = max(0, wmin - w - padding)
                wmax = min(w + padding + numThreads, width + padding + xtra)
                wdiff2 = max(w + padding + numThreads - wmax, 0)

                y_hat_crop = y_hat[:, :, h + padding: h + padding + numThreads, w + padding: w + padding + numThreads] \
                    .reshape(1, self.N, 1, -1)
                y_hat_crop[:, :, :, numThreads - 1:-1:numThreads - 1] = y_hat2
                y_hat[:, :, h + padding: h + padding + numThreads,
                w + padding + wdiff: w + padding + numThreads - wdiff2] \
                    = y_hat_crop.view(1, self.N, numThreads, numThreads)[:, :, :, wdiff:numThreads - wdiff2]
        y_hat = F.pad(y_hat, (-padding - xtra, -padding - xtra, -padding, -padding - padDown))
        self.entropy_parameters.float()
        # start of channel offsetting
        if self.channelOffsetsTool:
            wpad = ((width + self.offsetSplit_w - 1) // self.offsetSplit_w) * self.offsetSplit_w - width
            hpad = ((height + self.offsetSplit_h - 1) // self.offsetSplit_h) * self.offsetSplit_h - height
            kernel_w = (width + wpad) // self.offsetSplit_w
            kernel_h = (height + hpad) // self.offsetSplit_h
            self.decChannelOffsets = self.decChannelOffsets.to(device).unsqueeze(0)
            self.decChannelOffsets = einops.repeat(self.decChannelOffsets, 'a b c d -> a b (c repeat1) (d repeat2)',
                                                   repeat1 = kernel_h, repeat2 = kernel_w)
            self.decChannelOffsets = F.pad(self.decChannelOffsets, (0, -wpad, 0, -hpad))
            y_hat += self.decChannelOffsets
        # End of channel offsetting

        if MeanAndResidualScale:
            means_full = y_hat - rv_add
            for i in range(self.numfilters[2]):
                mask, scale = self.get_mask(scales_hat, dict(self.filterCoeffs3[i]))
                intermediate = y_hat + (scale[0] * means_full + scale[1] * rv_add)
                y_hat = torch.where(mask, intermediate, y_hat)
        return y_hat
    
    def _compress_ar_scale(self, y_hat, params, height, width, padding, scales_hat, forward = False):
        if ~forward:
            cdf = self.gaussian_conditional.quantized_cdf.tolist()
            cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
            offsets = self.gaussian_conditional.offset.tolist()

            encoder = BufferedRansEncoder()

        count = ((height + self.numthreads_min - 1) // self.numthreads_min)
        numThreads = self.numthreads_min
        for n in range(self.numthreads_min, self.numthreads_max + 1):
            if ((height + n - 1) // n) < count:
                numThreads = n
                count = ((height + n - 1) // n)

        threadNum = range(numThreads)
        xtra = (numThreads - 1) * self.waveShift
        padDown = ((height + numThreads - 1) // numThreads) * numThreads - height
        y_q_full = torch.zeros_like(scales_hat)
        params = F.pad(params, (xtra, xtra, 0, padDown))
        y_org = y_hat.clone()
        y_hat = F.pad(y_hat, (padding + xtra, padding + xtra, padding, padding + padDown))

        batch_size, params_channel_size, _, _ = params.shape
        ctx_p_channel_shape = self.context_prediction.bias.shape
        entropy_input = torch.zeros((batch_size, int(params_channel_size) + int(ctx_p_channel_shape[0]), 1, numThreads),
                                    device = params.device)
        mask1 = []
        scale1 = []
        for i in range(self.numfilters[0]):
            mask, scale = self.get_mask(scales_hat, dict(self.filterCoeffs1[i]))
            mask = F.pad(mask, (padding + xtra, padding + xtra, padding, padding + padDown))
            mask1.append(mask)
            scale1.append(torch.tensor(scale[0]))
        torch.backends.cudnn.deterministic = False if self.DeterminismSpeedup else True

        cout, cin, _, _ = self.context_prediction.weight.shape
        ctxlayer = nn.Conv2d(cin, cout, (self.kernel_height, self.kernel_width), stride = self.kernel_width)
        ws = self.context_prediction.weight[:, :, 0:self.kernel_height, 0:self.kernel_width] * \
             self.context_prediction.mask[:, :, 0:self.kernel_height, 0:self.kernel_width]
        ctxlayer.weight = nn.Parameter(ws)
        ctxlayer.bias = nn.Parameter(self.context_prediction.bias)
        if self.DoublePrecisionProcessing:
            ctxlayer = ctxlayer.double()
            self.entropy_parameters.double()

        for h in range(0, height + padDown, numThreads):
            for w in range(width + xtra):

                params_crop = params[:, :, h:h + numThreads, w:w + numThreads]
                entropy_input[:, :params_channel_size, :, :] = einops.rearrange(params_crop,
                                                                                'a b c d ->a b (c d)').unsqueeze(2)[:,
                                                               :, :, numThreads - 1:-1:numThreads - 1]

                y_hat_crop = y_hat[:, :, h: h + numThreads + self.kernel_height - 1,
                             w: w + numThreads + self.kernel_width - 1].clone()
                if h > 0:
                    y_hat_crop[:, :, :2, :] = 0
                y_hat_crop = F.unfold(y_hat_crop, kernel_size = (self.kernel_height, self.kernel_width))
                y_hat_crop = einops.rearrange(y_hat_crop[:, :, numThreads - 1:-(numThreads - 1):numThreads - 1],
                                              'a (C r e) d->a C r (d e)', C = self.N, e = self.kernel_width,
                                              r = self.kernel_height)

                if self.DoublePrecisionProcessing:
                    y_hat_crop = y_hat_crop.double()
                    entropy_input = entropy_input.double()
                entropy_input[:, params_channel_size:, :, :] = ctxlayer(y_hat_crop)

                means_hat = self.entropy_parameters(entropy_input)
                means_hat = means_hat.float()

                y_crop = y_hat[:, :, h + padding: h + numThreads + padding, w + padding: w + numThreads + padding]
                y_crop = einops.rearrange(y_crop, 'a b c d ->a b (c d)').unsqueeze(2)[:, :, :,
                         numThreads - 1:-1:numThreads - 1]
                y_q = (y_crop - means_hat)
                mask1_crop = []
                for i in range(self.numfilters[0]):
                    mask1_crop.append(
                        mask1[i][:, :, h + padding: h + numThreads + padding, w + padding: w + numThreads + padding])
                    mask1_crop[i] = einops.rearrange(mask1_crop[i], 'a b c d ->a b (c d)').unsqueeze(2)[:, :, :,
                                    numThreads - 1:-1:numThreads - 1]
                    y_q = torch.where(mask1_crop[i], y_q * scale1[i], y_q)

                y_q = y_q.round()
                for thr in threadNum:
                    if w < (width + thr * self.waveShift) and w >= (thr * self.waveShift):
                        y_q_full[:, :, h + thr: h + 1 + thr,
                        w - thr * self.waveShift: w - thr * self.waveShift + 1] = y_q[:, :, :, thr:thr + 1]

                for i in range(self.numfilters[0] - 1, -1, -1):
                    y_q = torch.where(mask1_crop[i], y_q / scale1[i], y_q)

                wmin = max(w + padding, padding + xtra)
                wdiff = max(0, wmin - w - padding)
                wmax = min(w + padding + numThreads, width + padding + xtra)
                wdiff2 = max(w + padding + numThreads - wmax, 0)

                y_hat_crop = y_hat[:, :, h + padding: h + padding + numThreads,
                             w + padding: w + padding + numThreads].reshape(batch_size, self.N, 1, -1)
                y_hat_crop[:, :, :, numThreads - 1:-1:numThreads - 1] = means_hat + y_q
                y_hat[:, :, h + padding: h + padding + numThreads,
                w + padding + wdiff: w + padding + numThreads - wdiff2] = y_hat_crop.view(batch_size, self.N, numThreads,
                                                                                          numThreads)[:, :, :, wdiff:numThreads - wdiff2]
        y_hat = F.pad(y_hat, (-padding - xtra, - padding - xtra, -padding, -padding - padDown))
        if forward:
            return y_hat

        self.y_hat = y_hat.clone()
        # Start of channel offsetting
        if self.channelOffsetsTool:
            wpad = ((width + self.offsetSplit_w - 1) // self.offsetSplit_w) * self.offsetSplit_w - width
            hpad = ((height + self.offsetSplit_h - 1) // self.offsetSplit_h) * self.offsetSplit_h - height
            kernel_w = (width + wpad) // self.offsetSplit_w
            kernel_h = (height + hpad) // self.offsetSplit_h
            diff = F.pad(y_org - y_hat, (0, wpad, 0, hpad))
            avg = torch.nn.AvgPool2d(kernel_size = (kernel_h, kernel_w), stride = (kernel_h, kernel_w), padding = 0)
            self.encChannelOffsets = avg(diff).squeeze()
            if self.offsetSplit_h == 1:
                self.encChannelOffsets.unsqueeze_(1)
            if self.offsetSplit_w == 1:
                self.encChannelOffsets.unsqueeze_(2)
            self.encChannelOffsets = self.encChannelOffsets.permute(1, 2, 0)
        # End of channel offsetting
        torch.backends.cudnn.deterministic = True

        mask2 = torch.ones_like(scales_hat, dtype = torch.bool)
        for i in range(self.numfilters[1]):
            mask, _ = self.get_mask(scales_hat, dict(self.filterCoeffs2[i]))
            mask2 = mask2 * mask
        mask1 = []
        scale1 = []
        for i in range(self.numfilters[0]):
            mask, scale = self.get_mask(scales_hat, dict(self.filterCoeffs1[i]))
            mask1.append(mask)
            scale1.append(scale)

        for i in range(self.numfilters[0]):
            if self.Quantized:
                scales_hat = scales_hat.double()
                scales_hat[mask1[i]] *= self.h_s_scale.h_s_scale[-1].scale_v
                scales_hat[mask1[i]].round().clamp(-(2 ** (self.h_s_scale.h_s_scale[-1].abit - 1)),
                                                2 ** (self.h_s_scale.h_s_scale[-1].abit - 1) - 1)
                scalesfactor = round(scale1[i][0] * 2 ** self.sbit)
                scales_hat[mask1[i]] *= scalesfactor
                scales_hat[mask1[i]] /= (self.h_s_scale.h_s_scale[-1].scale_v * (2 ** self.sbit))
                scales_hat = scales_hat.float()
            else:
                scales_hat[mask1[i]] *= scale1[i][0]
        self.y_q_full = y_q_full.clone()

        y_q_full = y_q_full[mask2]
        scales_hat = scales_hat[mask2]

        symbols_list = y_q_full.to('cpu').numpy()
        indexes_full = self.gaussian_conditional.build_indexes(scales_hat)
        indexes_list = indexes_full.to('cpu').numpy()

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        string = encoder.flush()
        return string

    def _decompress_ar_scale(self, y_string, params, height, width, padding, scales_hat, device):
        count = ((height + self.numthreads_min - 1) // self.numthreads_min)
        numThreads = self.numthreads_min
        for n in range(self.numthreads_min, self.numthreads_max + 1):
            if ((height + n - 1) // n) < count:
                numThreads = n
                count = ((height + n - 1) // n)
        xtra = (numThreads - 1) * self.waveShift
        padDown = ((height + numThreads - 1) // numThreads) * numThreads - height
        numTiles = (height + padDown) // numThreads

        y_hat = torch.zeros(
            (scales_hat.size(0), self.M, height + 2 * padding + padDown + (numTiles - 1) * (self.kernel_height - 1),
             width + 2 * (padding + xtra)),
            device = scales_hat.device,
        )

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_string)

        mask2 = torch.ones_like(scales_hat, dtype = torch.bool)
        for i in range(self.numfilters[1]):
            mask, _ = self.get_mask(scales_hat, dict(self.filterCoeffs2[i]))
            mask2 = mask2 * mask
        mask1 = []
        scale1 = []
        for i in range(self.numfilters[0]):
            mask, scale = self.get_mask(scales_hat, dict(self.filterCoeffs1[i]))
            mask1.append(mask)
            scale1.append(scale)
        for i in range(self.numfilters[0]):
            # Quantized scale operation
            if self.Quantized:
                scales_hat = scales_hat.double()
                scales_hat[mask1[i]] *= self.h_s_scale.h_s_scale[-1].scale_v
                scales_hat[mask1[i]].round().clamp(-(2 ** (self.h_s_scale.h_s_scale[-1].abit - 1)),
                                                2 ** (self.h_s_scale.h_s_scale[-1].abit - 1) - 1)
                scalesfactor = round(scale1[i][0] * 2 ** self.sbit)
                scales_hat[mask1[i]] *= scalesfactor
                scales_hat[mask1[i]] /= (self.h_s_scale.h_s_scale[-1].scale_v * (2 ** self.sbit))
                scales_hat = scales_hat.float()
            else:
                scales_hat[mask1[i]] *= scale1[i][0]

        indexes_full = self.gaussian_conditional.build_indexes(scales_hat[mask2])
        indexes = indexes_full.to('cpu').numpy()

        rv = decoder.decode_stream(
            indexes, cdf, cdf_lengths, offsets
        )
        rv = torch.Tensor(rv)
        rv_full = torch.zeros_like(scales_hat)
        rv_full[mask2] = rv.to(device)

        for i in range(self.numfilters[0] - 1, -1, -1):
            rv_full = torch.where(mask1[i], rv_full / scale1[i][0], rv_full)

        rv_add = rv_full.clone()

        params = F.pad(params, (xtra, xtra, 0, padDown + (numTiles - 1) * (self.kernel_height - 1)))
        rv_full = F.pad(rv_full, (xtra, xtra, 0, padDown + (numTiles - 1) * (self.kernel_height - 1)))

        torch.backends.cudnn.deterministic = False if self.DeterminismSpeedup else True

        _, params_channel_size, _, _ = params.shape
        ctx_p_channel_shape = self.context_prediction.bias.shape
        entropy_input = torch.zeros((1, int(params_channel_size) + int(ctx_p_channel_shape[0]),
                                     numThreads * numTiles + (numTiles - 1) * (self.kernel_height - 1), 1),
                                    device = device)

        cout, cin, _, _ = self.context_prediction.weight.shape
        ctxlayer = nn.Conv2d(cin, cout, (self.kernel_height, self.kernel_width), stride = (1, self.kernel_width))
        ws = self.context_prediction.weight[:, :, 0:self.kernel_height,
             0:self.kernel_width] * self.context_prediction.mask[:, :, 0:self.kernel_height, 0:self.kernel_width]
        ws_new = torch.zeros_like(ws)

        for i in range(3):
            ws_new[:, :, i, i:] = ws[:, :, i, :self.kernel_width - i]
        ctxlayer.weight = nn.Parameter(ws_new)
        ctxlayer.bias = nn.Parameter(self.context_prediction.bias)
        if self.DoublePrecisionProcessing:
            ctxlayer = ctxlayer.double()
            self.entropy_parameters.double()

        rv_new = torch.zeros_like(rv_full)
        params_new = torch.zeros_like(params)
        _, _, _, params_crop_with = params.shape

        # 1. prepare the data
        for index, h in enumerate(range(0, height + padDown, numThreads)):
            for j in range(numThreads):
                h_index = h + j
                rv_new[:, :, h_index + 2 * index, 0:params_crop_with - numThreads + 1] = rv_full[:, :, h_index,
                                                                                         numThreads - 1 - j:params_crop_with - j]
                params_new[:, :, h_index + 2 * index, 0:params_crop_with - numThreads + 1] = params[:, :, h_index,
                                                                                             numThreads - 1 - j:params_crop_with - j]

        # 2. prediction with regular neural layers
        for w in range(width + xtra):
            y_hat_crop = y_hat[:, :, :numTiles * numThreads + numTiles * (self.kernel_height - 1),
                         w: w + self.kernel_width]
            entropy_input[:, :params_channel_size, :, :] = params_new[:, :, :, w:w + 1]

            if self.DoublePrecisionProcessing:
                y_hat_crop = y_hat_crop.double()
                entropy_input = entropy_input.double()
            entropy_input[:, params_channel_size:, :, :] = ctxlayer(y_hat_crop)
            means_hat = self.entropy_parameters(entropy_input)
            if self.DoublePrecisionProcessing:
                means_hat = means_hat.float()

            y_hat2 = rv_new[:, :, :, w:w + 1] + means_hat

            wmin = max(w + padding, padding + xtra)
            wdiff = max(0, wmin - w - padding)
            wmax = min(w + padding + numThreads, width + padding + xtra)
            wdiff2 = max(w + padding + numThreads - wmax, 0)
            for index, h in enumerate(range(0, height + padDown, numThreads)):
                y_hat[:, :, wdiff2 + h + 2 * (index + 1): h + numThreads + 2 * (index + 1) - wdiff,
                w + 4: w + 5] = y_hat2[:, :, 2 * index + h + wdiff2: 2 * index + h + numThreads - wdiff, :]

        # 3. converting the y_hat to initial shape
        for index, h in enumerate(range(0, height + padDown, numThreads)):
            for j in range(numThreads):
                h_index = h + j
                y_hat[:, :, 2 + h_index, 0:-2 * (xtra + padding)] = y_hat[:, :, 2 * (index + 1) + h_index,
                                                                    4 + j: -2 * (xtra + padding) + j + 4]

        y_hat = F.pad(y_hat, (
        0, 2 * (-padding - xtra), -padding, -padding - padDown - (numTiles - 1) * (self.kernel_height - 1)))
        self.entropy_parameters.float()
        # start of channel offsetting
        if self.channelOffsetsTool:
            wpad = ((width + self.offsetSplit_w - 1) // self.offsetSplit_w) * self.offsetSplit_w - width
            hpad = ((height + self.offsetSplit_h - 1) // self.offsetSplit_h) * self.offsetSplit_h - height
            kernel_w = (width + wpad) // self.offsetSplit_w
            kernel_h = (height + hpad) // self.offsetSplit_h
            self.decChannelOffsets = self.decChannelOffsets.to(device).unsqueeze(0)
            self.decChannelOffsets = einops.repeat(self.decChannelOffsets, 'a b c d -> a b (c repeat1) (d repeat2)',
                                                   repeat1 = kernel_h, repeat2 = kernel_w)
            self.decChannelOffsets = F.pad(self.decChannelOffsets, (0, -wpad, 0, -hpad))
            y_hat += self.decChannelOffsets
        # End of channel offsetting

        if MeanAndResidualScale:
            means_full = y_hat - rv_add
            self.decParameters = [y_hat, rv_add, scales_hat]
            for i in range(self.numfilters[2]):
                mask, scale = self.get_mask(scales_hat, dict(self.filterCoeffs3[i]))
                intermediate = y_hat + (scale[0] * means_full + scale[1] * rv_add)
                y_hat = torch.where(mask, intermediate, y_hat)
        return y_hat

    def compress(self, x, h, w, quality, numIte=0, device = "cuda"):
        if self.encCompleteSkip:
            return self.encOutput
        else:
            self.likely = None  # initialize the tensor TODO: implement a less error prone initialization.
            device = device if torch.cuda.is_available() else "cpu"
            x = x.to(device)
            m = [self.g_a, self.h_a, self.entropy_bottleneck, self.h_s, self.h_s_scale,
                 self.g_s, self.entropy_parameters, self.context_prediction, self.gaussian_conditional,
                 self.g_s_extension]

            for mm in m:
                mm = mm.to(device)

            if self.encParams == None or not self.encSkip:
                torch.backends.cudnn.deterministic = False if self.DeterminismSpeedup else True
                RGB2YUV = RGB2YCbCr()
                yuvd = RGB2YUV(x)
                y = self.splitFunc(yuvd, self.g_a, 4, 64, 4)
                z = self.h_a(y)

                ########TODO: Copy a model to GPU and run the online optimization there.
                # TODO: big pictures cannot be handled due to memory.
                # TODO: modify the loss function according to final training loss.
                # TODO use new quantizer in offline training.

                x_true_org = crop(yuvd, [h, w])
                x_true_org = x_true_org.to(device)
                if x.shape[2] * x.shape[3] < 9000000:
                    y, z = self.y_refiner(y.clone(), z, x_true_org, quality, numIte)

                ########
                torch.backends.cudnn.deterministic = True
                z_strings = self.entropy_bottleneck.compress(z)
                z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

                params = self.h_s(z_hat)
                scales_hat = self.h_s_scale(z_hat)
                torch.backends.cudnn.deterministic = False if self.DeterminismSpeedup else True
                s = 4  # scaling factor between z and y
                kernel_size = 5  # context prediction kernel size
                padding = (kernel_size - 1) // 2

                y_height = z_hat.size(2) * s
                y_width = z_hat.size(3) * s
                y_hat = y.clone()
                encParams = [z, z_strings, y, y_hat[0: 0 + 1], params[0: 0 + 1], y_height, y_width, kernel_size,
                             padding, scales_hat]
                self.encParams = encParams
            else:
                encParams = self.encParams
            z, z_strings, y, y_hat, params, y_height, y_width, kernel_size, padding, scales_hat = encParams
            y_strings = []

            for i in range(y.size(0)):
                if self.oldversion:
                    string = self._compress_ar_scale_old(
                        y_hat[i: i + 1],
                        params[i: i + 1],
                        y_height,
                        y_width,
                        padding,
                        scales_hat
                    )
                else:
                    string = self._compress_ar_scale(
                        y_hat[i: i + 1],
                        params[i: i + 1],
                        y_height,
                        y_width,
                        padding,
                        scales_hat
                    )
                y_strings.append(string)
            self.encOutput = {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
            return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, device):
        assert isinstance(strings, list) and len(strings) == 2
        torch.backends.cudnn.deterministic = True
        self.likely = None
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = z_hat.to(device)
        m = [self.entropy_bottleneck, self.h_s, self.h_s_scale,
             self.gaussian_conditional, self.entropy_parameters, self.context_prediction, self.g_s,self.g_s_extension]
        for mm in m:
            mm = mm.to(device)
        params = self.h_s(z_hat)
        scales_hat = self.h_s_scale(z_hat)
        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        for i, y_string in enumerate(strings[0]):
            if self.oldversion:
                y_hat = self._decompress_ar_scale_old(
                                y_string,
                                params[i: i + 1],
                                y_height,
                                y_width,
                                padding,
                                scales_hat,
                                device)
            else:
                y_hat = self._decompress_ar_scale(
                                y_string,
                                params[i: i + 1],
                                y_height,
                                y_width,
                                padding,
                                scales_hat,
                                device)
        del z_hat, y_string, params, scales_hat
        torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = False if self.DeterminismSpeedup else True
        if device == "cuda":
            self.g_s.half()
            self.g_s_extension.half()
            y_hat = y_hat.half()
        x_hat = self.splitFunc(y_hat, self.g_s,self.decSplit1,4,16)
        x_hat = self.splitFunc(x_hat,self.g_s_extension,self.decSplit2,4,16).clamp_(0, 1)
        if device == "cuda":
            x_hat = x_hat.to(torch.float32)
            self.g_s.to(torch.float32)
            self.g_s_extension.to(torch.float32)
        torch.backends.cudnn.deterministic = True
        x_hat = self.yuv2rgb(x_hat).clip(0, 1)
        return {"x_hat": x_hat, "y_hat": y_hat}

    def _compress_ar2(self, y_hat, params, height, width, kernel_size, padding):
        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        # masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = self.context_prediction(y_crop)

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                ctx_p = ctx_p[:, :, 2:3, 2:3]
                means_hat = self.entropy_parameters(torch.cat((p, ctx_p), dim = 1)).squeeze(3).squeeze(2)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat


class YUV444DecoupledV0(QuantYUV444Decoupled):
    def __init__(self, N=192, M=192, refine=4, **kwargs):
        super().__init__(N = N, M = N, refine = refine, **kwargs)
        device = 'cpu' if kwargs.get("device") == None else kwargs.get("device")
        self.Quantized = False
        self.h_s_scale = nn.Sequential(
            deconv(N, M, stride = 2, kernel_size = 5, device = device),
            nn.LeakyReLU(inplace = True),
            deconv(M, M * 3 // 2, stride = 2, kernel_size = 5, device = device),
            nn.LeakyReLU(inplace = True),
            conv(M * 3 // 2, M * 2, stride = 1, kernel_size = 3, device = device),
            nn.LeakyReLU(inplace = True),
            conv(M * 2, M * 5 // 3, stride = 1, kernel_size = 1, device = device),
            nn.LeakyReLU(inplace = True),
            conv(M * 5 // 3, M * 4 // 3, stride = 1, kernel_size = 1, device = device),
            nn.LeakyReLU(inplace = True),
            conv(M * 4 // 3, M * 3 // 3, stride = 1, kernel_size = 1, device = device),
        )