# MIT License

# Copyright (c) 2017 Christian Cosgrove

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”). All Bytedance Modifications are Copyright 2022 Bytedance Inc.

import torch
import torch.nn as nn
import numpy as np


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self.l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = self.l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad = True)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad = True)
        u.data = self.l2normalize(u.data)
        v.data = self.l2normalize(v.data)
        w_bar = nn.Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class CompareGAN_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, stride=1, padding=1, initstd=0.02, sn=True, paddingmode='symmetric'):
        super(CompareGAN_Conv2d, self).__init__()
        self.sn = sn
        self.padding = padding
        self.net = nn.Sequential(SpectralNorm(nn.Conv2d(cin, cout, kernel_size, stride, 0)),
                                 nn.LeakyReLU(negative_slope = 0.2))

        # Compute padding region
        if paddingmode == 'symmetric':
            Padtop, Padleft, Padright, Padbottom = self.padding, self.padding, self.padding, self.padding
        else:
            Padtop, Padleft, Padright, Padbottom = self.padding, self.padding, self.padding + 1, self.padding + 1
        self.Pad = nn.ZeroPad2d((Padleft, Padright, Padtop, Padbottom))

    def forward(self, input):
        # Pad to implement same padding in torch
        if self.padding:
            input = self.Pad(input)
        return self.net(input)


class CompareGAN_TConv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, stride=1, padding=1, output_padding=0, initstd=0.02, sn=True,
                 paddingmode='symmetric'):
        super(CompareGAN_TConv2d, self).__init__()
        self.sn = sn
        self.net = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding)),
            nn.LeakyReLU(negative_slope = 0.2))

        # Compute padding region
        Padtop, Padleft, Padright, Padbottom = 0, 0, 0, 0
        if paddingmode == 'Asymmetric':
            Padtop, Padleft, Padright, Padbottom = 0, 0, 1, 1

        self.Pad = nn.ZeroPad2d((Padleft, Padright, Padtop, Padbottom))

    def forward(self, input):
        # Pad to implement same padding in torch
        input = self.Pad(input)
        return self.net(input)


class HiFLCdiscriminator(nn.Module):
    def __init__(self, Cy=220, baseChannel=64, model_dir=None):
        super(HiFLCdiscriminator, self).__init__()
        self.upnet = nn.Sequential(CompareGAN_Conv2d(Cy, 12, 3, 1, 1))

        self.discriminator = nn.Sequential(CompareGAN_Conv2d(15, baseChannel, 4, 2, 1, paddingmode='symmetric'),
                                           CompareGAN_Conv2d(baseChannel, baseChannel*2, 4, 2, 1, paddingmode='symmetric'),
                                           CompareGAN_Conv2d(baseChannel*2, baseChannel*4, 4, 2, 1, paddingmode='symmetric'),
                                           CompareGAN_Conv2d(baseChannel*4, baseChannel*8, 4, 1, 1, paddingmode='Asymmetric'),
                                           SpectralNorm(nn.Conv2d(baseChannel*8, 1, 1, 1, 0)))
        self.CrossEntropy = nn.BCELoss()
        if model_dir:
            checkpoint = torch.load(model_dir)
            self.load_state_dict(checkpoint)

    def cal_gloss(self, gpu_x_yuv, gpu_org_yuv, data_range, y_hat):
        with torch.no_grad():
            _, _, h, w = gpu_x_yuv.shape
            H_pad, W_pad = int(np.ceil(h / 64) * 64), int(np.ceil(w / 64) * 64)
            pad_F1 = nn.ConstantPad2d((0, W_pad - w, 0, H_pad - h), 0)
            pad_F2 = nn.ConstantPad2d((0, W_pad - w, 0, H_pad - h), 128)
            pad_x_org = torch.cat([pad_F1(gpu_org_yuv[:, 0:1, :, :]), pad_F2(gpu_org_yuv[:, 1:3, :, :])], dim = 1)
            pad_x = torch.cat([pad_F1(gpu_x_yuv[:, 0:1, :, :]), pad_F2(gpu_x_yuv[:, 1:3, :, :])], dim = 1)
            g_loss_pack, _, _ = self.forward([pad_x_org.div(data_range), pad_x.div(data_range)], y_hat)
            _, _, g_loss = g_loss_pack
        return g_loss

    def forward(self, input, y_hat, gradient_to_generator=True):
        # shape of the x_hat and y x->[2*N, 3, H, W] y->[2*N, C, H/16, W/16] 1:N real input N:-1 fake input
        x, x_hat = input

        latent = self.upnet(y_hat.detach())
        latent = nn.functional.interpolate(latent, scale_factor=16, mode='nearest')
        if not gradient_to_generator:
            x_hat = x_hat.detach()
        image_in = torch.cat([x, x_hat], dim=0)
        condition_in = torch.cat([latent, latent], dim=0)
        discriminator_in = torch.cat([image_in, condition_in], dim=1)
        out_logits = self.discriminator(discriminator_in).view(-1, 1)
        out = torch.sigmoid(out_logits)
        # Calculate loss
        d_real_logits, d_fake_logits = torch.chunk(out, 2, dim=0)
        d_loss_real = self.CrossEntropy(d_real_logits, torch.ones_like(d_real_logits))
        d_loss_fake = self.CrossEntropy(d_fake_logits, torch.zeros_like(d_fake_logits))
        g_loss = self.CrossEntropy(d_fake_logits, torch.ones_like(d_fake_logits))
        loss_pack = [d_loss_real, d_loss_fake, g_loss]
        return loss_pack, d_real_logits, d_fake_logits
