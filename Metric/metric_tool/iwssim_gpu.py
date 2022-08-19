# BSD License

# Copyright (c) 2019, Xinyu Guo
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”). All Bytedance Modifications are Copyright 2022 Bytedance Inc.

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import convolve


class IW_SSIM():
    def __init__(self, iw_flag=True, Nsc=5, blSzX=3, blSzY=3, parent=True,
                 sigma_nsq=0.4, device="cuda", use_double=False, data_range=255.0):
        # MS-SSIM parameters
        self.K = [0.01, 0.03]
        self.L = data_range
        self.weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.winsize = 11
        self.sigma = 1.5

        # IW-SSIM parameters
        self.iw_flag = iw_flag
        self.Nsc = Nsc    # scales
        self.blSzX = blSzX  # Neighbor size
        self.blSzY = blSzY
        self.parent = parent
        self.sigma_nsq = sigma_nsq

        self.bound = np.ceil((self.winsize-1)/2)
        self.bound1 = self.bound - np.floor((self.blSzX-1)/2)
        self.device = device
        self.use_double = use_double

        self.samplet = torch.tensor([1.0]).to(self.device)
        if self.use_double:
            self.samplet = self.samplet.double()
        kernel = np.sqrt(2) * self.binomial_filter(5)
        self.kernel = torch.Tensor(kernel * kernel.T).unsqueeze(0).unsqueeze(0).type(self.samplet.type())
        self.weight = torch.tensor(self.weight).type(self.samplet.type())

    def fspecial(self, fltr, ws, **kwargs):
        if fltr == 'uniform':
            return np.ones((ws, ws)) / ws**2

        elif fltr == 'gaussian':
            x, y = np.mgrid[-ws//2 + 1:ws//2 + 1, -ws//2 + 1:ws//2 + 1]
            g = np.exp(-((x**2 + y**2)/(2.0*kwargs['sigma']**2)))
            g[g < np.finfo(g.dtype).eps*g.max()] = 0
            assert g.shape == (ws, ws)
            den = g.sum()
            if den != 0:
                g /= den
            return g

        return None

    def binomial_filter(self, order_plus_one):
        if order_plus_one < 2:
            raise Exception("Error: order_plus_one argument must be at least 2")

        kernel = np.array([[0.5], [0.5]])
        for i in range(order_plus_one - 2):
            kernel = convolve(np.array([[0.5], [0.5]]), kernel)
        return kernel

    # Torch vision of Laplacian Pyramid, binom5 is utilized as the kernel to
    # upscale and downscale the input image.
    def LaplacianPyramid(self, img, num_layers=5):
        pyr = {}
        # Down sampling
        pad_size = 2
        for i in range(1, num_layers):
            imgsize = img.shape
            tmp = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
            nex_img = F.conv2d(tmp, self.kernel, stride=2)
            # Up Sampling, different padding strategy for odd/even number of pixels
            tmp = F.pad(nex_img, (pad_size // 2, 0, pad_size // 2, 0), mode='reflect')
            tmp = F.pad(tmp, (0, 0, 0, pad_size // 2), mode='reflect' if imgsize[2] % 2 == 1 else 'replicate')
            tmp = F.pad(tmp, (0, pad_size // 2, 0, 0), mode='reflect' if imgsize[3] % 2 == 1 else 'replicate')
            up_img = F.conv_transpose2d(tmp, self.kernel, stride=2)[:, :, 4:4 + imgsize[2], 4:4 + imgsize[3]]
            err = img - up_img
            pyr[i] = err
            img = nex_img
        pyr[num_layers] = img
        return pyr

    def get_pyrd(self, imgo, imgd):
        return self.LaplacianPyramid(imgo), self.LaplacianPyramid(imgd)

    def scale_qualty_maps(self, imgopr, imgdpr):
        ms_win = self.fspecial('gaussian', ws=self.winsize, sigma=self.sigma)
        ms_win = torch.from_numpy(ms_win).unsqueeze(0).unsqueeze(0).type(self.samplet.type())
        C1 = (self.K[0]*self.L)**2
        C2 = (self.K[1]*self.L)**2
        cs_map = {}
        for i in range(1, self.Nsc+1):
            imgo = imgopr[i]
            imgd = imgdpr[i]
            mu1 = F.conv2d(imgo, ms_win)
            mu2 = F.conv2d(imgd, ms_win)
            sigma12 = F.conv2d(imgo*imgd, ms_win) - mu1*mu2
            sigma1_sq = F.conv2d(imgo**2, ms_win) - mu1*mu1
            sigma2_sq = F.conv2d(imgd**2, ms_win) - mu2*mu2
            sigma1_sq = torch.max(torch.zeros(sigma1_sq.shape).type(self.samplet.type()), sigma1_sq)
            sigma2_sq = torch.max(torch.zeros(sigma2_sq.shape).type(self.samplet.type()), sigma2_sq)
            cs_map[i] = (2*sigma12+C2) / (sigma1_sq + sigma2_sq + C2)
            if i == self.Nsc:
                l_map = (2*mu1*mu2+C1) / (mu1**2+mu2**2+C1)
        return l_map, cs_map

    def roll(self, x, shift, dim):
        if dim == 0:
            return torch.cat((x[-shift:, :], x[:-shift, :]), dim)
        else:
            return torch.cat((x[:, -shift:], x[:, :-shift]), dim)

    def roll2(self, x, shift, dim):
        if dim == 1:
            return torch.cat((x[:, -shift:, :], x[:, :-shift, :]), dim)
        else:
            return torch.cat((x[:, :, -shift:], x[:, :, :-shift]), dim)

    def imenlarge2(self, im):
        B, _, M, N = im.shape
        t1 = F.interpolate(im, size=(int(4*M-3), int(4*N-3)), mode='bilinear', align_corners=False)
        t2 = torch.zeros([B, 1, 4*M-1, 4*N-1]).type(self.samplet.type())
        t2[:, :, 1: -1, 1:-1] = t1
        t2[:, :, 0, :] = 2*t2[:, :, 1, :] - t2[:, :, 2, :]
        t2[:, :, -1, :] = 2*t2[:, :, -2, :] - t2[:, :, -3, :]
        t2[:, :, :, 0] = 2*t2[:, :, :, 1] - t2[:, :, :, 2]
        t2[:, :, :, -1] = 2*t2[:, :, :, -2] - t2[:, :, :, -3]
        imu = t2[:, :, ::2, ::2]
        return imu

    def info_content_weight_map(self, imgopr, imgdpr):
        tol = torch.finfo(self.samplet.dtype).eps
        iw_map = {}
        for scale in range(1, self.Nsc):

            imgo = imgopr[scale]
            imgd = imgdpr[scale]
            win = np.ones([self.blSzX, self.blSzY])
            win = win / np.sum(win)
            win = torch.from_numpy(win).unsqueeze(0).unsqueeze(0).type(self.samplet.type())
            padding = int((self.blSzX-1)/2)

            # Prepare for estimating IW-SSIM parameters
            mean_x = F.conv2d(imgo, win, padding=padding)
            mean_y = F.conv2d(imgd, win, padding=padding)
            cov_xy = F.conv2d(imgo*imgd, win, padding=padding) - mean_x*mean_y
            ss_x = F.conv2d(imgo**2, win, padding=padding) - mean_x**2
            ss_y = F.conv2d(imgd**2, win, padding=padding) - mean_y**2

            ss_x = torch.relu(ss_x)
            ss_y = torch.relu(ss_y)

            # Estimate gain factor and error
            g = cov_xy / (ss_x + tol)
            vv = (ss_y - g*cov_xy)

            zerotensor = torch.zeros(g.shape).type(self.samplet.type())
            g = torch.where(ss_x >= tol, g, zerotensor)
            tempss_y = torch.where(ss_x < tol, ss_y, vv)
            vv = torch.where(ss_x < tol, tempss_y, vv)
            ss_x = torch.where(ss_x >= tol, ss_x, zerotensor)
            g = torch.where(ss_y >= tol, g, zerotensor)
            vv = torch.where(vv >= tol, vv, zerotensor)

            # Prepare parent band
            aux = imgo
            _, _, Nsy, Nsx = aux.shape
            prnt = (self.parent and scale < self.Nsc-1)
            BL = torch.zeros([aux.shape[0], 1, aux.shape[2], aux.shape[3], 1+prnt]).to(self.device)
            if self.use_double:
                BL = BL.double()

            BL[:, :, :, :, 0] = aux
            if prnt:
                auxp = imgopr[scale+1]
                auxp = self.imenlarge2(auxp)
                BL[:, :, :, :, 1] = auxp[:, :, 0:Nsy, 0:Nsx]
            imgo = BL
            nbz, _, nv, nh, nb = imgo.shape

            block = torch.tensor([win.shape[2], win.shape[3]]).to(self.device)
            
            # Group neighboring pixels
            nblv = nv-block[0]+1
            nblh = nh-block[1]+1
            nexp = nblv*nblh
            N = torch.prod(block) + prnt
            Ly = torch.div(block[0]-1, 2, rounding_mode='trunc')
            Lx = torch.div(block[1]-1, 2, rounding_mode='trunc')
            Y = torch.zeros([nbz, nexp, N]).type(self.samplet.type())

            n = -1
            for ny in range(-Ly, Ly+1):
                for nx in range(-Lx, Lx+1):
                    n = n + 1
                    temp = imgo[:, 0, :, :, 0]
                    foo1 = self.roll2(temp, ny, 1)
                    foo = self.roll2(foo1, nx, 2)
                    foo = foo[:, Ly: Ly + nblv, Lx: Lx + nblh]
                    Y[:, :, n] = foo.flatten(start_dim=1)
            if prnt:
                n = n + 1
                temp = imgo[:, 0, :, :, 1]
                foo = temp
                foo = foo[:, Ly: Ly+nblv, Lx: Lx+nblh]
                Y[:, :, n] = foo.flatten(start_dim=1)

            C_u = torch.matmul(torch.transpose(Y, 1, 2), Y) / nexp.type(self.samplet.type())
            C_u_inv_list = []
            eig_values_list = []
            #TODO improve the parallel about the mm/eig/diag/inverse1
            for i in range(nbz):
                # Much precision version
                # SingleY = Y[i, :, :]
                # singleC_u = torch.mm(torch.transpose(SingleY, 0, 1), SingleY) / nexp.type(self.samplet.type())
                singleC_u = C_u[i]
                eig_values, H = torch.linalg.eig(singleC_u)
                eig_values = torch.cat([eig_values.real.unsqueeze(1), eig_values.imag.unsqueeze(1)],dim=1)
                eig_values = eig_values.type(self.samplet.type())
                H = H.real.type(self.samplet.type())
                if self.use_double:
                    L = torch.diag(eig_values[:, 0] * (eig_values[:, 0] > 0).double()) * torch.sum(eig_values) / ((torch.sum(eig_values[:,0] * (eig_values[:, 0] > 0).double())) + (torch.sum(eig_values[:, 0] * (eig_values[:, 0] > 0).double())==0))
                else:
                    L = torch.diag(eig_values[:, 0] * (eig_values[:, 0] > 0).float()) * torch.sum(eig_values) /\
                        ((torch.sum(eig_values[:, 0] * (eig_values[:, 0] > 0).float())) +
                         (torch.sum(eig_values[:, 0] * (eig_values[:, 0] > 0).float()) == 0))
                C_u1 = torch.mm(torch.mm(H, L), torch.transpose(H, 0, 1))
                if torch.det(C_u1) == 0:
                    C_u1 = C_u1 + torch.diag(torch.ones(C_u1.shape[-1])).type(self.samplet.type()) * 1e-6
                try:
                    C_u_inv_list.append(torch.inverse(C_u1).unsqueeze(0))
                except:
                    print(C_u1)
                eig_values_list.append(eig_values.unsqueeze(0).unsqueeze(0))
            C_u_inv = torch.cat(C_u_inv_list, dim=0)
            eig_values = torch.cat(eig_values_list, dim=0)
            ss = (torch.matmul(Y, C_u_inv))*Y / N.type(self.samplet.type())
            ss = torch.sum(ss, 2)
            ss = ss.view(nbz, nblv, nblh)
            ss = ss.unsqueeze(1)
            g = g[:, :, Ly: Ly+nblv, Lx: Lx+nblh]
            vv = vv[:, :, Ly: Ly+nblv, Lx: Lx+nblh]

            # Calculate mutual information
            infow = torch.zeros(g.shape).type(self.samplet.type())
            for j in range(eig_values.shape[2]):
                infow = infow + torch.log2(1 + ((vv + (1 + g*g)*self.sigma_nsq)*ss*eig_values[:, :, j:j+1, 0:1]+self.sigma_nsq*vv) / (self.sigma_nsq*self.sigma_nsq))
            infow[infow < tol] = 0
            iw_map[scale] = infow
        return iw_map

    def test(self, imgo, imgd):
        imgopr, imgdpr = self.get_pyrd(imgo, imgd)
        l_map, cs_map = self.scale_qualty_maps(imgopr, imgdpr)
        if self.iw_flag:
            iw_map = self.info_content_weight_map(imgopr, imgdpr)

        wmcs = []
        for s in range(1, self.Nsc+1):
            cs = cs_map[s]
            if s == self.Nsc:
                cs = cs_map[s]*l_map

            if self.iw_flag:
                if s < self.Nsc:
                    iw = iw_map[s]
                    if self.bound1 != 0:
                        iw = iw[:, :, int(self.bound1): -int(self.bound1), int(self.bound1): -int(self.bound1)]
                    else:
                        iw = iw[:, :, int(self.bound1):, int(self.bound1):]
                else:
                    iw = torch.ones(cs.shape).type(self.samplet.type())
                if s == 1:
                    wmcs = torch.sum(cs*iw, (2, 3)) / torch.sum(iw, (2, 3))
                else:
                    wmcs = torch.cat((wmcs, torch.sum(cs*iw, (2, 3)) / torch.sum(iw, (2, 3))), dim=1)
            else:
                raise NotImplemented
        score = torch.prod((torch.abs(wmcs))**(self.weight), dim=1)
        return score


class iwssimgpu(nn.Module):
    def __init__(self, device="cuda", data_range=255.0):
        super().__init__()
        self.eval_net = IW_SSIM(device=device, data_range=data_range)

    def forward(self, imgo, imgd):
        score = self.eval_net.test(imgo, imgd)
        return score
