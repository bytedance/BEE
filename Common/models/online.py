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

def yuvPSNR(x, org):
    import Common.utils.csc as func
    x_yuv = func.rgb2ycbcr(x * 255)
    org_yuv = func.rgb2ycbcr(org * 255)
    _, __, h, w = x.shape
    mse = ((x_yuv - org_yuv) ** 2) / (h * w)
    mse = mse.squeeze()
    mse = torch.sum(torch.sum(mse, dim = 1), dim = 1)
    psnrY = 10 * math.log10(255 ** 2 / mse[0])
    psnrU = 10 * math.log10(255 ** 2 / mse[1])
    psnrV = 10 * math.log10(255 ** 2 / mse[2])
    return psnrY, psnrU, psnrV


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, quality=3):
        super().__init__()
        lambdaMSEQualityMap = {1: 0.0018, 2: 0.0035, 3: 0.0067, 4: 0.0130, 5: 0.0250, 6:0.0483, 7: 0.0932, 8: 0.1800}
        # lambdaMSSSIMQualityMap = {1: 2.40, 2: 4.58, 3: 8.73, 4: 16.64, 5: 31.73, 6: 60.50, 7: 115.37, 8: 220.00}
        # lambdaVIFQualityMap = {1: 0.150654756, 2: 0.240743544, 3: 0.390877797, 4: 0.596304349, 5: 0, 6: 1.762218837}
        # lambdaFSIMQualityMap = {1: 7.142105295, 2: 16.8133195, 3: 40.45175483, 4: 88.78172771, 5: 0, 6: 535.2276656}
        self.lmbda = lambdaMSEQualityMap[quality]
        # self.lmbda_msssim = lambdaMSSSIMQualityMap[quality]
        # self.lmbda_vif = lambdaVIFQualityMap[quality]
        # self.lmbda_fsim = lambdaFSIMQualityMap[quality]
        # self.colorswith = RGB2YCbCr()
        # self.ms_ssim = ms_ssim
        self.mse = nn.MSELoss()
        # self.fsim = FSIM(channels=3).to('cuda')
        # self.vif = VIFs(channels=1).to('cuda')

    def forward(self, output, target):
        out = {}

        if output["x_hat"] is not None and target is not None:
            recon_yuv = output["x_hat"]
            ori_yuv = target
            #out["distortion_vif"] = torch.mean(1 - self.vif(recon_y, ori_y, as_loss=False))*self.lmbda_vif
            #out["distortion_msssim"] = torch.mean(1 - self.ms_ssim(recon_yuv[:,0:1,:,:]*255, ori_yuv[:,0:1,:,:]*255, data_range=255))*self.lmbda_msssim
            #out["distortion_msssim2"] = torch.mean(1 - self.ms_ssim(recon_yuv[:,1:3,:,:]*255, ori_yuv[:,1:3,:,:]*255, data_range=255))*self.lmbda_msssim
            #out["distortion_fsim"] = torch.mean(1 - self.fsim(recon_y, ori_y, as_loss=False))*self.lmbda_fsim
            out["mse_loss"] = self.mse(recon_yuv[:, 1:3, :, :], ori_yuv[:, 1:3, :, :]) * 255 ** 2 * self.lmbda
            out["mse_loss2"] = self.mse(recon_yuv[:, 0:1, :, :], ori_yuv[:, 0:1, :, :]) * 255 ** 2 * self.lmbda
            out["loss"] = out["mse_loss"]/6 + out["mse_loss2"]/3
        return out


class latentUpdater(nn.Module):
    def __init__(self, y):
        super(latentUpdater, self).__init__()
        self.y_opt = torch.nn.parameter.Parameter(y)
        self.y_opt.requires_grad = True

    def forward(self):
        return self.y_opt

    def update(self, y):
        self.y_opt = torch.nn.parameter.Parameter(y)
        self.y_opt.requires_grad = True
