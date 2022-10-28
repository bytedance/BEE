import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
from Common.models.entropy_models import EntropyBottleneck, GaussianConditional
from Common.models.quantmodule import quantHSS
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

from Common.HighLevel_Function.Bits_Bin import Bits2w_hat

class BEE_Bin2Symbol(nn.Module):
    def __init__(self, N=192, M=192, refine=4, init_weights=True, Quantized=False, oldversion=False, **kwargs):
        super().__init__()
        device = 'cpu' if kwargs.get("device") == None else kwargs.get("device")
        self.N = int(N)
        self.M = int(M)
        self.filterCoeffs1 = []
        self.filterCoeffs2 = []
        self.filterCoeffs3 = []
        self.numfilters = [0, 0, 0]
        self.numthreads_min = 50
        self.numthreads_max = 100
        self.waveShift = 1
        self.kernel_height = 3
        self.kernel_width = 4
        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size = 5, padding = 2, stride = 1, device = device
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.entropy_bottleneck = EntropyBottleneck(self.N)
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
        self.h_s = nn.Sequential(
            deconv(N, M, stride = 2, kernel_size = 5, device = device),
            nn.LeakyReLU(inplace = True),
            deconv(M, M * 3 // 2, stride = 2, kernel_size = 5, device = device),
            nn.LeakyReLU(inplace = True),
            conv(M * 3 // 2, M * 2, stride = 1, kernel_size = 3, device = device),
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

    def Models2device(self,device):
        m = [self.entropy_bottleneck, self.h_s, self.h_s_scale,
            self.gaussian_conditional, self.entropy_parameters, self.context_prediction, self.g_s,self.g_s_extension]
        for mm in m:
            mm = mm.to(device)
        return


    def LatentSampleRestruction(self ,w_hat, w, h, scale_hat,padding,params,device):
        numThreads = self.numthreads_min
        ctxlayer = self.ContextModel()
        mu,rv_new,y_hat,xtra,padDown,numTiles = self.PredictionFusion(w_hat,h,w,padding,scale_hat,params,ctxlayer,device)
        y_hat2 = rv_new[:, :, :, w:w + 1] + mu
        wmin = max(w + padding, padding + xtra)
        wdiff = max(0, wmin - w - padding)
        wmax = min(w + padding + numThreads, w + padding + xtra)
        wdiff2 = max(w + padding + numThreads - wmax, 0)
        for index, h in enumerate(range(0, h + padDown, numThreads)):
            y_hat[:, :, wdiff2 + h + 2 * (index + 1): h + numThreads + 2 * (index + 1) - wdiff,
            w + 4: w + 5] = y_hat2[:, :, 2 * index + h + wdiff2: 2 * index + h + numThreads - wdiff, :]
        # converting the y_hat to initial shape
        for index, h in enumerate(range(0, h + padDown, numThreads)):
            for j in range(numThreads):
                h_index = h + j
                y_hat[:, :, 2 + h_index, 0:-2 * (xtra + padding)] = y_hat[:, :, 2 * (index + 1) + h_index,
                                                                    4 + j: -2 * (xtra + padding) + j + 4]

        y_hat = F.pad(y_hat, (
        0, 2 * (-padding - xtra), -padding, -padding - padDown - (numTiles - 1) * (self.kernel_height - 1)))
        self.entropy_parameters.float()
        return y_hat


    def HyperDecoding(self,z_hat):#->FeatureMap1
        FeatureMap1 = self.HyperDecoder(self,z_hat)
        return FeatureMap1


    def HyperScaleDecoding(self,z_hat):#->scale_hat
        scale_hat = self.HyperScaleDecoder(self,z_hat)
        return scale_hat


    def HyperDecoder(self,z_hat):
        fMap = self.h_s(z_hat)
        return fMap

    def HyperScaleDecoder(self,z_hat):
        scale_hat = self.h_s_scale(z_hat)
        return scale_hat


    # def ContextModel(self): #?
    #     cout, cin, _, _ = self.context_prediction.weight.shape
    #     ctxlayer = nn.Conv2d(cin, cout, (self.kernel_height, self.kernel_width), stride = (1, self.kernel_width))
    #     ws = self.context_prediction.weight[:, :, 0:self.kernel_height,
    #             0:self.kernel_width] * self.context_prediction.mask[:, :, 0:self.kernel_height, 0:self.kernel_width]
    #     ws_new = torch.zeros_like(ws)

    #     for i in range(3):
    #         ws_new[:, :, i, i:] = ws[:, :, i, :self.kernel_width - i]
    #     ctxlayer.weight = nn.Parameter(ws_new)
    #     ctxlayer.bias = nn.Parameter(self.context_prediction.bias)
    #     return ctxlayer

    def ContextModel(self,quant_hyper_latent):
        FeatureMap2 = self.context_prediction(quant_hyper_latent)
        return FeatureMap2

        
    # def PredictionFusion(self,rv_full,scales_hat,params,ctxlayer):
    #     count = ((height + self.numthreads_min - 1) // self.numthreads_min)
    #     numThreads = self.numthreads_min
    #     for n in range(self.numthreads_min, self.numthreads_max + 1):
    #         if ((height + n - 1) // n) < count:
    #             numThreads = n
    #             count = ((height + n - 1) // n)
    #     xtra = (numThreads - 1) * self.waveShift
    #     padDown = ((height + numThreads - 1) // numThreads) * numThreads - height
    #     numTiles = (height + padDown) // numThreads  
    #     params = F.pad(params, (xtra, xtra, 0, padDown + (numTiles - 1) * (self.kernel_height - 1)))
    #     rv_full = F.pad(rv_full, (xtra, xtra, 0, padDown + (numTiles - 1) * (self.kernel_height - 1)))
    #     torch.backends.cudnn.deterministic = False if self.DeterminismSpeedup else True
    #     params_new = torch.zeros_like(params)
    #     _, params_channel_size, _, _ = params.shape
    #     ctx_p_channel_shape = self.context_prediction.bias.shape
    #     entropy_input = torch.zeros((1, int(params_channel_size) + int(ctx_p_channel_shape[0]),
    #                                     numThreads * numTiles + (numTiles - 1) * (self.kernel_height - 1), 1),
    #                                 device = device)
    #     _, _, _, params_crop_with = params.shape
    #     rv_new = torch.zeros_like(rv_full)
    #     y_hat = torch.zeros(
    #         (scales_hat.size(0), self.M, height + 2 * padding + padDown + (numTiles - 1) * (self.kernel_height - 1),
    #             width + 2 * (padding + xtra)),
    #         device = scales_hat.device,
    #     )    
    #     for index, h in enumerate(range(0, height + padDown, numThreads)):
    #         for j in range(numThreads):
    #             h_index = h + j
    #             rv_new[:, :, h_index + 2 * index, 0:params_crop_with - numThreads + 1] = rv_full[:, :, h_index,
    #                                                                                         numThreads - 1 - j:params_crop_with - j]
    #             params_new[:, :, h_index + 2 * index, 0:params_crop_with - numThreads + 1] = params[:, :, h_index,
    #                                                                                             numThreads - 1 - j:params_crop_with - j]                                                                                     
    #     for w in range(width + xtra):
    #             y_hat_crop = y_hat[:, :, :numTiles * numThreads + numTiles * (self.kernel_height - 1),
    #                             w: w + self.kernel_width]
    #             entropy_input[:, :params_channel_size, :, :] = params_new[:, :, :, w:w + 1]

    #             if self.DoublePrecisionProcessing:
    #                 y_hat_crop = y_hat_crop.double()
    #                 entropy_input = entropy_input.double()
    #             entropy_input[:, params_channel_size:, :, :] = ctxlayer(y_hat_crop)
    #             means_hat = self.entropy_parameters(entropy_input)
    #             if self.DoublePrecisionProcessing:
    #                 means_hat = means_hat.float()
    #             #y_hat2 = rv_new[:, :, :, w:w + 1] + means_hat
    #     return means_hat,rv_new,y_hat,xtra,padDown,numTiles

    def Prediction_Fusion(self,featureMap1,featureMap2):
        latent_channelNum = 192
        Input = torch.zeros(latent_channelNum*3,1,1)
        for i in range(0,latent_channelNum*2):
            Input[i][0][0] = featureMap1[i][0][0]
        for i in range(0,latent_channelNum*2):
            Input[i+latent_channelNum*2][0][0] = featureMap2[i][0][0]
        mu = self.PredictionFusion(Input)

    def PredictionFusion(self,Input):
        mu = self.entropy_parameters(Input)
        return mu


    def AdaptiveQuant(self,w_hat,scale_hat,filterList):
        mask1 = []
        scale1 = []
        for i in range(filterList[0]):
            mask, scale = self.get_mask(scale_hat, dict(self.filterCoeffs1[i]))
            mask1.append(mask)
            scale1.append(scale)
        for i in range(filterList[0]):
            # Quantized scale operation
            if net.Quantized:
                scales_hat = scale_hat.double()
                scales_hat[mask1[i]] *= net.h_s_scale.h_s_scale[-1].scale_v
                scales_hat[mask1[i]].round().clamp(-(2 ** (net.h_s_scale.h_s_scale[-1].abit - 1)),
                                                2 ** (net.h_s_scale.h_s_scale[-1].abit - 1) - 1)
                scalesfactor = round(scale1[i][0] * 2 ** net.sbit)
                scales_hat[mask1[i]] *= scalesfactor
                scales_hat[mask1[i]] /= (net.h_s_scale.h_s_scale[-1].scale_v * (2 ** net.sbit))
                scales_hat = scales_hat.float()
            else:
                scale_hat[mask1[i]] *= scale1[i][0]
        for i in range(self.numfilters[0] - 1, -1, -1):
            w_hat = torch.where(mask1[i], w_hat / scale1[i][0], w_hat)
        return w_hat
    
    def update(self,header):
        self.numfilters = [header.picture.mask_scale.num_adaptive_quant_params,
                           header.picture.mask_scale.num_block_based_skip_params,
                           header.picture.mask_scale.num_latent_post_process_params]
        if self.numfilters[0]:
            self.filterCoeffs1 = header.picture.mask_scale.filterList['filterCoeffs'][0:self.numfilters[0]]
        if self.numfilters[1]:
            self.filterCoeffs2 = header.picture.mask_scale.filterList['filterCoeffs'][self.numfilters[0]:self.numfilters[0]+self.numfilters[1]]
        if self.numfilters[2]:
            self.filterCoeffs3 = header.picture.mask_scale.filterList['filterCoeffs'][self.numfilters[0]+self.numfilters[1]:]

    def decode(self, varlist, header):
        latent_channelNum = 192
        w = header.picture.latent_code_shape_w
        h = header.picture.latent_code_shape_h
        z_hat = varlist[0]
        w_hat = varlist[1]
        featureMap1 = self.HyperDecoding(z_hat)
        scale_hat = self.HyperScaleDecoding(z_hat)
        w_hat = self.AdaptiveQuant(w_hat,scale_hat,self.numfilters)
        sQL = torch.zeros(latent_channelNum,h+4,w+4)
        quant_latent = torch.zeros(latent_channelNum,h,w)
        for i in range(0,h):
            for j in range(0,w):
                featureMap2 = self.ContextModel(sQL[0:latent_channelNum-1][i:i+3][j:j+3])
                mu = self.PredictionFusion(featureMap2,featureMap1[0:latent_channelNum-1][i][j])
                for c in range(0,latent_channelNum):
                    sQL[c][i+2][j+2] = mu[c][0][0] + w_hat[c][i][j]
        for i in range(0,h):
            for j in range(0,w):
                for c in range(0,latent_channelNum-1):
                    quant_latent[c][i][j] = sQL[c][i+2][j+2]
        return quant_latent
