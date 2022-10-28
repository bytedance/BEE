import torch.nn as nn
import torch
import torch.nn.functional as F
import einops
from Common.models.entropy_models import GaussianConditional
from Common.models.quantmodule import quantHSS
from Common.models.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    subpel_conv3x3,
    conv, 
    deconv, 
)


MeanAndResidualScale = True
class BEE_InverseTrans(nn.Module):
    def __init__(self, N=192, M=192, refine=4, init_weights=True, Quantized=False, oldversion=False, **kwargs):
        super().__init__()
        device = 'cpu' if kwargs.get("device") == None else kwargs.get("device")
        self.N = int(N)
        self.M = int(M)
        self.filterCoeffs1 = []
        self.filterCoeffs2 = []
        self.filterCoeffs3 = []
        self.numfilters = [0, 0, 0]
        self.channelOffsetsTool = False
        self.decParameters = None
        self.offsetSplit_w = 0
        self.offsetSplit_h = 0
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
        self.gaussian_conditional = GaussianConditional(None)
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

    def get_decode_variables(self,scale_hat,filterList):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        mask2 = torch.ones_like(scale_hat, dtype = torch.bool)
        for i in range(filterList[1]):
            mask, _ = self.get_mask(scale_hat, dict(self.filterCoeffs2[i]))
            mask2 = mask2 * mask
        mask1 = []
        scale1 = []
        for i in range(filterList[0]):
            mask, scale = self.get_mask(scale_hat, dict(self.filterCoeffs1[i]))
            mask1.append(mask)
            scale1.append(scale)
        for i in range(filterList[0]):
            # Quantized scale operation
            if self.Quantized:
                scales_hat = scale_hat.double()
                scales_hat[mask1[i]] *= self.h_s_scale.h_s_scale[-1].scale_v
                scales_hat[mask1[i]].round().clamp(-(2 ** (self.h_s_scale.h_s_scale[-1].abit - 1)),
                                                2 ** (self.h_s_scale.h_s_scale[-1].abit - 1) - 1)
                scalesfactor = round(scale1[i][0] * 2 ** self.sbit)
                scales_hat[mask1[i]] *= scalesfactor
                scales_hat[mask1[i]] /= (self.h_s_scale.h_s_scale[-1].scale_v * (2 ** self.sbit))
                scales_hat = scales_hat.float()
            else:
                scale_hat[mask1[i]] *= scale1[i][0]

        indexes_full = GaussianConditional(None).build_indexes(scale_hat[mask2])
        indexes = indexes_full.to('cpu').numpy()
        return cdf,cdf_lengths,offsets,indexes,mask1,mask2,scale1

    def LSBS(self, w_hat, quant_latent, scale_hat): #-> Tensor quant_latent
        if MeanAndResidualScale:
            means_full = quant_latent - w_hat
            self.decParameters = [quant_latent, w_hat, scale_hat]
            for i in range(self.numfilters[2]):
                mask, scale = self.get_mask(scale_hat, dict(self.filterCoeffs3[i]))
                intermediate = quant_latent + (scale[0] * means_full + scale[1] * w_hat)
                quant_latent = torch.where(mask, intermediate,quant_latent)
        return quant_latent

    def LDAO(self,quant_latent,height,width,device): #-> Tensor qunat_latent
        if self.channelOffsetsTool:
            wpad = ((width + self.offsetSplit_w - 1) // self.offsetSplit_w) * self.offsetSplit_w - width
            hpad = ((height + self.offsetSplit_h - 1) // self.offsetSplit_h) * self.offsetSplit_h - height
            kernel_w = (width + wpad) // self.offsetSplit_w
            kernel_h = (height + hpad) // self.offsetSplit_h
            self.decChannelOffsets = self.decChannelOffsets.to(device).unsqueeze(0)
            self.decChannelOffsets = einops.repeat(self.decChannelOffsets, 'a b c d -> a b (c repeat1) (d repeat2)',
                                                   repeat1 = kernel_h, repeat2 = kernel_w)
            self.decChannelOffsets = F.pad(self.decChannelOffsets, (0, -wpad, 0, -hpad))
            quant_latent += self.decChannelOffsets
        return quant_latent

    def SynTrans(self,quant_latent): #->Tensor imArray
        f_map = self.DecoderFirstPart(quant_latent,num_first_level_tile=1)
        imArray = self.DecoderSecondPart(f_map,num_second_level_tile=1)
        return imArray

    def DecoderFirstPart(self,quant_latent,num_first_level_tile = 1): #-> Tensor f_map
        f_map = self.splitFunc(quant_latent,self.g_s,num_first_level_tile,4,16)
        return f_map

    def DecoderSecondPart(self,f_map,num_second_level_tile = 1):#-> Tensor imArray
        imArray = self.splitFunc(f_map,self.g_s_extension,num_second_level_tile,4,16).clamp_(0, 1)
        return imArray
    
    def decode(self, quant_latent, header):
        imArray = self.SynTrans(quant_latent)
        return imArray

    def update(self, header):
        self.numfilters = [header.picture.mask_scale.num_adaptive_quant_params,
                           header.picture.mask_scale.num_block_based_skip_params,
                           header.picture.mask_scale.num_latent_post_process_params]
        if self.numfilters[0]:
            self.filterCoeffs1 = header.picture.mask_scale.filterList['filterCoeffs'][0:self.numfilters[0]]
        if self.numfilters[1]:
            self.filterCoeffs2 = header.picture.mask_scale.filterList['filterCoeffs'][self.numfilters[0]:self.numfilters[0]+self.numfilters[1]]
        if self.numfilters[2]:
            self.filterCoeffs3 = header.picture.mask_scale.filterList['filterCoeffs'][self.numfilters[0]+self.numfilters[1]:]
        
        self.channelOffsetsTool = header.picture.adaptive_offset_header.adaptive_offset_enabled_flag
        if self.channelOffsetsTool:
            self.offsetSplit_w = header.picture.adaptive_offset_header.num_horizontal_split
            self.offsetSplit_h = header.picture.adaptive_offset_header.num_vertical_split
        
        if header.picture.adaptive_offset.offset is not None:   
            self.decChannelOffsets = header.picture.adaptive_offset.offset
        else:
            self.decChannelOffsets = []
