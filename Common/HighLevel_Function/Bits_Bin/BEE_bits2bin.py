import torch
import torch.nn as nn
from Common import available_entropy_coders
from Common.models.ans import RansDecoder
from Common.models.entropy_models import GaussianConditional
from Common.models.quantmodule import quantHSS
from Common import set_entropy_coder
from Common.models.layers import (
    conv, 
    deconv, 
)

class BEE_Bits2Bin(nn.Module):
    def __init__(self, N=192, M=192, refine=4, init_weights=True, Quantized=False, oldversion=False, **kwargs):
        super().__init__()
        device = 'cpu' if kwargs.get("device") == None else kwargs.get("device")
        self.Quantized = Quantized
        self.filterCoeffs1 = []
        self.filterCoeffs2 = []
        self.filterCoeffs3 = []
        self.numfilters = [0, 0, 0]
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
            
        set_entropy_coder(available_entropy_coders()[0])

    def bits2z_hat(self,string,shape,device):
        z_hat = self.entropy_bottleneck.decompress(string, shape)
        z_hat = z_hat.to(device)
        return z_hat

    def bits2w_hat(y_string,indexes, cdf, cdf_lengths, offsets,scales_hat):
        decoder = RansDecoder()
        decoder.set_stream(y_string)
        w = decoder.decode_stream(
                indexes, cdf, cdf_lengths, offsets
            )
        w = torch.Tensor(w)
        w_hat = torch.zeros_like(scales_hat)
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
    
    def decode(self,header):
        first_substream_data = header.picture.first_substream_data
        second_substream_data = header.picture.second_substream_data
        z_hat = self.bits2z_hat(first_substream_data,self.header.picture.latent_code_shape_h,self.header.picture.latent_code_shape_w)
        # start: get bits2w_hat variables
        scales_hat = self.h_s_scale(z_hat)
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        mask2 = torch.ones_like(scales_hat, dtype = torch.bool)
        for i in range(self.numfilters[1]):
            mask, _ = self.get_mask(scales_hat, dict(self.filterCoeffs2[i]))
            mask2 = mask2 * mask
        mask1 = []
        scale1 = []
        for i in range(self.numfilters[1]):
            mask, scale = self.get_mask(scales_hat, dict(self.filterCoeffs1[i]))
            mask1.append(mask)
            scale1.append(scale)
        for i in range(self.numfilters[1]):
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
        #end

        w_hat = self.bits2w_hat(second_substream_data,indexes, cdf, cdf_lengths, offsets,scales_hat)
        return [z_hat, w_hat]