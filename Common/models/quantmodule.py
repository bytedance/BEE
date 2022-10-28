import torch.nn as nn
import torch
from Common.models.layers import conv, deconv

__all__ = [
    "QuantPrototype",
    "QuantConv",
    "QuantTransposeConv",
    "QuantLeakyReLU",
    "quantHSS"
]

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class QuantPrototype(nn.Module):
    def __init__(self, use_float=False,device='cpu'):
        super().__init__()
        self.layers = None
        self.register_buffer('convertabile', torch.ones(1,device=device).type(torch.bool))
        self.register_buffer('value_max', torch.zeros(1,device=device))
        self.register_buffer('value_min', torch.zeros(1,device=device))
        self.register_buffer('scale_v', torch.ones(1,device=device))
        self.register_buffer('use_float',
                             torch.ones(1,device=device).type(torch.bool) if use_float else torch.zeros(1,device=device).type(torch.bool))

    def load_weights(self, premodule):
        raise NotImplemented

    def calfloatbits(self, value, bitdepth):
        intv = torch.ceil(value).int() + 1
        intbits = torch.ceil(torch.log2(intv)).int()
        floatbits = bitdepth - intbits
        return floatbits - 1 - 6 if self.use_float else floatbits - 1

    def chk_cvtb(self):
        if not self.convertabile:
            print('Module has already been quantized')
            raise SystemError
        else:
            return

    def quant_weights(self):
        self.chk_cvtb()
        raise NotImplemented

    def forward(self, x):
        raise NotImplemented


class QuantTransposeConv(QuantPrototype):
    def __init__(self, in_ch, out_ch, stride, kernel_size, vbit=16, cbit=16, use_float=False,device='cpu'):
        super().__init__(use_float = use_float,device=device)
        self.layers = deconv(in_ch, out_ch, stride = stride, kernel_size = kernel_size,device=device)
        self.register_buffer('scale_c', torch.ones([1, out_ch, 1, 1],device=device))
        self.register_buffer('bias', torch.zeros([1, out_ch, 1, 1],device=device))
        self.vbit = vbit
        self.cbit = cbit

    def load_weights(self, premodule):
        self.layers.weight = premodule.weight
        self.layers.bias = premodule.bias

    def quant_weights(self):
        self.chk_cvtb()
        device = self.convertabile.device
        self.convertabile = torch.zeros(1).type(torch.bool).to(device)
        max_v = torch.max(torch.abs(self.value_max), torch.abs(self.value_min))
        max_v_bits = self.calfloatbits(max_v, bitdepth = self.vbit)
        self.scale_v = torch.exp2(max_v_bits)
        w_max = torch.max(torch.max(torch.max(self.layers.weight, dim = 0, keepdim = True).values, dim = 2,
                                    keepdim = True).values, dim = 3, keepdim = True).values
        w_min = torch.min(torch.min(torch.min(self.layers.weight, dim = 0, keepdim = True).values, dim = 2,
                                    keepdim = True).values, dim = 3, keepdim = True).values
        max_c = torch.max(torch.cat([torch.abs(w_max), torch.abs(w_min)], dim = 3), dim = 3, keepdim = True).values
        max_c_bits = self.calfloatbits(max_c, bitdepth = self.cbit)
        self.scale_c = torch.exp2(max_c_bits)
        self.bias = self.layers.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.layers.bias = nn.Parameter(torch.zeros_like(self.layers.bias))
        self.layers.weight = nn.Parameter(torch.round(self.layers.weight * self.scale_c))

    def forward(self, x):
        if self.training:
            out = self.layers(x)
            self.value_max = torch.max(torch.cat([torch.max(x).reshape((1)), self.value_max], dim = 0)).reshape((1))
            self.value_min = torch.min(torch.cat([torch.min(x).reshape((1)), self.value_min], dim = 0)).reshape((1))
            return out
        else:
            if not self.use_float:
                self.layers.double()
                self.bias.double()
                x = x.double()
            x = torch.round(x * self.scale_v).clamp(-(2**(self.vbit-1)), 2**(self.vbit-1)-1)
            out = (self.layers(x) / (self.scale_v * self.scale_c) + self.bias).float()
            return out


class QuantConv(QuantTransposeConv):
    def __init__(self, in_ch, out_ch, stride, kernel_size, vbit=16, cbit=16, use_float=False,device='cpu'):
        super().__init__(in_ch, out_ch, stride, kernel_size, vbit = vbit, cbit = cbit, use_float = use_float,device=device)
        self.layers = conv(in_ch, out_ch, stride = stride, kernel_size = kernel_size,device=device)

    def quant_weights(self):
        self.chk_cvtb()
        device = self.convertabile.device
        self.convertabile = torch.zeros(1).type(torch.bool).to(device)
        max_v = torch.max(torch.abs(self.value_max), torch.abs(self.value_min))
        max_v_bits = self.calfloatbits(max_v, bitdepth = self.vbit)
        self.scale_v = torch.exp2(max_v_bits)
        w_max = torch.max(torch.max(torch.max(self.layers.weight, dim = 1, keepdim = True).values, dim = 2,
                                    keepdim = True).values, dim = 3, keepdim = True).values
        w_min = torch.min(torch.min(torch.min(self.layers.weight, dim = 1, keepdim = True).values, dim = 2,
                                    keepdim = True).values, dim = 3, keepdim = True).values
        max_c = torch.max(torch.cat([torch.abs(w_max), torch.abs(w_min)], dim = 3), dim = 3, keepdim = True).values
        max_c_bits = self.calfloatbits(max_c, bitdepth = self.cbit)
        self.scale_c = torch.exp2(max_c_bits)
        self.bias = self.layers.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.layers.bias = nn.Parameter(torch.zeros_like(self.layers.bias))
        self.layers.weight = nn.Parameter(torch.round(self.layers.weight * self.scale_c))
        self.scale_c = self.scale_c.permute(1, 0, 2, 3)


class QuantLeakyReLU(QuantPrototype):
    def __init__(self, abit=16, use_float=False):
        super().__init__(use_float = use_float)
        self.layers = nn.LeakyReLU(inplace = True)
        self.abit = abit

    def load_weights(self, premodule):
        return 0

    def quant_weights(self):
        self.chk_cvtb()
        device = self.convertabile.device
        self.convertabile = torch.zeros(1).type(torch.bool).to(device)
        max_v = torch.max(torch.abs(self.value_max), torch.abs(self.value_min))
        max_v_bits = self.calfloatbits(max_v, bitdepth = self.abit)
        self.scale_v = torch.exp2(max_v_bits)

    def forward(self, x):
        if self.training:
            out = self.layers(x)
            self.value_max = torch.max(torch.cat([torch.max(x).reshape((1)), self.value_max], dim = 0)).reshape((1))
            self.value_min = torch.min(torch.cat([torch.min(x).reshape((1)), self.value_min], dim = 0)).reshape((1))
            return out
        else:
            if not self.use_float:
                x = x.double()
            x = torch.round(x * self.scale_v).clamp(-(2**(self.abit-1)), 2**(self.abit-1)-1)
            out = (torch.round(self.layers(x)) / (self.scale_v)).float()
            return out


class QuantIdentity(QuantPrototype):
    def __init__(self, abit=16, use_float=False):
        super().__init__(use_float = use_float)
        self.layers = nn.LeakyReLU(inplace = True)
        self.abit = abit

    def load_weights(self, premodule):
        return 0

    def quant_weights(self):
        self.chk_cvtb()
        device = self.convertabile.device
        self.convertabile = torch.zeros(1).type(torch.bool).to(device)
        max_v = torch.max(torch.abs(self.value_max), torch.abs(self.value_min))
        max_v_bits = self.calfloatbits(max_v, bitdepth = self.abit)
        self.scale_v = torch.exp2(max_v_bits)

    def forward(self, x):
        if self.training:
            out = self.layers(x)
            self.value_max = torch.max(torch.cat([torch.max(x).reshape((1)), self.value_max], dim = 0)).reshape((1))
            self.value_min = torch.min(torch.cat([torch.min(x).reshape((1)), self.value_min], dim = 0)).reshape((1))
            return out
        else:
            return x


class quantHSS(nn.Module):
    def __init__(self, N, M, vbit=16, cbit=16, abit=16, use_float=False, device = 'cpu'):
        super().__init__()
        self.h_s_scale = nn.Sequential(
            QuantTransposeConv(N, M, stride = 2, kernel_size = 5, vbit = vbit, cbit = cbit, use_float = use_float,device=device),
            QuantLeakyReLU(abit = abit, use_float = use_float),
            QuantTransposeConv(M, M, stride = 2, kernel_size = 5, vbit = vbit, cbit = cbit, use_float = use_float,device=device),
            QuantLeakyReLU(abit = abit, use_float = use_float),
            QuantConv(M, M, stride = 1, kernel_size = 3, vbit = vbit, cbit = cbit, use_float = use_float,device=device),
            QuantLeakyReLU(abit = abit, use_float = use_float),
            QuantConv(M, M, stride = 1, kernel_size = 3, vbit = vbit, cbit = cbit, use_float = use_float,device=device),
            QuantIdentity()
        )

    def loadmodel(self, totalmodel):
        for idx in range(len(totalmodel)):
            self.h_s_scale[idx].load_weights(totalmodel[idx])

    def quantlayers(self):
        for idx in range(len(self.h_s_scale)):
            self.h_s_scale[idx].quant_weights()

    def forward(self, x):
        out = self.h_s_scale(x)
        return out
