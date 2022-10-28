import torch.nn as nn

class BaseBits2Bin(nn.Module):
    def __init__(self):
        super().__init__()
    
    def decode(self, header):
        # output: bits2bin_varlist (Example: [z_hat, w_hat])
        raise NotImplementedError
    
        