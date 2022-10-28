import torch.nn as nn

class BaseBin2Symbol(nn.Module):
    def __init__(self):
        super().__init__()
        
    def decode(self, bits2bin_varlist, header):
        # return quantized latent y_hat (example: y_hat)
        raise NotImplementedError
        