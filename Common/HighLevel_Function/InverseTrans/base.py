import torch.nn as nn

class BaseInverseTrans(nn.Module):
    def __init__(self):
        super().__init__()
        
    def decode(self, quant_latent, header):
        # return imArray (example: x_hat)
        raise NotImplementedError
        