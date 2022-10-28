import torch.nn as nn

class BasePostprocess(nn.Module):
    def __init__(self):
        super().__init__()
        
    def decode(self, imArray, header, recon_path):
        # return 0 enhance/process reconstruction & save image in recon_path
        raise NotImplementedError
        