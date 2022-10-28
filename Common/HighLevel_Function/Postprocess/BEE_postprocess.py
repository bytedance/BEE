from Common.utils.csc import YCbCr2RGB
from Common.utils.tensorops import crop,resizeTensorFast,resizeTensor
import cv2
import numpy as np
import torch.nn as nn





class BEE_Postprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def YUV2RGB(self, imArray): #-》 imArrayRGB
        imArrayRGB = YCbCr2RGB()(imArray).clip(0,1)
        return imArrayRGB

    def ImageCropping(self, imArray, resized_size):
        imArray = crop(imArray, resized_size)
        return imArray

    def AdaptiveResampling(self, imArray, fast_resize_flag, original_size):
        if fast_resize_flag:
            imArray = resizeTensorFast(imArray, original_size)
        else:
            imArray = resizeTensor(imArray, original_size)
        return imArray
    
    def decode(self, imArray, header, recon_path):
        imArray = self.ImageCropping(imArray, [header.picture.resized_size_h, header.picture.resized_size_w])
        imArray = self.AdaptiveResampling(imArray, header.picture.fast_resize_flag, [header.picture.original_size_h, header.picture.original_size_w])
        imArray = self.YUV2RGB(imArray)
        self.writePngOutput(imArray, header.picture.outputBitDepth, header.picture.outputBitShift, recon_path)
        return 0
        
    def writePngOutput(pytorchIm,bitDepth,bitshift, outFileName):
        output = outFileName
        if output is not None:
            img = pytorchIm.squeeze().permute(1, 2, 0).cpu().numpy()
            img = (img*((1<<bitDepth) - 1)).round()
            img = img.clip(0, (1<<bitDepth)-1)
            img *=  (1<<bitshift)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if bitDepth <= 8:
                cv2.imwrite(output, img.astype(np.uint8))
            else:
                cv2.imwrite(output, img.astype(np.uint16))
        else:
            print("not writing output.")
    
