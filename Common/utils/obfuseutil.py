import torch
from Common.utils.zoo import models


def model_obfuscator(state_dict=None, removeEncoder = False):
    new_state_dict = state_dict.copy()
    for i, key in enumerate(state_dict.keys()):
        if removeEncoder and ("h_a." in key or "g_a." in key):
            del(new_state_dict[key])
            
    return new_state_dict
        

def DecModelGenerator(ModelFolder, DecModelFolder, model="quantyuv444-decoupled",  metric="mse"):
    from os import listdir
    from os.path import isfile, join
    from pathlib import Path
    onlyfiles = [f for f in listdir(ModelFolder) if isfile(join(ModelFolder, f))]
    for file in onlyfiles: 
        quality = int(file.split("-")[-1])
        net = models[model](quality, metric=metric, decoderOnly=False, device='cpu', Quantized=True)
        state_dict = torch.load(Path(ModelFolder ,file))
        net.load_state_dict(state_dict)
        net.g_s.half()
        net.g_s_extension.half()
        dec_model = model_obfuscator(state_dict=net.state_dict(), removeEncoder=True)
        torch.save(dec_model, Path(DecModelFolder, "dec_model.ckpt-{:02d}".format(quality)))


def model_deObfuscator(state_dict):
    return state_dict


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    modelDir = "ModelDirectory"
    decModelDir = "DecModelDirectory"
    if args:
        modelDir = args[0]
        decModelDir = args[1]
    
    import os
    if not os.path.exists(decModelDir):
        os.mkdir(decModelDir)
    DecModelGenerator(modelDir,decModelDir)