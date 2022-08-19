# Copyright 2022 Bytedance Inc.

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

import os
import struct
import time
import numpy as np
from .obfuseutil import model_deObfuscator
from pathlib import Path
import cv2
import torch
from ptflops.flops_counter import flops_to_string, add_flops_counting_methods, batch_counter_hook
from PIL import Image
from torchvision.transforms import ToTensor

import Common
import sys
from Common.utils.zoo import models
from Common.utils.tensorops import pad, crop, resizeTensor, resizeTensorFast
import warnings
import re

torch.backends.cudnn.deterministic = True

model_ids = {k: i for i, k in enumerate(models.keys())}

metric_ids = {
    "mse": 0,
}

def init_ptflops_calc(model):
    ans = add_flops_counting_methods(model)
    ans.start_flops_count(ost=sys.stdout, verbose=False, ignore_list=[])
    batch_counter_hook(model,[[1]],[])
    return ans

def finish_ptflops_calc(model, size=None):
    """
    Get flops of the model

    Args:
        model (nn.Module): model, which we examine
        size (list, optional): shape of the input image (width, height). Defaults to None.

    Returns:
        float: MACs
        float: MACs per pixel
    """
    # Flops calculation
    flops_count, _ = model.compute_average_flops_cost()
    flops_per_pixel = None
    if size is not None:
        flops_per_pixel = flops_count / (size[0] * size[1])
    return flops_count, flops_per_pixel

def inverse_dict(d):
    # We assume dict values are unique...
    assert len(d.keys()) == len(set(d.keys()))
    return {v: k for k, v in d.items()}

def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size

def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")

def img2torch(img: Image.Image) -> torch.Tensor:
    return ToTensor()(img).unsqueeze(0)

def torch2img(x: torch.Tensor):
    img = x.mul(255).round().squeeze().permute(1, 2, 0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def readPngToTorchIm (inputFileName):
    bitShiftMap = {
            8:[8,0],
            16:[10,6],
        }
    x = cv2.imread(inputFileName,cv2.IMREAD_UNCHANGED )
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    if x.dtype == np.uint8:
        bitDepth = bitShiftMap[8][0]
        bitShift = bitShiftMap[8][1]
    elif x.dtype == np.uint16:
        bitDepth = bitShiftMap[16][0]
        bitShift = bitShiftMap[16][1]
    else:
        print("unknown input bitDepth")
        raise NotImplementedError
    x = x.astype(np.float32)
    x = torch.from_numpy(x).permute(2, 0 , 1).unsqueeze(0)
    x /= (1<<bitShift)
    x /= ((1<<bitDepth) - 1)
    return x, bitDepth, bitShift

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

def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))

def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))

def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))

def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

def get_header(model_name, metric, quality):
    """Format header information:
    - 1 byte for model id
    - 1 byte for metric
    - 1 byte for quality param
    """
    metric = metric_ids[metric]
    return model_ids[model_name], metric, quality - 1

def parse_header(header, oldversion):
    """Read header information from 2 bytes:
    - 1 byte for model id
    - 1 byte for metric
    - 1 byte for quality param
    """
    model_id, metric, quality = header
    quality += 1
    if oldversion:
        model_id = model_id - 8
    return (
        inverse_dict(model_ids)[model_id],
        inverse_dict(metric_ids)[metric],
        quality,
    )

def show_image(img: Image.Image):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.title.set_text("Decoded image")
    ax.imshow(img)
    fig.tight_layout()
    plt.show()

def update_weights(net,weights = None):
    if weights == None:
        net.numfilters = [0,0,0]
    else:
        net.numfilters = weights['numfilters']
        net.filterCoeffs1 = []
        net.filterCoeffs2 = []
        net.filterCoeffs3 = []
        if net.numfilters[0]:
            net.filterCoeffs1 = weights['filterCoeffs'][0:net.numfilters[0]]
        if net.numfilters[1]:
            net.filterCoeffs2 = weights['filterCoeffs'][net.numfilters[0]:net.numfilters[0]+net.numfilters[1]]
        if net.numfilters[2]:
            net.filterCoeffs3 = weights['filterCoeffs'][net.numfilters[0]+net.numfilters[1]:]
        net.decSplit1 = weights["decSplit"][0]
        net.decSplit2 = weights["decSplit"][1]
    net.channelOffsetsTool = weights["channelOffsetTool"]
    if net.channelOffsetsTool:
        net.offsetSplit_w = weights["offsetSplit_w"]
        net.offsetSplit_h = weights["offsetSplit_h"]
    if 'meansRoundDec' in weights.keys():   
        net.decChannelOffsets = weights['meansRoundDec']
    else:
        net.decChannelOffsets = []
    net.numthreads_min = weights["numthreads_min"]
    net.numthreads_max = weights["numthreads_max"]
    net.waveShift = weights["waveShift"]
    net.outputBitDepth = weights["outputBitDepth"]
    net.outputBitShift = weights["outputBitShift"]
    net.DoublePrecisionProcessing = weights["DoublePrecisionProcessing"]
    net.DeterminismSpeedup = weights["DeterminismSpeedup"]
    net.FastResize = weights["FastResize"] 

def read_weights(f):
    def reader(f,precise): 
        if precise: 
            return float(read_uints(f,1)[0])/100000
        else:
            return float(read_uchars(f,1)[0])/100
    weights = {}
    code = read_uchars(f,1)[0]
    weights["outputBitDepth"] = (code>>4) + 1
    weights["outputBitShift"] = code & 0x0F
    code = read_uchars(f,1)[0]
    weights["DoublePrecisionProcessing"] = True if int((code & 0x80)>>7) else False
    weights["DeterminismSpeedup"] = True if int((code & 0x40)>>6) else False
    weights["FastResize"] = True if int((code & 0x20)>>5) else False
    weights['numfilters'] = list(read_uchars(f,3))
    filter1 = []
    for i in range(sum(weights['numfilters'])):
        code = read_uchars(f,1)[0]
        blkSize = (code & 0x08) + 1
        greater = True if ((code & 0x04)) else False
        precise1 = True if ((code & 0x02)) else False
        precise2 = True if ((code & 0x01)) else False
        mode = code >> 4
        if blkSize > 1:
            blkSize = read_uchars(f,1)[0]
        thr = reader(f,precise1)
        b1 = reader(f,precise2)
        if mode == 5:
            b2 = reader(f,precise2)
            scale = [b1,b2]
        else:
            scale = [b1]
        channels = []
        if mode == 4:
            numFilters = read_uchars(f,1)[0]
            if numFilters > 0:
                channels = list(read_uchars(f,numFilters))
        filter = {"thr":thr,"scale":scale,"greater":greater,"mode":mode,"block_size":blkSize,"channels":channels}
        filter1.append(filter)
        weights['filterCoeffs'] = filter1

    weights["decSplit"] = list(read_uchars(f,2))

    weights["channelOffsetTool"] = True if read_uchars(f,1)[0] else 0
    if weights["channelOffsetTool"]:
        weights["offsetSplit_w"] = list(read_uchars(f,1))[0]
        weights["offsetSplit_h"] = list(read_uchars(f,1))[0]
        weights["offsetPrecision"] = read_uints (f,1)[0]
        means_full = torch.zeros((weights["offsetSplit_h"],weights["offsetSplit_w"],192))


        nonZeroChannels = []
        for i in range(0,192,8):
            byte = read_uchars(f,1)[0]
            nonZeroChannels.append((byte & 0x80)>>7)
            nonZeroChannels.append((byte & 0x40)>>6)
            nonZeroChannels.append((byte & 0x20)>>5)
            nonZeroChannels.append((byte & 0x10)>>4)
            nonZeroChannels.append((byte & 0x08)>>3)
            nonZeroChannels.append((byte & 0x04)>>2)
            nonZeroChannels.append((byte & 0x02)>>1)
            nonZeroChannels.append(byte & 0x01)
        for x in range(weights["offsetSplit_h"]): 
            for y in range (weights["offsetSplit_w"]):
                means = torch.zeros(192)
                for i in range(0,192):
                    if nonZeroChannels[i] >0:
                        byte = read_uchars(f,1)[0]
                        sign = -1 if ((byte & 0x80)>>7) else 1
                        val = (byte & 0x7F)
                        means[i] = sign*val
                means_full[x,y,:]=means
        weights['meansRoundDec'] = means_full.permute(2,0,1)/weights["offsetPrecision"] 
    weights["numthreads_min"] = numFilters = read_uchars(f,1)[0]
    weights["numthreads_max"] = numFilters = read_uchars(f,1)[0]
    weights["waveShift"] = numFilters = read_uchars(f,1)[0]

    return weights

def dec_adap(inputpath, coder, model_list, output=None, net=None, device="cuda", calcFlops = False, oldversion=False):
    Common.set_entropy_coder(coder)
    if device == "cuda":
        torch.cuda.synchronize()
    dec_start = time.time()
    with Path(inputpath).open("rb") as f:
        model, metric, quality = parse_header(read_uchars(f, 3), oldversion)
        original_size = read_uints(f, 2)
        resized_size = read_uints(f, 2)
        weightsReceived = read_weights(f)

        shape = read_uints(f, 2)
        strings = []
        n_strings = read_uints(f, 1)[0]
        for _ in range(n_strings):
            s = read_bytes(f, read_uints(f, 1)[0])
            strings.append([s])

    ckpt = model_list[len(model_list) - quality]

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    if net is None:
        if oldversion:
            net = models[model](quality=quality, metric=metric, decoderOnly=True, device=device, Quantized=True, oldversion=oldversion)
        else:
            net = models[model](quality=quality, metric=metric, decoderOnly=True, device=device, Quantized=True, oldversion=oldversion)
        checkpoint = torch.load(ckpt, map_location=torch.device(device))
        if "state_dict" in checkpoint.keys():
            net.load_state_dict(model_deObfuscator(checkpoint["state_dict"]))
        else:
            net.load_state_dict(model_deObfuscator(checkpoint))
        net.eval()
        net.update()
    update_weights(net, weightsReceived)
    if device == "cuda":
        torch.cuda.synchronize()
    load_time = time.time() - start

    with torch.no_grad():
        if calcFlops:
            init_ptflops_calc(net)
        out = net.decompress(strings, shape, device)
        if calcFlops:
            flops_count, flops_per_pixel = finish_ptflops_calc(
            net, original_size)
            print('Recommended model: Flops: {0}, i.e {1} / pxl'.format(
                flops_to_string(flops_count, units=None),
                flops_to_string(flops_per_pixel, units='Mac')))


    x_hat = crop(out["x_hat"], resized_size)
    if net.FastResize:
        x_hat = resizeTensorFast(x_hat, original_size)
    else:
        x_hat = resizeTensor(x_hat, original_size)
    if device == "cuda":
        torch.cuda.synchronize()
    dec_time = time.time() - dec_start
    print(f"Decoded in {dec_time:.2f}s (model loading: {load_time:.2f}s), (processing time: {dec_time - load_time:.2f}s)")

    writePngOutput(x_hat,net.outputBitDepth,net.outputBitShift,output)
    return dec_time - load_time, x_hat

def clear_trash(file_list):
    for file in file_list:
        if os.path.exists(file):
            os.system(f"rm {file}")

def read_recipe(file, img, rate):
    pattern = img + f"@{rate:.2f}"
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip()
            if l.startswith(pattern):
                s = l.split(": ")[-1]
                s = s.split(", ")
                if len(s) == 2:
                    return int(s[0]), None
                if len(s) == 3:
                    return int(s[0]), (int(s[1]), int(s[2]))

def get_device(args_device):
    device = "cuda" if args_device == "cuda" and torch.cuda.is_available() else "cpu"
    if args_device == "cuda" and not torch.cuda.is_available():
        warnings.warn("Cuda is not available, running with cpu ..")
    return device

def print_args(args):
    for k, v in vars(args).items():
        print(f"{k:<15}: {v}")

def get_model_list(ckptdir):
    model_list = os.listdir(ckptdir)
    model_list = [os.path.join(ckptdir, v) for v in model_list if 'ckpt' in v and not v.startswith('.')]
    qualities = []
    for k, m in enumerate(model_list):
        r = re.search("ckpt-\d+", m).group()
        qualities.append(int(r[5:]))
    idx = np.argsort(qualities)[::-1]
    sorted_list = [model_list[v] for v in idx]
    return sorted_list