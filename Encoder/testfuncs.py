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
from pathlib import Path
import cv2
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import Common
from Common.utils.zoo import models
from Common.utils.tensorops import pad, resizeTensor
import warnings
import re
torch.backends.cudnn.deterministic = True

model_ids = {k: i for i, k in enumerate(models.keys())}

metric_ids = {
    "mse": 0,
}

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


def get_header(model_name, metric, quality, oldversion):
    """Format header information:
    - 1 byte for model id
    - 1 byte for metric
    - 1 byte for quality param
    """
    metric = metric_ids[metric]
    m_id = model_ids[model_name]+8 if oldversion else model_ids[model_name]
    return m_id, metric, quality - 1


def parse_header(header):
    """Read header information from 2 bytes:
    - 1 byte for model id
    - 1 byte for metric
    - 1 byte for quality param
    """
    model_id, metric, quality = header
    quality += 1
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


def savetimeinfo(enctime, dectime, args):
    timeinfo = args.bin[:-4] + '.txt'
    infohandle = open(timeinfo, mode = 'w')
    infohandle.write('Enctime:{}\n'.format(enctime))
    infohandle.write('Dectime:{}\n'.format(dectime))
    infohandle.close()
    return timeinfo


def construct_weights(json_file,qp,image_name, org_h = None, org_w = None, GAN=False):
    r'''
    looks into the json file. if image_name and rate matches for an entry, the recipe is loaded. 
    the recipe can be partial, missing entries are loaded as defaults.
    '''
    qp2quality = {22: 6, 28: 4, 34: 3, 40: 2, 46: 1}
    quality_mapped = qp2quality[qp]
    initial_weights = prepare_weights(quality_mapped,org_h,org_w)
    if org_h is not None and org_w is not None:
        initial_weights['resized_size'] = [org_h, org_w]
    initial_weights['image'] = image_name
    initial_weights['model_idx'] = 16 - (quality_mapped * 2 if quality_mapped<6 else 10)
    if GAN:
        initial_weights['model_idx'] = 2  # Tentatively, use the middle one
    initial_weights['QP'] = qp
    initial_weights['numRefIte'] = 1


    import json
    partial_weights = []
    if os.path.isfile(json_file):
        with open(json_file) as json_file:
            data = json.load(json_file)
            for d in data:
                if d["image"] == image_name and d['qp']==qp:
                    partial_weights = d

    if partial_weights == []:
        print("could not find the image or rate in the encoding recipe, using initial weights")
        return initial_weights

    for entry in partial_weights.keys():
        initial_weights[entry] = partial_weights[entry]
    return initial_weights

def prepare_weights(quality, h, w):

    mean_weight_list = [0.002, 0.004, 0.006, 0.008, 0.01,0.012]
    residual_weight_list = [0.004, 0.006, 0.008, 0.010, 0.012,0.014]
    filterCoeffs = [
            # first set of filter coefficients control the quantization of the residual. the filters will be applied one after the other.
            {"thr": 1.30, "scale": [1.20], "greater": True, "mode": 0, "block_size": 1, "channels": [], "precise": [True, False]},
            {"thr": 3.98, "scale": [1.20], "greater": True, "mode": 0, "block_size": 1, "channels": [], "precise": [True, False]},
            {"thr": 0.95, "scale": [0.95], "greater": False, "mode": 0, "block_size": 1, "channels": [], "precise": [True, False]},
            # second set of filter coefficients control skipping condition of entropy coding. if a first filter OR the the second filter decides to skip a sample, that sample is skipped.
            {"thr": 0.1132, "scale": [0], "greater": True, "mode": 3, "block_size": 16, "channels": [], "precise": [True, False]},
            # third set of filters determine scaling at the decoder after y_hat is reconstructed. scale[0] controls weight of mean, scale[1] controls weight of residual.
            {"thr": 0.95, "scale": [mean_weight_list[6 - quality] / 1.5, residual_weight_list[6 - quality] / 1.5], "greater": True, "mode": 5, "block_size": 1, "channels": [], "precise": [True, True]},
            {"thr": 0.95, "scale": [mean_weight_list[6 - quality] / 3, residual_weight_list[6 - quality] / 3], "greater": False, "mode": 5, "block_size": 1, "channels": [], "precise": [True, True]},
    ]
    numfilters = [3, 1, 2]
    weights= {"numfilters": numfilters}
    weights['filterCoeffs'] = filterCoeffs
    weights["decSplit"] = [w//1000 + 1,w//400 + 1]
    weights["channelOffsetTool"] = True
    if weights["channelOffsetTool"]:
        weights["offsetSplit_w"] = w//1600 + 1
        weights["offsetSplit_h"] = h//600 + 1
        weights["numChannels4offset"] = 20
        weights["offsetPrecision"] = 400
    weights["numthreads_min"] = 50
    weights["numthreads_max"] = 100
    weights["waveShift"] = 1
    weights["outputBitDepth"] = 8
    weights["outputBitShift"] = 0
    weights["DoublePrecisionProcessing"] = False
    weights["DeterminismSpeedup"] = True
    weights["FastResize"] = True
    return weights


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

    
def write_weights(f,weights):
    def writer (num, precise ):
        if precise:
            write_uints(f,[int(num*100000)])
        else:
            write_uchars(f,[int(num*100)])
    code = ((weights["outputBitDepth"] - 1) << 4) | ((weights["outputBitShift"])  & 0x0F)
    write_uchars(f,tuple([code]))
    aa = 1 if weights["DoublePrecisionProcessing"] else 0
    bb = 1 if weights["DeterminismSpeedup"] else 0
    cc = 1 if weights["FastResize"] else 0
    code = aa<<7 | bb<<6 | cc<<5
    write_uchars(f,tuple([code]))
    write_uchars(f,tuple(weights['numfilters']))
    
    for i in range(sum(weights['numfilters'])):
        filt2Write = weights['filterCoeffs'][i]
        code = ( filt2Write["mode"] << 4) |\
             ((1 if filt2Write["block_size"]>1 else 0)<<3)  & 0x0F |\
             ((1 if filt2Write["greater"] else 0)<<2)  & 0x0F |\
             ((1 if filt2Write["precise"][0] else 0)<<1)  & 0x0F |\
             ((1 if filt2Write["precise"][1] else 0))  & 0x0F

        write_uchars(f,tuple([code]))
        if filt2Write["block_size"] > 1:
            write_uchars(f,tuple([filt2Write["block_size"]]))
        writer(filt2Write["thr"],filt2Write["precise"][0])
        if filt2Write["mode"] == 5:
            writer(filt2Write["scale"][0],filt2Write["precise"][1])
            writer(filt2Write["scale"][1],filt2Write["precise"][1])
        else:
            writer(filt2Write["scale"][0],filt2Write["precise"][1])
        if filt2Write["mode"] == 4:
            write_uchars(f, tuple([len(filt2Write["channels"])]))
            if filt2Write["channels"]:
                write_uchars(f, tuple(filt2Write["channels"]))
    write_uchars(f, tuple(weights["decSplit"]))
    
    write_uchars(f,tuple([1 if weights["channelOffsetTool"] else 0]))
    if weights["channelOffsetTool"]:
        write_uchars(f,tuple([weights["offsetSplit_w"],weights["offsetSplit_h"]]))
        write_uints (f,tuple([ weights["offsetPrecision"]]))
        means_full = weights['ChannelOffsets']
        dummy = means_full*weights["offsetPrecision"]**2
        
        nonZeroChannels = torch.sum(torch.sum(torch.abs((dummy).round()),dim=0),dim=0)
        
        _,indices = torch.sort(nonZeroChannels,descending=True)
        nonZeroChannels = nonZeroChannels.to("cpu")
        indices = indices.to("cpu")
        means_full = (means_full*weights["offsetPrecision"]).round()
        means_full = means_full.to("cpu")
        for ind,val in enumerate(indices):
            if ind>weights["numChannels4offset"] :
                nonZeroChannels[val] = 0
        bytestream = []
        for i in range(0,192,8):
            byte = (0 if nonZeroChannels[i] == 0 else 1)<<7 |\
                (0 if nonZeroChannels[i+1] == 0 else 1)<<6 |\
                (0 if nonZeroChannels[i+2] == 0 else 1)<<5 |\
                (0 if nonZeroChannels[i+3] == 0 else 1)<<4 |\
                (0 if nonZeroChannels[i+4] == 0 else 1)<<3 |\
                (0 if nonZeroChannels[i+5] == 0 else 1)<<2 |\
                (0 if nonZeroChannels[i+6] == 0 else 1)<<1 |\
                (0 if nonZeroChannels[i+7] == 0 else 1)
            bytestream.append(byte)
        for x in range(weights["offsetSplit_h"]):
            for y in range(weights["offsetSplit_w"]):
                means = means_full[x,y,:]
                for i in range(192):
                    if not nonZeroChannels[i] == 0:
                        sign = 1 if means[i]<0 else 0
                        val = int(abs(means[i])) if int(abs(means[i])) < 127 else 127
                        byte = (sign << 7) | (val & 0x7F)
                        bytestream.append(byte)
        write_uchars(f,tuple(bytestream))
    write_uchars(f,tuple([weights["numthreads_min"]]))
    write_uchars(f,tuple([weights["numthreads_max"]]))
    write_uchars(f,tuple([weights["waveShift"]]))
  
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


def enc_adap(image, model, metric, coder, output, ckpt, numRefIte=0, im=None, net=None,
             resized_size=None, QP=None, device="cuda", recipe=None, oldversion=False):
    """
    - resized_size: (height, width)
    """
    qp2quality = {22: 6, 28: 4, 34: 3, 40: 2, 46: 1}
    quality_mapped = qp2quality[QP]
    quality = int(ckpt.split('ckpt-')[-1])

    Common.set_entropy_coder(coder)
    enc_start = time.time()
    if im is not None:
        x_org = img2torch(im).to(device)
    else:
        x_org, _, _ = readPngToTorchIm(image)
        x_org = x_org.to(device)
    start = time.time()
    if net is None:
        net = models[model](quality=quality, metric=metric, device=device, Quantized=True, oldversion=oldversion).to(device)
        print('Load latest model')
        checkpoint = torch.load(ckpt, map_location=torch.device(device))
        if "state_dict" in checkpoint.keys():
            net.load_state_dict(checkpoint["state_dict"])
        else:
            net.load_state_dict(checkpoint)
        net.eval()
        net.update()
    print('Load finished')
    load_time = time.time() - start

    h0, w0 = x_org.size(2), x_org.size(3)
    x = x_org if (resized_size is None or (resized_size[0] == h0 and resized_size[1] == w0)) else resizeTensor(x_org, resized_size)
    x = x.to(device)
    h, w = x.size(2), x.size(3)
    if recipe is None:
        weights2transmit = prepare_weights(quality_mapped, h, w)
    else:
        weights2transmit = recipe
    if oldversion:
        weights2transmit["FastResize"] = False
    update_weights(net, weights2transmit)
    p = 64  # maximum 6 strides of 2
    x = pad(x, p)

    with torch.no_grad():
        out = net.compress(x, h, w, quality_mapped, numRefIte, device = device)
    weights2transmit['ChannelOffsets'] = net.encChannelOffsets

    shape = out["shape"]
    header = get_header(model, metric, quality, oldversion)

    with Path(output).open("wb") as f:
        write_uchars(f, header)
        # write original image size
        write_uints(f, (h0, w0))
        # write resized image size
        write_uints(f, (h, w))
        write_weights(f, weights2transmit)
        # write shape and number of encoded latents
        write_uints(f, (shape[0], shape[1], len(out["strings"])))
        for s in out["strings"]:
            write_uints(f, (len(s[0]),))
            write_bytes(f, s[0])

    enc_time = time.time() - enc_start
    size = filesize(output)
    bpp = float(size) * 8 / (h0 * w0)
    print(
        f"{bpp:.3f} bpp |"
        f" Encoded in {enc_time:.2f}s (model loading: {load_time:.2f}s)"
    )
    return bpp, enc_time - load_time, net


def clear_trash(file_list):
    for file in file_list:
        if os.path.exists(file):
            os.system(f"rm {file}")



def get_size(org_size, ratio):
    if ratio == 1.0:
        return [org_size]

    h0, w0 = org_size
    num_pixels = h0 * w0 * ratio
    hr = int(round(np.sqrt(num_pixels * h0 / w0)))
    wr = int(round(num_pixels / hr))

    # height x64
    hr1 = hr // 64 * 64
    wr1 = int(round(num_pixels / hr1))

    # width x64
    wr2 = wr // 64 * 64
    hr2 = int(round(num_pixels / wr2))

    # with and height both x64
    hr3 = hr // 64 * 64
    wr3 = (wr // 64 + 1) * 64
    wr4 = wr // 64 * 64
    hr4 = (hr // 64 + 1) * 64
    wr5 = wr // 64 * 64
    hr5 = hr // 64 * 64

    return [(hr, wr), (hr1, wr1), (hr2, wr2), (hr3, wr3), (hr4, wr4), (hr5, wr5)]


def resizeCand(image, model, metric, coder, bitstream, ModelList, lower_bound, step,
               numRefIte, target_rate, recipe, device):
    w0, h0 = Image.open(image).convert("RGB").size

    indices = []
    for k, ckpt in enumerate(ModelList):
        bpp, _, _ = enc_adap(image, model, metric, coder, bitstream, ckpt, numRefIte,
                             None, None, (h0, w0), target_rate, device, recipe=recipe)
        # torch.cuda.reset_max_memory_allocated()
        torch.cuda.empty_cache()
        if bpp >= lower_bound * target_rate:
            indices.append(k)
            if len(indices) == 5:
                indices.pop(0)  # Keep only 4 models
        else:
            break

    if not indices:
        return [0, (h0, w0)]

    candidates = []
    for k in indices:
        ckpt = ModelList[k]
        for p in range(100, 24, -5):
            ratio = p / 100.
            running_bpp = []
            for resized_size in get_size((h0, w0), ratio):
                bpp, _, _ = enc_adap(image, model, metric, coder, bitstream, ckpt, numRefIte, None,
                                     None, resized_size, target_rate, device, recipe)
                # torch.cuda.reset_max_memory_allocated()
                torch.cuda.empty_cache()
                if lower_bound * target_rate <= bpp <= 1.1 * target_rate:
                    candidates.append([k, resized_size])
                running_bpp.append(bpp)
            if max(running_bpp) < lower_bound * target_rate:
                break
    if not candidates:
        return [[indices[-1], (h0, w0)]]
    return candidates


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


def read_anchor_metrics(file, img_name, target_rate):
    rate2idx = {0.75: 0, 0.50: 1, 0.25: 2, 0.12: 3, 0.06: 4}
    metrics = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "\t" in line.strip():
                l = line.strip().split("\t")
            else:
                l = line.strip().split(" ")
            l = list(filter(None, l))
            if l[0] == img_name:
                l[1:] = list(map(float, l[1:]))
                metrics.append(l)
    m = metrics[rate2idx[target_rate]]
    metrics = {'image': m[0], 'q': m[1], 'bpp': m[2],
               'y_msssim': m[3], 'psnr': {'y': m[4], 'u': m[5], 'v': m[6]},
               'vif': m[7], 'fsim': m[8], 'nlpd': m[9], 'iw_ssim': m[10],
               'vmaf': m[11], 'psnr_vhsm': m[12], 'enc': m[13], 'dec': m[14], 'dec_mem': m[15]}
    return metrics
