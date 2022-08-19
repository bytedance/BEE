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

import sys
import argparse
import glob, os
from os.path import abspath, join, dirname, pardir
sys.path.append(join(abspath(dirname(__file__)), pardir))

import torch
import Common
from Common.utils.zoo import models
from Encoder.testfuncs import enc_adap, construct_weights
from Common.utils.testfuncs import print_args, get_device, get_model_list, read_recipe,readPngToTorchIm

torch.backends.cudnn.deterministic = True


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Decoder arguments parser.")
    parser.add_argument('-i', '--input', type=str, help='Input image file path.', default=None)
    parser.add_argument('--inputPath', type=str, help='Input image file path.', default=None)
    parser.add_argument('-o', '--output', type=str, default=None, help='Output bin file name')
    parser.add_argument('--outputPath', type=str, default='bitstreams', help='Output bin file path')
    parser.add_argument("--ckpt", type=str, help='Pretrained model path.', default=None)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='CPU or GPU device.')
    parser.add_argument("--model", choices=models.keys(), default="quantyuv444-decoupled", help="NN model to use (default: %(default)s)")
    parser.add_argument("--metric", choices=["mse"], default="mse", help="metric trained against (default: %(default)s")
    parser.add_argument("--coder", choices=Common.available_entropy_coders(), default=Common.available_entropy_coders()[0], help="Entropy coder (default: %(default)s)")
    parser.add_argument("--ckptdir", type=str, help='Checkpoint folder containing multiple pretrained models.')
    parser.add_argument("--cfg", type=str, help='Path to the CfG file', default = "Encoder/AllRecipes.json")
    parser.add_argument("--target_rate", type=float, nargs='+', default=[0.75,0.50,0.25,0.12,0.06], help="Target bpp (default: %(default)s)")
    parser.add_argument("--online_off", action='store_true', help="Flag to turn off online refinement")
    parser.add_argument("--oldversion", action='store_true', help="flag to compress bitstream of old version")
    args = parser.parse_args(argv)
    return args


def encode(args):
    device = get_device(args.device)
    if args.input == None and args.inputPath == None:
        print("either the input image name or the folder path needs to be provided.")
        return []
    if args.ckptdir == None and args.ckpt == None:
        print("either the ckptdir or ckpt needs to be provided.")
        return []
    if args.inputPath:
        images = glob.glob(os.path.join(args.inputPath, "*.png"))
        if images == []:
            print("No files found found in the images directory: ",args.inputPath)
            return []
        images.sort()
    else:
        images = [args.input]
    result = []
    if not os.path.exists(args.outputPath):
        os.mkdir(args.outputPath)
    for s in images:
        for rate in args.target_rate:
            image = s.split("/")[-1]
            x_org, _, _ = readPngToTorchIm(s)
            _,_,h,w = x_org.shape
            recipe = construct_weights(args.cfg,rate,image,h,w)
            if args.ckpt is not None:
                stateDictPath = args.ckpt
            elif args.ckptdir:
                model_list = get_model_list(args.ckptdir)
                stateDictPath = model_list[recipe['model_idx']]
            else:
                raise NotImplementedError("either the checkpoint path or the folder containing the state dictionaries must be given.")

            if args.output is not None:
                binName = args.output
                binName = os.path.join(args.outputPath,binName)
            else:
                dummy = image.split(".")[0]
                binName = os.path.join(args.outputPath,"BEE_"+dummy+"_"+f'{rate:0.02f}'+".bin")
            _, enc_time, _ = enc_adap(s, args.model, args.metric, args.coder, binName, stateDictPath,
                                      recipe['numRefIte'] if not args.online_off else 0, None, None, recipe['resized_size'], 
                                      rate, device,recipe=recipe, oldversion=args.oldversion)
            result.append([image, rate, enc_time])
            torch.cuda.empty_cache()
            print(f"{result[-1][0]} Rate: {result[-1][1]:.2f} EncTime: {result[-1][2]:.3f}")
    return result


def main(argv):
    args = parse_args(argv[1:])
    print_args(args)
    torch.set_num_threads(1)  # just to be sure

    result = encode(args)
    if result:
        print("All Results:")
        for s in result:
            print(f"{s[0]} Rate: {s[1]:.2f} EncTime: {s[2]:.3f}")

#Usage examples:
# encode a single image in 2 different rate point, use default recipes:
# python3 CoreEncApp.py --ckptdir ModelDirectory --input cfp-test-set/00011_TE_1784x1296_8bit_sRGB.png --outputPath bitstreams --target_rate 0.06 0.12
# Encode all images in cfp-test-set folder using 5, use default recipes:
# python3 CoreEncApp.py --ckptdir ModelDirectory --inputPath cfp-test-set
# Encode a single image using a given state dictionary:
# python3 CoreEncApp.py --input cfp-test-set/00011_TE_1784x1296_8bit_sRGB.png --output semih.bin --target_rate 0.06 --ckpt ModelDirectory/quant_model.ckpt-12

if __name__ == '__main__':
    #Before running the Encoder, make sure you copy the encoder models and images to local machine.
    if False:
        os.system("mkdir ModelDirectory")
        os.system("hdfs dfs -copyToLocal hdfs://harunava/home/byte_videoarch_AIIC/quantFinetune/* ModelDirectory/")
        os.system("hdfs dfs -copyToLocal " + "hdfs://harunava/home/byte_videoarch_AIIC/cfp-test-set/" + " .")
    main(sys.argv)