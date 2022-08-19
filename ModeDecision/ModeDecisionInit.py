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

"""
Generate initial recipe when no recipe and anchor results are available.
"""
import os, glob
import torch
import argparse
from copy import deepcopy
from os import listdir
from os.path import isfile, join
import Common
from Metric.metric_tool import calculateMetrics as cm
from Encoder.testfuncs import readPngToTorchIm, construct_weights, enc_adap, get_device, resizeCandOld
from Common.utils.testfuncs import dec_adap
import sys
import warnings
from contextlib import contextmanager
import json
import time


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def ModeDecision(image, rate, modelDir, decModelPath, cfgPath="Encoder/NotRecipes.json",
                 AnchorResultPath="Encoder/NotAFile.txt", lowerBound=0.95, step=64, device="cuda"):
    encodingTimeStart = time.time()
    global AllWeights
    Common.set_entropy_coder(Common.available_entropy_coders()[0])
    model = "quantyuv444-decoupled"
    metric = "mse"
    coder = Common.available_entropy_coders()[0]
    bitstream = "str.bin"

    rate2quality = {0.75: 6, 0.5: 4, 0.25: 3, 0.12: 2, 0.06: 1}
    quality_mapped = rate2quality[rate]
    x_true_org, _, _ = readPngToTorchIm(image)

    onlyfiles = [join(modelDir, f) for f in listdir(modelDir) if isfile(join(modelDir, f))]
    ModelList = sorted(onlyfiles, reverse=True)
    onlyfiles = [join(decModelPath, f) for f in listdir(decModelPath) if isfile(join(decModelPath, f))]
    DecModelList = sorted(onlyfiles, reverse=True)

    _, _, h, w = x_true_org.shape

    candidates = [
        [0, 0],
    ]

    net = None
    prev_rate = {0.75: 0.50, 0.50: 0.25, 0.25: 0.12, 0.12: 0.06, 0.06: 0.06}
    # metrics_anchor = read_anchor_metrics(AnchorResultPath, os.path.basename(image), prev_rate[rate])
    # metrics_best = read_anchor_metrics(AnchorResultPath, os.path.basename(image), rate)
    loss_best = 100
    recipe = construct_weights(cfgPath, rate, image.split("/")[-1], h, w)
    recipe_best = deepcopy(recipe)
    recipe_init = deepcopy(recipe)
    best_updated = False
    for k in [2]:
        for i in range(len(candidates)):
            # 1. construct the encoding recipe
            recipe = deepcopy(recipe_init)

            # 2. update the parameters that is to be checked.
            recipe['filterCoeffs'][k]['thr'] += candidates[i][0]
            recipe['filterCoeffs'][k]['scale'][0] += candidates[i][1]

            # 3. Get resized candidates under the given params in 2
            start_time = time.time()
            resize_candidates = resizeCandOld(image, model, metric, coder, bitstream, ModelList, lowerBound,
                                           step, recipe['numRefIte'], rate, recipe, device)
            elapsed_time = time.time() - start_time
            print(
                f"{len(resize_candidates)} resized candidates are found, elapsed time {elapsed_time // 60:.0f} minutes.")

            # 4. Find best candidate among resized_candidates
            for j, rcand in enumerate(resize_candidates):
                recipe['model_idx'] = rcand[0]
                recipe['resized_size'] = rcand[1]

                # do the encoding and decoding
                _, enctime, net = enc_adap(image, model, metric, coder, bitstream, ModelList[recipe['model_idx']],
                                           recipe['numRefIte'], None, None, recipe['resized_size'], rate, device, recipe=recipe)
                # torch.cuda.reset_max_memory_allocated()
                torch.cuda.empty_cache()
                dec_time, _ = dec_adap(bitstream, coder, DecModelList, output="reco.png", net=net, device=device)

                # 5. calculate the quality metrics
                recoIm, _, _ = readPngToTorchIm("reco.png")
                metrics = cm(recoIm, x_true_org, bitstream, recipe["model_idx"], " ")
                metrics['enc'] = enctime
                metrics['dec'] = dec_time
                metrics['image'] = image.split("/")[-1]
                metrics['dec_mem'] = torch.cuda.max_memory_allocated('cuda') / 1000000000

                # if (metrics['vmaf'] - metrics_anchor['vmaf']) / (metrics['bpp'] - metrics_anchor['bpp']) > 0:
                    # 6. calculate loss
                    # loss = calAverageBdRate(AnchorResultPath, metrics, quality_mapped)
                loss = 0. - metrics['y_msssim']

                # 7. update the best recipe
                if loss < loss_best:
                    loss_best = loss
                    metrics_best = deepcopy(metrics)
                    recipe_best = deepcopy(recipe)
                    best_updated = True
            recipe_init = deepcopy(recipe_best)
    metrics_best['enc'] = time.time() - encodingTimeStart
    if 'ChannelOffsets' in recipe_best:
        del recipe_best['ChannelOffsets']  # channel offsets weights are calculated using mse, no need to store them.
    if not best_updated:
        print(f"The best metrics are from anchor.")
    return metrics_best, recipe_best


def ModeDecisionGAN(image, rate, modelDir, decModelPath, cfgPath="Encoder/AllRecipes.json", lowerBound=0.95, step=64, device="cuda"):
    Common.set_entropy_coder(Common.available_entropy_coders()[0])
    model = "quantyuv444-decoupled"
    metric = "mse"
    coder = Common.available_entropy_coders()[0]
    bitstream = "str.bin"

    x_true_org, _, _ = readPngToTorchIm(image)

    onlyfiles = [join(modelDir, f) for f in listdir(modelDir) if isfile(join(modelDir, f))]
    ModelList = sorted(onlyfiles, reverse=True)
    onlyfiles = [join(decModelPath, f) for f in listdir(decModelPath) if isfile(join(decModelPath, f))]
    DecModelList = sorted(onlyfiles, reverse=True)

    _, _, h, w = x_true_org.shape

    recipe = construct_weights(cfgPath, rate, image.split("/")[-1], h, w, True)

    for k in range(5):
        m = ModelList[k]
        bpp, enc_time, _ = enc_adap(image, model, metric, coder, bitstream, m,
                                    recipe['numRefIte'], None, None, (h, w), rate, device, recipe=recipe)
        torch.cuda.empty_cache()

        if k == 0 and bpp < rate * lowerBound:
            recipe['model_idx'] = 0
            recipe['resized_size'] = [h, w]
            break

        wr, hr = w, h
        if k == 4 and bpp > rate * 1.1:
            while bpp > rate * 1.1:
                wr -= step
                hr = int(round(wr / w * h))
                bpp, enc_time, _ = enc_adap(image, model, metric, coder, bitstream, m,
                                            recipe['numRefIte'], None, None, (hr, wr), rate, device, recipe=recipe)
                torch.cuda.empty_cache()
            recipe['model_idx'] = 4
            recipe['resized_size'] = [hr, wr]
            break

        if bpp <= rate * 1.1:
            if bpp >= rate * lowerBound:
                recipe['model_idx'] = k
                recipe['resized_size'] = [h, w]
                break
            else:  # current bpp is too low, use the previous model with resize
                wr, hr = w, h
                bpp, enc_time, _ = enc_adap(image, model, metric, coder, bitstream, ModelList[k-1],
                                            recipe['numRefIte'], None, None, (h, w), rate, device, recipe=recipe)
                torch.cuda.empty_cache()
                while bpp > rate * 1.1:
                    wr -= step
                    hr = int(round(wr / w * h))
                    bpp, enc_time, _ = enc_adap(image, model, metric, coder, bitstream, ModelList[k-1],
                                                recipe['numRefIte'], None, None, (hr, wr), rate, device, recipe=recipe)
                    torch.cuda.empty_cache()
                recipe['model_idx'] = k - 1
                recipe['resized_size'] = [hr, wr]
                break

    dec_time, _ = dec_adap(bitstream, coder, DecModelList, output="reco.png", net=None, device=device)
    recoIm, _, _ = readPngToTorchIm("reco.png")
    metrics = cm(recoIm, x_true_org, bitstream, recipe["model_idx"], " ")
    metrics['enc'] = enc_time
    metrics['dec'] = dec_time
    metrics['image'] = image.split("/")[-1]
    metrics['dec_mem'] = torch.cuda.max_memory_allocated('cuda') / 1000000000

    if 'ChannelOffsets' in recipe:
        del recipe['ChannelOffsets']  # channel offsets weights are calculated using mse, no need to store them.
    return metrics, recipe


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Model Decision arguments parser.")
    parser.add_argument('-i', '--input', type=str, help='Input image file path.', default=None)
    parser.add_argument('-ip', '--inputPath', type=str, help='Input image file path.', default=None)
    parser.add_argument('-c', "--ckptdir", type=str, required=True,
                        help='Checkpoint folder containing multiple pretrained models.')
    parser.add_argument('-dc', "--decCkptdir", type=str, required=True,
                        help='Checkpoint folder containing multiple decoder models.')
    parser.add_argument('-t', "--target_rate", type=float, nargs='+', default=[0.75, 0.50, 0.25, 0.12, 0.06],
                        help="Target bpp (default: %(default)s)")
    parser.add_argument('-cfg', '--cfgPath', type=str, help='config (recipe) file path.',
                        default="Encoder/AllRecipesLB080.json")
    parser.add_argument('-a', '--anchorResultPath', type=str, help='Anchor Result file.', default="Encoder/Anchor_LB080.txt")
    parser.add_argument('-n', '--newCfgPath', type=str, help='output config (recipe) file path.',
                        default="Encoder/newAllRecipes.json")
    parser.add_argument('-l', '--lowerBound', type=float, default=0.80,
                        help='Only search for candidates with bpp >= lower_bound * rate.')
    parser.add_argument('-s', '--step', type=int, default=64, help='Search step in resizing.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='CPU or GPU device.')
    parser.add_argument('--gan', action='store_true', help='Use GAN model if true.')

    args = parser.parse_args(argv)
    return args


def Runner(argv):
    args = parse_args(argv[1:])
    device = get_device(args.device)
    # inputImagePath, modelDir = "ModelDirectory",decModelPath = "ModelDirectory", cfgPath = "Encoder/AllRecipes.json", AnchorResultPath = "Encoder/Anchor.txt"
    if args.input is None and args.inputPath is None:
        print("either the input image name or the folder path needs to be provided.")
        return []
    args = parse_args(argv[1:])
    if args.inputPath:
        images = glob.glob(os.path.join(args.inputPath, "*.png"))
        if not images:
            print("No files found found in the images directory: ", args.inputPath)
            return -1
        images.sort()
    else:
        images = [args.input]

    process_list = []
    for im in images:
        for rate in args.target_rate:
            process_list.append({'image': im, 'rate': rate})

    AllRecipes = []
    for _, list_entry in enumerate(process_list):
        with suppress_stdout():
            if args.gan:
                metrics, recipe_best = ModeDecisionGAN(list_entry["image"], list_entry["rate"], args.ckptdir,
                                                       args.decCkptdir, args.cfgPath, args.lowerBound, args.step,
                                                       device)
            else:
                metrics, recipe_best = ModeDecision(list_entry["image"], list_entry["rate"], args.ckptdir, args.decCkptdir,
                                                    args.cfgPath, args.anchorResultPath, args.lowerBound, args.step, device)

        AllRecipes.append(recipe_best)
        print(
            f"{metrics['image']} {metrics['q']} {metrics['bpp']:.8f} "
            f"{metrics['y_msssim']:.8f} {metrics['psnr']['y']:.8f} {metrics['psnr']['u']:.8f} {metrics['psnr']['v']:.8f} "
            f"{metrics['vif']:.8f} {metrics['fsim']:.8f} {metrics['nlpd']:.8f} {metrics['iw_ssim']:.8f} {metrics['vmaf']:.8f} {metrics['psnr_vhsm']:.8f} "
            f"{metrics['enc']:.2f} {metrics['dec']:.2f}  {metrics['dec_mem']:.2f}"
        )
        with open(args.newCfgPath, "w") as outfile:
            json.dump(AllRecipes, outfile)


# Usage examples:
# Do the mode decision for all images in cfp-test-set/ folder (using all 5 rate points by default):
# python3 Encoder/ModeDecision.py -c ModelDirectory -dc DecModelDirectory -ip cfp-test-set/
# Do the mode decision for a single image (using all 5 rate points by default): 
# python3 Encoder/ModeDecision.py -c ModelDirectory -dc DecModelDirectory -i cfp-test-set/00001_TE_1192x832_8bit_sRGB.png
# Do the mode decision for a single image using single target rate point: 
# python3 Encoder/ModeDecision.py -c ModelDirectory -dc DecModelDirectory -i cfp-test-set/00001_TE_1192x832_8bit_sRGB.png -t 0.75
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    if False:  # If you are running this locally, download the encoder and decoder models in two folders, also download the input images. The below code does that for me.
        os.system("mkdir ModelDirectory")
        os.system("hdfs dfs -copyToLocal hdfs://harunava/home/byte_videoarch_AIIC/quantFinetune/* ModelDirectory/")
        os.system("python3 Common/testfunc/util.py ModelDirectory DecModelDirectory")
        os.system("hdfs dfs -copyToLocal " + "hdfs://harunava/home/byte_videoarch_AIIC/cfp-test-set/" + " .")
    Runner(sys.argv)
