import os, glob
import torch
import argparse
from os import listdir
from os.path import isfile, join
import Common
from Metric.metric_tool import calculateMetrics as cm
from Encoder.testfuncs import readPngToTorchIm, construct_weights, enc_adap, get_device
from Common.utils.testfuncs import dec_adap
import sys
import warnings
from contextlib import contextmanager
import json


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def ModeDecisionGAN(image, rate, modelDir, decModelPath, cfgPath="Encoder/AllRecipes.json", lowerBound=0.95, step=64, device="cuda", metriclabel='LPIPS'):
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
    candidates = []
    if recipe['rate']<=0.25:
        for k in range(5):
            rh, rw = h, w
            ratio = w/h
            m = ModelList[k]
            bpp, enc_time, _ = enc_adap(image, model, metric, coder, bitstream, m,
                                        recipe['numRefIte'], None, None, (rh, rw), rate, device, recipe=recipe)
            print(bpp, m)
            torch.cuda.empty_cache()
            if bpp < recipe['rate']*lowerBound:
                break
            else:
                while bpp:
                    if bpp > recipe['rate'] * 1.1:
                        # Reduce the resolution of image
                        rh = rh - step if (rh-step)-(rh-step)//64*64 > 32 else rh-step+32
                        rw = int(rh * ratio) if int(rh * ratio)-int(rh * ratio)//64*64 > 32 else int(rh * ratio)+32
                        # Encode
                        bpp, enc_time, _ = enc_adap(image, model, metric, coder, bitstream, m,
                                                    recipe['numRefIte'], None, None, (rh, rw), rate, device, recipe=recipe)
                    elif bpp < recipe['rate']*lowerBound:
                        break
                    else:
                        # Reduce the resolution of image
                        candidates.append([rh, rw, k])
                        rh = rh - step
                        rw = int(rh * ratio)
                        # Encode
                        bpp, enc_time, _ = enc_adap(image, model, metric, coder, bitstream, m,
                                                    recipe['numRefIte'], None, None, (rh, rw), rate, device, recipe=recipe)
        best_metric = 1 if metriclabel=='LPIPS' else 0
        best_cand = []
        from Metric.metric_tool import LPIPS
        LPIPS_lossfn = LPIPS(net = 'alex', spatial = True).to(device)
        for candidate in candidates:
            bpp, enc_time, _ = enc_adap(image, model, metric, coder, bitstream, ModelList[candidate[2]],
                                        recipe['numRefIte'], None, None, (candidate[0], candidate[1]), rate, device, recipe=recipe)
            dec_time, _ = dec_adap(bitstream, coder, DecModelList, output="reco.png", net=None, device=device)
            recoIm, _, _ = readPngToTorchIm("reco.png")
            metrics = cm(recoIm, x_true_org, bitstream, recipe["model_idx"], " ")
            metrics['LPIPS'] = LPIPS_lossfn(recoIm.to(device), x_true_org.to(device)).mean().item()
            print(metriclabel, best_metric, metrics[metriclabel], candidate)
            if metrics[metriclabel] < best_metric and metriclabel=='LPIPS':
                best_metric = metrics[metriclabel]
                best_cand = candidate
            if metrics[metriclabel] > best_metric and metriclabel!='LPIPS':
                best_metric = metrics[metriclabel]
                best_cand = candidate
        if len(best_cand):
            recipe['resized_size'] = best_cand[0:-1]
            recipe['model_idx'] = best_cand[-1]
        else:
            recipe['model_idx'] = recipe['model_idx'] + 5
        bpp, enc_time, _ = enc_adap(image, model, metric, coder, bitstream, ModelList[recipe["model_idx"]],
                                    recipe['numRefIte'], None, None, (recipe['resized_size'][0], recipe['resized_size'][1]), rate, device, recipe=recipe)
    else:
        recipe['model_idx'] = recipe['model_idx'] + 5
        k = recipe['model_idx']
        bpp, enc_time, _ = enc_adap(image, model, metric, coder, bitstream, ModelList[k],
                                    recipe['numRefIte'], None, None, (recipe['resized_size'][0], recipe['resized_size'][1]), rate, device, recipe=recipe)
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
    parser.add_argument('-m','--metric', type=str, default='LPIPS', help='Metric used for selection.')

    args = parser.parse_args(argv)
    return args


def Runner(argv):
    args = parse_args(argv[1:])
    device = get_device(args.device)
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
            metrics, recipe_best = ModeDecisionGAN(list_entry["image"], list_entry["rate"], args.ckptdir,
                                                    args.decCkptdir, args.cfgPath, args.lowerBound, args.step,
                                                    device, args.metric)
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
