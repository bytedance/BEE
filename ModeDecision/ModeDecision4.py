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

import os, glob
import torch
import argparse
from copy import deepcopy
from os import listdir
from os.path import isfile, join

from torch.autograd.grad_mode import F
import Common
from Metric.metric_tool.converted_code import calAverageBdRate,calculateAnchorMetric,bdRateExtend
from Metric.metric_tool import calculateMetrics as cm
from Encoder.testfuncs import readPngToTorchIm, read_recipe, construct_weights, enc_adap 
from Common.utils.testfuncs import dec_adap
import sys
import warnings
from contextlib import contextmanager
import json
import time
import numpy as np
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout



def calAverageBdRate2 (metric_file, metricList, skip = [], MonotonicCheck = True, includeChroma = False):
    bdrate = []
    with open(metric_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "\t" in line.strip():
                l = line.strip().split("\t")
            else:
                l = line.strip().split(" ")
            l = list(filter(None,l))
            if l[0] == metricList[0]['image']:
                l[1] = l[1].split(".p")[0]
                l[1:] = list(map(float, l[1:]))
                bdrate.append(l)
    bdrate_compare = []
    for metric2compare in metricList:
        toCompare = [
                        metric2compare['image'],metric2compare['q'],\
                        metric2compare['bpp'],metric2compare['y_msssim'],metric2compare['psnr']['y'],metric2compare['psnr']['u'],\
                        metric2compare['psnr']['v'],metric2compare['vif'],metric2compare['fsim'],metric2compare['nlpd'],\
                        metric2compare['iw_ssim'],metric2compare['vmaf'],metric2compare['psnr_vhsm'],\
                        metric2compare['enc'],metric2compare['dec'],metric2compare['dec_mem'],
                    ]
        bdrate_compare.append(toCompare)
    
    bdrate_compare_t = list(zip(*bdrate_compare))
    bdrate_t = list(zip(*bdrate))
    summ = 0
    devide = 0
    count = 0
    for kk in range(2):
        if bdrate_compare_t[4][kk] < bdrate_compare_t[4][kk+1]:
            count += 1
    for j in range(3,len(bdrate_compare_t)-3):
        if 'y_msssim' in skip and j == 3:
            continue
        if j == 4 or j == 5 or j == 6:
            continue
        if 'vif' in skip and j == 7:
            continue
        if 'fsim' in skip and j == 8:
            continue
        if 'nlpd' in skip and j == 9:
            continue
        if 'iw_ssim' in skip and j == 10:
            continue   
        if 'vmaf' in skip and j == 11:
            continue
        if 'psnr_hvsm' in skip and j == 12:
            continue  
        gain5 = bdRateExtend(bdrate_t[2][0:],bdrate_t[j][0:],bdrate_compare_t[2][0:],bdrate_compare_t[j][0:])
        sanityCheck = True
        if j != 9:
            for kk in range(4):
                if bdrate_compare_t[j][kk] < bdrate_compare_t[j][kk+1]:
                    count += 1
        if count>0:
            return 100, []
        devide += 1
        summ += gain5
    if devide > 0:
        summ = summ/devide
    else:
        return 100, []

    if includeChroma:
        secondary = ((sum(bdrate_compare_t[5]) + sum(bdrate_compare_t[6])) - (sum(bdrate_t[5]) + sum(bdrate_t[6])))/(sum(bdrate_t[5]) + sum(bdrate_t[6]))
        secondary = -secondary/3
        summ += secondary
    return summ, bdrate_compare


class ModeDecider():
    model = "quantyuv444-decoupled"
    metric = "mse"
    coder = Common.available_entropy_coders()[0]
    rate2quality = {0.75: 6, 0.5: 4, 0.25: 3, 0.12: 2, 0.06: 1}
    quality2rate = {6: 0.75, 4: 0.5, 3: 0.25, 2: 0.12, 1: 0.06}
    pruneLossThreshold = 0.01
    maxPruneCandidate = 3
    lowerRateLimit = 0.95
    higherRateLimit = 1.1
    skipInPrune = ['fsim','vmaf','iw_ssim']
    skipIn2ndStage = ['psnr_hvsm','nlpd','vif','y_msssim']
    
    AllCandidates = [[]]*5
    def __init__(self, modelDir, decModelPath, cfgPath = "Encoder/AllRecipes.json", AnchorResultPath = "Encoder/Anchor.txt"):
        self.modelDir = modelDir
        self.decModelPath = decModelPath
        self.cfgPath = cfgPath
        self.AnchorResultPath = AnchorResultPath
        Common.set_entropy_coder(Common.available_entropy_coders()[0])
        onlyfiles = [join(self.modelDir, f) for f in listdir(self.modelDir) if isfile(join(self.modelDir, f))]
        self.ModelList = sorted(onlyfiles, reverse = True)
        onlyfiles = [join(self.decModelPath, f) for f in listdir(self.decModelPath) if isfile(join(self.decModelPath, f))]
        self.DecModelList = sorted(onlyfiles, reverse = True)


    
    def FindBestCombination (self):
        loss_best = 0
        if not self.AllCandidates[0] == []:
            _,m1,r1  = self.AllCandidates[0][0][0]
            _,m2,r2  = self.AllCandidates[1][0][0]
            _,m3,r3  = self.AllCandidates[2][0][0]
            _,m4,r4  = self.AllCandidates[3][0][0]
            _,m5,r5  = self.AllCandidates[4][0][0]
            metricList = [m1,m2,m3,m4,m5]
            recipeList = [r1,r2,r3,r4,r5]
            ml_best = deepcopy(metricList)
            recipeList_best = deepcopy(recipeList)
            for k in range(3):
                for i in range(5):
                    metricList = deepcopy(ml_best)
                    recipeList = deepcopy(recipeList_best)
                    for cand in self.AllCandidates[i]:
                        _,m,r = cand[0]
                        metricList[i] = deepcopy(m)
                        recipeList[i] = deepcopy(r)
                        loss,_ = calAverageBdRate2(self.AnchorResultPath,metricList,includeChroma=True)
                        if (loss) < (loss_best):
                            loss_best = (loss)
                            ml_best = deepcopy(metricList)
                            recipeList_best = deepcopy(recipeList)
            return loss_best, ml_best, recipeList_best
        return [], [], []


    def PreStoreCandidates(self,image, filename = "recipes"):
        self.x_true_org,_,_ = readPngToTorchIm(image)
        _, _, self.h, self.w = self.x_true_org.shape
        self.anchorRecipe = []
        pixelnum = self.h * self.w
        rates = {0.75:0,0.5:1,0.25:2,0.12:3,0.06:4}
        indices = {0:0.75,1:0.5,2:0.25,3:0.12,4:0.06}
        self.AllCandidates = [0]*5
        self.x_true_org,_,_ = readPngToTorchIm(image)
        _, _, self.h, self.w = self.x_true_org.shape
        for i in range(5):
            self.AllCandidates[i] = []
        bitstream = "str.bin"
        reconName = "recon.png"

        for index in [0,1,2,3,4]:
            qualityMapped = 5 - index if index > 0 else 6
            anchorRecipe = construct_weights(self.cfgPath, self.quality2rate[qualityMapped],image.split("/")[-1],self.h,self.w)
            anchorMetrics = calculateAnchorMetric(self.AnchorResultPath, image.split("/")[-1],qualityMapped)
            with suppress_stdout():
                net = None
                _, enctime, net = enc_adap(image, self.model, self.metric, self.coder, bitstream, self.ModelList[anchorRecipe['model_idx']], anchorRecipe['numRefIte'], \
                    None, net, anchorRecipe['resized_size'], indices[index], recipe=anchorRecipe)
                if 'ChannelOffsets' in anchorRecipe:
                    del anchorRecipe['ChannelOffsets'] 
                dec_time, _ = dec_adap(bitstream, self.coder, self.DecModelList, net = net, output = reconName)
                recoIm,_,_ = readPngToTorchIm(reconName)
                anchorMetrics = cm(recoIm, self.x_true_org, bitstream, anchorRecipe["model_idx"], " ",skip=[])
                anchorMetrics['enc'] = enctime; anchorMetrics['dec'] = dec_time; anchorMetrics['image'] = image.split("/")[-1]
                anchorMetrics['dec_mem'] = 100
            self.AllCandidates[index].append([(self.quality2rate[qualityMapped],deepcopy(anchorMetrics),deepcopy(anchorRecipe))])
        
        
        ####to include the anchors from Zhaobin
        for index in [0,1,2,3,4]:
            qualityMapped = 5 - index if index > 0 else 6
            temporaryRecipes = self.cfgPath.split(".js")[0] + "Newx64.json"
            anchorRecipe = construct_weights(temporaryRecipes, self.quality2rate[qualityMapped],image.split("/")[-1],self.h,self.w)
            anchorMetrics = calculateAnchorMetric(self.AnchorResultPath, image.split("/")[-1],qualityMapped)
            with suppress_stdout():
                
                net = None
                _, enctime, net = enc_adap(image, self.model, self.metric, self.coder, bitstream, self.ModelList[anchorRecipe['model_idx']], anchorRecipe['numRefIte'], \
                    None, net, anchorRecipe['resized_size'], indices[index], recipe=anchorRecipe)
                if 'ChannelOffsets' in anchorRecipe:
                    del anchorRecipe['ChannelOffsets'] 
                dec_time, _ = dec_adap(bitstream, self.coder, self.DecModelList, net = net, output = reconName)
                recoIm,_,_ = readPngToTorchIm(reconName)
                anchorMetrics = cm(recoIm, self.x_true_org, bitstream, anchorRecipe["model_idx"], " ",skip=[])
                anchorMetrics['enc'] = enctime; anchorMetrics['dec'] = dec_time; anchorMetrics['image'] = image.split("/")[-1]
                anchorMetrics['dec_mem'] = 200
            self.AllCandidates[index].append([(self.quality2rate[qualityMapped],deepcopy(anchorMetrics),deepcopy(anchorRecipe))])
        ####to include the anchors from Zhaobin

        model_idx_list = [15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
        rate_list = [0.75,0.5,0.25,0.12,0.06]
        rates = {0.75:0,0.5:1,0.25:2,0.12:3,0.06:4}
        self.resizeCandidates = []
        self.resizeCandidates.append([self.h,self.w])
        for i in np.arange(1,0.5,-0.05):
            h = (1 + (self.h * i - 1) // 64) * 64
            w = (1 + (self.w * i - 1) // 64) * 64
            if self.resizeCandidates[-1][0] == h and self.resizeCandidates[-1][1] == w:
                continue
            if not(self.resizeCandidates[-1][0] == h) and not(self.resizeCandidates[-1][1] == w):
                self.resizeCandidates.append([int(h+64),int(w)])
                self.resizeCandidates.append([int(h),int(w+64)])
                self.resizeCandidates.append([int(h),int(w)])
            else:
                self.resizeCandidates.append([int(h),int(w)])
        for rate in rate_list:

            anchorRecipe = construct_weights(self.cfgPath,rate, image.split("/")[-1],self.h,self.w)
            newFilter = {'thr':0.00, 'scale':[1.0], 'greater':True,'mode':0,'block_size':1,'channels':[],'precise':[False,True]}
            anchorRecipe['numfilters'][0] = 4
            anchorRecipe['filterCoeffs'].insert(3, newFilter) 
            initial_value = anchorRecipe['filterCoeffs'][3]['scale'][0]
            for model_idx in model_idx_list:
                anchorRecipe['model_idx'] = model_idx
                anchorRecipe['resized_size'] = [self.h, self.w]
                net = None
                with suppress_stdout():
                    _, enctime, net = enc_adap(image, self.model, self.metric, self.coder, bitstream, self.ModelList[anchorRecipe['model_idx']], anchorRecipe['numRefIte'], \
                        None, net, anchorRecipe['resized_size'], rate, recipe=anchorRecipe)
                    torch.cuda.empty_cache()
                binsize = os.path.getsize(bitstream) * 8
                bpp = binsize / pixelnum
                if bpp > 2*rate or bpp < rate*0.8:
                    continue
                for size in self.resizeCandidates:
                    anchorRecipe['resized_size'] = [size[0],size[1]]
                    with suppress_stdout():
                        _, enctime, net = enc_adap(image, self.model, self.metric, self.coder, bitstream, self.ModelList[anchorRecipe['model_idx']], anchorRecipe['numRefIte'], \
                            None, net, anchorRecipe['resized_size'], rate, recipe=anchorRecipe)
                        torch.cuda.empty_cache()
                    binsize = os.path.getsize(bitstream) * 8
                    bpp = binsize / pixelnum
                    if bpp > 1.2*rate or bpp < rate*0.8:
                        continue
                    for i in np.arange(0.2,-0.2,-0.04):
                        anchorRecipe['filterCoeffs'][3]['scale'][0] = round((initial_value + i)*100)/100
                        with suppress_stdout():
                            _, enctime, net = enc_adap(image, self.model, self.metric, self.coder, bitstream, self.ModelList[anchorRecipe['model_idx']], anchorRecipe['numRefIte'], \
                                None, net, anchorRecipe['resized_size'], rate, recipe=anchorRecipe)
                            torch.cuda.empty_cache()
                        binsize = os.path.getsize(bitstream) * 8
                        bpp = binsize / pixelnum
                        if bpp > 1.1*rate or bpp < rate*0.95:
                            continue
                        with suppress_stdout():
                            dec_time, _ = dec_adap(bitstream, self.coder, self.DecModelList, net = net, output = reconName)
                            torch.cuda.empty_cache()
                        if 'ChannelOffsets' in anchorRecipe:
                            del anchorRecipe['ChannelOffsets'] 
                        recoIm,_,_ = readPngToTorchIm(reconName)
                        anchorMetrics = cm(recoIm, self.x_true_org, bitstream, anchorRecipe["model_idx"], " ",skip=[])
                        anchorMetrics['enc'] = enctime; anchorMetrics['dec'] = dec_time; anchorMetrics['image'] = image.split("/")[-1]
                        anchorMetrics['dec_mem'] = 0
                        index = rates[rate]
                        self.AllCandidates[index].append([(rate,deepcopy(anchorMetrics),deepcopy(anchorRecipe))])

            index = rates[rate]
            print(image.split("/")[-1].split(".")[0], rate, len(self.AllCandidates[index]))
        
        torch.save(self.AllCandidates,filename)




def parse_args(argv):
    parser = argparse.ArgumentParser(description="Model Decision arguments parser.")
    parser.add_argument('-i', '--input', type=str, help='Input image file path.', default=None)
    parser.add_argument('-ip', '--inputPath', type=str, help='Input image file path.', default=None)
    parser.add_argument('-c', "--ckptdir", type=str, required = True, help='Checkpoint folder containing multiple pretrained models.')
    parser.add_argument('-dc', "--decCkptdir", type=str, required = True, help='Checkpoint folder containing multiple decoder models.')
    parser.add_argument('-t', "--target_rate", type=float, nargs='+', default=[0.75,0.50,0.25,0.12,0.06], help="Target bpp (default: %(default)s)")
    parser.add_argument('-cfg', '--cfgPath', type=str, help='config (recipe) file path.', default="Encoder/AllRecipes.json")
    parser.add_argument('-a', '--anchorResultPath', type=str, help='Anchor Result file.', default="Encoder/Anchor.txt")
    parser.add_argument('-n', '--newCfgPath', type=str, help='output config (recipe) file path.', default="Encoder/newAllRecipes.json")
    
    args = parser.parse_args(argv)
    return args

def Runner (argv):
    args = parse_args(argv[1:])
    #inputImagePath, modelDir = "ModelDirectory",decModelPath = "ModelDirectory", cfgPath = "Encoder/AllRecipes.json", AnchorResultPath = "Encoder/Anchor.txt"
    if args.input == None and args.inputPath == None:
        print("either the input image name or the folder path needs to be provided.")
        return []
    args = parse_args(argv[1:])
    if args.inputPath:
        images = glob.glob(os.path.join(args.inputPath, "*.png"))
        if images == []:
            print("No files found found in the images directory: ",args.inputPath)
            return -1
        images.sort()
    else:
        images = [args.input]

    process_list = []
    for _, im in enumerate(images):
        process_list.append({'image': im})
    MD = ModeDecider(args.ckptdir,args.decCkptdir,args.cfgPath,args.anchorResultPath)
    for _, list_entry in enumerate(process_list):
        #with suppress_stdout():
        MD.PreStoreCandidates(list_entry["image"],filename = list_entry["image"].split("/")[-1].split(".")[0])


def Runner2 (argv):
    args = parse_args(argv[1:])
    #inputImagePath, modelDir = "ModelDirectory",decModelPath = "ModelDirectory", cfgPath = "Encoder/AllRecipes.json", AnchorResultPath = "Encoder/Anchor.txt"
    if args.input == None and args.inputPath == None:
        print("either the input image name or the folder path needs to be provided.")
        return []
    args = parse_args(argv[1:])
    if args.inputPath:
        images = glob.glob(os.path.join(args.inputPath, "*.png"))
        if images == []:
            print("No files found found in the images directory: ",args.inputPath)
            return -1
        images.sort()
    else:
        images = [args.input]

    process_list = []
    for idx, im in enumerate(images):
        process_list.append({'image': im})
    MD = ModeDecider(args.ckptdir,args.decCkptdir,args.cfgPath,args.anchorResultPath)
    recipesList = [
        '00001_TE_1192x832_8bit_sRGB',   
        '00002_TE_1280x848_8bit_sRGB',   
        '00003_TE_3032x1856_8bit_sRGB',  
        '00004_TE_1920x1080_8bit_sRGB',  
        '00005_TE_3680x2456_8bit_sRGB',  
        '00006_TE_2192x1520_8bit_sRGB',  
        '00007_TE_1248x832_8bit_sRGB',
        '00008_TE_2464x1640_8bit_sRGB',
        '00009_TE_1536x1024_8bit_sRGB',
        '00010_TE_1984x1320_8bit_sRGB',
        '00011_TE_1784x1296_8bit_sRGB',
        '00012_TE_3680x2456_8bit_sRGB',
        '00013_TE_800x1200_8bit_sRGB',
        '00014_TE_2000x2496_8bit_sRGB',
        '00015_TE_976x1472_8bit_sRGB',
        '00016_TE_560x888_8bit_sRGB',
        '00017_TE_1752x1856_8bit_sRGB',
        '00018_TE_7680x5120_8bit_sRGB',
        '00019_TE_2120x1608_8bit_sRGB',
        '00020_TE_1072x928_8bit_sRGB',
        '00021_TE_2048x1080_8bit_sRGB',
    ]
    AllRecipes = []
    
    for idx, list_entry in enumerate(process_list):
        #with suppress_stdout():
        if os.path.exists('Encoder/prestoredCandidates/' + recipesList[idx]):
            MD.AllCandidates = torch.load('Encoder/prestoredCandidates/' + recipesList[idx])
            _, metricList_best, recipeList_best = MD.FindBestCombination()
            AllRecipes.extend(recipeList_best)
            for metrics in metricList_best:
                print(
                    f"{metrics['image']} {metrics['q']} {metrics['bpp']:.5f} "
                    f"{metrics['y_msssim']:.8f} {metrics['psnr']['y']:.8f} {metrics['psnr']['u']:.8f} {metrics['psnr']['v']:.8f} "
                    f"{metrics['vif']:.8f} {metrics['fsim']:.8f} {metrics['nlpd']:.8f} {metrics['iw_ssim']:.8f} {metrics['vmaf']:.8f} {metrics['psnr_vhsm']:.8f} "
                    f"{metrics['dec_mem']:0.2f}"
                )
        else:
            for index in [0,1,2,3,4]:
                qualityMapped = 5 - index if index > 0 else 6
                anchorRecipe = construct_weights("Encoder/AllRecipesNewx64.json", MD.quality2rate[qualityMapped],list_entry['image'].split("/")[-1],0,0)
                AllRecipes.extend([anchorRecipe])
        with open(args.newCfgPath, "w") as outfile:
            json.dump(AllRecipes, outfile)


# Usage examples:
# Do the mode decision for all images in cfp-test-set/ folder (using all 5 rate points by default):
# python3 Encoder/ModeDecision3.py -c ModelDirectory -dc DecModelDirectory -ip cfp-test-set/
# Do the mode decision for a single image (using all 5 rate points by default): 
# python3 Encoder/ModeDecision3.py -c ModelDirectory -dc DecModelDirectory -i cfp-test-set/00001_TE_1192x832_8bit_sRGB.png
# Do the mode decision for a single image using single target rate point: 
# python3 Encoder/ModeDecision3.py -c ModelDirectory -dc DecModelDirectory -i cfp-test-set/00001_TE_1192x832_8bit_sRGB.png -t 0.75
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    if False: #If you are running this locally, download the encoder and decoder models in two folders, also download the input images. The below code does that for me. 
        os.system("mkdir ModelDirectory")
        os.system("hdfs dfs -copyToLocal hdfs://harunava/home/byte_videoarch_AIIC/quantFinetune/* ModelDirectory/")
        os.system("python3 Common/testfunc/util.py ModelDirectory DecModelDirectory")
        os.system("hdfs dfs -copyToLocal " + "hdfs://harunava/home/byte_videoarch_AIIC/cfp-test-set/" + " .")
        os.system("mkdir reconstructed")
        os.system("chmod +x Metric/metric_tool/vmaf.linux")
    Runner2(sys.argv)

#python3 Encoder/ModeDecision3.py -c ModelDirectory -dc ModelDirectory -ip cfp-test-set