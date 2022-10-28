import os, glob
import torch
import argparse
from copy import deepcopy
from os import listdir
from os.path import isfile, join
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

    return summ, bdrate_compare
class ModeDecider():
    model = "quantyuv444-decoupled"
    metric = "mse"
    coder = Common.available_entropy_coders()[0]
    rate2quality = {0.75: 6, 0.5: 4, 0.25: 3, 0.12: 2, 0.06: 1}
    quality2rate = {6: 0.75, 4: 0.5, 3: 0.25, 2: 0.12, 1: 0.06}
    pruneLossThreshold = 0.001
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

    def populateCandidateList(self,image,rate, candidates,minRate):
        bitstream = "str.bin"
        net = None
        recipe_init = self.anchorRecipe
        recipeShortList = []
        reconstructedShortList = []
        metricsShortList = []
        lossShortList = []
        for _, cand in enumerate(candidates): 
            recipe = deepcopy(recipe_init)
            if cand[0]<3:
                recipe['filterCoeffs'][cand[0]]['thr'] += cand[1]
                recipe['filterCoeffs'][cand[0]]['scale'][0] += cand[2]
                recipe['filterCoeffs'][cand[0]]['thr'] = round(recipe['filterCoeffs'][cand[0]]['thr']*100)/100
                recipe['filterCoeffs'][cand[0]]['scale'][0] = round(recipe['filterCoeffs'][cand[0]]['scale'][0]*100)/100
            else:
                recipe['filterCoeffs'][cand[0]]['scale'][0] *= cand[1]
                recipe['filterCoeffs'][cand[0]]['scale'][1] *= cand[2]
                recipe['filterCoeffs'][cand[0]]['scale'][0] = round(recipe['filterCoeffs'][cand[0]]['scale'][0]*100000)/100000
                recipe['filterCoeffs'][cand[0]]['scale'][1] = round(recipe['filterCoeffs'][cand[0]]['scale'][1]*100000)/100000
            _, enctime, net = enc_adap(image, self.model, self.metric, self.coder, bitstream, self.ModelList[recipe['model_idx']], recipe['numRefIte'], \
                None, net, recipe['resized_size'], rate, recipe=recipe)
            if cand[0]<3:
                net.ACSkip = True
                net.encSkip = True


            pixelnum = self.h * self.w
            binsize = os.path.getsize(bitstream) * 8
            bpp = binsize / pixelnum
            if bpp > self.higherRateLimit*rate or bpp < min(minRate,recipe['rate']):
                continue
            
            torch.cuda.reset_max_memory_allocated()
            dec_time, _ = dec_adap(bitstream, self.coder, self.DecModelList, net = net, output = "reco.png")

            if cand[0]>3:
                net.encCompleteSkip = True
                net.decCompleteSkip = True
            recoIm,_,_ = readPngToTorchIm("reco.png")
            metrics = cm(recoIm, self.x_true_org, bitstream, recipe["model_idx"], " ",skip=self.skipInPrune)
            metrics['enc'] = enctime; metrics['dec'] = dec_time; metrics['image'] = image.split("/")[-1]
            metrics['dec_mem'] = torch.cuda.max_memory_allocated('cuda')/1000000000
            loss = calAverageBdRate (self.AnchorResultPath, metrics,self.quality_mapped,skip=self.skipInPrune)

            recipeShortList.append(recipe)
            reconstructedShortList.append(recoIm)
            metricsShortList.append(metrics)
            lossShortList.append(loss)
        return list(zip(reconstructedShortList,metricsShortList,recipeShortList,lossShortList))

    def pruneAndDecide(self, candidates = [], minRate = 0):
        metrics_best = self.anchorMetrics
        recipe_best = self.anchorRecipe
        if len(candidates) == 0:
            return metrics_best, recipe_best
        bitstream = "str.bin"
        candidates.sort(key = lambda x: x[3])
        validEntries = []
        minLoss = min([ i for _,_,_,i in candidates ])
        minLoss = 0 if minLoss > 0 else minLoss
        for i,cand in enumerate(candidates):
            _,m,recipe,loss = cand
            if len(validEntries) >= self.maxPruneCandidate or loss > minLoss + self.pruneLossThreshold:
                continue
            if m['bpp']>= (recipe['rate'] if recipe['rate']> 0.06 else max(recipe['rate'],minRate)):
                validEntries.append(i)
        candidates = [candidates[i] for i in validEntries]
        #Third stage: calculate full metrics and decide

        for i,cand in enumerate(candidates):
            recoIm,m,recipe,loss = cand
            if 'ChannelOffsets' in recipe:
                del recipe['ChannelOffsets'] 

            metrics = cm(recoIm, self.x_true_org, bitstream, recipe["model_idx"], " ",skip = self.skipIn2ndStage)
            for j in self.skipInPrune:
                 m[j] = metrics[j]
            metrics = m
            # 5. calculate loss
            loss = calAverageBdRate (self.AnchorResultPath, metrics,self.quality_mapped)
            index = 4 - (4 if self.quality_mapped == 6 else self.quality_mapped - 1)

            self.AllCandidates[index].append((self.quality_mapped,deepcopy(metrics),deepcopy(recipe)))

            # 6. update the best recipe
            if loss < self.loss_best:
                self.loss_best = loss
                metrics_best = deepcopy(metrics)
                recipe_best = deepcopy(recipe)

        self.anchorMetrics = metrics_best
        self.anchorRecipe = recipe_best
        return metrics_best, recipe_best
    
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


    def ModeDecision(self,image,rate,reset = False):

        global times
        encodingStart = time.time()
        self.quality_mapped = self.rate2quality[rate]
        self.anchorMetrics = calculateAnchorMetric(self.AnchorResultPath, image.split("/")[-1],self.quality_mapped)
        self.x_true_org,_,_ = readPngToTorchIm(image)
        _, _, self.h, self.w = self.x_true_org.shape
        self.anchorRecipe = construct_weights(self.cfgPath, rate,image.split("/")[-1],self.h,self.w)


        if reset:
            self.AllCandidates = [[]]*5
            for index in [0,1,2,3,4]:
                qualityMapped = 5 - index if index > 0 else 6
                anchorRecipe = construct_weights(self.cfgPath, self.quality2rate[qualityMapped],image.split("/")[-1],self.h,self.w)
                anchorMetrics = calculateAnchorMetric(self.AnchorResultPath, image.split("/")[-1],qualityMapped)
                self.AllCandidates[index] =[(qualityMapped,deepcopy(anchorMetrics),deepcopy(anchorRecipe))]
        #first stage: populate metrics without computing vmaf and fsim (that take the most time):
        self.loss_best = 0
        self.skipInPrune = []
        minBpp = self.anchorMetrics['bpp']
        self.skipInPrune = []
        self.skipIn2ndStage = ['psnr_hvsm','nlpd','vif','y_msssim','iw_ssim','fsim','vmaf',]
        candidates = [
            [0,0.0,-0.100],
            [0,0.0,-0.050],
            [0,0.0,-0.025],
            [0,0.0,0.025],
            [0,0.0,0.050],
            [0,0.0,0.100], 
            [0,0.0,0.150],
            [0,0.0,0.200],
        ]
        candidates = self.populateCandidateList(image,rate, candidates,minBpp)
        metrics_best, recipe_best = self.pruneAndDecide(candidates,minBpp)
        candidates = [
            [0,0.100,0.0],
            [0,0.050,0.0],
            [0,-0.050,0.0],
            [0,-0.100,0.0], 
            [0,-0.150,0.0],
            [0,-0.200,0.0], 
            [0,-0.250,0.0], 
        ]
        candidates = self.populateCandidateList(image,rate, candidates,minBpp)
        metrics_best, recipe_best = self.pruneAndDecide(candidates,minBpp)
        candidates = [
            [2,0.0,-0.150],
            [2,0.0,-0.100],
            [2,0.0,-0.050],
            [2,0.0,0.050],
            [2,0.0,0.100], 
            [2,0.0,0.150],
            [2,0.0,0.200],
        ]
        candidates = self.populateCandidateList(image,rate, candidates,minBpp)
        metrics_best, recipe_best = self.pruneAndDecide(candidates,minBpp)
        candidates = [
            [2,0.200,0.0],
            [2,0.150,0.0],
            [2,0.100,0.0],
            [2,0.050,0.0],
            [2,-0.050,0.0],
            [2,-0.100,0.0], 
            [2,-0.150,0.0],
            [2,-0.200,0.0],
        ]
        candidates = self.populateCandidateList(image,rate, candidates,minBpp)
        metrics_best, recipe_best = self.pruneAndDecide(candidates,minBpp)
        candidates = [
            [1,0.0,-0.100],
            [1,0.0,-0.050],
            [1,0.0,0.050],
            [1,0.0,0.100], 
            [1,0.0,0.150],
            [1,0.0,0.200],
            [1,0.0,0.250], 
        ]
        candidates = self.populateCandidateList(image,rate, candidates,minBpp)
        metrics_best, recipe_best = self.pruneAndDecide(candidates,minBpp)
        candidates = [
            [1,0.300,0.0],
            [1,0.200,0.0],
            [1,0.100,0.0],
            [1,-0.100,0.0],
            [1,-0.200,0.0], 
            [1,-0.300,0.0],
            [1,-0.400,0.0], 
        ]
        candidates = self.populateCandidateList(image,rate, candidates,minBpp)
        metrics_best, recipe_best = self.pruneAndDecide(candidates,minBpp)
        metrics_best['enc'] = time.time() - encodingStart
        return metrics_best, recipe_best


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
    for im in images:
        for rate in args.target_rate:
            process_list.append({'image': im, 'rate':rate})
    AllRecipes = []
    MD = ModeDecider(args.ckptdir,args.decCkptdir,args.cfgPath,args.anchorResultPath)
    count = 0
    enctimes = []
    for idx, list_entry in enumerate(process_list):
        with suppress_stdout():
            metrics,recipe_best = MD.ModeDecision(list_entry["image"],list_entry["rate"],count==0)
        enctimes.append(metrics['enc'])
        if count == 4:
            _, metricList_best, recipeList_best = MD.FindBestCombination()
            AllRecipes.extend(recipeList_best)
            for enc,metrics in zip(enctimes,metricList_best):
                for idx,j in enumerate(metrics):
                    if isinstance(j,float):
                        if idx == 13:
                            print(f"{enc:.2f}", end =" ")
                        else:
                            print(f"{j:.6f}", end =" ")
                    else:
                        print(j, end =" ")
                print()
            count = 0
            enctimes = []
        else:
            count += 1
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
    if False: #If you are running this locally, download the encoder and decoder models in two folders, also download the input images. The below code does that for me. 
        os.system("mkdir ModelDirectory")
        os.system("hdfs dfs -copyToLocal hdfs://harunava/home/byte_videoarch_AIIC/quantFinetune/* ModelDirectory/")
        os.system("python3 Common/testfunc/util.py ModelDirectory DecModelDirectory")
        os.system("hdfs dfs -copyToLocal " + "hdfs://harunava/home/byte_videoarch_AIIC/cfp-test-set/" + " .")
    Runner(sys.argv)

#python3 Encoder/ModeDecision2.py -c ModelDirectory -dc ModelDirectory -ip cfp-test-set