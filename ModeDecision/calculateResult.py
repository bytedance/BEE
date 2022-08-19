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

import argparse
import sys
from Metric.metric_tool import calculateMetrics as cm
from Common.utils.testfuncs import readPngToTorchIm
def parse_args(argv):
    parser = argparse.ArgumentParser(description="metric Calculator arguments parser.")

    parser.add_argument('--binpath', type=str, default="bitstreams", help="Bitstream path.")
    parser.add_argument('--recpath', type=str, default="reconstructed", help="Reconstructed path where the decoded images are to be saved.")
    parser.add_argument("--orgDir", type=str, default="cfp-test-set", help='Checkpoint folder containing multiple pretrained models.')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [join(args.binpath, f) for f in listdir(args.binpath) if isfile(join(args.binpath, f))]
    bitstreams = sorted(onlyfiles, reverse = False)
    onlyfiles = [join(args.recpath, f) for f in listdir(args.recpath) if isfile(join(args.recpath, f))]
    reco_images = sorted(onlyfiles, reverse = False)
    toPrint = []
    for idx,testing in enumerate(zip(reco_images,bitstreams)):
        orgImage = testing[0].split("BEE_")[-1].split("_8bit")[0]
        orgImage = join(args.orgDir,orgImage+"_8bit_sRGB.png")
        recoIm,_,_ = readPngToTorchIm(testing[0])
        x_true_org,_,_ = readPngToTorchIm(orgImage)
        metrics = cm(recoIm, x_true_org, testing[1], 0, " ")
        metrics['enc'] = 0
        metrics['dec'] = 0
        metrics['image'] = testing[0].split("BEE_")[-1].split("_8bit")[0]
        metrics['q'] = testing[0].split("BEE_")[-1].split("_")[-1].split(".png")[0]
        metrics['dec_mem'] = 0
        toPrint.insert(0,metrics)
        if len(toPrint) == 5:
            for metrics in toPrint:
                print(
                    f"{metrics['image']} {metrics['q']} {metrics['bpp']:.5f} "
                    f"{metrics['y_msssim']:.8f} {metrics['psnr']['y']:.8f} {metrics['psnr']['u']:.8f} {metrics['psnr']['v']:.8f} "
                    f"{metrics['vif']:.8f} {metrics['fsim']:.8f} {metrics['nlpd']:.8f} {metrics['iw_ssim']:.8f} {metrics['vmaf']:.8f} {metrics['psnr_hvsm']:.8f} {metrics['psnr_rgb']:.8f} "
                )
                toPrint=[]