
import sys
from os.path import abspath, join, dirname, pardir
import argparse
sys.path.append(join(abspath(dirname(__file__)), pardir))
import torch.nn as nn
import torch

from Train.datasets.dataIO import get_dataset
from Common.utils.csc import RGB2YCbCr
import numpy as np
import os
from Common.utils.zoo import models
from Common.models.quantmodule import quantHSS


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "--model",
        type = str,
        help = "Model dir",
    )
    parser.add_argument(
        "--saver",
        type = str,
        help = "dir to save model",
    )
    parser.add_argument(
        "--data",
        type = str,
        help = "dir of the data",
        default = 'tempdir/lmdbdata/jpegai256x256_lmdb'
    )
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    testpath = args.data
    loaderkwargs = {'num_workers': 8}
    print('Testing dataset is:')
    test_set = get_dataset(is_training=False, patch_size=256, sets_path=[testpath])
    test_dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, **loaderkwargs)
    ckpt = args.model
    saverdir = args.saver
    quantmodule = quantHSS(192, 192, vbit = 16, cbit = 16, use_float = False).cuda()
    net1 = models['quantyuv444-decoupled'](quality = 1, Quantized=False)
    checkpoint = torch.load(ckpt)
    if "state_dict" in checkpoint.keys():
        net1.load_state_dict(checkpoint["state_dict"])
    else:
        net1.load_state_dict(checkpoint)
    net1.cuda()
    net1.eval()
    quantmodule.loadmodel(net1.h_s_scale)
    RGB2YUV = RGB2YCbCr()
    quantmodule.train()
    for i, data in enumerate(test_dataloader):
        if (i + 1) % 5000 == 0:
            print("Progress: {}/{}".format(i+1, len(test_dataloader)))
        with torch.no_grad():
            d = data.cuda()
            yuvd = RGB2YUV(d)
            _, _, H, W = yuvd.shape
            H_pad, W_pad = int(np.ceil(H / 64) * 64), int(np.ceil(W / 64) * 64)
            pad_F1 = nn.ConstantPad2d((0, W_pad - W, 0, H_pad - H), 0)
            pad_F2 = nn.ConstantPad2d((0, W_pad - W, 0, H_pad - H), 0.5)
            pad_d = torch.cat([pad_F1(yuvd[:, 0:1, :, :]), pad_F2(yuvd[:, 1:3, :, :])], dim = 1)
            proxy_z_hat = net1.proxyfunc(pad_d)
            quantmodule(proxy_z_hat)
    print('Start Quantization')
    quantmodule.quantlayers()
    quantmodule.eval()
    net1.h_s_scale = quantmodule
    savename = 'quant_{}'.format(os.path.basename(ckpt))
    torch.save(net1.state_dict(), savename)
