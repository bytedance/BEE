import sys
import argparse
from os.path import abspath, join, dirname, pardir

sys.path.append(join(abspath(dirname(__file__)),pardir))
import glob, os
from pathlib import Path
import torch
from Common.utils.zoo import models
from Common.utils.testfuncs import  print_args, get_device, get_model_list
from Common.model_engine import ModelEngine
torch.backends.cudnn.deterministic = True


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Decoder arguments parser.")
    parser.add_argument('-i', '--input', type=str, default='str.bin', help='Input bin file path.')
    parser.add_argument('-o', '--output', type=str, default='rec.png', help='Decoded image file save path.')
    parser.add_argument('--binpath', type=str, default=None, help="Bitstream path.")
    parser.add_argument('--recpath', type=str, default="reconstructed", help="Reconstructed path where the decoded images are to be saved.")
    parser.add_argument("--ckptdir", type=str, default="DecModelDirectory", help='Checkpoint folder containing multiple pretrained models.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='CPU or GPU device.')
    args = parser.parse_args(argv)
    return args


def decode(args):
    #Single bitstream decoding
    device = get_device(args.device)
    model_list = get_model_list(args.ckptdir)
    dec_engine = ModelEngine()
    dec_engine.decode(args.i, args.o, model_list, device)
    return 


def decodes(args):
    #Traversal bitstream decoding
    suffixset = ['.bin','.bits']
    for suffix in suffixset:
        files = glob.glob(os.path.join(args.binpath, "*"+suffix))
        if len(files):
            break
    if files == []:
        print("No files found found in the bitstream directory: ",args.binpath)
    files.sort()
    device = get_device(args.device)
    model_list = get_model_list(args.ckptdir)
    for k, bits in enumerate(files):
        output = os.path.join(args.recpath, os.path.basename(bits).replace(suffix, '.png'))
        print("reconstructed picture:", output)
        with Path(bits).open("rb") as f:
            dec_engine = ModelEngine()
            dec_engine.decode(bits, output, model_list, device)
    return 


def main(argv):
    args = parse_args(argv[1:])
    print_args(args)
    os.system("mkdir "+args.recpath)
    torch.set_num_threads(1)  # just to be sure
    if args.binpath is not None:
        decodes(args)
    else:
        decode(args)


if __name__ == '__main__':
    main(sys.argv)
