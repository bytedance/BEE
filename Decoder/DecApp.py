import sys
import argparse
from os.path import abspath, join, dirname, pardir
sys.path.append(join(abspath(dirname(__file__)),pardir))
import glob, os
from pathlib import Path
import torch
import Common
from Common.utils.zoo import models
from Common.utils.testfuncs import dec_adap, print_args, get_device, get_model_list

torch.backends.cudnn.deterministic = True

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Decoder arguments parser.")
    parser.add_argument('-i', '--input', type=str, default='str.bin', help='Input bin file path.')
    parser.add_argument('-o', '--output', type=str, default='rec.png', help='Decoded image file save path.')
    parser.add_argument('--binpath', type=str, default=None, help="Bitstream path.")
    parser.add_argument('--recpath', type=str, default="reconstructed", help="Reconstructed path where the decoded images are to be saved.")
    parser.add_argument("--ckptdir", type=str, default="DecModelDirectory", help='Checkpoint folder containing multiple pretrained models.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='CPU or GPU device.')
    parser.add_argument("--model", choices=models.keys(), default="quantyuv444-decoupled", help="NN model to use (default: %(default)s)")
    parser.add_argument("--metric", choices=["mse"], default="mse", help="metric trained against (default: %(default)s")
    parser.add_argument("--coder", choices=Common.available_entropy_coders(), default=Common.available_entropy_coders()[0], help="Entropy coder (default: %(default)s)")
    parser.add_argument("--rate_adap", type=int, default=1, choices=[0, 1], help="The bit stream is encoded with rate adaptation if it is 1.")
    parser.add_argument("--calcFlops", type=int, default=0, choices=[0, 1], help="Calculate the kMacs according to recommended model")
    parser.add_argument("--oldversion", action='store_true', help="flag to decompress bitstream of old version")
    args = parser.parse_args(argv)
    return args


def decode(args):
    device = get_device(args.device)
    model_list = get_model_list(args.ckptdir)
    if args.rate_adap:
        dec_time, _ = dec_adap(args.input, args.coder, model_list, args.output, None, device, args.calcFlops, oldversion = args.oldversion)
    else:
        raise NotImplementedError
    return dec_time


def decodes(args):
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
    dec_times = []
    for k, f in enumerate(files):
        output = os.path.join(args.recpath, os.path.basename(f).replace(suffix, '.png'))
        print("reconstructed picture:", output)
        if args.rate_adap:
            dec_time, _ = dec_adap(f, args.coder, model_list, output, None, device, args.calcFlops, args.oldversion)
        else:
            raise NotImplementedError
        dec_times.append(dec_time)
    return dec_times


def main(argv):
    args = parse_args(argv[1:])
    print_args(args)
    os.system("mkdir "+args.recpath)
    torch.set_num_threads(1)  # just to be sure
    if args.binpath is not None:
        dec_times = decodes(args)
    else:
        dec_time = decode(args)


if __name__ == '__main__':
    main(sys.argv)
