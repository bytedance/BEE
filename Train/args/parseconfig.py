# Copyright 2020 Bytedance, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import configparser
import sys


def ArgsAnalyse(argv):
    conf_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
        )
    conf_parser.add_argument("-c", "--conf_file", help="Specify TrainConfig file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args(argv)

    defaults = {}

    if args.conf_file:
        config = configparser.SafeConfigParser()
        config.optionxform=str
        config.read([args.conf_file])
        defaults.update(dict(config.items("Defaults")))
    defaults['conf_file'] = args.conf_file

    parser = argparse.ArgumentParser(
        parents=[conf_parser]
        )
    parser.add_argument(
        "-m",
        "--model",
        help = "Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2, 3],
        help='Training stage (default: %(default)s)')
    parser.add_argument(
        "-e",
        "--epochs",
        type = int,
        help = "Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type = float,
        help = "Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        type = int,
        help = "Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lmbda",
        dest = "lmbda",
        type = float,
        help = "Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size", type = int, help = "Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test_batch_size",
        type = int,
        help = "Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux_learning_rate",
        type = float,
        help = "Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch_size",
        type = int,
        help = "Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--quality", type = int, help = "quality of the model(default: %(default)s)")
    parser.add_argument("--cuda", action = "store_true", help = "Use cuda")
    parser.add_argument(
        "--save", action = "store_true", help = "Save model to disk"
    )
    parser.add_argument(
        "--fineTune", action = "store_true", help = "Save model to disk"
    )
    parser.add_argument(
        "--loadmodel", action = "store_true", help = "load model from checkpoint"
    )
    parser.add_argument("--checkpoint", type = str, help = "Path to a checkpoint")

    parser.add_argument(
        "--seed", type = float, help = "Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        type = float,
        help = "gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--metric",
        help = "distortion metric",
    )
    parser.add_argument(
        "--PretrainModel",
        help = "Directory of the PretrainModel",
    )
    parser.add_argument(
        "--InitModel",
        help="Pretrained model used to initialize stage-2, stage-3 or stage-4 before training.",
    )
    parser.add_argument('--hdfs_savepath', type=str, default=None, help="HDFS save path.")
    parser.add_argument('--hdfs_trainpath', type=str, default=None, help='Training data path on HDFS')
    parser.add_argument('--hdfs_evalpath', type=str, default=None, help='Evaluation data path on HDFS')
    parser.add_argument("--k_msssim", type = float)
    parser.add_argument("--k_mse", type = float)
    parser.add_argument("--k_nlpd", type = float)
    parser.add_argument("--k_fsim", type = float)
    parser.add_argument("--k_vif", type = float)
    parser.add_argument("--k_iwssim", type = float)
    parser.add_argument("--trainpath",
        help="hdfs path to training dataset",
    )
    parser.add_argument("--evalpath",
        help="hdfs path to evalution dataset",
    )
    parser.add_argument(
        "--YUV",
        help = "weight of the loss on different color channel",
    )
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)
    args.loadmodel = True if args.loadmodel in {'True', 'Yes', True} else False
    args.save = True if args.save in {'True', 'Yes', True} else False
    args.cuda = True if args.cuda in {'True', 'Yes', True} else False
    args.hdfs_savepath = None if args.hdfs_savepath in {None, 'None'} else args.hdfs_savepath
    args.hdfs_trainpath = None if args.hdfs_trainpath in {None, 'None'} else args.hdfs_trainpath
    args.hdfs_evalpath = None if args.hdfs_evalpath in {None, 'None'} else args.hdfs_evalpath
    args.checkpoint = None if args.checkpoint in {None, 'None'} else args.checkpoint
    args.PretrainModel = None if args.PretrainModel in {None, 'None'} else args.PretrainModel
    args.fineTune = False if args.fineTune in {False, 'False', None, 'None'} else True
    args.InitModel = None if args.InitModel in {None, 'None'} else args.InitModel
    print(args)
    return args


if __name__ == "__main__":
    ArgsAnalyse(sys.argv[1:])
