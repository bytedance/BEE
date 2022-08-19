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

import os
import sys
import numpy as np
import re
import torch
from Train.train_YUV444_arnold_comb import main
from Train.args import ArgsAnalyse


def check_exist(path_to_hdfs):
    command = 'hdfs dfs -ls {}'.format(path_to_hdfs)
    r = os.popen(command)
    info = r.readlines()
    best_exist, second_best_exist = False, False
    for i in info:
        if 'second_best.pth' in i.strip():
            second_best_exist = True
        elif 'best.pth' in i.strip():
            best_exist = True
    return best_exist, second_best_exist


def mkdir_hdfs(PATH, exist_ok=False):
    command = 'hdfs dfs -ls {}'.format(PATH)
    r = os.popen(command)
    info = r.readlines()
    if info == []:
        command = 'hdfs dfs -mkdir -p {}'.format(PATH)
        print('Command is excute:{}'.format(command))
        os.system(command)
        return
    for i in info:
        if 'File exists' in i:
            if exist_ok:
                return
            else:
                raise FileExistsError


def modelzoo(PATH):
    command = 'hdfs dfs -ls {}'.format(PATH)
    r = os.popen(command)
    info = r.readlines()
    ckpt = []
    epoch = []
    for i in info:
        if 'hdfs://' in i:
            exist = re.search("\d+.pth", i.strip().split()[-1])
            if exist:
                epoch.append(int(exist.group()[:-4]))
                ckpt.append(i.strip().split()[-1])
    if len(ckpt) == 0:
        return 0
    idx = np.argmax(np.array(epoch))
    print('Downloading model : {}'.format(ckpt[idx]))
    command = 'hdfs dfs -copyToLocal {} .'.format(ckpt[idx])
    os.system(command)
    print('Model {} is downloaded!'.format(ckpt[idx]))
    return os.path.basename(ckpt[idx])


def prepare_checkpoints(args):
    # Copy training data and evaluation data
    os.system(f"hdfs dfs -copyToLocal {args.hdfs_trainpath} .")
    os.system(f"hdfs dfs -copyToLocal {args.hdfs_evalpath} .")
    args.trainpath = os.path.basename(args.hdfs_trainpath)
    args.evalpath = os.path.basename(args.hdfs_evalpath)

    if args.InitModel:
        os.system(f"hdfs dfs -copyToLocal {args.InitModel} ./init_model.pth")
        args.InitModel = "init_model.pth"

    # Copy pretrained model from HDFS
    if args.PretrainModel:
        os.system(f"hdfs dfs -copyToLocal {args.PretrainModel} .")
        args.PretrainModel = os.path.basename(args.PretrainModel)
    else:
        # Search for the latest checkpoint on HDFS
        mkdir_hdfs(args.hdfs_savepath, True)
        epoch0, epoch1, epoch2 = -1, -1, -1
        model_with_largest_epoch = modelzoo(args.hdfs_savepath)
        if model_with_largest_epoch:
            epoch0 = torch.load(model_with_largest_epoch, map_location='cpu')['epoch']
        best_exist, second_best_exist = check_exist(args.hdfs_savepath)
        if best_exist:
            os.system(f"hdfs dfs -copyToLocal {os.path.join(args.hdfs_savepath, 'best.pth')} .")
            epoch1 = torch.load('best.pth', map_location='cpu')['epoch']
        if second_best_exist:
            os.system(f"hdfs dfs -copyToLocal {os.path.join(args.hdfs_savepath, 'second_best.pth')} .")
            epoch2 = torch.load('second_best.pth', map_location='cpu')['epoch']
        if np.max([epoch0, epoch1, epoch2]) >= 0:
            pmodels = [model_with_largest_epoch, 'best.pth', 'second_best.pth']
            idx = np.argmax([epoch0, epoch1, epoch2])
            args.PretrainModel = pmodels[idx]
    return args


if __name__ == '__main__':

    args = ArgsAnalyse(sys.argv[1:])
    if args.hdfs_savepath:
        args = prepare_checkpoints(args)
    for k, v in vars(args).items():
        print(f"{k:<18}: {v}")

    main(args)
    print("Training is finished!")
