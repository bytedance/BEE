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
import six
import lmdb
import random

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.utils.data as data
from torchvision import transforms


def convert_raw(x):
    norm_params = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
    mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
    std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
    return x * std + mean


def get_dataset(is_training, patch_size, sets_path):
    datasets_list = []
    print(sets_path)
    transform = get_transform(is_training, patch_size)
    for set_path in sets_path:
        datasets_list.append(
            ImgLMDBDataset(
                db_path=set_path,
                transform=transform
            )
        )
    return data.ConcatDataset(datasets_list)


def get_transform(is_training, patch_size):
    if is_training:
        return transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
        ])


class ImgLMDBDataset(data.Dataset):
    def __init__(self, db_path, transform, shuffle=False):
        self.db_path = db_path
        self.env = lmdb.open(db_path,
                             max_readers=1,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        self.txn = self.env.begin(write=False)
        self.nSamples = int(self.txn.get('num-samples'.encode()))
        self.indices = range(self.nSamples)
        if shuffle:
            random.shuffle(self.indices)
        self.transform = transform

    def __getitem__(self, index):
        imgKey = 'image-{:0>9}'.format(self.indices[index] + 1)
        imageBin = self.txn.get(imgKey.encode())
        buf = six.BytesIO()
        buf.write(imageBin)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return self.nSamples

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
