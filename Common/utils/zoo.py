# Copyright 2020 InterDigital Communications, Inc.
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

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”). All Bytedance Modifications are Copyright 2022 Bytedance Inc.

from Common.models import (
    QuantYUV444Decoupled,
    YUV444DecoupledV0,
)

__all__ = ['models']

model_architectures = {
    'quantyuv444-decoupled': QuantYUV444Decoupled,
    'yuv444-decoupled-v0': YUV444DecoupledV0,
}

cfgs = {
    "quantyuv444-decoupled": {
        1: (192,),
        2: (192,),
        3: (192,),
        4: (192,),
        5: (192,),
        6: (192,),
        7: (192,),
        8: (192,),
        9: (192,),
        10: (192,),
        11: (192,),
        12: (192,),
        13: (192,),
        14: (192,),
        15: (192,),
        16: (192,),
        17: (192,),
        18: (192,),
        19: (192,),
        20: (192,),
        21: (192,),
        22: (192,),
        23: (192,),
        24: (192,),
        25: (192,),
        26: (192,),
        27: (192,),
        28: (192,),
        29: (192,),
        30: (192,),
        31: (192,),
        32: (192,),
    },
    "yuv444-decoupled-v0": {
        1: (192,),
        2: (192,),
        3: (192,),
        4: (192,),
        5: (192,),
        6: (192,),
        7: (192,),
        8: (192,),
        9: (192,),
        10: (192,),
        11: (192,),
        12: (192,),
        13: (192,),
        14: (192,),
        15: (192,),
        16: (192,),
        17: (192,),
        18: (192,),
        19: (192,),
        20: (192,),
        21: (192,),
        22: (192,),
        23: (192,),
        24: (192,),
        25: (192,),
        26: (192,),
        27: (192,),
        28: (192,),
        29: (192,),
        30: (192,),
        31: (192,),
        32: (192,),
    },
}


def _load_model(architecture, metric, quality, **kwargs):
    if architecture not in model_architectures:
        raise ValueError(f'Invalid architecture name "{architecture}"')

    if quality not in cfgs[architecture]:
        raise ValueError(f'Invalid quality value "{quality}"')

    model = model_architectures[architecture](*cfgs[architecture][quality], **kwargs)
    return model


def quantyuv444_decoupled(quality, metric="mse", **kwargs):
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 32:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 32)')

    return _load_model(
        "quantyuv444-decoupled", metric, quality, **kwargs
    )


def yuv444_decoupled_v0(quality, metric="mse", **kwargs):
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 32:
        raise ValueError(f'Invalid quality "{quality}", should be between (1,32)')

    return _load_model(
        "yuv444-decoupled-v0", metric, quality, **kwargs
    )

models = {
    'quantyuv444-decoupled': quantyuv444_decoupled,
    'yuv444-decoupled-v0': yuv444_decoupled_v0,
}