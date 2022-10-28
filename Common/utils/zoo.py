# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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