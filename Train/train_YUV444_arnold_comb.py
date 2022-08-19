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

import math
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Metric.metric_tool.ssim import ms_ssim

from Common.utils.zoo import models
from Common.utils.csc import RGB2YCbCr, YCbCr2RGB, YUV420To444
from Metric.metric_tool import iwssimgpu, IW_SSIM
from Train.args import ArgsAnalyse
from Train.datasets.dataIO import get_dataset
import glob
import warnings
import os

warnings.filterwarnings("ignore")


def search_local_ckpt(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None
    Ckpt = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth')))
    if len(Ckpt) == 0:
        return None
    epochs = [torch.load(ckpt, map_location='cpu')['epoch'] for ckpt in Ckpt]
    idx = np.argmax(epochs)
    return Ckpt[idx]


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


def copy_to_hdfs(is_best, ckptfile, hdfs_savepath):
    if not is_best:
        os.system(f"hdfs dfs -copyFromLocal {ckptfile} {hdfs_savepath}")
    else:
        best_exist, second_best_exist = check_exist(hdfs_savepath)
        if not best_exist and not second_best_exist:
            os.system(f"hdfs dfs -copyFromLocal {ckptfile} {hdfs_savepath}")
            print(f"{ckptfile} is copied to {hdfs_savepath}.")
        elif best_exist and not second_best_exist:
            os.system(f"hdfs dfs -mv {os.path.join(hdfs_savepath, 'best.pth')} {os.path.join(hdfs_savepath, 'second_best.pth')}")
            os.system(f"hdfs dfs -copyFromLocal {ckptfile} {hdfs_savepath}")
            print(f"HDFS: best.pth is renamed to second_best.pth. {ckptfile} is copied to {hdfs_savepath}.")
        elif best_exist and second_best_exist:
            os.system(f"hdfs dfs -rm {os.path.join(hdfs_savepath, 'second_best.pth')}")
            os.system(f"hdfs dfs -mv {os.path.join(hdfs_savepath, 'best.pth')} {os.path.join(hdfs_savepath, 'second_best.pth')}")
            os.system(f"hdfs dfs -copyFromLocal {ckptfile} {hdfs_savepath}")
            print(f"HDFS: second_best.pth is removed, best.pth is renamed to second_best.pth. {ckptfile} is copied to {hdfs_savepath}.")


def init_net_stage2(model, init_model):
    state_dict = torch.load(init_model, map_location='cpu')
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    exclude = "h_s_scale"
    state_dict_new = state_dict.copy()
    for k in state_dict:
        if exclude in k:
            del state_dict_new[k]
    model.load_state_dict(state_dict_new)
    model.update(force=True)
    print(f"{init_model} loaded")
    return model


def init_net_stage3(model, init_model):
    state_dict = torch.load(init_model, map_location='cpu')
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.update(force=True)
    print(f"{init_model} loaded")
    return model


def init_net_stage4(model, init_model):
    state_dict = torch.load(init_model, map_location='cpu')
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.update(force=True)
    print(f"{init_model} loaded")
    return model


def init_net(model, init_model, stage):
    if stage == 2:
        new_net = init_net_stage2(model, init_model)
    elif stage == 3:
        new_net = init_net_stage3(model, init_model)
    elif stage == 4:
        new_net = init_net_stage4(model, init_model)
    return new_net


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, args, distortion='mse', device="cuda", YUVweight=[1, 1, 1]):
        super().__init__()
        lambdaMSEQualityMap = {1: 0.0016, 2: 0.0035, 3: 0.0050, 4: 0.0067, 5: 0.0100, 6: 0.0130, 7: 0.0200, 8: 0.0250,
                               9: 0.0483, 10: 0.0932, 11: 0.1366, 12: 0.1800, 13: 0.2500, 14: 0.3600, 15: 0.4300,
                               16: 0.5000}
        self.lmbda_mse = lambdaMSEQualityMap[args.quality]
        self.lmbda_msssim = self.lmbda_mse * 1275

        self.distortion = distortion
        self.mse = nn.MSELoss()
        self.iwssim = iwssimgpu(device=device, data_range=255.0)
        self.iwssim2 = IW_SSIM()
        self.colorswith = RGB2YCbCr()
        self.colorswith2 = YCbCr2RGB()
        self.YUVupsample = YUV420To444()
        self.ms_ssim = ms_ssim

        self.k_msssim = args.k_msssim
        self.k_mse = args.k_mse
        self.YUVweight = YUVweight

    def forward(self, output, target, Test=False, Testi=torch.Tensor([0])):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        # TODO The input of the loss is quite different, need to refine related code.
        if not Test:
            out["bpp_loss"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()
            )

            out["distortion_loss_Y"] = self.mse(output["x_hat"][:, 0:1], target[:, 0:1])
            out["distortion_loss_Cb"] = self.mse(output["x_hat"][:, 1:2], target[:, 1:2])
            out["distortion_loss_Cr"] = self.mse(output["x_hat"][:, 2:3], target[:, 2:3])
            out["distortion_mse"] = (self.YUVweight[0] * out["distortion_loss_Y"] +
                                     self.YUVweight[1] * out["distortion_loss_Cb"] +
                                     self.YUVweight[2] * out["distortion_loss_Cr"]) / sum(self.YUVweight)
            out["distortion_mse"] = out["distortion_mse"] * self.lmbda_mse * 255 ** 2
            recon_y = output["x_hat"][:, 0:1]
            ori_y = target[:, 0:1]
            if self.k_msssim == 0:
                out["distortion_msssim"] = out["distortion_mse"]
            else:
                out["distortion_msssim"] = torch.mean(
                    1 - self.ms_ssim(recon_y, ori_y, data_range=1.0)) * self.lmbda_msssim
            out["distortion_loss"] = out["distortion_msssim"] * self.k_msssim + out["distortion_mse"] * self.k_mse
            out["loss"] = out["bpp_loss"] + out["distortion_loss"]

        if Test:
            ori = target.mul(255).round()
            x_hat = output["x_hat"].clip(0, 1).mul(255).round()
            out["bpp_loss"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()
            )
            color_componet = ['Y', 'U', 'V']
            for i in range(3):
                c_output = x_hat[:, i:i + 1, :, :]
                c_target = ori[:, i:i + 1, :, :]
                out["{}_mse".format(color_componet[i])] = self.mse(c_output, c_target)
                out["{}_psnr".format(color_componet[i])] = 10 * torch.log10(
                    255.0 ** 2 / out["{}_mse".format(color_componet[i])])
                out["{}_msssim".format(color_componet[i])] = self.ms_ssim(c_target, c_output, data_range=255.0)
                out["{}_iwssim".format(color_componet[i])] = self.iwssim2.test(
                    c_output[0, 0, :, :].detach().cpu().numpy(),
                    c_target[0, 0, :, :].detach().cpu().numpy()).item()
            out["mse_loss"] = (self.YUVweight[0] * out["{}_mse".format(color_componet[0])] +
                               self.YUVweight[1] * out["{}_mse".format(color_componet[1])] +
                               self.YUVweight[2] * out["{}_mse".format(color_componet[2])]) / sum(self.YUVweight)
            out["loss"] = (self.lmbda_mse * out["mse_loss"] +
                           self.lmbda_msssim * (1 - out["Y_msssim"])) / 2 + out["bpp_loss"]

            rgb_ori = Testi.mul(255).round()
            rgb_hat = self.colorswith2(output["x_hat"]).clip(0, 1).mul(255).round()
            out['psnr'] = 10 * torch.log10(255.0 ** 2 / self.mse(rgb_hat, rgb_ori))
            yuv_hat = self.colorswith(rgb_hat / 255.0).clip(0, 1).mul(255).round()
            yuv_recon = ori
            for i in range(3):
                c_output = yuv_hat[:, i:i + 1, :, :]
                c_target = yuv_recon[:, i:i + 1, :, :]
                out["Final{}_mse".format(color_componet[i])] = self.mse(c_output, c_target)
                out["Final{}_psnr".format(color_componet[i])] = 10 * torch.log10(
                    255.0 ** 2 / out["Final{}_mse".format(color_componet[i])])
                out["Final{}_msssim".format(color_componet[i])] = self.ms_ssim(c_target, c_output, data_range=255.0)
                out["Final{}_iwssim".format(color_componet[i])] = self.iwssim2.test(
                    c_output[0, 0, :, :].detach().cpu().numpy(), c_target[0, 0, :, :].detach().cpu().numpy()).item()
        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    if args.fineTune:
        net.g_a.requires_grad = False
        net.h_a.requires_grad = False

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, fineTune = False):
    model.train()
    device = next(model.parameters()).device
    RGB2YUV = RGB2YCbCr()

    for i, data in enumerate(train_dataloader):
        with torch.no_grad():
            d = data.to(device)
            yuvd = RGB2YUV(d)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        if fineTune:
            out_net = model.forward_fineTune(yuvd)
        else:
            out_net = model(yuvd)

        out_criterion = criterion(out_net, yuvd)
        if not torch.isnan(out_criterion["loss"]):
            out_criterion["loss"].backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

        if (i + 1) % 5000 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{(i + 1) * len(data)}/{len(train_dataloader.dataset)}"
                f" ({100. * (i + 1) / len(train_dataloader):.0f}%)]"
                f'Loss: {out_criterion["loss"].item():.6f} |'
                f'distortion loss: {out_criterion["distortion_loss"].item():.6f} |'
                f'Bpp loss: {out_criterion["bpp_loss"].item():.4f} |'
                f"Aux loss: {aux_loss.item():.4f} |"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    model.update()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    psnr_loss = AverageMeter()
    cpsnr_loss = [AverageMeter() for _ in range(3)]
    ciwssim_loss = [AverageMeter() for _ in range(3)]
    cmsssim_loss = [AverageMeter() for _ in range(3)]
    cpsnr_loss2 = [AverageMeter() for _ in range(3)]
    ciwssim_loss2 = [AverageMeter() for _ in range(3)]
    cmsssim_loss2 = [AverageMeter() for _ in range(3)]
    aux_loss = AverageMeter()
    color_componet = ['Y', 'U', 'V']
    RGB2YUV = RGB2YCbCr()

    with torch.no_grad():
        for data in test_dataloader:
            d = data.to(device)
            yuvd = RGB2YUV(d)
            _, _, H, W = yuvd.shape
            H_pad, W_pad = int(np.ceil(H / 64) * 64), int(np.ceil(W / 64) * 64)
            pad_F1 = nn.ConstantPad2d((0, W_pad - W, 0, H_pad - H), 0)
            pad_F2 = nn.ConstantPad2d((0, W_pad - W, 0, H_pad - H), 0.5)
            pad_d = torch.cat([pad_F1(yuvd[:, 0:1, :, :]), pad_F2(yuvd[:, 1:3, :, :])], dim=1)
            out_net = model.evalEstimate(pad_d)

            out_net["x_hat"] = out_net["x_hat"][:, :, :H, :W]
            out_criterion = criterion(out_net, yuvd, Test=True, Testi=d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            psnr_loss.update(out_criterion["psnr"])
            for i in range(3):
                cpsnr_loss[i].update(out_criterion["{}_psnr".format(color_componet[i])])
                ciwssim_loss[i].update(out_criterion["{}_iwssim".format(color_componet[i])])
                cmsssim_loss[i].update(out_criterion["{}_msssim".format(color_componet[i])])
                cpsnr_loss2[i].update(out_criterion["Final{}_psnr".format(color_componet[i])])
                ciwssim_loss2[i].update(out_criterion["Final{}_iwssim".format(color_componet[i])])
                cmsssim_loss2[i].update(out_criterion["Final{}_msssim".format(color_componet[i])])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"Loss: {loss.avg:.6f} |"
        f"RGB PSNR: {psnr_loss.avg:.4f} |"
        f"Bpp loss: {bpp_loss.avg:.4f} |"
        f"Aux loss: {aux_loss.avg:.4f} |"
    )
    for i in range(3):
        print(
            f"{color_componet[i]} PSNR: {cpsnr_loss[i].avg:.4f} |"
            f"{color_componet[i]} PSNR after Conversion: {cpsnr_loss2[i].avg:.4f} |"
            f"{color_componet[i]} IWSSIM : {ciwssim_loss[i].avg:.6f} |"
            f"{color_componet[i]} IWSSIM after Conversion: {ciwssim_loss2[i].avg:.6f} |"
            f"{color_componet[i]} MSSSIM : {cmsssim_loss[i].avg:.6f} |"
            f"{color_componet[i]} MSSSIM after Conversion: {cmsssim_loss2[i].avg:.6f} |"
        )
    return loss.avg


def main(args):

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # Dataset IO
    trainpath = args.trainpath
    testpath = args.evalpath

    loaderkwargs = {'num_workers': 8}
    print('Training dataset is:')
    train_set = get_dataset(is_training=True, patch_size=args.patch_size, sets_path=[trainpath])
    train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, **loaderkwargs)
    print('Evaluation dataset is:')
    test_set = get_dataset(is_training=False, patch_size=args.patch_size, sets_path=[testpath])
    test_dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, **loaderkwargs)

    # Initialize model
    net = models[args.model](quality=args.quality, metric=args.metric, decoderOnly=False, device=device, Quantized=False)
    CkptsDir = args.checkpoint
    os.makedirs(CkptsDir, exist_ok=True)
    ckpt = search_local_ckpt(CkptsDir)
    if args.PretrainModel:
        print("Loading {}".format(args.PretrainModel))
        checkpoint = torch.load(args.PretrainModel, map_location=device)
        if 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)
    elif ckpt:
        print("Loading {}".format(ckpt))
        checkpoint = torch.load(ckpt, map_location=device)
        if 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)
    elif args.stage in [2, 3, 4]:
        net = init_net(net, args.InitModel, args.stage)
    net = net.to(device)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    # Initialize optimizer
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    yuvw = [int(args.YUV[:-2]), int(args.YUV[-2]), int(args.YUV[-1])]
    print("Current YUV weights is {}".format(yuvw))
    criterion = RateDistortionLoss(args=args, distortion=args.metric, device=device, YUVweight=yuvw)

    # Initialize parameters for results recording
    loss, best_loss = float('inf'), float('inf')
    last_epoch, best_loss_epoch = 0, 0

    # Load model trained before
    if args.PretrainModel or ckpt:
        if 'state_dict' in checkpoint:
            loss = checkpoint['loss']
            best_loss = checkpoint['best_loss']
            last_epoch = checkpoint['epoch'] + 1
            best_loss_epoch = checkpoint['best_loss_epoch']
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            print("Test results on {}:".format(testpath.split('/')[-1]))
            _ = test_epoch(last_epoch - 1, test_dataloader, net, criterion)

    # Initial mask conv
    net.context_prediction.mask[:, :, 1, 4] = 0
    net.context_prediction.mask[:, :, 1, 3] = 0
    net.context_prediction.mask[:, :, 0, 4] = 0
    # Train model
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm,args.fineTune)
        print("Test results on {}:".format(testpath.split('/')[-1]))
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        print('Best loss:{}, Cur loss:{}'.format(best_loss, loss))
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        ckptfile = os.path.join(CkptsDir, "{:05d}.pth".format(epoch))
        if is_best:
            best_loss_epoch = epoch
            print('{} has better model now.'.format(testpath.split('/')[-1]))
            ckptfile = os.path.join(CkptsDir, "best.pth")
            ckptfile_second = os.path.join(CkptsDir, "second_best.pth")
            if os.path.exists(ckptfile):
                if os.path.exists(ckptfile_second):
                    os.system(f"rm {ckptfile_second}")
                os.system(f"mv {ckptfile} {ckptfile_second}")

        torch.save(
            {
                "loss": loss,
                'best_loss': best_loss,
                "epoch": epoch,
                'best_loss_epoch': best_loss_epoch,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict()
            }, ckptfile
        )

        old_ckptfile = os.path.join(CkptsDir, "{:05d}.pth".format(epoch-3))
        if os.path.exists(old_ckptfile):
            os.system(f"rm {old_ckptfile}")

        # Copy checkpoint to HDFS
        if args.hdfs_savepath:
            copy_to_hdfs(is_best, ckptfile, args.hdfs_savepath)


if __name__ == "__main__":
    args = ArgsAnalyse(sys.argv[1:])
    for k, v in vars(args).items():
        print(f"{k:<18}: {v}")
    main(args)
