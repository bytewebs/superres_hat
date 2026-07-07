from math import floor, ceil

import numpy as np
import torch
import torch.nn as nn

from losses.cross_correlation import xcorr_torch as ccorr


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class SpectralLoss(nn.Module):
    def __init__(self, mtf, net_scope, pan_shape, ratio, device, mask=None):

        # Class initialization
        super(SpectralLoss, self).__init__()
        kernel = mtf[0]
        # Parameters definition
        self.nbands = kernel.shape[-1]
        self.net_scope = net_scope
        self.device = device
        self.ratio = ratio

        # Conversion of filters in Tensor
        self.MTF_r = mtf[1]
        self.MTF_c = mtf[2]
        self.pad = floor((kernel.shape[0] - 1) / 2)

        kernel = np.moveaxis(kernel, -1, 0)
        kernel = np.expand_dims(kernel, axis=1)

        kernel = torch.from_numpy(kernel).type(torch.float32)

        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(in_channels=self.nbands,
                                   out_channels=self.nbands,
                                   groups=self.nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)

        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        self.loss = nn.L1Loss(reduction='sum')
        # self.loss = nn.SmoothL1Loss(reduction='sum')

        # Mask definition
        if mask is not None:
            self.mask = mask
        else:
            self.mask = torch.ones((1, self.nbands, pan_shape[-2] - (self.net_scope + self.pad) * 2,
                                    pan_shape[-1] - (self.net_scope + self.pad) * 2), device=self.device)

    def forward(self, outputs, labels):

        x = self.depthconv(outputs)

        labels = labels[:, :, self.pad:-self.pad, self.pad:-self.pad]
        y = torch.zeros(x.shape, device=self.device)
        W_ = torch.zeros(x.shape, device=self.device)

        for b in range(self.nbands):
            y[:, b, self.MTF_r[b]::self.ratio, self.MTF_c[b]::self.ratio] = labels[:, b, 2::self.ratio, 2::self.ratio]
            W_[:, b, self.MTF_r[b]::self.ratio, self.MTF_c[b]::self.ratio] = self.mask[:, b, 2::self.ratio, 2::self.ratio]

        W_ = W_ / torch.sum(W_)

        x = x * W_
        y = y * W_
        L = self.loss(x, y)

        return L


class SpectralLossNocorr(nn.Module):
    def __init__(self, mtf, net_crop, pan_shape, ratio, device, mask=None):

        # Class initialization
        super(SpectralLossNocorr, self).__init__()
        kernel = mtf[0]
        # Parameters definition
        self.nbands = kernel.shape[-1]
        self.net_scope = net_crop
        self.device = device
        self.ratio = ratio

        # Conversion of filters in Tensor
        self.MTF_r = 2
        self.MTF_c = 2
        self.pad = floor((kernel.shape[0] - 1) / 2)

        kernel = np.moveaxis(kernel, -1, 0)
        kernel = np.expand_dims(kernel, axis=1)

        kernel = torch.from_numpy(kernel).type(torch.float32)

        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(in_channels=self.nbands,
                                   out_channels=self.nbands,
                                   groups=self.nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)

        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        self.loss = nn.L1Loss(reduction='sum')
        # self.loss = nn.SmoothL1Loss(reduction='sum')

        # Mask definition
        if mask is not None:
            self.mask = mask
        else:
            self.mask = torch.ones((1, self.nbands, pan_shape[-2] - (self.net_scope + self.pad) * 2,
                                    pan_shape[-1] - (self.net_scope + self.pad) * 2), device=self.device)

    def forward(self, outputs, labels):

        x = self.depthconv(outputs)

        labels = labels[:, :, self.pad:-self.pad, self.pad:-self.pad]
        y = torch.zeros(x.shape, device=self.device)
        W_ = torch.zeros(x.shape, device=self.device)

        for b in range(self.nbands):
            y[:, b, self.MTF_r::self.ratio, self.MTF_c::self.ratio] = labels[:, b, 2::self.ratio, 2::self.ratio]
            W_[:, b, self.MTF_r::self.ratio, self.MTF_c::self.ratio] = self.mask[:, b, 2::self.ratio, 2::self.ratio]

        W_ = W_ / torch.sum(W_)

        x = x * W_
        y = y * W_
        L = self.loss(x, y)

        return L


class StructuralThreshold(nn.Module):

    def __init__(self, mtf, ratio, sensor, kernel, device):
        # Class initialization
        super(StructuralThreshold, self).__init__()

        # Parameters definition:
        self.kernel = kernel
        self.device = device

        MTF_kern = np.expand_dims(mtf[:,:,0], axis=(0, 1))
        MTF_kern = torch.from_numpy(MTF_kern).type(torch.float32)
        pad = floor((MTF_kern.shape[-1] - 1) / 2)

        self.padding = nn.ReflectionPad2d(pad)
        self.depthconv = nn.Conv2d(in_channels=1,
                            out_channels=1,
                            groups=1,
                            kernel_size=MTF_kern.shape,
                            bias=False)

        self.depthconv.weight.data = MTF_kern
        self.depthconv.weight.requires_grad = False

    def forward(self, PAN, MS):
        PAN = torch.unsqueeze(PAN[:, :, :], dim=1)
        PAN = self.padding(PAN)
        PAN = self.depthconv(PAN)
        mask = torch.clamp(ccorr(PAN, MS, self.kernel, self.device), min=-1)  ## Change made
        mask = 1.0 - mask

        return mask


class StructuralThresholdSent(nn.Module):

    def __init__(self, mtf, ratio, sensor, kernel, device):
        # Class initialization
        super(StructuralThresholdSent, self).__init__()

        # Parameters definition:
        self.kernel = kernel
        self.device = device

        MTF_kern = np.expand_dims(mtf[:,:,0], axis=(0, 1))
        MTF_kern = torch.from_numpy(MTF_kern).type(torch.float32)
        pad = floor((MTF_kern.shape[-1] - 1) / 2)

        self.padding = nn.ReflectionPad2d(pad)
        self.depthconv = nn.Conv2d(in_channels=4,
                            out_channels=4,
                            groups=1,
                            kernel_size=MTF_kern.shape,
                            bias=False)   

        self.depthconv.weight.data = MTF_kern
        self.depthconv.weight.requires_grad = False

    def forward(self, PAN, MS):
        b, c, h, w = PAN.shape
        PAN = PAN.reshape(b*c, 1, h, w)
        PAN = self.padding(PAN)
        PAN = self.depthconv(PAN)
        _, _, h, w = PAN.shape
        PAN = PAN.reshape(b, c, h, w)
        mask = torch.clamp(ccorr(PAN, MS, self.kernel, self.device), min=-1)  ## Change made
        mask = 1.0 - mask

        return mask


class StructuralLoss(nn.Module):

    def __init__(self, sigma, device):
        # Class initialization
        super(StructuralLoss, self).__init__()

        # Parameters definition:

        self.scale = ceil(sigma / 2)
        self.device = device

    def forward(self, outputs, labels, xcorr_thr):
        X_corr = torch.clamp(ccorr(outputs, labels, self.scale, self.device), min=-1)
        X = 1.0 - X_corr

        with torch.no_grad():
            Lxcorr_no_weights = torch.mean(X)

        worst = X.gt(xcorr_thr)
        Y = X * worst
        Lxcorr = torch.mean(Y)

        return Lxcorr, Lxcorr_no_weights.item()
