import cv2
import numpy as np
import torch
import random
import math
from config import config
from torch.nn import functional as F
from datasets.degradations import random_add_gaussian_noise_pt, circular_lowpass_kernel, random_mixed_kernels

kernel_range = [2 * v + 1 for v in range(3, 9)]


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.75, threshold=5):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img


def sinc_filter(inp):
    #  Generate sinc kernel
    kernels = []
    for i in range(inp.shape[0]):
      kernel_size = random.choice(kernel_range)
      omega_c = np.random.uniform(np.pi / 3, np.pi)
      sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
      sinc_kernel = torch.FloatTensor(sinc_kernel)
      kernels.append(sinc_kernel)
    
    sinc_kernels = torch.stack(kernels)
    out = filter2D(inp, sinc_kernels)
    return out


def blur_filter(inp):
    kernels = []
    kernel_size = random.choice(kernel_range)
    for i in range(inp.shape[0]):
        kernel = random_mixed_kernels(
                config.TRAIN.DATA.kernel_list, config.TRAIN.DATA.kernel_prob, kernel_size,
                [0.2, 1], [0.2, 1], [-math.pi, math.pi],
                [0.5, 1], [0.5, 1], noise_range=None)
        kernel = torch.FloatTensor(kernel)
        kernels.append(kernel)

    kernels = torch.stack(kernels)
    out = filter2D(inp, kernels)
    return out
    

def data_degradation(data):
    
    # Add noise points
    if np.random.uniform() < config.TRAIN.DATA.gaussian_noise_prob:
        data = random_add_gaussian_noise_pt(
                    data, sigma_range=[1, 25], clip=True, rounds=False, gray_prob=0.4)

    # Adding blur
    if np.random.uniform() < config.TRAIN.DATA.blur_prob:
        data = blur_filter(data)

    return data


def blur_filter_single(inp):
    # kernels = []
    kernel_size = random.choice(kernel_range)
    # for i in range(inp.shape[0]):
    kernel = random_mixed_kernels(
            config.TRAIN.DATA.kernel_list, config.TRAIN.DATA.kernel_prob, kernel_size,
            [0.2, 5], [0.2, 5], [-math.pi, math.pi],
            [0.5, 2], [0.5, 2], noise_range=None)
    kernel = torch.FloatTensor(kernel)
    out = filter2D(inp.unsqueeze(0), kernel.unsqueeze(0))
    return out[0]

def sinc_filter_single(inp):
    #  Generate sinc kernel
    # for i in range(inp.shape[0]):
    kernel_size = random.choice(kernel_range)
    omega_c = np.random.uniform(np.pi / 3, np.pi)
    sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
    sinc_kernel = torch.FloatTensor(sinc_kernel)
    out = filter2D(inp.unsqueeze(0), sinc_kernel.unsqueeze(0))
    return out[0]


def deform_single(inp, out_scale=4):
    _, H, W = inp.shape

    out = inp.clone()

    # First degradation
    if np.random.uniform() < 0.7:
        out = blur_filter_single(inp)
    updown_type = random.choices(['up', 'down', 'keep'], [0.2, 0.7, 0.1])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, 1.5)
    elif updown_type == 'down':
        scale = np.random.uniform(0.5, 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    # print('1', out.shape)
    out = F.interpolate(out.unsqueeze(0), scale_factor=scale, mode=mode)[0]
    # print(out.shape)
    if np.random.uniform() < 0.7:
        out = random_add_gaussian_noise_pt(out.unsqueeze(0), sigma_range=[0.1,1], clip=True, rounds=False, gray_prob=0.4)[0]

    # Second degradation
    if np.random.uniform() < 0.4:
        out = blur_filter_single(out)
    updown_type = random.choices(['up', 'down', 'keep'], [0.3, 0.4, 0.3])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, 1.2)
    elif updown_type == 'down':
        scale = np.random.uniform(0.75, 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out.unsqueeze(0), scale_factor=scale, mode=mode)[0]
    if np.random.uniform() < 0.4:
        out = random_add_gaussian_noise_pt(out.unsqueeze(0), sigma_range=[0.1,2], clip=True, rounds=False, gray_prob=0.2)[0]

    # Resize to target size
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    # print(H // out_scale, W // out_scale, out.shape)
    out = F.interpolate(out.unsqueeze(0), size=(H // out_scale, W // out_scale), mode=mode)[0]
    # out = sinc_filter_single(out)
    out = torch.clamp(out, 0, 1)

    return out


def deform_single_refine(inp, out_scale=4):
    _, H, W = inp.shape

    out = inp.clone()

    # First degradation
    if np.random.uniform() < 0.5:
        out = blur_filter_single(inp)
    updown_type = random.choices(['up', 'down', 'keep'], [0.2, 0.7, 0.1])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, 1.3)
    elif updown_type == 'down':
        scale = np.random.uniform(0.7, 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    # print('1', out.shape)
    out = F.interpolate(out.unsqueeze(0), scale_factor=scale, mode=mode)[0]
    # print(out.shape)
    if np.random.uniform() < 0.7:
        out = random_add_gaussian_noise_pt(out.unsqueeze(0), sigma_range=[0.1,3], clip=True, rounds=False, gray_prob=0.4)[0]

    # Second degradation
    if np.random.uniform() < 0.3:
        out = blur_filter_single(out)
    updown_type = random.choices(['up', 'down', 'keep'], [0.3, 0.4, 0.3])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, 1.2)
    elif updown_type == 'down':
        scale = np.random.uniform(0.75, 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out.unsqueeze(0), scale_factor=scale, mode=mode)[0]
    if np.random.uniform() < 0.4:
        out = random_add_gaussian_noise_pt(out.unsqueeze(0), sigma_range=[0.1,3], clip=True, rounds=False, gray_prob=0.4)[0]

    # Resize to target size
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    # print(H // out_scale, W // out_scale, out.shape)
    out = F.interpolate(out.unsqueeze(0), size=(H // out_scale, W // out_scale), mode=mode)[0]
    # out = sinc_filter_single(out)
    out = torch.clamp(out, 0, 1)

    return out
