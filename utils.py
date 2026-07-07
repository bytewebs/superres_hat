import math, os
from math import floor
import random
import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from torch import Tensor

from torchvision.transforms import functional as F
from losses.cross_correlation import xcorr_torch
from losses.spectral_tools import gen_mtf
from datasets.interpolator_tools import interp23tap
import scipy.io as io
from plot_MS_PAN import plot_visual_comp, plot_visual_comp_senven, plot_histogram_comp
from sewar.full_ref import uqi, ergas, sam, scc
from sewar.no_ref import d_lambda, d_s

NORM_DATA = np.array([[ 120, 1241],
                    [ 271, 1853],
                    [ 198, 2732],
                    [1402, 3824],
                    [ 455, 2966],
                    [1127, 3359],
                    [1346, 3713],
                    [1598, 4078],
                    ]).astype('float')

# NORM_DATA = np.array([[ 20, 1241],
#                     [ 171, 1853],
#                     [ 98, 2732],
#                     [1102, 3824],
#                     [ 255, 2966],
#                     [927, 3359],
#                     [1146, 3713],
#                     [1398, 4078],
#                     ]).astype('float')

#--------------
def normalise_bandwise_RGB(img, norm_data=NORM_DATA):
    out_img = img.clone()
    out_img = out_img.float()
    for i in range(img.shape[0]):
        norm = norm_data[i]
        mini = min(norm[0], out_img[i].mean()*0.7)
        maxi = max(norm[1], out_img[i].mean()/0.7)
        out_img[i] = (out_img[i] - mini)/(maxi - mini)

    return out_img


# def normalise_bandwise_RGB(img, norm_data=NORM_DATA):
#     # out_img = img.clone()
#     img = img.float()
#     for i in range(img.shape[0]):
#         norm = norm_data[i]
#         mini = min(norm[0], img[i].mean()*0.7)
#         maxi = max(norm[1], img[i].mean()/0.7)
#         img[i] = (img[i] - mini)/(maxi - mini)

#     return img


def normalise_bandwise_RGB_batch(img, norm_data=NORM_DATA):
    img = img.float()
    for i in range(img.shape[1]):
        norm = norm_data[i]
        mini = min(norm[0], img[:,i].mean()*0.7)
        img[:,i] = (img[:,i] - mini)/(norm[1] - mini)

    return img


def recover_bandwise_RGB(img, norm_data=NORM_DATA):
    if isinstance(img, np.ndarray):
        out_img = img.copy()
    else:
        out_img = img.clone()
        
    for i in range(img.shape[1]):
        norm = norm_data[i]
        out_img[:,i] = (out_img[:,i]*(norm[1] - norm[0])) + norm[0]
        
    return out_img

#-------
# def normalise_bandwise_20m(img, norm_data=NORM_DATA):
#     out_img = img.clone()
#     out_img = out_img.float()
#     for i in range(img.shape[0]):
#         norm = norm_data[i+4]
#         # print(img[i].mean(), 'in', i, img.shape[0], np.array(norm_data).shape)
#         mini = min(norm[0], img[i].mean()*0.7)
#         out_img[i] = (img[i] - mini)/(norm[1] - mini)
#         print(out_img[i].mean())

#     return out_img


def normalise_bandwise_20m(img, norm_data=NORM_DATA):
    img = img.float()
    for i in range(img.shape[0]):
        norm = norm_data[i+4]
        # print(img[i].mean(), 'in', i, img.shape[0], np.array(norm_data).shape)
        mini = min(norm[0], img[i].mean()*0.7)
        img[i] = (img[i] - mini)/(norm[1] - mini)
        # print(img[i].mean())

    return img


def normalise_bandwise_20m_batch(img, norm_data=NORM_DATA):
    img = img.float()
    for i in range(img.shape[1]):
        norm = norm_data[i+4]
        mini = min(norm[0], img[:,i].mean()*0.7)
        img[:,i] = (img[:,i] - mini)/(norm[1] - norm[0])

    return img


def recover_bandwise_20m(img, norm_data=NORM_DATA):
    if isinstance(img, np.ndarray):
        out_img = img.copy()
    else:
        out_img = img.clone()
    
    for i in range(img.shape[1]):
        norm = norm_data[i+4]
        out_img[:,i] = (out_img[:,i]*(norm[1] - norm[0])) + norm[0]

    return out_img


def recover_bandwise_all(img, norm_data=NORM_DATA):
    if isinstance(img, np.ndarray):
        out_img = img.copy()
    else:
        out_img = img.clone()
    
    out_img[:, :4] = recover_bandwise_RGB(img[:, :4], norm_data=norm_data)
    out_img[:, 4:] = recover_bandwise_20m(img[:, 4:], norm_data=norm_data)
    
    return out_img


def normalise_quantile(img, Q=None):
    return img/(2**13), 1

# def normalise_quantile(img, Q=None):
#     img = np.array(img)
#     if Q is None:
#         Q = np.quantile(img, (0.01, 0.85), (1, 2), keepdims=True)
#     print(Q)
#     img_norm = (img - Q[0, :, :]) / (Q[1, :, :] - Q[0, :, :])
#     return torch.tensor(img_norm, dtype= torch.float), Q

def normalise_quantile2(img, idx, Q=None):
    # img = np.array(img)
    # if Q is None:
    #     Q = np.quantile(img, (0.01, 0.99), (2, 3), keepdims=True).mean(0)
    # print(Q.shape)
    # img_norm = (img[idx] - Q[0, :, :]) / (Q[1, :, :] - Q[0, :, :])
    return img[idx]/(2**13), Q


def test_single(sr_img, hr_img, niqe, ssim_module):

    # netG.eval()
    # ms, ven_up, ms_rgb_up, ms_up = data

    # ven_up = ven_up.to(device)
    # ms = ms.to(device)
    # ms_rgb_up = ms_rgb_up.to(device)
    # ms_up = ms_up.to(device)
    # # ms_rgb_up = netG_rgb(ms_rgb)
    
    # # ms_fake = net_g(ms_up, ms_rgb_up)
    # ms_fake = net_g(ms, ms_rgb_up)
    # d_gen = torch.sigmoid(net_d(ms_fake)).detach().cpu()
    
    # ms_fake = ms_fake.detach().cpu()
    # ms_up = ms_up.detach().cpu()
    # ms = ms.detach().cpu()
    # ven_up = ven_up.detach().cpu()
    # ms_rgb_up = ms_rgb_up.detach().cpu()

    # ms_fake_crop = center_crop(ms_fake, 110)
    # ven_up_crop = center_crop(ven_up, 110)
    # ms_up_crop = center_crop(ms_up, 110)

    metrics = {}
    metrics['PSNR'] = psnr(sr_img, hr_img)

    metrics['RMSE'] = rmse(recover_bandwise_20m(sr_img), recover_bandwise_20m(hr_img))
    
    metrics['SSIM'] = ssim_module(sr_img, hr_img)
    
    # print(sr_img.flip(1)[:,1:].shape, ms_up_crop.flip(1)[:,1:].shape)
    metrics['NIQE'] = niqe(sr_img.flip(1)[:,1:]*255).mean().item()    
    
    metrics['SAM'] = sam_batch(hr_img[:,:3].detach().cpu(), sr_img[:,:3].detach().cpu())
    
    metrics['UQI'] = uqi_batch(hr_img[:,:].detach().cpu(), sr_img[:,:].detach().cpu())
    
    metrics['SCC'] = scc_batch(hr_img[:,:].detach().cpu(), sr_img[:,:].detach().cpu())
    
    
    return metrics



def bgr2ycbcr(image: np.ndarray, use_y_channel: bool = False) -> np.ndarray:
    """Implementation of bgr2ycbcr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in BGR format.
        use_y_channel (bool): Extract Y channel separately. Default: ``False``.

    Returns:
        ndarray: YCbCr image array data.
    """

    if use_y_channel:
        image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        image = np.matmul(image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


def stack_ycbcr(image_y, image_cbcr):
    return torch.stack((image_y, image_cbcr), dim=1)


def ycbcr2bgr_torch(image):
    # image_dtype = image.dtype
    image *= 255.
    transf = torch.tensor([[0.00456621, 0.00456621, 0.00456621],
                            [0.00791071, -0.00153632, 0],
                            [0, -0.00318811, 0.00625893]])
    
    transf_b = torch.tensor([-276.836, 135.576, -222.921])

    for i in range(image.shape[0]):
        # print(image[i].shape)
        image[i] = torch.matmul(image[i], transf) * 255.0 + transf_b

    image /= 255.
    # image = image.astype(image_dtype)
    return image


def ycbcr_stack2BGR(image_y, image_cbcr):
    image = torch.cat((image_y, image_cbcr), dim=1)
    image = torch.moveaxis(image, 1, -1)
    image = ycbcr2bgr_torch(image)
    image = torch.moveaxis(image, -1, 1)
    return image


def on_load_checkpoint(model, checkpoint: dict, strict=False):
        state_dict = checkpoint #["state_dict"]
        model_state_dict = model.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        # if is_changed:
        #     checkpoint.pop("optimizer_states", None)
        
        model.load_state_dict(state_dict, strict=strict)
        
        return model, state_dict


def psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    

def psnr_bands(img1, img2):
        mse = torch.mean((img1 - img2) ** 2, dim=[0, 2, 3])
        return 20 * torch.log10(1.0 / torch.sqrt(mse))


def rmse(img1, img2):
    criterion = nn.MSELoss()
    loss = torch.sqrt(criterion(img1, img2))
    return loss


def rmse_bands(predicted_x , target ):
    loss = torch.sum(torch.square(predicted_x - target) , axis=[0, 2, 3])/(predicted_x.size()[0]*predicted_x.size()[2]*predicted_x.size()[3])
    loss = torch.sqrt(loss)
    return loss


def transform(Mf, s):
    Mfs = Mf.cpu().detach().numpy()
    Mfs = np.moveaxis(Mfs, 1, -1)       # (B, H, W, C)
    Mfs = Mfs * (2 ** s.nbits)          # Convert to [0, 2^s.nbits]
    Mfs = np.clip(Mfs, 0, Mfs.max())
    Mfs = Mfs.astype(np.uint16)
    return Mfs


def transform_norm_RGB(Mf, s, norm_data=NORM_DATA):
    Mfs = Mf.cpu().detach().numpy()
    Mfs = recover_bandwise_RGB(Mfs, norm_data=norm_data)
    Mfs = np.moveaxis(Mfs, 1, -1)       # (B, H, W, C)
    Mfs = np.clip(Mfs, 0, Mfs.max())
    Mfs = Mfs.astype(np.uint16)
    return Mfs


def transform_norm_20m(Mf, s, norm_data=NORM_DATA):
    Mfs = Mf.cpu().detach().numpy()
    Mfs = recover_bandwise_20m(Mfs, norm_data=norm_data)
    Mfs = np.moveaxis(Mfs, 1, -1)       # (B, H, W, C)
    Mfs = np.clip(Mfs, 0, Mfs.max())
    Mfs = Mfs.astype(np.uint16)
    return Mfs


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def random_flip(sentinel, venus, venus_up, sentinel_up, p_vflip=0.5, p_hflip=0.5):
    if torch.rand(1) < p_vflip:
        sentinel = F.vflip(sentinel)
        sentinel_up = F.vflip(sentinel_up)
        venus = F.vflip(venus)
        venus_up = F.vflip(venus_up)
    if torch.rand(1) < p_hflip:
        sentinel = F.hflip(sentinel)
        sentinel_up = F.hflip(sentinel_up)
        venus = F.hflip(venus)
        venus_up = F.hflip(venus_up)

    return sentinel, venus, venus_up, sentinel_up


def random_flip2(sentinel, venus, p_vflip=0.5, p_hflip=0.5):
    if torch.rand(1) < p_vflip:
        sentinel = F.vflip(sentinel)
        venus = F.vflip(venus)
    if torch.rand(1) < p_hflip:
        sentinel = F.hflip(sentinel)
        venus = F.hflip(venus)

    return sentinel, venus

def savemat_plot(MS_gen, PAN, MS_up, sample_dir, s, filename='plot.png'):
    Mfs = transform(MS_gen, s)
    M_ups = transform(MS_up, s)
    Ps = PAN.cpu().detach().numpy()* (2 ** s.nbits)
    Ps = np.clip(Ps, 0, Ps.max())
    Ps = Ps.astype(np.uint16)
    mat_path = f'{sample_dir}/data_out_results.mat'
    io.savemat(
            mat_path,
            {
                'MS': M_ups,
                'PAN': Ps,
                'MS_out': Mfs
            }
        )
    
    plot_visual_comp(mat_path, save_dir=sample_dir, num_imgs=5, fused_present=False, filename=filename)


def savemat_plot_senven(MS_gen, MS_up, Ven, sample_dir, s, Disc=None, abs_diff=False, filename='plot.png'):
    Mfs = transform(MS_gen, s)
    M_ups = transform(MS_up, s)
    Vens = transform(Ven, s)
    uid = filename[:-4].split('_')[-1]
    mat_path = f'{sample_dir}/data_out_results_{uid}.mat'
    
    if Disc==None:
        io.savemat(
            mat_path,
            {
                'MS': M_ups,
                'MS_out': Mfs,
                'Fused': Vens
            }
        )
        
        plot_visual_comp_senven(mat_path, save_dir=sample_dir, num_imgs=5, disc_present=False, fused_present=True, filename=filename)
    else:
        # print(Disc.min(), Disc.max(), Disc.mean(), 'stat')
        if Disc.shape[1]==1:
            disc = Disc.cpu().detach().numpy()[:,0]
        elif Disc.ndim==3:
            disc = Disc.cpu().detach().numpy()
        else:
            disc = transform(Disc, s)
        io.savemat(
                mat_path,
                {
                    'MS': M_ups,
                    'Disc': disc,
                    'MS_out': Mfs,
                    'Fused': Vens
                }
            )
        if abs_diff==True:
            plot_visual_comp_senven(mat_path, save_dir=sample_dir, num_imgs=5, disc_present=False, abs_diff=True, fused_present=True, filename=filename)
        else:
            plot_visual_comp_senven(mat_path, save_dir=sample_dir, num_imgs=5, fused_present=True, filename=filename)
        
    os.remove(mat_path)


def savemat_plot_senven_RGBnorm(MS_gen, MS_up, Ven, sample_dir, s, Disc=None, abs_diff=False, norm=NORM_DATA, histo=True, filename='plot.png'):
    Mfs = transform_norm_RGB(MS_gen, s, norm_data=norm)
    M_ups = transform_norm_RGB(MS_up, s, norm_data=norm)
    Vens = transform_norm_RGB(Ven, s, norm_data=norm)
    uid = filename[:-4].split('_')[-1]
    mat_path = f'{sample_dir}/data_out_results_{uid}.mat'
    
    if Disc==None:
        io.savemat(
            mat_path,
            {
                'MS': M_ups,
                'MS_out': Mfs,
                'Fused': Vens,
                'NORM': NORM_DATA
            }
        )
        
        plot_visual_comp_senven(mat_path, save_dir=sample_dir, num_imgs=5, disc_present=False, fused_present=True, filename=filename)
    else:
        # print(Disc.min(), Disc.max(), Disc.mean(), 'stat')
        if Disc.shape[1]==1:
            disc = Disc.cpu().detach().numpy()[:,0]
        elif Disc.ndim==3:
            disc = Disc.cpu().detach().numpy()
        else:
            disc = transform(Disc, s)
        io.savemat(
                mat_path,
                {
                    'MS': M_ups,
                    'Disc': disc,
                    'MS_out': Mfs,
                    'Fused': Vens,
                    'NORM': NORM_DATA
                }
            )
        if abs_diff==True:
            plot_visual_comp_senven(mat_path, save_dir=sample_dir, num_imgs=5, disc_present=False, abs_diff=True, fused_present=True, filename=filename)
        else:
            plot_visual_comp_senven(mat_path, save_dir=sample_dir, num_imgs=5, fused_present=True, filename=filename)

        if histo:
            plot_histogram_comp(mat_path, save_dir=sample_dir, num_imgs=5, filename=filename.replace('.png', '_hist.png'))
    
    os.remove(mat_path)
    
    
def savemat_plot_senven_20mnorm(MS_gen, MS_up, Ven, sample_dir, s, Disc=None, abs_diff=False, norm=NORM_DATA, filename='plot.png'):
    Mfs = transform_norm_20m(MS_gen, s, norm_data=norm)
    M_ups = transform_norm_20m(MS_up, s, norm_data=norm)
    Vens = transform_norm_20m(Ven, s, norm_data=norm)
    uid = filename[:-4].split('_')[-1]
    mat_path = f'{sample_dir}/data_out_results_{uid}.mat'
    
    if Disc==None:
        io.savemat(
            mat_path,
            {
                'MS': M_ups,
                'MS_out': Mfs,
                'Fused': Vens,
                'NORM': NORM_DATA
            }
        )
        
        plot_visual_comp_senven(mat_path, save_dir=sample_dir, num_imgs=5, disc_present=False, fused_present=True, filename=filename)
    else:
        if Disc.shape[1]==1:
            disc = Disc.cpu().detach().numpy()[:,0]
        elif Disc.ndim==3:
            disc = Disc.cpu().detach().numpy()
        else:
            disc = transform_norm_RGB(Disc, s)
        io.savemat(
                mat_path,
                {
                    'MS': M_ups,
                    'Disc': disc,
                    'MS_out': Mfs,
                    'Fused': Vens,
                    'NORM': NORM_DATA
                }
            )

        if abs_diff==True:
            plot_visual_comp_senven(mat_path, save_dir=sample_dir, num_imgs=5, disc_present=False, abs_diff=True, fused_present=True, filename=filename)
        else:
            plot_visual_comp_senven(mat_path, save_dir=sample_dir, num_imgs=5, fused_present=True, filename=filename)

        plot_histogram_comp(mat_path, save_dir=sample_dir, num_imgs=5, filename=filename.replace('.png', '_hist.png'))
    
    os.remove(mat_path)


def net_scope(kernel_size):
    """
        Compute the network scope.

        Parameters
        ----------
        kernel_size : List[int]
            A list containing the kernel size of each layer of the network.

        Return
        ------
        scope : int
            The scope of the network

        """

    scope = 0
    for i in range(len(kernel_size)):
        scope += math.floor(kernel_size[i] / 2)
    return scope


def local_corr_mask(img_in, ratio, sensor, device, kernel=8):
    """
        Compute the threshold mask for the structural loss.

        Parameters
        ----------
        img_in : Torch Tensor
            The test image, already normalized and with the MS part upsampled with ideal interpolator.
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        device : Torch device
            The device on which perform the operation.
        kernel : int
            The semi-width for local cross-correlation computation.
            (See the cross-correlation function for more details)

        Return
        ------
        mask : PyTorch Tensor
            Local correlation field stack, composed by each MS and PAN. Dimensions: Batch, B, H, W.

        """

    I_PAN = torch.unsqueeze(img_in[:, -1, :, :], dim=1)
    I_MS = img_in[:, :-1, :, :]

    MTF_kern = gen_mtf(ratio, sensor)[:, :, 0]
    MTF_kern = np.expand_dims(MTF_kern, axis=(0, 1))
    MTF_kern = torch.from_numpy(MTF_kern).type(torch.float32)
    pad = floor((MTF_kern.shape[-1] - 1) / 2)

    padding = nn.ReflectionPad2d(pad)

    depthconv = nn.Conv2d(in_channels=1,
                          out_channels=1,
                          groups=1,
                          kernel_size=MTF_kern.shape,
                          bias=False)

    depthconv.weight.data = MTF_kern
    depthconv.weight.requires_grad = False

    I_PAN = padding(I_PAN)
    I_PAN = depthconv(I_PAN)
    mask = xcorr_torch(I_PAN, I_MS, kernel, device)
    mask = 1.0 - mask

    return mask


def random_resize_torch(img, resize_scales=[1, 2, 4]):
    """
    resize_scales: List of integers
    """
    idxs = random.randint(0, len(resize_scales)-1)
    chnl_idx = random.randint(0, img.shape[0]-1)
    scale = resize_scales[idxs]
    if scale == 1:
        return img
    else:
        tmp = img[chnl_idx]
        img = img[:,::scale, ::scale]
        img_up = interp23tap(np.moveaxis(np.array(img), 0, -1), scale)
        img_up[:,:,chnl_idx] = tmp
        img_up = np.moveaxis(img_up, -1, 0)
        return img_up


def random_crop_torch(
        gt_images: Tensor,
        lr_images: Tensor,
        gt_patch_size: int,
        upscale_factor: int,
    ):
    """Randomly intercept two images in the specified area

    Args:
        gt_images (ndarray | Tensor | list[ndarray] | list[Tensor]): ground truth images read by PyTorch
        lr_images (ndarray | Tensor | list[ndarray] | list[Tensor]): Low resolution images read by PyTorch
        gt_patch_size (int): the size of the ground truth image after interception
        upscale_factor (int): the ground truth image size is a magnification of the low resolution image size

    Returns:
        gt_images (ndarray or Tensor or): the intercepted ground truth image
        lr_images (ndarray or Tensor or): low-resolution intercepted images

    """

    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if input_type == "Tensor":
        lr_image_height, lr_image_width = lr_images[0].size()[-2:]
    else:
        lr_image_height, lr_image_width = lr_images[0].shape[0:2]

    # Calculate the size of the low-resolution image that needs to be intercepted
    lr_patch_size = gt_patch_size // upscale_factor

    # Just need to find the top and left coordinates of the image
    lr_top = random.randint(0, lr_image_height - lr_patch_size)
    lr_left = random.randint(0, lr_image_width - lr_patch_size)

    # Capture low-resolution images
    if input_type == "Tensor":
        lr_images = [lr_image[
                     :,
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size] for lr_image in lr_images]
    else:
        lr_images = [lr_image[
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size,
                     ...] for lr_image in lr_images]

    # Intercept the ground truth image
    gt_top, gt_left = int(lr_top * upscale_factor), int(lr_left * upscale_factor)

    if input_type == "Tensor":
        gt_images = [v[
                     :,
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size] for v in gt_images]
    else:
        gt_images = [v[
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size,
                     ...] for v in gt_images]

    # When the input has only one image
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images


def random_crop_rgb_torch(
        gt_images: Tensor,
        lr_images: Tensor,
        rgb_images: Tensor,
        gt_patch_size: int,
        upscale_factor: int,
    ):
    """Randomly intercept two images in the specified area

    Args:
        gt_images (ndarray | Tensor | list[ndarray] | list[Tensor]): ground truth images read by PyTorch
        lr_images (ndarray | Tensor | list[ndarray] | list[Tensor]): Low resolution images read by PyTorch
        gt_patch_size (int): the size of the ground truth image after interception
        upscale_factor (int): the ground truth image size is a magnification of the low resolution image size

    Returns:
        gt_images (ndarray or Tensor or): the intercepted ground truth image
        lr_images (ndarray or Tensor or): low-resolution intercepted images

    """

    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]
    if not isinstance(rgb_images, list):
        rgb_images = [rgb_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if input_type == "Tensor":
        lr_image_height, lr_image_width = lr_images[0].size()[-2:]
    else:
        lr_image_height, lr_image_width = lr_images[0].shape[0:2]

    # Calculate the size of the low-resolution image that needs to be intercepted
    lr_patch_size = gt_patch_size // upscale_factor

    # Just need to find the top and left coordinates of the image
    lr_top = random.randint(0, lr_image_height - lr_patch_size)
    lr_left = random.randint(0, lr_image_width - lr_patch_size)

    # Capture low-resolution images
    if input_type == "Tensor":
        lr_images = [lr_image[
                     :,
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size] for lr_image in lr_images]
    else:
        lr_images = [lr_image[
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size,
                     ...] for lr_image in lr_images]

    # Intercept the ground truth image
    gt_top, gt_left = int(lr_top * upscale_factor), int(lr_left * upscale_factor)

    if input_type == "Tensor":
        gt_images = [v[
                     :,
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size] for v in gt_images]
    else:
        gt_images = [v[
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size,
                     ...] for v in gt_images]

    # When the input has only one image
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images, rgb_images


def center_crop(img, crop_size):
    _, _, h, w = img.shape
    mid_x, mid_y = int(w/2), int(h/2)
    cw2, ch2 = int(crop_size/2), int(crop_size/2) 
    crop_img = img[:,:,mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def uqi_batch(target, pred):
    target = np.moveaxis(np.array(target), 1, -1)
    pred = np.moveaxis(np.array(pred), 1, -1)
    num = target.shape[0]
    uqis = []
    for i in range(num):
        uq = uqi(target[i], pred[i])
        uqis.append(uq)
    return np.array(uqis).mean()


def sam_batch(target, pred):
    target = np.moveaxis(np.array(target), 1, -1)
    pred = np.moveaxis(np.array(pred), 1, -1)
    num = target.shape[0]
    ergass = []
    for i in range(num):
        uq = sam(target[i], pred[i])
        ergass.append(uq)
    return np.array(ergass).mean()


def scc_batch(target, pred):
    target = np.moveaxis(np.array(target), 1, -1)
    pred = np.moveaxis(np.array(pred), 1, -1)
    num = target.shape[0]
    ergass = []
    for i in range(num):
        uq = scc(target[i], pred[i])
        ergass.append(uq)
    return np.array(ergass).mean()


def d_s_batch(target, ms, pred):
    target = np.moveaxis(np.array(target), 1, -1)
    pred = np.moveaxis(np.array(pred), 1, -1)
    ms = np.moveaxis(np.array(ms), 1, -1)
    # print(target.dtype, pred.dtype, ms.dtype)
    num = target.shape[0]
    ergass = []
    for i in range(num):
        uq = d_s(target[i], ms[i], pred[i])
        ergass.append(uq)
    return np.array(ergass).mean()

# def corrcoef(a,b):
#     a = a - numpy.mean(a)
#     b = b - numpy.mean(b)
#     r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
#     return r 


def cc_batch(target, sr):
    sr = np.moveaxis(sr.numpy(), 1, -1)
    target = np.moveaxis(target.numpy(), 1, -1)

    target_mean = np.mean(target, (1, 2))
    sr_mean = np.mean(sr, (1, 2))
    # a_mean = np.mean(a, (2, 3))
    # b_mean = np.mean(b, (2, 3))
    CC = np.zeros(sr.shape[-1])
    nbatch = sr.shape[0]
    for i in range(nbatch):
        sr[i] = sr[i] - sr_mean[i]
        target[i] = target[i] - target_mean[i]
        r = (sr[i]*target[i]).sum((0, 1)) / np.sqrt((sr[i]*sr[i]).sum((0, 1)) * (target[i]*target[i]).sum((0, 1)))
        CC+=r
    CC /= nbatch
    
    return CC


# def d_s_batch(target, ms, pred):
#     target = np.moveaxis(np.array(target*255).astype(np.uint8), 1, -1)
#     pred = np.moveaxis(np.array(pred*255).astype(np.uint8), 1, -1)
#     ms = np.moveaxis(np.array(ms*255).astype(np.uint8), 1, -1)
#     print(target.dtype, pred.dtype, ms.dtype)
#     num = target.shape[0]
#     ergass = []
#     for i in range(num):
#         uq = d_s(target[i], ms[i], pred[i])
#         ergass.append(uq)
#     return np.array(ergass).mean()


def d_lambda_batch(target, pred):
    target = np.moveaxis(np.array(target), 1, -1)
    pred = np.moveaxis(np.array(pred), 1, -1)
    num = target.shape[0]
    ergass = []
    for i in range(num):
        uq = d_lambda(target[i], pred[i])
        ergass.append(uq)
    return np.array(ergass).mean()
