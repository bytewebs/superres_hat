import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))
sys.path.append(str(Path('.').absolute().parent.parent))

os.environ['TL_BACKEND'] = 'torch'
os.environ['WANDB_API_KEY']='ENTER WANDB KEY'
os.environ['WANDB_ENTITY']='ENTER USERNAME'

import time
import random
import torch
import functools
from torch import nn
import numpy as np
from torch.cuda import amp
from torch.utils.data import DataLoader
from config import config
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from losses.LDL_loss import get_refined_artifact_map
import glob
from losses.spectral_tools import gen_mtf
from sensor import Sensor
import losses
from losses.vgg_loss import ContentLoss
from networks import UNetDiscriminatorSN
from utils import savemat_plot_senven_20mnorm, savemat_plot_senven_RGBnorm, psnr, psnr_bands, rmse, rmse_bands, center_crop, sam_batch, uqi_batch, scc_batch
from datasets.sen2ven_data_all_norm import TrainData, ValidData
from datasets.img_process_utils import USMSharp
from datasets.niqe_iqa import NIQE
from utils import recover_bandwise_all, on_load_checkpoint
from pytorch_msssim import SSIM, MS_SSIM
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_single(data, net_g, net_d, iters):

    netG.eval()
    ms, ven_up, ms_rgb_up, ms_up = data

    ven_up = ven_up.to(device)
    ms = ms.to(device)
    ms_rgb_up = ms_rgb_up.to(device)
    ms_up = ms_up.to(device)
    
    ms_fake = net_g(ms, ms_rgb_up)
    d_gen = torch.sigmoid(net_d(ms_fake)).detach().cpu()
    
    ms_fake = ms_fake.detach().cpu()
    ms_up = ms_up.detach().cpu()
    ms = ms.detach().cpu()
    ven_up = ven_up.detach().cpu()
    ms_rgb_up = ms_rgb_up.detach().cpu()

    ms_fake_crop = center_crop(ms_fake, 110)
    ven_up_crop = center_crop(ven_up, 110)
    ms_up_crop = center_crop(ms_up, 110)

    metrics, metrics_bi = {}, {}
    metrics['PSNR'] = psnr(ms_fake_crop, ven_up_crop)
    metrics_bi['PSNR'] = psnr(ms_up_crop, ven_up_crop)

    psnr_bands_res = psnr_bands(ms_fake_crop.detach().cpu(), ven_up_crop.detach().cpu())
    psnr_bands_res_bi = psnr_bands(ms_up_crop.detach().cpu(), ven_up_crop.detach().cpu())
    print('PSNR_bands:', psnr_bands_res, '/', psnr_bands_res_bi)
    
    metrics['RMSE'] = rmse(recover_bandwise_all(ms_fake_crop), recover_bandwise_all(ven_up_crop))    # Caution
    metrics_bi['RMSE'] = rmse(recover_bandwise_all(ms_up_crop), recover_bandwise_all(ven_up_crop))      # Caution
    
    rmse_bands_res = rmse_bands(recover_bandwise_all(ms_fake_crop), recover_bandwise_all(ven_up_crop))
    rmse_bands_res_bi = rmse_bands(recover_bandwise_all(ms_up_crop), recover_bandwise_all(ven_up_crop))
    print('RMSE_bands:', rmse_bands_res, '/', rmse_bands_res_bi)
    
    metrics['SSIM'] = ssim_module(ms_fake_crop, ven_up_crop)
    metrics_bi['SSIM'] = ssim_module(ms_up_crop, ven_up_crop)
    
    metrics['NIQE_RGB'] = niqe(ms_fake_crop[:,:3].flip(1)*255).mean().item()
    metrics_bi['NIQE_RGB'] = niqe(ms_up_crop[:,:3].flip(1)*255).mean().item()
    
    metrics['NIQE_20m'] = niqe4(ms_fake_crop[:,4:7].flip(1)*255).mean().item()
    metrics_bi['NIQE_20m'] = niqe4(ms_up_crop[:,4:7].flip(1)*255).mean().item()
    
    metrics['SAM'] = sam_batch(ven_up_crop[:,:3].detach().cpu(), ms_fake_crop[:,:3].detach().cpu())
    metrics_bi['SAM'] = sam_batch(ven_up_crop[:,:3].detach().cpu(), ms_up_crop[:,:3].detach().cpu())

    metrics['UQI'] = uqi_batch(ven_up_crop[:,:].detach().cpu(), ms_fake_crop[:,:].detach().cpu())
    metrics_bi['UQI'] = uqi_batch(ven_up_crop[:,:].detach().cpu(), ms_up_crop[:,:].detach().cpu())
    
    metrics['SCC'] = scc_batch(ven_up_crop[:,:].detach().cpu(), ms_fake_crop[:,:].detach().cpu())
    metrics_bi['SCC'] = scc_batch(ven_up_crop[:,:].detach().cpu(), ms_up_crop[:,:].detach().cpu())
    
    savemat_plot_senven_RGBnorm(ms_fake[:,:4], ms_up[:,:4], ven_up[:,:4], save_dir, s, d_gen, filename=f'plot_rgbn_{iters}.png')
    savemat_plot_senven_20mnorm(ms_fake[:,4:], ms_up[:,4:], ven_up[:,4:], save_dir, s, d_gen, filename=f'plot_20m_{iters}.png')

    return metrics, metrics_bi


def _atomic_torch_save(obj, path):
    """Write to a temp file then rename, so a killed job cannot leave a 0-byte .pt."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + f'.tmp.{os.getpid()}'
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def save_resume_checkpoint(path, epoch, iters, netG, ema_netG, optimizerG, schedulerG,
                           netD=None, optimizerD=None, schedulerD=None):
    """Consolidated, SenHAT-compatible resume checkpoint.

    Stores everything needed to continue training bit-for-bit: generator weights,
    EMA generator state, optimizer/scheduler state, and the epoch/iteration counters.
    Discriminator artifacts are included when present (unused in the fidelity stage).
    Uses an atomic write so walltime kills cannot truncate the previous good file.
    """
    ckpt = {
        'epoch': epoch,
        'iters': iters,
        'netG': netG.state_dict(),
        'ema_netG': ema_netG.state_dict(),
        'optimizerG': optimizerG.state_dict(),
        'schedulerG': schedulerG.state_dict(),
    }
    if netD is not None:
        ckpt['netD'] = netD.state_dict()
    if optimizerD is not None:
        ckpt['optimizerD'] = optimizerD.state_dict()
    if schedulerD is not None:
        ckpt['schedulerD'] = schedulerD.state_dict()
    _atomic_torch_save(ckpt, path)


def _strip_module_prefix(state, model):
    """Align DataParallel ``module.`` prefixes between checkpoint and model."""
    model_has = any(k.startswith('module.') for k in model.state_dict())
    ckpt_has = any(k.startswith('module.') for k in state)
    if ckpt_has and not model_has:
        return {k.replace('module.', '', 1): v for k, v in state.items()}
    if (not ckpt_has) and model_has:
        return {'module.' + k: v for k, v in state.items()}
    return state


def load_resume_checkpoint(path, netG, ema_netG, optimizerG, schedulerG,
                           netD=None, optimizerD=None, schedulerD=None, device='cpu'):
    """Restore a consolidated resume checkpoint OR a plain gens_*.pt state_dict.

    Returns ``(start_epoch, iters)`` where ``start_epoch`` is the next epoch to run.
    """
    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        raise RuntimeError(
            f'Checkpoint is missing or corrupt (empty/tiny file): {path} '
            f'({os.path.getsize(path) if os.path.isfile(path) else "missing"} bytes). '
            'Use gens_7_389.pt instead, or restore a non-empty resume_7.pt backup.')

    ckpt = torch.load(path, map_location=torch.device(device))

    # Case A: consolidated resume written by save_resume_checkpoint
    if isinstance(ckpt, dict) and 'netG' in ckpt:
        netG.load_state_dict(_strip_module_prefix(ckpt['netG'], netG))
        if ema_netG is not None and 'ema_netG' in ckpt:
            ema_netG.load_state_dict(ckpt['ema_netG'])
            # after loading EMA, also sync if shapes match; AveragedModel wraps netG
        if 'optimizerG' in ckpt:
            try:
                optimizerG.load_state_dict(ckpt['optimizerG'])
            except ValueError as e:
                print(f'[resume] optimizerG not restored ({e}); using fresh optimizer.')
        if schedulerG is not None and 'schedulerG' in ckpt:
            try:
                schedulerG.load_state_dict(ckpt['schedulerG'])
            except Exception as e:
                print(f'[resume] schedulerG not restored ({e}); using fresh scheduler.')
        if netD is not None and 'netD' in ckpt:
            try:
                netD.load_state_dict(_strip_module_prefix(ckpt['netD'], netD))
            except Exception as e:
                print(f'[resume] netD not restored ({e}); using random-init discriminator.')
        if optimizerD is not None and 'optimizerD' in ckpt:
            try:
                optimizerD.load_state_dict(ckpt['optimizerD'])
            except ValueError as e:
                print(f'[resume] optimizerD not restored ({e}); using fresh optimizerD.')
        if schedulerD is not None and 'schedulerD' in ckpt:
            try:
                schedulerD.load_state_dict(ckpt['schedulerD'])
            except Exception as e:
                print(f'[resume] schedulerD not restored ({e}).')
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        iters = int(ckpt.get('iters', 0))
        return start_epoch, iters

    # Case B: plain generator weights (gens_7_{epoch}.pt)
    if isinstance(ckpt, dict):
        netG.load_state_dict(_strip_module_prefix(ckpt, netG))
        # EMA was constructed before resume (random weights); sync it to the loaded G.
        if ema_netG is not None:
            with torch.no_grad():
                for p_ema, p in zip(ema_netG.parameters(), netG.parameters()):
                    p_ema.copy_(p)
            if hasattr(ema_netG, 'n_averaged'):
                ema_netG.n_averaged.zero_()
        # Prefer epoch encoded in filename: gens_7_389.pt -> resume at 390
        start_epoch = 0
        import re
        m = re.search(r'gens_7_(\d+)\.pt$', os.path.basename(path))
        if m:
            start_epoch = int(m.group(1)) + 1
        print(f'[resume] loaded generator-only weights from {path}; '
              f'optimizer/EMA/scheduler start fresh; start_epoch={start_epoch}.')
        return start_epoch, 0

    raise RuntimeError(f'Unrecognized checkpoint format at {path}')


def train(args, netG, netD):

    train_ds = TrainData(config.TRAIN.data_path, s)
    valid_ds = ValidData(config.VALID.data_path, s)
    
    I_S2, I_VE, _, _ = next(iter(train_ds))
    I_S2_np = np.moveaxis(np.array(I_S2), 0, -1)
    I_VE_np = np.array(I_VE)

    mtf_h = gen_mtf(s.ratio, 'S2')
    # spectral_loss = losses.SpectralLossNocorr([mtf_h], 0, np.array(I_VE_np[0].shape), s.ratio, device).to(device)
    spectral_loss = nn.L1Loss().to(device)
    # spectral_loss = nn.HuberLoss().to(device)
    feature_criterion = ContentLoss("vgg19", False, 1000, config.TRAIN.vgg_weights_path, config.TRAIN.vgg_features, config.TRAIN.vgg_weights, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)
    tv_loss = losses.TVLoss()
    cri_artifacts = nn.L1Loss()
    ndvi_loss = nn.L1Loss()

    train_ds_img_nums = len(train_ds)
    dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    valloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    optimizerG = optim.Adam(netG.parameters(), config.TRAIN.lr_init)
    # adversarial_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # NOTE: SenHAT-compatible checkpoint resume is performed after the optimizers,
    # schedulers, and EMA model exist (see the load_resume_checkpoint call below).
    # The old StyleGAN-specific key remapping (conv_rgb1/.trunk./module.generator/
    # stylegan_latest_all.pt) has been removed -- it does not apply to SenHAT.

    start_time = time.time()
    iters = 0

    I_D = netD(I_VE.unsqueeze(0).to(device))
    _, _, H, W = I_D.shape
    
    # target_real = torch.full([batch_size, 1], 1.0, dtype=torch.float, device=device)
    # target_fake = torch.full([batch_size, 1], 0.0, dtype=torch.float, device=device)
    target_real = torch.full([batch_size, 1, H, W], 1.0, dtype=torch.float, device=device)
    target_fake = torch.full([batch_size, 1, H, W], 0.0, dtype=torch.float, device=device)

    # SenHAT has no separate pretrained latent bank, so the generator trains as a
    # single parameter group (the old StyleGAN low-LR split no longer applies).
    optimizerD = optim.Adam(netD.parameters(), config.TRAIN.lr_adv)
    optimizerG = optim.Adam(netG.parameters(), config.TRAIN.lr_init)

    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=100, gamma=0.8)
    # schedulerG = optim.lr_scheduler.CyclicLR(optimizerG, base_lr=1e-5, max_lr=1e-4, step_size_up=50, cycle_momentum=False)
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=100, gamma=0.8)
    # schedulerD = optim.lr_scheduler.CyclicLR(optimizerD, base_lr=1e-5, max_lr=1e-4, step_size_up=50, cycle_momentum=False)

    # ---- SenHAT-compatible resume (generator + EMA + optimizer + scheduler + epoch/iters) ----
    start_epoch = 0
    if args.resume:
        resume_path = args.checkpoint if args.checkpoint is not None else f'{checkpoint_dir}resume_7.pt'
        if os.path.exists(resume_path):
            start_epoch, iters = load_resume_checkpoint(
                resume_path, netG, ema_netG, optimizerG, schedulerG,
                netD, optimizerD, schedulerD, device=device)
            print(f'[resume] loaded {resume_path}; resuming at epoch {start_epoch} (iters={iters}).')
        else:
            print(f'[resume] no checkpoint at {resume_path}; starting from scratch.')

    print('Started Training ...')

    for epoch in range(start_epoch, n_epoch):
        lr = schedulerG.optimizer.param_groups[0]['lr']

        for i, data in enumerate(dataloader):
            iters += 1

            netG.train()
            netD.train()
            ema_netG.eval()
            
            ms, ven_up, ms_rgb_up, ms_up = data

            ven_up = ven_up.to(device)
            ms = ms.to(device)
            ms_rgb_up = ms_rgb_up.to(device)
            ms_up = ms_up.to(device)
            
            ms_fake = netG(ms, ms_rgb_up)
            ms_fake_ema = ema_netG(ms, ms_rgb_up)

            if not fidelity_only:
                D_real = netD(ven_up)
                alpha = 0.8
                D_gen2 = netD(ms_fake)
                D_gen2_map = 1 - torch.sigmoid(D_gen2)*alpha
            
            # Sharpening
            ms_up_usm = usm_sharpener(ms_up)
            ven_up_usm = usm_sharpener(ven_up)

            # Generator
            # start training the generator model
            # Disable discriminator backpropagation during generator training
            for d_parameters in netD.parameters():
                d_parameters.requires_grad = False

            netG.zero_grad(set_to_none=True)
            
            # print(ms_fake.shape, ven_up.shape)
            # print(ms_fake[:,3].unsqueeze(1).shape, ven_up[:,3].unsqueeze(1).shape)
            # with amp.autocast():
            if do_usm:
                if disc_spec:
                    errG_spec_ven = spectral_loss(torch.mul(ms_fake, D_gen2_map), torch.mul(ven_up_usm, D_gen2_map))*0.8
                    errG_spec_ms = spectral_loss(torch.mul(ms_fake, D_gen2_map), torch.mul(ms_up_usm, D_gen2_map))*0.2
                else:
                    errG_spec_ven = spectral_loss(ms_fake, ven_up_usm)*0.8
                    errG_spec_ms = spectral_loss(ms_fake, ms_up_usm)*0.2
                    
                content_loss_rgb = feature_criterion(ms_fake[:,:3].flip(1), ven_up_usm[:,:3].flip(1))[0].sum()*0.35
                content_loss_nir = feature_criterion(ms_fake[:,3].unsqueeze(1).repeat(1, 3, 1, 1), ven_up_usm[:,3].unsqueeze(1).repeat(1, 3, 1, 1))[0].sum()*0.15
                content_loss_20m = feature_criterion(ms_fake[:,4:7].flip(1), ven_up_usm[:,4:7].flip(1))[0].sum()*0.35
                content_loss_vnir = feature_criterion(ms_fake[:,7].unsqueeze(1).repeat(1, 3, 1, 1), ven_up_usm[:,7].unsqueeze(1).repeat(1, 3, 1, 1))[0].sum()*0.15
                content_loss = content_loss_nir + content_loss_rgb + content_loss_vnir + content_loss_20m
            else: 
                if disc_spec:
                    errG_spec_ven = spectral_loss(torch.mul(ms_fake, D_gen2_map), torch.mul(ven_up, D_gen2_map))*0.8
                    errG_spec_ms = spectral_loss(torch.mul(ms_fake, D_gen2_map), torch.mul(ms_up, D_gen2_map))*0.2
                else:
                    errG_spec_ven = spectral_loss(ms_fake, ven_up)*0.8
                    errG_spec_ms = spectral_loss(ms_fake, ms_up)*0.2
                    
                # content_loss = feature_criterion(ms_fake.flip(1)[:,1:], ven_up.flip(1)[:,1:])[0].sum()
                content_loss_rgb = feature_criterion(ms_fake[:,:3].flip(1), ven_up[:,:3].flip(1))[0].sum()*0.35
                content_loss_nir = feature_criterion(ms_fake[:,3].unsqueeze(1).repeat(1, 3, 1, 1), ven_up[:,3].unsqueeze(1).repeat(1, 3, 1, 1))[0].sum()*0.15
                content_loss_20m = feature_criterion(ms_fake[:,4:7].flip(1), ven_up[:,4:7].flip(1))[0].sum()*0.35
                content_loss_vnir = feature_criterion(ms_fake[:,7].unsqueeze(1).repeat(1, 3, 1, 1), ven_up[:,7].unsqueeze(1).repeat(1, 3, 1, 1))[0].sum()*0.15
                content_loss = content_loss_nir + content_loss_rgb + content_loss_vnir + content_loss_20m
            
            errG_spec = errG_spec_ven + errG_spec_ms
            
            errG_tv = tv_loss(ms_fake)
            errG_ndvi_10m = ndvi_loss((ms_fake[:,3].unsqueeze(1)-ms_fake[:,2].unsqueeze(1)), (ven_up[:,3].unsqueeze(1)-ven_up[:,2].unsqueeze(1)))
            errG_ndvi_20m = ndvi_loss((ms_fake[:,7].unsqueeze(1)-ms_fake[:,6].unsqueeze(1)), (ven_up[:,7].unsqueeze(1)-ven_up[:,6].unsqueeze(1)))
            errG_ndvi = errG_ndvi_10m + errG_ndvi_20m
            #gen_spatial_loss, gen_struct_no_threshold = LStruct(ms_fake, ven_up, thresh)
            pixel_weight = get_refined_artifact_map(ven_up, ms_fake, ms_fake_ema, 7)
            errG_artifacts = cri_artifacts(torch.mul(pixel_weight, ms_fake), torch.mul(pixel_weight, ven_up))

            if fidelity_only:
                # Fidelity stage: MS-SSIM and adversarial terms are disabled.
                errG_ssim = torch.zeros((), device=device)
                errG_adv = torch.zeros((), device=device)
            else:
                errG_ssim = 1 - ms_ssim_module(ms_fake, ven_up)
                errG_adv_fake = adversarial_criterion(D_gen2, target_real)*0.5
                errG_adv_real = adversarial_criterion(D_real, target_fake)*0.5
                # Relativistic Disc
                # errG_adv_fake = adversarial_criterion(D_gen2 - torch.mean(D_real), target_real)*0.5
                # errG_adv_real = adversarial_criterion(D_real - torch.mean(D_gen2), target_fake)*0.5
                errG_adv = errG_adv_fake + errG_adv_real
            # errG = (8*errG_spec) + (0.25*errG_adv) + (0.05*content_loss) + (5*errG_tv) + (1e3*errG_artifacts) #+ (0.1*errG_spat)
            # errG = (80*errG_spec) + (0.2*errG_adv) + (3*content_loss) + (5e3*errG_artifacts)  # Current best
            # errG = (10*errG_spec) + (0.2*errG_adv) + (2*content_loss) + (1e3*errG_artifacts)   # 6_479
            
            # errG = (15*errG_spec) + (0.3*errG_adv) + (2*content_loss) + (5e2*errG_artifacts) + (errG_ssim*10) + (0*errG_tv) + (errG_ndvi*5)
            errG = (15*errG_spec) + (0.5*errG_adv) + (1.5*content_loss) + (5e2*errG_artifacts) + (errG_ssim*4) + (0*errG_tv) + (errG_ndvi*5)
            # errG = (15*errG_spec) + (0.3*errG_adv) + (3*content_loss) + (5e2*errG_artifacts) + (errG_ssim*10) + (7*errG_tv) + (errG_ndvi*10)
            # (6*errG_spec) + (0.45*errG_adv) + (0.06*content_loss) + (2*errG_tv) + (5e3*errG_artifacts)
            # (4*errG_spec) + (0.25*errG_adv) + (0.04*content_loss) + (8*errG_tv) + (1e3*errG_artifacts)
            
            errG.backward()
            optimizerG.step()

            if fidelity_only:
                # Discriminator kept unchanged but inactive during the fidelity stage.
                errD = torch.zeros((), device=device)
                D_gen = None
                d_gt_probability = torch.zeros(())
                d_sr_probability = torch.zeros(())
            else:
                # Discriminator
                # start training the discriminator model
                # During discriminator model training, enable discriminator model backpropagation
                for d_parameters in netD.parameters():
                    d_parameters.requires_grad = True

                # Initialize the discriminator model gradients
                netD.zero_grad(set_to_none=True)

                # Calculate the classification score of the discriminator model for real samples
                D_real = netD(ven_up)
                errD_real = adversarial_criterion(D_real, target_real)
                # # Relativistic Discriminator
                # errD_real = adversarial_criterion(D_real - torch.mean(D_gen), target_real)
                errD_real.backward()

                # Calculate the classification score of the discriminator model for fake samples
                D_gen = netD(ms_fake.detach().clone())
                errD_fake = adversarial_criterion(D_gen, target_fake)
                # # Relativistic Discriminator
                # errD_fake = adversarial_criterion(D_gen - torch.mean(D_real), target_fake)
                errD = errD_real + errD_fake
                errD_fake.backward()
                optimizerD.step()

                d_gt_probability = torch.sigmoid_(torch.mean(errD_real.detach().cpu()))
                d_sr_probability = torch.sigmoid_(torch.mean(errD_fake.detach().cpu()))

            # update exponential average model weights (kept: the LDL artifact loss
            # depends on the EMA generator's outputs)
            ema_netG.update_parameters(netG)
            
            if i%10 == 0:
                # print(errD.item(), errG.item(), errG_spec.item(), 0.001*errG_adv.item())
                # Testing with bicubic interpolation
                test_data = next(iter(valloader))

                metrics, metrics_inter = test_single(test_data, netG, netD, iters)
                print(f'[{epoch}/{n_epoch}][{i}/{len(dataloader)}] errD: {errD.item():.4f} Disc_Prob: ({d_gt_probability.item():.3f}/{d_sr_probability.item():.3f})'
                    f' errG: {errG.item():.4f} Spec/TV/Adv/Cont/Artif/SSIM/NDVI: ({errG_spec.item():.4f}/{errG_tv.item():.4f}/{errG_adv.item():.4f}/{content_loss.item():.4f}/{errG_artifacts.item():.7f}/{errG_ssim.item():.4f}/{errG_ndvi.item():.4f})\n'
                f' RMSE(our/interp): ({metrics["RMSE"]:.4f}/{metrics_inter["RMSE"]:.4f})\n'
                f' PSNR(our/interp): ({metrics["PSNR"]:.4f}/{metrics_inter["PSNR"]:.4f})\n'
                f' SSIM(our/interp): ({metrics["SSIM"]:.4f}/{metrics_inter["SSIM"]:.4f})\n'
                f' NIQE_RGB(our/interp): ({metrics["NIQE_RGB"]:.4f}/{metrics_inter["NIQE_RGB"]:.4f})\n'
                f' NIQE_20m(our/interp): ({metrics["NIQE_20m"]:.4f}/{metrics_inter["NIQE_20m"]:.4f})\n'
                f' SAM(our/interp): ({metrics["SAM"]:.4f}/{metrics_inter["SAM"]:.4f})\n'
                f' UQI(our/interp): ({metrics["UQI"]:.4f}/{metrics_inter["UQI"]:.4f})\n'
                f' SCC(our/interp): ({metrics["SCC"]:.4f}/{metrics_inter["SCC"]:.4f})\n'
                )
                
                ms_fake = ms_fake.detach().cpu()
                ms_up = ms_up.detach().cpu()
                ven_up = ven_up.detach().cpu()
                ms_rgb_up = ms_rgb_up.detach().cpu()
                if i%20==0 and (D_gen is not None):
                    d_gen = torch.sigmoid(D_gen.detach().cpu())
                    savemat_plot_senven_RGBnorm(ms_fake[:,:4], ms_up[:,:4], ven_up[:,:4], save_dir, s, d_gen, filename=f'plot_train_rgbn_{iters}.png')
                    savemat_plot_senven_20mnorm(ms_fake[:,4:], ms_up[:,4:], ven_up[:,4:], save_dir, s, d_gen, filename=f'plot_train_20m_{iters}.png')
                else:
                    d_gen = torch.abs(ms_fake - ms_up).mean(1)
                    savemat_plot_senven_RGBnorm(ms_fake[:,:4], ms_up[:,:4], ven_up[:,:4], save_dir, s, d_gen, abs_diff=True, filename=f'plot_train_rgbn_{iters}.png')
                    savemat_plot_senven_20mnorm(ms_fake[:,4:], ms_up[:,4:], ven_up[:,4:], save_dir, s, d_gen, abs_diff=True, filename=f'plot_train_20m_{iters}.png')
                
                if args.do_log:
                    image_rgb = wandb.Image(os.path.join(save_dir, f'plot_rgbn_{iters}.png'), caption=f"Iteration RGBN {iters}")
                    image_20m = wandb.Image(os.path.join(save_dir, f'plot_20m_{iters}.png'), caption=f"Iteration 20m {iters}")
                    train_rgb_image = wandb.Image(os.path.join(save_dir, f'plot_train_rgbn_{iters}.png'), caption=f"Iteration RGBN {iters}")
                    train_20m_image = wandb.Image(os.path.join(save_dir, f'plot_train_20m_{iters}.png'), caption=f"Iteration 20m {iters}")
                    test_hist = wandb.Image(os.path.join(save_dir, f'plot_rgbn_{iters}_hist.png'), caption=f"Histogram Iteration {iters}")
                    wandb.log({"errD_loss": errD.item(), 'errG_loss':errG.item(), 'Spectral loss':errG_spec.item(), 'TV loss':errG_tv.item(),
                            'Content loss':content_loss.item(), "Artifact Loss":errG_artifacts.item(), "SSIM Loss":errG_ssim.item(), 'RMSE':metrics["RMSE"], 'RMSE_interpolate':metrics_inter["RMSE"], 'PSNR':metrics["PSNR"], 
                            'PSNR_interpolate':metrics_inter["PSNR"], 'SSIM': metrics["SSIM"], 'SSIM_interpolate': metrics_inter["SSIM"], 'Learning Rate':lr, 
                            "Plots_RGB":image_rgb, "Plots_20m":image_20m, "Plots RGB training":train_rgb_image, "Plots 20m training":train_20m_image, "Histogram testing":test_hist})
                
            
        if epoch%10 == 9:
            # Generator-only weights kept for eval/inference compatibility.
            _atomic_torch_save(netG.state_dict(), f'{checkpoint_dir}gens_7_{epoch}.pt')
            _atomic_torch_save(netD.state_dict(), f'{checkpoint_dir}discs_7_{epoch}.pt')
            # Consolidated, SenHAT-compatible resume checkpoint (rolling latest).
            save_resume_checkpoint(f'{checkpoint_dir}resume_7.pt', epoch, iters,
                                   netG, ema_netG, optimizerG, schedulerG,
                                   netD, optimizerD, schedulerD)
        schedulerG.step()
        schedulerD.step()
        print(f'Time Elapsed: {(time.time()-start_time): .2f}')



if __name__ == '__main__':
    import argparse
    # os.environ["CUDA_LAUNCH_BLOCKING"] = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--logging', action='store_true', dest='do_log', help='do wandb logging or not')
    parser.add_argument('--disc_spec', action='store_true', dest='disc_spec', help='do wandb logging or not')
    parser.add_argument('--do_usm', action='store_true', dest='do_usm', help='do wandb logging or not')
    parser.add_argument('--name', type=str, default='SRGAN', help='w&b runtime name in UI')
    parser.add_argument('--resume', action='store_true', dest='resume', help='load pretrained weights or not')
    parser.add_argument('--enable_gan', action='store_true', dest='enable_gan', help='enable adversarial + MS-SSIM (GAN stage); default off = fidelity-only (L1+VGG+LDL+NDI)')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--batch', type=int, default=None, help='path to checkpoint')
    args = parser.parse_args()

    # tlx.global_flag['mode'] = args.mode

    # if tlx.global_flag['mode'] == 'train':
    if args.do_log:
        import wandb

        run = wandb.init(
            # Set the project where this run will be logged
            project="ESRGAN_sentinel_glean_all",
            # resume=True,
            name=args.name,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": config.TRAIN.lr_adv,
                "epochs": config.TRAIN.n_epoch,
        })

    ###====================== HYPER-PARAMETERS ===========================###
    if args.batch == None:
        batch_size = config.TRAIN.batch_size
    else:
        batch_size = args.batch

    n_epoch_init = config.TRAIN.n_epoch_init
    n_epoch = config.TRAIN.n_epoch
    
    if args.do_usm == True:
        do_usm = args.do_usm
    else:
        do_usm = config.TRAIN.use_usm
    print(do_usm)
    
    # Output directories. On HPC, set SENHAT_CHECKPOINT_DIR and SENHAT_SAVE_DIR
    # to persistent /scratch paths; training data is staged separately to
    # $SLURM_TMPDIR via SENHAT_TRAIN_DATA / SENHAT_VALID_DATA (see config.py).
    save_dir = os.environ.get('SENHAT_SAVE_DIR', '../srgan_samples_sent_glean_all/')
    checkpoint_dir = os.environ.get('SENHAT_CHECKPOINT_DIR', '../weights/')
    # Ensure trailing slash so f'{checkpoint_dir}gens_...' does not become weightsgens_...
    if not checkpoint_dir.endswith(os.sep):
        checkpoint_dir = checkpoint_dir + os.sep
    if not save_dir.endswith(os.sep):
        save_dir = save_dir + os.sep
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    print(f"train data: {config.TRAIN.data_path}")
    print(f"valid data: {config.VALID.data_path}")
    print(f"checkpoints: {checkpoint_dir}")
    print(f"samples: {save_dir}")

    s = Sensor('S2')   ## Hardcoded
    # Use whatever GPUs are actually available. On CPU this is [], and DataParallel
    # with an empty device list acts as a transparent passthrough while still
    # exposing `.module` (which the training loop relies on).
    if torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = []

    # Generative latent bank swapped from StyleGAN to a HAT backbone (SenHAT).
    # Same forward(lq, rgb) contract and output shape as SenGLEANStyleGANv2.
    from networks.senhat import SenHAT
    netG = SenHAT(64, 256, 4, 64, num_rrdbs=config.NETWORK.NUM_RRDB, rgb_channels=4,
                  hat_variant=config.NETWORK.HAT_VARIANT).to(device)
    netG = nn.DataParallel(netG, device_ids=gpu_ids)
    
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

    model_ema_decay = 0.99
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - model_ema_decay) * averaged_model_parameter + model_ema_decay * model_parameter
    ema_netG = AveragedModel(netG, avg_fn=ema_avg)

    # Unet based discriminator (kept unchanged; inactive during the fidelity stage)
    netD = UNetDiscriminatorSN(num_in_ch=8).to(device)
    netD = nn.DataParallel(netD,  device_ids=gpu_ids)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    usm_sharpener = USMSharp().to(device)
    ssim_module = SSIM(data_range=1, size_average=True, channel=8)
    ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=8)
    niqe = NIQE(s.ratio//2, './model/niqe_model.mat').to(device)
    niqe4 = NIQE(s.ratio, './model/niqe_model.mat').to(device)
    # Fidelity-only stage by default: only L1 + VGG + LDL + NDI (+ TV@0) are active;
    # adversarial and MS-SSIM are disabled and the discriminator stays inactive.
    fidelity_only = not args.enable_gan
    # disc_spec masks the pixel loss with the discriminator output, so it is
    # GAN-coupled and cannot be used without the discriminator active.
    disc_spec = args.disc_spec and (not fidelity_only)
    if args.disc_spec and fidelity_only:
        print("[warn] --disc_spec ignored in fidelity-only mode (requires GAN).")
    print("DS", disc_spec, "| fidelity_only", fidelity_only)

    train(args, netG, netD)
