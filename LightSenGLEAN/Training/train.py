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
    feature_criterion = ContentLoss("vgg19", False, 1000, None, config.TRAIN.vgg_features, config.TRAIN.vgg_weights, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)
    tv_loss = losses.TVLoss()
    cri_artifacts = nn.L1Loss()
    ndvi_loss = nn.L1Loss()

    train_ds_img_nums = len(train_ds)
    dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    valloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    optimizerG = optim.Adam(netG.parameters(), config.TRAIN.lr_init)
    # adversarial_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCEWithLogitsLoss()

    if (args.checkpoint != None)&(args.resume == True):
        new_params = netG.state_dict()
        params = torch.load(args.checkpoint, map_location=torch.device(device))
        params['module.conv_rgb2.weight'] = params['module.conv_rgb1.weight']
        params['module.conv_rgb2.bias'] = params['module.conv_rgb1.bias']
        params['module.upsampling2.0.weight'] = params['module.upsampling1.0.weight']
        params['module.upsampling2.0.bias'] = params['module.upsampling1.0.bias']
        for key in new_params.keys():
            if '.trunk.' in key:
                new_key = key.replace('.trunk.', '.trunk2.')
                params[new_key] = params[key]
        # params['module.trunk2.weight'] = params['module.trunk.weight']
        # params['module.trunk2.bias'] = params['module.trunk.bias']
        
        netG.load_state_dict(params, strict=False)
        
    elif (args.resume==True):
        print('Loading pre-defined default models')
        # Adversarial Training
        checkpoint_dict = torch.load(f'{checkpoint_dir}gens_3_99.pt', map_location=torch.device(device))
        # checkpoint_dict = torch.load(f'../checkpoints_glean_all/gens_7_69.pt', map_location=torch.device(device))
        netG, _ = on_load_checkpoint(netG, checkpoint_dict, strict=False)
        
        # netG.module.generator.load_state_dict(torch.load('./stylegan/stylegan_latest_all.pt', map_location=torch.device(device)), strict=False)
        # netG.generator.requires_grad_(False)
        # netG.module.generator.requires_grad_(False)
        netG.module.generator.requires_grad_(True)
        # netG.load_state_dict(torch.load(f'{checkpoint_dir}gens_1_29.pt', map_location=torch.device(device)), strict=False)
        # 0_399 {checkpoint_dir}
    
    params = torch.load(f'{checkpoint_dir}discs_3_99.pt', map_location=torch.device(device))
    # checkpoint_dict = torch.load(f'../checkpoints_glean_all/discs3n_12_49.pt', map_location=torch.device(device))
    # netD, _ = on_load_checkpoint(netD, checkpoint_dict, strict=True)
    # params = torch.load(f'../checkpoints_glean_all/discs_7_69.pt', map_location=torch.device(device))
    # new_params = netD.state_dict()
    # new_params['module.conv0.weight'][:,:3] = params['module.conv0.weight'][:,:3]
    # new_params['module.conv0.weight'][:,3] = params['module.conv0.weight'][:,0]
    # params['module.conv0.weight'] = new_params['module.conv0.weight']
    netD.load_state_dict(params)
    
    # netD.module.load_state_dict(torch.load(f'{checkpoint_dir}RealESRGAN_x4plus_netD.pth', map_location=torch.device(device))['params'])
    # discs22_6_399.pt
    
    # for param in netG.parameters():
    #     param.requires_grad = False
    # for param in netG.module.conv1.parameters():
    #     param.requires_grad = True
    # for param in netG.module.conv4.parameters():
    #     param.requires_grad = True
        
    start_time = time.time()
    iters = 0

    I_D = netD(I_VE.unsqueeze(0).to(device))
    _, _, H, W = I_D.shape
    
    # target_real = torch.full([batch_size, 1], 1.0, dtype=torch.float, device=device)
    # target_fake = torch.full([batch_size, 1], 0.0, dtype=torch.float, device=device)
    target_real = torch.full([batch_size, 1, H, W], 1.0, dtype=torch.float, device=device)
    target_fake = torch.full([batch_size, 1, H, W], 0.0, dtype=torch.float, device=device)

    optimizer_dict = torch.load(f'{checkpoint_dir}optimizer_2_.pt', map_location=torch.device(device))
    all_parameters = set(netG.parameters())
    generator_params = set(netG.module.generator.parameters())

    not_gen = all_parameters - generator_params
    gen_param = list(generator_params)
    not_gen_param = list(not_gen)
    optimizerD = optim.Adam(netD.parameters(), config.TRAIN.lr_adv)
    optimizerG = optim.Adam(
                    [{"params": not_gen_param},
                    {"params": gen_param, "lr": 1e-7}  # 公用层learning_rate应取平均
                    ], config.TRAIN.lr_adv)
    # optimizerG.load_state_dict(optimizer_dict['optimizerG'])
    optimizerD.load_state_dict(optimizer_dict['optimizerD'])
    
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=100, gamma=0.8)
    # schedulerG = optim.lr_scheduler.CyclicLR(optimizerG, base_lr=1e-5, max_lr=1e-4, step_size_up=50, cycle_momentum=False)
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=100, gamma=0.8)
    # schedulerD = optim.lr_scheduler.CyclicLR(optimizerD, base_lr=1e-5, max_lr=1e-4, step_size_up=50, cycle_momentum=False)
    
    print('Started Training ...')

    for epoch in range(n_epoch):
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
            errG_ssim = 1 - ms_ssim_module(ms_fake, ven_up)
            #gen_spatial_loss, gen_struct_no_threshold = LStruct(ms_fake, ven_up, thresh)
            pixel_weight = get_refined_artifact_map(ven_up, ms_fake, ms_fake_ema, 7)
            errG_artifacts = cri_artifacts(torch.mul(pixel_weight, ms_fake), torch.mul(pixel_weight, ven_up))

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

            # update exponential average model weights
            ema_netG.update_parameters(netG)
            
            d_gt_probability = torch.sigmoid_(torch.mean(errD_real.detach().cpu()))
            d_sr_probability = torch.sigmoid_(torch.mean(errD_fake.detach().cpu()))
            
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
                if i%20==0:
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
            torch.save(netG.state_dict(), f'{checkpoint_dir}gens_4_{epoch}.pt')
            torch.save(netD.state_dict(), f'{checkpoint_dir}discs_4_{epoch}.pt')
            torch.save({'optimizerG': optimizerG.state_dict(),
                        'optimizerD': optimizerD.state_dict()},
                        f'{checkpoint_dir}optimizer_4_.pt')
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
    parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--batch', type=int, default=None, help='path to checkpoint')
    args = parser.parse_args()

    # tlx.global_flag['mode'] = args.mode

    # if tlx.global_flag['mode'] == 'train':
    if args.do_log:
        import wandb

        run = wandb.init(
            # Set the project where this run will be logged
            project="ESRGAN_sentinel_Lglean_all",
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
    
    # create folders to save result images and trained models
    save_dir = "../srgan_samples_sent_Lglean_all/"
    checkpoint_dir = "../weights/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    s = Sensor('S2')   ## Hardcoded
    gpu_ids = [0, 1, 2, 3]

    from networks.glean import LightSenGLEANStyleGANv2
    netG = LightSenGLEANStyleGANv2(64, 256, 4, 64, 23, rgb_channels=4).to(device)
    # netG.generator.load_state_dict(torch.load('./stylegan/stylegan_latest9.pt', map_location=torch.device(device)))
    # # netG.generator.requires_grad_(False)
    # netG.generator.requires_grad_(True)
    netG = nn.DataParallel(netG,  device_ids = gpu_ids)
    
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

    model_ema_decay = 0.99
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - model_ema_decay) * averaged_model_parameter + model_ema_decay * model_parameter
    ema_netG = AveragedModel(netG, avg_fn=ema_avg)

    # Unet based discriminator
    netD = UNetDiscriminatorSN(num_in_ch=8).to(device)
    netD = nn.DataParallel(netD,  device_ids=gpu_ids)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    usm_sharpener = USMSharp().to(device)
    ssim_module = SSIM(data_range=1, size_average=True, channel=8)
    ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=8)
    niqe = NIQE(s.ratio//2, './model/niqe_model.mat').to(device)
    niqe4 = NIQE(s.ratio, './model/niqe_model.mat').to(device)
    disc_spec = args.disc_spec
    print("DS", disc_spec)

    train(args, netG, netD)
