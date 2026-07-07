import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))
sys.path.append(str(Path('.').absolute().parent.parent))

os.environ['TL_BACKEND'] = 'torch'
os.environ['WANDB_API_KEY']='ENTER WANDB KEY'
os.environ['WANDB_ENTITY']='ENTER USERNAME'

#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
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
from utils import savemat_plot_senven_RGBnorm, psnr, psnr_bands, rmse, rmse_bands, center_crop, sam_batch, uqi_batch, scc_batch
from datasets.sen2ven_data_RGBN2x_norm import TrainData, ValidData
from datasets.img_process_utils import USMSharp
from datasets.niqe_iqa import NIQE
from pytorch_msssim import SSIM, MS_SSIM
from utils import on_load_checkpoint, recover_bandwise_RGB
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_single(data, net_g, net_d, iters):

    netG.eval()
    ms, ven, ven_up, ms_up = data

    ven = ven.to(device)
    ms = ms.to(device)
    ven_up = ven_up.to(device)
    ms_up = ms_up.to(device)
    batch_size = ven.size()[0]
    ms_fake = net_g(ms)
    d_gen = torch.sigmoid(net_d(ms_fake)).detach().cpu()
    
    ms_fake = ms_fake.detach().cpu()
    ms_up = ms_up.detach().cpu()
    ven_up = ven_up.detach().cpu()
    _, _, H, W = ms_fake.shape

    ms_fake_crop = center_crop(ms_fake, int(H*0.9))
    ven_up_crop = center_crop(ven_up, int(H*0.9))
    ms_up_crop = center_crop(ms_up, int(H*0.9))

    metrics, metrics_bi = {}, {}
    metrics['PSNR'] = psnr(ms_fake_crop, ven_up_crop)
    metrics_bi['PSNR'] = psnr(ms_up_crop, ven_up_crop)
    
    psnr_bands_res = psnr_bands(ms_fake_crop.detach().cpu(), ven_up_crop.detach().cpu())
    psnr_bands_res_bi = psnr_bands(ms_up_crop.detach().cpu(), ven_up_crop.detach().cpu())
    print('PSNR_bands:', psnr_bands_res, '/', psnr_bands_res_bi)

    metrics['RMSE'] = rmse(recover_bandwise_RGB(ms_fake_crop), recover_bandwise_RGB(ven_up_crop))    # Caution
    metrics_bi['RMSE'] = rmse(recover_bandwise_RGB(ms_up_crop), recover_bandwise_RGB(ven_up_crop))      # Caution
    print('-----------------')
    rmse_bands_res = rmse_bands(recover_bandwise_RGB(ms_fake_crop), recover_bandwise_RGB(ven_up_crop))
    rmse_bands_res_bi = rmse_bands(recover_bandwise_RGB(ms_up_crop), recover_bandwise_RGB(ven_up_crop))
    print('RMSE_bands:', rmse_bands_res, '/', rmse_bands_res_bi)
    
    metrics['SSIM'] = ssim_module(ms_fake_crop, ven_up_crop)
    metrics_bi['SSIM'] = ssim_module(ms_up_crop, ven_up_crop)
    
    metrics['NIQE'] = niqe(ms_fake_crop.flip(1)[:,1:]*255).mean().item()
    metrics_bi['NIQE'] = niqe(ms_up_crop.flip(1)[:,1:]*255).mean().item()
    
    metrics['SAM'] = sam_batch(ven_up_crop[:,:3].detach().cpu(), ms_fake_crop[:,:3].detach().cpu())
    metrics_bi['SAM'] = sam_batch(ven_up_crop[:,:3].detach().cpu(), ms_up_crop[:,:3].detach().cpu())

    metrics['UQI'] = uqi_batch(ven_up_crop[:,:].detach().cpu(), ms_fake_crop[:,:].detach().cpu())
    metrics_bi['UQI'] = uqi_batch(ven_up_crop[:,:].detach().cpu(), ms_up_crop[:,:].detach().cpu())
    
    metrics['SCC'] = scc_batch(ven_up_crop[:,:].detach().cpu(), ms_fake_crop[:,:].detach().cpu())
    metrics_bi['SCC'] = scc_batch(ven_up_crop[:,:].detach().cpu(), ms_up_crop[:,:].detach().cpu())
    
    savemat_plot_senven_RGBnorm(ms_fake, ms_up, ven_up, save_dir, s, d_gen, filename=f'plot_{iters}.png')

    return metrics, metrics_bi


def train(args, batch_size, netG, netD):

    train_ds = TrainData(config.TRAIN.data_path, s)
    valid_ds = ValidData(config.VALID.data_path, s)
    
    I_S2, _, I_VE, _ = next(iter(train_ds))
    I_S2_np = np.moveaxis(np.array(I_S2), 0, -1)
    I_VE_np = np.array(I_VE)

    mtf_h = gen_mtf(s.ratio, 'S2')
    spectral_loss = nn.L1Loss().to(device)
    feature_criterion = ContentLoss("vgg19", False, 1000, None, config.TRAIN.vgg_features, config.TRAIN.vgg_weights, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)
    tv_loss = losses.TVLoss()
    cri_artifacts = nn.L1Loss()
    ndvi_loss = nn.L1Loss()
    # psnr_criterion = nn.MSELoss().to(device)

    train_ds_img_nums = len(train_ds)
    dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)
    valloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)

    optimizerG = optim.Adam(netG.parameters(), config.TRAIN.lr_init)
    # adversarial_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCEWithLogitsLoss()

    if (args.checkpoint != None)&(args.resume == True):
        checkpoint_dict = torch.load(f'./weights/gens2x_10_99.pt', map_location=torch.device(device))
        netG, _ = on_load_checkpoint(netG, checkpoint_dict, strict=False)
        
    elif (args.resume==True):
        print('Loading pre-defined default models')
        # Adversarial Training
        netG.load_state_dict(torch.load(f'{checkpoint_dir}gens2xn_15_89.pt', map_location=torch.device(device)), strict=False)
        # netG.load_state_dict(torch.load(f'{checkpoint_dir}gens2xesrgan_t299.pt', map_location=torch.device(device)), strict=False)
        # 0_399
    
    # checkpoint_dict = torch.load(f'../checkpoints_RGB/discs2x_10_99.pt', map_location=torch.device(device))
    # netD, _ = on_load_checkpoint(netD, checkpoint_dict, strict=True)
        
    netD.load_state_dict(torch.load(f'../weights/discs_rgbn.pt', map_location=torch.device(device)))
    
    start_time = time.time()
    iters = 0
    # scaler = amp.GradScaler()

    I_D = netD(I_VE.unsqueeze(0).to(device))
    _, _, H, W = I_D.shape
    
    target_real = torch.full([batch_size, 1, H, W], 1.0, dtype=torch.float, device=device)
    target_fake = torch.full([batch_size, 1, H, W], 0.0, dtype=torch.float, device=device)
# 
    optimizer_dict = torch.load(f'{checkpoint_dir}optimizer2xn_15_.pt', map_location=torch.device(device))
    # optimizer_dict = torch.load(f'{checkpoint_dir}optimizer2xesrgan_t.pt', map_location=torch.device(device))

    optimizerG = optim.Adam(netG.parameters(), 1e-4)
    optimizerD = optim.Adam(netD.parameters(), 5e-5)   # config.TRAIN.lr_adv/2
    optimizerG.load_state_dict(optimizer_dict['optimizerG'])
    optimizerD.load_state_dict(optimizer_dict['optimizerD'])
    
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=100, gamma=0.8)
    # schedulerG = optim.lr_scheduler.CyclicLR(optimizerG, base_lr=4e-5, max_lr=1e-4, step_size_up=50, cycle_momentum=False)
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=100, gamma=0.8)
    # schedulerD = optim.lr_scheduler.CyclicLR(optimizerD, base_lr=4e-5, max_lr=1e-4, step_size_up=50, cycle_momentum=False)

    for epoch in range(n_epoch):
        lr = schedulerG.optimizer.param_groups[0]['lr']

        for i, data in enumerate(dataloader):
            iters += 1

            netG.train()
            netD.train()
            ema_netG.eval()
            
            ms, ven, ven_up, ms_up = data
            
            # Adding degradation
            # ms = data_degradation(ms)

            ven = ven.to(device)
            ms = ms.to(device)
            ven_up = ven_up.to(device)
            ms_up = ms_up.to(device)
            batch_size = ven.size()[0]
            
            ms_fake = netG(ms)
            ms_fake_ema = ema_netG(ms)
            D_real = netD(ven_up)

            alpha = 0.8
            D_real_map = 1 - torch.sigmoid(D_real)*alpha
            D_gen2 = netD(ms_fake)
            D_gen2_map = 1 - torch.sigmoid(D_gen2)*alpha
            # print(ms.shape, ven.shape, ven_up.shape, ms_up.shape)

            # Sharpening
            ms_up_usm = usm_sharpener(ms_up)
            ven_up_usm = usm_sharpener(ven_up)

            # Generator
            # start training the generator model
            # Disable discriminator backpropagation during generator training
            for d_parameters in netD.parameters():
                d_parameters.requires_grad = False

            netG.zero_grad(set_to_none=True)
            
            # with amp.autocast():
            if do_usm:
                # print(ms_fake.shape, ven_up_usm.shape, "dfd")
                if disc_spec:
                    errG_spec_ven = spectral_loss(torch.mul(ms_fake, D_gen2_map), torch.mul(ven_up_usm, D_gen2_map))*0.8
                    errG_spec_ms = spectral_loss(torch.mul(ms_fake, D_gen2_map), torch.mul(ms_up_usm, D_gen2_map))*0.2
                else:
                    errG_spec_ven = spectral_loss(ms_fake, ven_up_usm)*0.8
                    errG_spec_ms = spectral_loss(ms_fake, ms_up_usm)*0.2
                    
                content_loss_rgb = feature_criterion(ms_fake[:,:3].flip(1), ven_up_usm[:,:3].flip(1))[0].sum()*0.5
                content_loss_nir = feature_criterion(ms_fake[:,3:].repeat(1, 3, 1, 1), ven_up_usm[:,3:].repeat(1, 3, 1, 1))[0].sum()*0.5
                content_loss = content_loss_nir + content_loss_rgb
            else:
                if disc_spec:
                    errG_spec_ven = spectral_loss(torch.mul(ms_fake, D_gen2_map), torch.mul(ven_up, D_gen2_map))*0.8
                    errG_spec_ms = spectral_loss(torch.mul(ms_fake, D_gen2_map), torch.mul(ms_up, D_gen2_map))*0.2
                else:
                    errG_spec_ven = spectral_loss(ms_fake, ven_up)*0.9
                    errG_spec_ms = spectral_loss(ms_fake, ms_up)*0.1
                    
                # content_loss = feature_criterion(ms_fake[:,:3].flip(1), ven_up[:,:3].flip(1))[0].sum()
                content_loss_rgb = feature_criterion(ms_fake[:,:3].flip(1), ven_up[:,:3].flip(1))[0].sum()*0.8
                content_loss_nir = feature_criterion(ms_fake[:,3:].repeat(1, 3, 1, 1), ven_up[:,3:].repeat(1, 3, 1, 1))[0].sum()*0.2
                content_loss = content_loss_nir + content_loss_rgb
            
            errG_spec = errG_spec_ven + errG_spec_ms
            
            # Try adding
            # errG_spat = 0
            ## thresh = threshold_calc(ven_up, ms_up)
            # chs = random.randint(0, 3)
            # # print(ven_up[:,chs].shape)
            # thresh = threshold_calc(ven_up[:,chs], ms_up)
            # errG_spat, _ = LStruct(ms_fake, torch.unsqueeze(ven_up[:,chs], dim=1), thresh)
            # print(errG_spat)
            
            errG_tv = tv_loss(ms_fake)
            errG_ndvi = ndvi_loss((ms_fake[:,3]-ms_fake[:,2]), (ven_up[:,3]-ven_up[:,2]))
            # print(ms_fake.shape, ven_up.shape)
            errG_ssim = 1 - ssim_module(ms_fake, ven_up)
            #gen_spatial_loss, gen_struct_no_threshold = LStruct(ms_fake, ven_up, thresh)
            pixel_weight = get_refined_artifact_map(ven_up, ms_fake, ms_fake_ema, 7)
            errG_artifacts = cri_artifacts(torch.mul(pixel_weight, ms_fake), torch.mul(pixel_weight, ven_up))

            errG_adv_fake = adversarial_criterion(D_gen2, target_real)*0.5
            errG_adv_real = adversarial_criterion(D_real, target_fake)*0.5
            # Relativistic Disc
            # errG_adv_fake = adversarial_criterion(D_gen2 - torch.mean(D_real), target_real)*0.5
            # errG_adv_real = adversarial_criterion(D_real - torch.mean(D_gen2), target_fake)*0.5
            
            errG_adv = errG_adv_fake + errG_adv_real
            
            errG = (10*errG_spec) + (errG_tv*8) + (0.1*errG_adv) + (2*content_loss) + (5e2*errG_artifacts) + (errG_ssim*5) + (errG_ndvi*10)
            
            # (15*errG_spec) + (errG_tv*2) + (0.5*errG_adv) + (4*content_loss) + (5e2*errG_artifacts) + (errG_ssim*5) + (errG_ndvi*10)  # Final
            
            # (35*errG_spec) + (0.2*errG_adv) + (1*content_loss) + (1e3*errG_artifacts) + (errG_ssim*3)    # 2x_9_219
            # scaler.scale(errG).backward()
            errG.backward()
            # scaler.step(optimizerG)
            # scaler.update()
            
            optimizerG.step()

            # Discriminator
            # start training the discriminator model
            # During discriminator model training, enable discriminator model backpropagation
            for d_parameters in netD.parameters():
                d_parameters.requires_grad = True
                
            # Initialize the discriminator model gradients
            netD.zero_grad(set_to_none=True)

            # Calculate the classification score of the discriminator model for real samples
            # with amp.autocast():
            D_real = netD(ven_up)
            errD_real = adversarial_criterion(D_real, target_real)
            # # Relativistic Discriminator
            # errD_real = adversarial_criterion(D_real - torch.mean(D_gen), target_real)
            # scaler.scale(errD_real).backward()
            errD_real.backward()
            
            # Calculate the classification score of the discriminator model for fake samples
            # with amp.autocast():
            D_gen = netD(ms_fake.detach().clone())
            errD_fake = adversarial_criterion(D_gen, target_fake)
            # # Relativistic Discriminator
            # errD_fake = adversarial_criterion(D_gen - torch.mean(D_real), target_fake)
            errD = errD_real + errD_fake
            errD_fake.backward()

            optimizerD.step()

            # update exponential average model weights
            ema_netG.update_parameters(netG)
            
            d_gt_probability = torch.sigmoid_(torch.mean(errD_real.detach()))
            d_sr_probability = torch.sigmoid_(torch.mean(errD_fake.detach()))
            
            if i%10 == 0:
                # print(errD.item(), errG.item(), errG_spec.item(), 0.001*errG_adv.item())
                # Testing with bicubic interpolation
                test_data = next(iter(valloader))

                metrics, metrics_inter = test_single(test_data, netG, netD, iters)
                print(f'[{epoch}/{n_epoch}][{i}/{len(dataloader)}] errD: {errD.item():.4f} Disc_Prob: ({d_gt_probability.item():.3f}/{d_sr_probability.item():.3f})'
                    f' errG: {errG.item():.4f} Spec/TV/Adv/Cont/Artif/ssim/ndvi: ({errG_spec.item():.4f}/{errG_tv.item():.4f}/{errG_adv.item():.4f}/{content_loss.item():.4f}/{errG_artifacts.item():.7f}/{errG_ssim.item():.4f}/{errG_ndvi.item():.4f})\n'
                f' RMSE(our/interp): ({metrics["RMSE"]:.4f}/{metrics_inter["RMSE"]:.4f})\n'
                f' PSNR(our/interp): ({metrics["PSNR"]:.4f}/{metrics_inter["PSNR"]:.4f})\n'
                f' SSIM(our/interp): ({metrics["SSIM"]:.4f}/{metrics_inter["SSIM"]:.4f})\n'
                f' NIQE(our/interp): ({metrics["NIQE"]:.4f}/{metrics_inter["NIQE"]:.4f})\n'
                f' SAM(our/interp): ({metrics["SAM"]:.4f}/{metrics_inter["SAM"]:.4f})\n'
                f' UQI(our/interp): ({metrics["UQI"]:.4f}/{metrics_inter["UQI"]:.4f})\n'
                f' SCC(our/interp): ({metrics["SCC"]:.4f}/{metrics_inter["SCC"]:.4f})\n'
                )

                # savemat_plot_senven(ms_fake, ms_up, ven_up, save_dir, s, filename=f'plot_{iters}.png') /{errG_spat.item():.4f}
                if epoch%2==0:
                    d_gen = torch.sigmoid(D_gen.detach().cpu()).mean(1)
                    savemat_plot_senven_RGBnorm(ms_fake, ms_up_usm, ven_up_usm, save_dir, s, d_gen, filename=f'plot_train_{iters}.png')
                else:
                    d_gen = torch.abs(ms_fake - ms_up).mean(1)
                    savemat_plot_senven_RGBnorm(ms_fake, ms_up, ven_up, save_dir, s, d_gen, abs_diff=True, filename=f'plot_train_{iters}.png')
                
                if args.do_log:
                    image = wandb.Image(os.path.join(save_dir, f'plot_{iters}.png'), caption=f"Iteration {iters}")
                    train_image = wandb.Image(os.path.join(save_dir, f'plot_train_{iters}.png'), caption=f"Iteration {iters}")
                    wandb.log({"errD_loss": errD.item(), 'errG_loss':errG.item(), 'Spectral loss':errG_spec.item(), 'TV loss':errG_tv.item(),
                            'Content loss':content_loss.item(), "Artifact Loss":errG_artifacts.item(), 'RMSE':metrics["RMSE"], 'RMSE_interpolate':metrics_inter["RMSE"], 'PSNR':metrics["PSNR"], 
                            'PSNR_interpolate':metrics_inter["PSNR"], 'SSIM': metrics["SSIM"], 'SSIM_interpolate': metrics_inter["SSIM"], 'Learning Rate':lr, 
                            "Plots":image, "Plots training":train_image})
                
            
            # G_losses.append(errG.item())
            # D_losses.append(errD.item())
        
        if epoch%70 == 19:
            torch.save(netG.state_dict(), f'{checkpoint_dir}gens2xn_15_{epoch}.pt')
            torch.save(netD.state_dict(), f'{checkpoint_dir}discs2xn_15_{epoch}.pt')
            torch.save({'optimizerG': optimizerG.state_dict(),
                        'optimizerD': optimizerD.state_dict()},
                        f'{checkpoint_dir}optimizer2xn_15_.pt')
        schedulerG.step()
        schedulerD.step()
        print(f'Time Elapsed: {(time.time()-start_time): .2f}')



if __name__ == '__main__':
    import argparse

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
            # Set the project where this run will be logged  _final
            project="ESRGAN_sentinel_RGBN",
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
    
    disc_spec = args.disc_spec
    print("DS", disc_spec)
    
    # create folders to save result images and trained models
    save_dir = "../srgan_samples_sent_RGBN2x/"
    checkpoint_dir = "./weights/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    s = Sensor('S2')   ## Hardcoded
    gpu_ids = [0, 1, 2, 3]

    # Creating generator network
    from networks import RRDBNet2 as RRDBNet
    netG = RRDBNet(in_channels=4, out_channels=4, channels=config.NETWORK.G_CHANNELS, 
                    growth_channels=config.NETWORK.GROWTH_CHANNELS, num_rrdb=13,  
                    upscale=2).to(device)
    netG = nn.DataParallel(netG,  device_ids = gpu_ids)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

    model_ema_decay = 0.99
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - model_ema_decay) * averaged_model_parameter + model_ema_decay * model_parameter
    ema_netG = AveragedModel(netG, avg_fn=ema_avg)

    ## Creating discriminator network
    # Unet based discriminator
    netD = UNetDiscriminatorSN(num_in_ch=4).to(device)
    netD = nn.DataParallel(netD,  device_ids=gpu_ids)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    usm_sharpener = USMSharp().to(device)
    ssim_module = SSIM(data_range=1, size_average=True, channel=4)
    ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=4)
    niqe = NIQE(2, './losses/niqe_model.mat').to(device)
    print(batch_size)
    
    train(args, batch_size, netG, netD)
