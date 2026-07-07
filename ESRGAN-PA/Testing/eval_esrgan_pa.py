import os
import sys
from pathlib import Path
print(str(Path('.').absolute()))
sys.path.insert(0, str(Path('..').absolute()))
sys.path.append(str(Path('.').absolute().parent.parent))

os.environ['TL_BACKEND'] = 'torch'
import time
import copy
import torch
from torch import nn
import numpy as np
from config import config
import torch.optim as optim
from torch.utils.data import DataLoader
from losses.spectral_tools import gen_mtf
from losses.vgg_loss import ContentLoss
from torch.optim.swa_utils import AveragedModel
from sensor import Sensor
import losses
# from model import RRDBNet
from networks import RRDBNet
from losses.LDL_loss import get_refined_artifact_map
from utils import savemat_plot_senven_RGBnorm, psnr, psnr_bands, rmse, rmse_bands, sam_batch, uqi_batch, scc_batch, cc_batch, center_crop
from pytorch_msssim import SSIM
from datasets.niqe_iqa import NIQE
from utils import recover_bandwise_RGB


def test(args):
    s = Sensor('S2')

    # Creating generator model
    from networks import RRDBNet2 as RRDBNet
    netG = RRDBNet(in_channels=4, out_channels=4, channels=config.NETWORK.G_CHANNELS,
                growth_channels=config.NETWORK.GROWTH_CHANNELS, num_rrdb=13,
                upscale=2).to(device)

    netG = nn.DataParallel(netG)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

    from networks import UNetDiscriminatorSN
    # Unet based discriminator
    netD = UNetDiscriminatorSN(num_in_ch=4).to(device)
    netD = nn.DataParallel(netD)
    # DiscriminatorForVGG(in_channels=1, out_channels=1, channels=config.NETWORK.D_CHANNELS, opt2pan=opt2pan).to(device)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    # Loading generator weights
    if args.checkpoint_g != None:
        print("Loading model:", args.checkpoint_g)
        netG.load_state_dict(torch.load(args.checkpoint_g))
    else:
        netG.load_state_dict(torch.load(config.TEST.checkpoint_path), strict=False)

    model_ema_decay = 0.99
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - model_ema_decay) * averaged_model_parameter + model_ema_decay * model_parameter
    ema_netG = AveragedModel(netG, avg_fn=ema_avg)

    # Reading Dataset
    if args.data_path is not None:
        from datasets.patch_data_RGBN2x_norm import TestData_single, TestData_TA_single
        test_ds = TestData_single(args.data_path, s, args.patch_size)
        test_ta = TestData_TA_single(args.data_path, s, args.patch_size)
    else:
        from datasets.sen2ven_data_RGBN2x_norm import TestData, TestData_TA

        print("Loading default dataloader")
        test_ds = TestData(config.TEST.hr_img_path, s)
        if args.evaluate:
            test_ta = TestData_TA(config.TEST.hr_img_path, s)
        else:
            test_ta = TestData(config.TEST.hr_img_path, s)

    test_ds_img_nums = len(test_ds)
    norm = test_ta.ms_norms
    dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    iter_ds = iter(dataloader)
    I_S2, I_VE = next(iter(test_ds))
    _, H, W = I_VE.shape
    data = next(iter_ds)

    ms_orig, ven_up_orig = data
    ms_orig = ms_orig.to(device)
    ven_up_orig = ven_up_orig.to(device)
    ssim_module = SSIM(data_range=1, size_average=True, channel=4)
    niqe = NIQE(2, './losses/niqe_model.mat').to(device)
    ta_dataloader = DataLoader(test_ta, batch_size=batch_size, shuffle=True, drop_last=True)

    if args.target_adapt:
        I_S2_np = np.moveaxis(np.array(I_S2), 0, -1)
        I_VE_np = np.array(I_VE)
        mtf_h = gen_mtf(s.ratio, 'S2')
        if args.evaluate:
            I_S2, I_VE, _ = next(iter(test_ta))
            _, H, W = I_VE.shape

        # Loading pre-trained discriminator
        netD.load_state_dict(torch.load(args.checkpoint_d))

        spectral_loss = nn.L1Loss().to(device)
        tv_loss = losses.TVLoss()
        cri_artifacts = nn.L1Loss()
        ndvi_loss = nn.L1Loss()
        feature_criterion = ContentLoss("vgg19", False, 1000, None, config.TRAIN.vgg_features, config.TRAIN.vgg_weights, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], use_huber=False).to(device)

        target_real = torch.full([batch_size, 1, H, W], 1.0, dtype=torch.float, device=device)
        target_fake = torch.full([batch_size, 1, H, W], 0.0, dtype=torch.float, device=device)
        adversarial_criterion = nn.BCEWithLogitsLoss()

        # Iterative Target Adaptation
        optimizerG = optim.Adam(netG.parameters(), 1e-4)
        optimizerD = optim.Adam(netD.parameters(), 1e-5)
        schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, [100], 0.6)
        start_time = time.time()
        iters = 0
        max_psnr = 0

        for epoch in range(n_epoch_init):

            if iters > n_epoch_init:
                break

            for i, data in enumerate(ta_dataloader):

                if iters > n_epoch_init:
                    break

                if args.evaluate:
                    ms, ven_up, ms_up = data
                else:
                    ms, ven_up = data
                    ms_up = ven_up.clone()

                ms = ms.to(device)
                ven_up = ven_up.to(device)
                ms_up = ms_up.to(device)
                ms_fake = netG(ms)
                D_fake = netD(ms_fake)
                ms_fake_ema = ema_netG(ms)

                # During discriminator model training, enable discriminator model backpropagation
                for d_parameters in netD.parameters():
                    d_parameters.requires_grad = False

                netG.zero_grad()
                gen_spectral_loss = spectral_loss(ms_fake, ven_up)
                errG_tv = tv_loss(ms_fake)
                content_loss_rgb = feature_criterion(ms_fake[:,:3].flip(1), ven_up[:,:3].flip(1))[0].sum()*0.6
                content_loss_nir = feature_criterion(ms_fake[:,3:].repeat(1, 3, 1, 1), ven_up[:,3:].repeat(1, 3, 1, 1))[0].sum()*0.4
                content_loss = content_loss_nir + content_loss_rgb
                adv_loss = adversarial_criterion(D_fake, target_real)
                errG_ssim = 1 - ssim_module(ms_fake, ven_up)

                pixel_weight = get_refined_artifact_map(ven_up, ms_fake, ms_fake_ema, 7)
                errG_artifacts = cri_artifacts(torch.mul(pixel_weight, ms_fake), torch.mul(pixel_weight, ven_up))
                errG_ndvi = ndvi_loss((ms_fake[:,3]-ms_fake[:,2]), (ven_up[:,3]-ven_up[:,2]))

                gen_total_loss = (10*gen_spectral_loss) + (errG_ndvi*10) + (2*content_loss) + (0.1*adv_loss) + (5e1*errG_artifacts) + (errG_ssim*7)

                gen_total_loss.backward()
                optimizerG.step()

                # update exponential average model weights
                ema_netG.update_parameters(netG)
                ms_fake = ms_fake.detach().cpu()
                ms_up = ms_up.detach().cpu()
                ven_up = ven_up.detach().cpu()

                _, _, H, W = ms_fake.shape
                ms_fake = center_crop(ms_fake, int(H*0.9))
                ven_up = center_crop(ven_up, int(H*0.9))
                ms_up = center_crop(ms_up, int(H*0.9))
                print(ms_fake.shape, ven_up.shape)

                psnr_results = psnr(ms_fake, ven_up)
                # print(ms_fake.shape, ms_up.shape)
                psnr_results_bi = psnr(ms_up, ven_up)
                psnr_bands_res = psnr_bands(ms_fake, ven_up)
                psnr_bands_res_bi = psnr_bands(ms_up, ven_up)

                rmse_results = rmse(recover_bandwise_RGB(ms_fake, norm_data=norm), recover_bandwise_RGB(ven_up, norm_data=norm))
                rmse_results_bi = rmse(recover_bandwise_RGB(ms_up, norm_data=norm), recover_bandwise_RGB(ven_up, norm_data=norm))
                rmse_bands_res = rmse_bands(recover_bandwise_RGB(ms_fake, norm_data=norm), recover_bandwise_RGB(ven_up, norm_data=norm))
                rmse_bands_res_bi = rmse_bands(recover_bandwise_RGB(ms_up, norm_data=norm), recover_bandwise_RGB(ven_up, norm_data=norm))

                ssim_results = ssim_module(ms_fake, ven_up)
                ssim_results_bi = ssim_module(ms_up, ven_up)
                sam_res = sam_batch(ven_up[:,:3], ms_fake[:,:3])
                sam_res_bi = sam_batch(ven_up[:,:3], ms_up[:,:3])
                uqi_res = uqi_batch(ven_up[:,:], ms_fake[:,:])
                uqi_res_bi = uqi_batch(ven_up[:,:], ms_up[:,:])
                scc_res = scc_batch(ven_up[:,:], ms_fake[:,:])
                scc_res_bi = scc_batch(ven_up[:,:], ms_up[:,:])

                cc_res = cc_batch(ven_up[:,:], ms_fake[:,:])
                cc_res_bi = cc_batch(ven_up[:,:], ms_up[:,:])

                niqe_res = niqe(ms_fake.flip(1)[:,1:]*255).mean().item()
                niqe_res_bi = niqe(ms_up.flip(1)[:,1:]*255).mean().item()
                niqe_res_at = niqe(ven_up.flip(1)[:,1:]*255).mean().item()

                print(f'[{iters}/{n_epoch_init}] ({gen_total_loss.item():.5f}) | Gen_Spectral: {gen_spectral_loss.item():.6f} | Content Loss: {content_loss.item():.6f} | Adv Loss: {adv_loss.item():.4f}' 
                      f'| RMSE: ({rmse_results.item():.4f}/{rmse_results_bi.item():.4f})\n'
                      f'| PSNR: ({psnr_results.item():.3f}/{psnr_results_bi.item():.3f})\n'
                      f'| SAM: ({sam_res:.4f}/{sam_res_bi: .4f})\n'
                      f'| SSIM: ({ssim_results:.4f}/{ssim_results_bi: .4f})\n'
                      f'| UQI: ({uqi_res:.4f}/{uqi_res_bi: .4f})\n'
                      f'| SCC: ({scc_res:.4f}/{scc_res_bi: .4f})\n'
                      f'| NIQE: ({niqe_res:.4f}/{niqe_res_bi: .4f}/{niqe_res_at: .4f})\n'
                      )
                print('RMSE_bands:', rmse_bands_res, '/', rmse_bands_res_bi)
                print('PSNR_bands:', psnr_bands_res, '/', psnr_bands_res_bi)
                print('CC_bands:', cc_res, '/', cc_res_bi)

                if iters%2 == 0:
                    if iters%4==0:
                        savemat_plot_senven_RGBnorm(ms_fake, ms_up, ven_up, save_dir, s, torch.sigmoid(D_fake), filename=f'plot_{iters}.png')
                    else:
                        d_gen = torch.abs(ms_fake - ms_up).mean(1)
                        savemat_plot_senven_RGBnorm(ms_fake, ms_up, ven_up, save_dir, s, d_gen, abs_diff=True, filename=f'plot_{iters}.png')
                iters += 1

                schedulerG.step()
                print(f'Time Elapsed: {(time.time()-start_time): .2f}')

        # Saving the target adaptive model
        torch.save(netG.state_dict(), f'{checkpoint_dir}gen_rgb2_{args.model}_TA.pt')

    # Final evaluation of model
    print("Starting evaluation .....")
    psnr_li, psnr_inter = [], []
    rmse_li, rmse_inter = [], []
    ssim_li, ssim_inter = [], []
    niqe_li, niqe_or, niqe_inter = [], [], []
    psnr_li_batch, psnr_inter_batch = [], []
    rmse_li_batch, rmse_inter_batch = [], []
    cc_li_batch, cc_inter_batch = [], []
    sam_li, sam_inter = [], []
    uqi_li, uqi_inter = [], []

    for i, data in enumerate(ta_dataloader):
        ms, ven_up, ms_up = data
        ms = ms.to(device)
        ven_up = ven_up.to(device)
        ms_up = ms_up.to(device)
        # netG_opt = copy.deepcopy(netG)
        # print(ms.shape)
        ms_fake = netG(ms)

        ms_fake = ms_fake.detach().cpu()
        ms_up = ms_up.detach().cpu()
        ven_up = ven_up.detach().cpu()

        _, _, H, W = ms_fake.shape
        ms_fake = center_crop(ms_fake, int(H*0.9))
        ven_up = center_crop(ven_up, int(H*0.9))
        ms_up = center_crop(ms_up, int(H*0.9))

        psnr_results = psnr(ms_fake, ven_up)
        psnr_results_bi = psnr(ms_up, ven_up)
        psnr_li.append(psnr_results.item())
        psnr_inter.append(psnr_results_bi.item())

        psnr_bands_res = psnr_bands(ms_fake.detach().cpu(), ven_up.detach().cpu())
        psnr_bands_res_bi = psnr_bands(ms_up.detach().cpu(), ven_up.detach().cpu())
        psnr_li_batch.append(psnr_bands_res.numpy())
        psnr_inter_batch.append(psnr_bands_res_bi.numpy())

        rmse_results = rmse(recover_bandwise_RGB(ms_fake, norm_data=norm), recover_bandwise_RGB(ven_up, norm_data=norm))
        rmse_results_bi = rmse(recover_bandwise_RGB(ms_up, norm_data=norm), recover_bandwise_RGB(ven_up, norm_data=norm))
        rmse_li.append(rmse_results.item())
        rmse_inter.append(rmse_results_bi.item())

        rmse_bands_res = rmse_bands(recover_bandwise_RGB(ms_fake), recover_bandwise_RGB(ven_up))
        rmse_bands_res_bi = rmse_bands(recover_bandwise_RGB(ms_up), recover_bandwise_RGB(ven_up))
        rmse_li_batch.append(rmse_bands_res.numpy())
        rmse_inter_batch.append(rmse_bands_res_bi.numpy())

        ssim_results = ssim_module(ms_fake, ven_up)
        ssim_results_bi = ssim_module(ms_up, ven_up)
        ssim_li.append(ssim_results.item())
        ssim_inter.append(ssim_results_bi.item())

        niqe_res = niqe(ms_fake.flip(1)[:,1:]*255).mean().item()
        niqe_res_bi = niqe(ms_up.flip(1)[:,1:]*255).mean().item()
        niqe_res_at = niqe(ven_up.flip(1)[:,1:]*255).mean().item()
        niqe_li.append(niqe_res)
        niqe_inter.append(niqe_res_bi)
        niqe_or.append(niqe_res_at)

        cc_res = cc_batch(ms_up[:,:], ms_fake[:,:])
        cc_res_bi = cc_batch(ven_up[:,:], ms_up[:,:])
        cc_li_batch.append(cc_res)
        cc_inter_batch.append(cc_res_bi)

        sam_li.append(sam_batch(ven_up[:,:3], ms_fake[:,:3]))
        sam_inter.append(sam_batch(ven_up[:,:3], ms_up[:,:3]))

        uqi_res = uqi_batch(ven_up[:,:], ms_fake[:,:])
        uqi_res_bi = uqi_batch(ven_up[:,:], ms_up[:,:])
        uqi_li.append(uqi_res)
        uqi_inter.append(uqi_res_bi)

    print(f'Final Evaluations \n'
            f'PSNR(our/inter): ({np.array(psnr_li).mean()}/{np.array(psnr_inter).mean()}) \n'
            f'RMSE(our/inter): ({np.array(rmse_li).mean()}/{np.array(rmse_inter).mean()}) \n'
            f'SSIM(our/inter): ({np.array(ssim_li).mean()}/{np.array(ssim_inter).mean()}) \n'
            f'NIQE(our/inter): ({np.array(niqe_li).mean()}/{np.array(niqe_or).mean()}/{np.array(niqe_inter).mean()}) \n'
            f'SAM(our/inter): ({np.array(sam_li).mean()}/{np.array(sam_inter).mean()}) \n'
            f'UQI(our/inter): ({np.array(uqi_li).mean()}/{np.array(uqi_inter).mean()}) \n'
            )

    psnr_li_batch = np.array(psnr_li_batch)
    psnr_inter_batch = np.array(psnr_inter_batch)
    rmse_li_batch = np.array(rmse_li_batch)
    rmse_inter_batch = np.array(rmse_inter_batch)
    cc_li_batch = np.array(cc_li_batch)
    cc_inter_batch = np.array(cc_inter_batch)

    print('RMSE band-wise', rmse_li_batch.mean(0),' | ', rmse_inter_batch.mean(0))
    print('PSNR band-wise', psnr_li_batch.mean(0), ' | ', psnr_inter_batch.mean(0))
    print('CC band-wise', cc_li_batch.mean(0), ' | ', cc_inter_batch.mean(0))

    # ms_fake = netG(ms_orig)
    # savemat_plot_senven_RGBnorm(ms_fake, ms_orig, ven_up_orig, save_dir, s, filename=f'plot_final.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_TA', dest='target_adapt', action='store_true')
    parser.add_argument('--evals', dest='evaluate', action='store_true')
    parser.add_argument('--checkpoint_g', type=str, default=None, help='Select weights by providing path')
    parser.add_argument('--checkpoint_d', type=str, default=None, help='Select weights by providing path')
    parser.add_argument('--data_path', type=str, default=None, help='Select data clips path')
    parser.add_argument('--model', type=str, default='esrgan', help='Select model type')
    parser.add_argument('--batch', type=int, default=None, help='path to checkpoint')
    parser.add_argument('--patch_size', type=int, default=64, help='path to checkpoint')
    parser.add_argument('--nepoch', type=int, default=None, help='Num of epoch for TA')
    parser.set_defaults(feature=False)

    args = parser.parse_args()

    if args.batch == None:
        batch_size = config.TRAIN.batch_size
    else:
        batch_size = args.batch
        print(batch_size)

    if args.nepoch==None:
        n_epoch_init = 150 #config.TRAIN.n_epoch_init
    else:
        n_epoch_init = args.nepoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create folders to save result images and trained models
    save_dir = "samples_test/"
    checkpoint_dir = "../weights/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    test(args)
