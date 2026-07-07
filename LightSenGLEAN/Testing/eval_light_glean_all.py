import os
import sys
from pathlib import Path
print(str(Path('.').absolute()))
sys.path.insert(0, str(Path('.').absolute()))
sys.path.append(str(Path('.').absolute().parent.parent))

os.environ['TL_BACKEND'] = 'torch'
import time
import copy
import torch
from torch import nn
import numpy as np
from config import config
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from losses.spectral_tools import gen_mtf
from sensor import Sensor
from torch.optim.swa_utils import AveragedModel
from losses.LDL_loss import get_refined_artifact_map
import losses
from losses.vgg_loss import ContentLoss
import torch.nn.functional as F
from utils import savemat_plot_senven_20mnorm, savemat_plot_senven_RGBnorm, psnr, psnr_bands, rmse, rmse_bands, sam_batch, uqi_batch, scc_batch, center_crop, cc_batch, d_lambda_batch, d_s_batch
from pytorch_msssim import SSIM
from datasets.niqe_iqa import NIQE
from utils import recover_bandwise_20m, recover_bandwise_all, recover_bandwise_RGB


def test(args):
    s = Sensor('S2')

    from networks.glean import LightSenGLEANStyleGANv2
    netG = LightSenGLEANStyleGANv2(64, 256, 4, 64, 23, rgb_channels=4).to(device)
    # netG.generator.load_state_dict(torch.load('./stylegan/stylegan_latest.pt', map_location=torch.device(device)))
    netG.generator.requires_grad_(False)
    # netG.encoder.requires_grad_(False)
    # netG.encoder2.requires_grad_(False)
    # netG.fusion_out.requires_grad_(False)
    # netG.fusion_skip.requires_grad_(False)
    netG = nn.DataParallel(netG)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

    if args.do_artif:
        model_ema_decay = 0.99
        ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - model_ema_decay) * averaged_model_parameter + model_ema_decay * model_parameter
        ema_netG = AveragedModel(netG, avg_fn=ema_avg)

    from networks import UNetDiscriminatorSN
    # Unet based discriminator
    netD = UNetDiscriminatorSN(num_in_ch=8).to(device)
    netD = nn.DataParallel(netD)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    # Loading generator weights
    if args.checkpoint_g != None:
        netG.load_state_dict(torch.load(args.checkpoint_g, map_location=device))
    else:
        netG.load_state_dict(torch.load(config.TEST.checkpoint_path))

    # Reading Dataset
    if args.data_path is not None:
        from datasets.patch_data_all_norm import TestData_single, TestData_TA_single, TestData_TA_single_RGB
        test_ds = TestData_single(args.data_path, s, 128)
        test_ta = TestData_TA_single(args.data_path, s, 512)
        test_ta_rgb = TestData_TA_single_RGB(args.data_path, s, 256)
    else:
        from datasets.sen2ven_data_20m_norm import TestData, TestData_TA
        print("Loading default dataloader")
        test_ds = TestData(config.TEST.hr_img_path, s)
        test_ta = TestData_TA(config.TEST.hr_img_path, s)

    norm = test_ta.ms_norms
    test_ds_img_nums = len(test_ds)
    dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    iter_ds = iter(dataloader)
    I_S2, I_VE = next(iter(test_ds))
    _, H, W = I_VE.shape
    data = next(iter_ds)
    # ms, ven, ven_up, ms_up = data

    ms_orig, ven_up_orig = data
    ms_orig = ms_orig.to(device)
    ven_up_orig = ven_up_orig.to(device)
    ssim_module = SSIM(data_range=1, size_average=True, channel=4)
    niqe_20m = NIQE(4, '../../niqe_model.mat').to(device)
    niqe_rgb = NIQE(2, '../../niqe_model.mat').to(device)

    ta_dataloader = DataLoader(test_ta, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_dataloader = DataLoader(test_ta, batch_size=1, shuffle=False, drop_last=False)
    eval_dataloader_rgb = DataLoader(test_ta_rgb, batch_size=1, shuffle=False, drop_last=False)
    print(args.target_adapt)

    if args.target_adapt:

        if args.data_path is not None:
            netD.load_state_dict(torch.load(args.checkpoint_d, map_location=device))
        else:
            netD.load_state_dict(torch.load(config.TEST.checkpoint_path.replace('gens', 'discs')))

        I_S2_np = np.moveaxis(np.array(I_S2), 0, -1)
        I_VE_np = np.array(I_VE)
        mtf_h = gen_mtf(s.ratio, 'S2')
        I_S2, _, I_VE, _ = next(iter(test_ta))
        _, H, W = I_VE.shape

        spectral_loss = nn.L1Loss().to(device)
        ndvi_loss = nn.L1Loss()
        ssim_module_train = SSIM(data_range=1, size_average=True, channel=8)
        tv_loss = losses.TVLoss()
        if args.do_artif:
            cri_artifacts = nn.L1Loss()
        feature_criterion = ContentLoss(None, config.TRAIN.vgg_weights, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], use_huber=False).to(device)

        target_real = torch.full([batch_size, 1, H, W], 1.0, dtype=torch.float, device=device)
        adversarial_criterion = nn.BCEWithLogitsLoss()

        spectral_loss = nn.L1Loss().to(device)
        tv_loss = losses.TVLoss()

        # Iterative Target Adaptation
        optimizerG = optim.Adam(netG.parameters(), 8e-5)
        schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, [100, 200], 0.8)
        start_time = time.time()
        iters = 0
        max_psnr = 0

        for epoch in range(n_epoch_init):
            # netG.eval()

            if iters > n_epoch_init:
                break

            for i, data in enumerate(ta_dataloader):

                if iters > n_epoch_init:
                    break

                ms, ms_rgb, ms_up, ms_interp = data #next(iter_ds)
                print(ms.shape ,ms_rgb.shape, ms_up.shape, ms_interp.shape)

                # if args.evaluate:
                #     ms, ven_up, ms_up = data
                # else:
                #     ms, ms_rgb = data
                #     ms_up = ven_up.clone()

                ms = ms.to(device)
                ms_rgb = ms_rgb.to(device)
                ms_up = ms_up.to(device)
                ms_fake = netG(ms, ms_rgb)
                if args.do_artif:
                    ms_fake_ema = ema_netG(ms, ms_rgb)
                D_fake = netD(ms_fake)

                # if args.data_path is not None:
                #     ms_fake = ms_fake[:,:,::2,::2]

                netG.zero_grad()
                gen_spectral_loss = spectral_loss(ms_fake, ms_up)
                errG_tv = tv_loss(ms_fake)
                errG_ndvi_rgb = ndvi_loss((ms_fake[:,3]-ms_fake[:,2]), (ms_up[:,3]-ms_up[:,2]))
                errG_ndvi_20m = ndvi_loss((ms_fake[:,7]-ms_fake[:,6]), (ms_up[:,7]-ms_up[:,6]))
                errG_ndvi = errG_ndvi_20m + errG_ndvi_rgb
                # content_loss_rgb = feature_criterion(ms_fake.flip(1)[:,1:], ms_up.flip(1)[:,1:])[0].sum()*0.7
                # content_loss_nir = feature_criterion(ms_fake[:,3:].repeat(1, 3, 1, 1), ms_up[:,3:].repeat(1, 3, 1, 1))[0].sum()*0.3
                # content_loss = content_loss_nir + content_loss_rgb

                content_loss_rgb = feature_criterion(ms_fake[:,:3].flip(1), ms_up[:,:3].flip(1))[0].sum()*0.35
                content_loss_nir = feature_criterion(ms_fake[:,3].unsqueeze(1).repeat(1, 3, 1, 1), ms_up[:,3].unsqueeze(1).repeat(1, 3, 1, 1))[0].sum()*0.15
                content_loss_20m = feature_criterion(ms_fake[:,4:7].flip(1), ms_up[:,4:7].flip(1))[0].sum()*0.35
                content_loss_vnir = feature_criterion(ms_fake[:,7].unsqueeze(1).repeat(1, 3, 1, 1), ms_up[:,7].unsqueeze(1).repeat(1, 3, 1, 1))[0].sum()*0.15
                content_loss = content_loss_nir + content_loss_rgb + content_loss_vnir + content_loss_20m

                # gen_total_loss = gen_spectral_loss + content_loss #+ (0*errG_tv)
                adv_loss = adversarial_criterion(D_fake, target_real)

                errG_ssim = 1 - ssim_module_train(ms_fake, ms_up)
                if args.do_artif:
                    pixel_weight = get_refined_artifact_map(ms_up, ms_fake, ms_fake_ema, 7)
                    errG_artifacts = cri_artifacts(torch.mul(pixel_weight, ms_fake), torch.mul(pixel_weight, ms_up))
                else:
                    errG_artifacts = 0
                # pixel_weight = get_refined_artifact_map(ven_up, ms_fake, ms_fake_ema, 7)
                # errG_artifacts = cri_artifacts(torch.mul(pixel_weight, ms_fake), torch.mul(pixel_weight, ven_up))

                gen_total_loss = (10*gen_spectral_loss) + (errG_tv*1) + (2*content_loss) + (0.3*adv_loss) + (5e2*errG_artifacts) + (errG_ssim*5) + (errG_ndvi*3)
                # gen_total_loss = (10*gen_spectral_loss) + (errG_tv*3) + (3*content_loss) + (0.1*adv_loss) + (5e2*errG_artifacts) + (errG_ssim*2) + (errG_ndvi*3)

                gen_total_loss.backward()
                optimizerG.step()

                ms_fake = ms_fake.detach().cpu()
                ms_up = ms_up.detach().cpu()
                ms_interp = ms_interp.detach().cpu()
                ms_rgb = ms_rgb.detach().cpu()

                _, _, H, W = ms_fake.shape
                # print(ms_fake.shape)
                ms_fake = center_crop(ms_fake, H-20-((H-20)%8))
                ms_up = center_crop(ms_up, H-20-((H-20)%8))
                ms_interp = center_crop(ms_interp, H-20-((H-20)%8))
                _, _, H2, W2 = ms_fake.shape
                ms = center_crop(ms, H2//4)

                ds_res = d_s_batch(ms_up[:,:4].detach().cpu(), ms[:,:4].detach().cpu(), ms_fake[:,:4].detach().cpu())
                ds_res_bi = d_s_batch(ms_up[:,:4].detach().cpu(), ms[:,:4].detach().cpu(), ms_interp[:,:4].detach().cpu())

                dlam_res = d_lambda_batch(recover_bandwise_all(ms_up, norm_data=norm)[:,:4], recover_bandwise_all(ms_fake, norm_data=norm)[:,:4])
                dlam_res_bi = d_lambda_batch(recover_bandwise_all(ms_up, norm_data=norm)[:,:4], recover_bandwise_all(ms_interp, norm_data=norm)[:,:4])

                psnr_results_rgb = psnr(ms_fake[:,:4], ms_up[:,:4])
                psnr_results_bi_rgb = psnr(ms_interp[:,:4], ms_up[:,:4])
                psnr_results_20m = psnr(ms_fake[:,4:], ms_up[:,4:])
                psnr_results_bi_20m = psnr(ms_interp[:,4:], ms_up[:,4:])
                psnr_bands_res = psnr_bands(ms_fake.detach().cpu(), ms_up.detach().cpu())
                psnr_bands_res_bi = psnr_bands(ms_interp.detach().cpu(), ms_up.detach().cpu())

                rmse_results_rgb = rmse(recover_bandwise_all(ms_fake, norm_data=norm)[:,:4], recover_bandwise_all(ms_up, norm_data=norm)[:,:4])
                rmse_results_bi_rgb = rmse(recover_bandwise_all(ms_interp, norm_data=norm)[:,:4], recover_bandwise_all(ms_up, norm_data=norm)[:,:4])
                rmse_results_20m = rmse(recover_bandwise_all(ms_fake, norm_data=norm)[:,4:], recover_bandwise_all(ms_up, norm_data=norm)[:,4:])
                rmse_results_bi_20m = rmse(recover_bandwise_all(ms_interp, norm_data=norm)[:,4:], recover_bandwise_all(ms_up, norm_data=norm)[:,4:])
                rmse_bands_res = rmse_bands(recover_bandwise_all(ms_fake, norm_data=norm), recover_bandwise_all(ms_up, norm_data=norm))
                rmse_bands_res_bi = rmse_bands(recover_bandwise_all(ms_interp, norm_data=norm), recover_bandwise_all(ms_up, norm_data=norm))

                ssim_results_rgb = ssim_module(ms_fake[:,:4], ms_up[:,:4])
                ssim_results_bi_rgb = ssim_module(ms_interp[:,:4], ms_up[:,:4])

                ssim_results_20m = ssim_module(ms_fake[:,4:], ms_up[:,4:])
                ssim_results_bi_20m = ssim_module(ms_interp[:,4:], ms_up[:,4:])

                sam_res = sam_batch(ms_up[:,:3].detach().cpu(), ms_fake[:,:3].detach().cpu())
                sam_res_bi = sam_batch(ms_up[:,:3].detach().cpu(), ms_interp[:,:3].detach().cpu())

                uqi_res = uqi_batch(ms_up[:,:].detach().cpu(), ms_fake[:,:].detach().cpu())
                uqi_res_bi = uqi_batch(ms_up[:,:].detach().cpu(), ms_interp[:,:].detach().cpu())
                scc_res = scc_batch(ms_up[:,:].detach().cpu(), ms_fake[:,:].detach().cpu())
                scc_res_bi = scc_batch(ms_up[:,:].detach().cpu(), ms_interp[:,:].detach().cpu())

                cc_res = cc_batch(ms_up[:,:], ms_fake[:,:])
                cc_res_bi = cc_batch(ms_up[:,:], ms_interp[:,:])

                niqe_rgb_res = niqe_rgb(ms_fake[:,:3].flip(1)*255).mean().item()
                niqe_rgb_res_bi = niqe_rgb(ms_up[:,:3].flip(1)*255).mean().item()
                niqe_rgb_res_at = niqe_rgb(ms_interp[:,:3].flip(1)*255).mean().item()

                niqe_20m_res = niqe_20m(ms_fake[:,4:7].flip(1)*255).mean().item()
                niqe_20m_res_bi = niqe_20m(ms_up[:,4:7].flip(1)*255).mean().item()
                niqe_20m_res_at = niqe_20m(ms_interp[:,4:7].flip(1)*255).mean().item()
                # print(d_gen.mean(), d_gen.min(), d_gen.max(), D_fake.shape, d_gen.shape, 'jkj')

                print(f'[{iters}/{n_epoch_init}] ({gen_total_loss.item():.5f}) | Gen_Spectral: {gen_spectral_loss.item():.6f} | Adv Loss: {adv_loss.item():.4f} | SSIM Loss: {errG_ssim.item():.4f}' # | Content Loss: {content_loss.item():.6f}' 
                      f'| RMSE_RGB: ({rmse_results_rgb.item():.4f}/{rmse_results_bi_rgb.item():.4f})\n'
                      f'| RMSE_20m: ({rmse_results_20m.item():.4f}/{rmse_results_bi_20m.item():.4f})\n'
                      f'| PSNR_RGB: ({psnr_results_rgb.item():.3f}/{psnr_results_bi_rgb.item():.3f})\n'
                      f'| PSNR_20m: ({psnr_results_20m.item():.3f}/{psnr_results_bi_20m.item():.3f})\n'
                      f'| SAM: ({sam_res:.4f}/{sam_res_bi: .4f})\n'
                      f'| SSIM_RGB: ({ssim_results_rgb:.4f}/{ssim_results_bi_rgb: .4f})\n'
                      f'| SSIM_20m: ({ssim_results_20m:.4f}/{ssim_results_bi_20m: .4f})\n'
                      f'| UQI: ({uqi_res:.4f}/{uqi_res_bi: .4f})\n'
                      f'| SCC: ({scc_res:.4f}/{scc_res_bi: .4f})\n'
                      f'| D_lambda: ({dlam_res:.4f}/{dlam_res_bi: .4f})\n'
                      f'| D_s: ({ds_res:.4f}/{ds_res_bi: .4f})\n'
                      f'| NIQE_rgb: ({niqe_rgb_res:.4f}/{niqe_rgb_res_bi: .4f}/{niqe_rgb_res_at: .4f})\n'
                      f'| NIQE_20m: ({niqe_20m_res:.4f}/{niqe_20m_res_bi: .4f}/{niqe_20m_res_at: .4f})\n'
                      )
                print('RMSE_bands:', rmse_bands_res, '/', rmse_bands_res_bi)
                print('PSNR_bands:', psnr_bands_res, '/', psnr_bands_res_bi)
                print('CC_bands:', cc_res, '/', cc_res_bi)

                # if iters%2 == 0:
                #     d_gen = torch.abs(ms_fake - ms_up).mean(1)
                #     savemat_plot_senven_RGBnorm(ms_fake[:,:4], ms_interp[:,:4], ms_up[:,:4], save_dir, s, d_gen, abs_diff=True, filename=f'plot_rgbn_{iters}.png')
                #     savemat_plot_senven_20mnorm(ms_fake[:,4:], ms_interp[:,4:], ms_up[:,4:], save_dir, s, d_gen, abs_diff=True, filename=f'plot_20m_{iters}.png')

                # elif iters%2 == 1:
                #     d_gen = torch.sigmoid(D_fake.detach().cpu()).mean(1)
                #     # savemat_plot_senven_20mnorm(ms_fake, ms_interp, ms_up, save_dir, s, d_gen, filename=f'plot_{iters}.png')
                #     savemat_plot_senven_RGBnorm(ms_fake[:,:4], ms_interp[:,:4], ms_up[:,:4], save_dir, s, d_gen, filename=f'plot_rgbn_{iters}.png')
                #     savemat_plot_senven_20mnorm(ms_fake[:,4:], ms_interp[:,4:], ms_up[:,4:], save_dir, s, d_gen, filename=f'plot_20m_{iters}.png')

                iters += 1

                schedulerG.step()
                print(f'Time Elapsed: {(time.time()-start_time): .2f}')

        # Saving the target adaptive model
        torch.save(netG.state_dict(), f'{args.checkpoint_g[:-3]}_TA.pt')

    # Final evaluation of model
    # netG = netG_opt.to(device)
    print("Starting evaluation .....  RGB")
    psnr_li, psnr_inter = [], []
    rmse_li, rmse_inter = [], []
    psnr_li_batch, psnr_inter_batch = [], []
    rmse_li_batch, rmse_inter_batch = [], []
    cc_li_batch, cc_inter_batch = [], []
    sam_li, sam_inter = [], []

    ssim_li, ssim_inter = [], []
    niqe_rgb_li, niqe_rgb_or, niqe_rgb_inter = [], [], []
    # niqe_20m_li, niqe_20m_or, niqe_20m_inter = [], [], []

    uqi_li, uqi_inter = [], []

    for i, data in enumerate(eval_dataloader_rgb):
        ms, ms_rgb, ms_up, ms_interp = data
        ms = ms.to(device)
        ms_rgb = ms_rgb.to(device)
        ms_up = ms_up.to(device)
        ms_fake = netG(ms, ms_rgb)

        ms_fake = ms_fake.detach().cpu()
        ms_up = ms_up.detach().cpu()
        ms_interp = ms_interp.detach().cpu()
        ms_rgb = ms_rgb.detach().cpu()

        _, _, H, W = ms_fake.shape
        ms_fake = center_crop(ms_fake, int(H*0.9))[:,:4]
        ms_interp = center_crop(ms_interp, int(H*0.9))[:,:4]
        ms_up = center_crop(ms_up, int(H*0.9))[:,:4]

        psnr_results = psnr(ms_fake, ms_up)
        psnr_results_bi = psnr(ms_interp, ms_up)
        psnr_li.append(psnr_results.item())
        psnr_inter.append(psnr_results_bi.item())

        psnr_bands_res = psnr_bands(ms_fake.detach().cpu(), ms_up.detach().cpu())
        psnr_bands_res_bi = psnr_bands(ms_interp.detach().cpu(), ms_up.detach().cpu())
        psnr_li_batch.append(psnr_bands_res.numpy())
        psnr_inter_batch.append(psnr_bands_res_bi.numpy())

        rmse_results = rmse(recover_bandwise_RGB(ms_fake, norm_data=norm), recover_bandwise_RGB(ms_up, norm_data=norm))
        rmse_results_bi = rmse(recover_bandwise_RGB(ms_interp, norm_data=norm), recover_bandwise_RGB(ms_up, norm_data=norm))
        rmse_li.append(rmse_results.item())
        rmse_inter.append(rmse_results_bi.item())

        rmse_bands_res = rmse_bands(recover_bandwise_RGB(ms_fake, norm_data=norm), recover_bandwise_RGB(ms_up, norm_data=norm))
        rmse_bands_res_bi = rmse_bands(recover_bandwise_RGB(ms_interp, norm_data=norm), recover_bandwise_RGB(ms_up, norm_data=norm))
        rmse_li_batch.append(rmse_bands_res.numpy())
        rmse_inter_batch.append(rmse_bands_res_bi.numpy())

        ssim_results = ssim_module(ms_fake, ms_up)
        ssim_results_bi = ssim_module(ms_interp, ms_up)
        ssim_li.append(ssim_results.item())
        ssim_inter.append(ssim_results_bi.item())

        niqe_rgb_res = niqe_rgb(ms_fake[:,:3].flip(1)*255).mean().item()
        niqe_rgb_res_bi = niqe_rgb(ms_up[:,:3].flip(1)*255).mean().item()
        niqe_rgb_res_at = niqe_rgb(ms_interp[:,:3].flip(1)*255).mean().item()

        niqe_rgb_li.append(niqe_rgb_res)
        niqe_rgb_inter.append(niqe_rgb_res_bi)
        niqe_rgb_or.append(niqe_rgb_res_at)

        cc_res = cc_batch(ms_up[:,:], ms_fake[:,:])
        cc_res_bi = cc_batch(ms_up[:,:], ms_interp[:,:])
        cc_li_batch.append(cc_res)
        cc_inter_batch.append(cc_res_bi)

        sam_li.append(sam_batch(ms_up[:,:3], ms_fake[:,:3]))
        sam_inter.append(sam_batch(ms_interp[:,:3], ms_fake[:,:3]))

        uqi_res = uqi_batch(ms_up[:,:].detach().cpu(), ms_fake[:,:].detach().cpu())
        uqi_res_bi = uqi_batch(ms_up[:,:].detach().cpu(), ms_interp[:,:].detach().cpu())
        uqi_li.append(uqi_res)
        uqi_inter.append(uqi_res_bi)

    print(f'Final Evaluations \n'
            f'PSNR(our/inter): ({np.array(psnr_li).mean()}/{np.array(psnr_inter).mean()}) \n'
            f'RMSE(our/inter): ({np.array(rmse_li).mean()}/{np.array(rmse_inter).mean()}) \n'
            f'SSIM(our/inter): ({np.array(ssim_li).mean()}/{np.array(ssim_inter).mean()}) \n'
            f'NIQE_RGB(our/inter): ({np.array(niqe_rgb_li).mean()}/{np.array(niqe_rgb_or).mean()}/{np.array(niqe_rgb_inter).mean()}) \n'
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

    print('-----------------------------------------------------------')
    print("Starting evaluation .....  20m")
    psnr_li, psnr_inter = [], []
    rmse_li, rmse_inter = [], []
    psnr_li_batch, psnr_inter_batch = [], []
    rmse_li_batch, rmse_inter_batch = [], []
    cc_li_batch, cc_inter_batch = [], []
    sam_li, sam_inter = [], []

    ssim_li, ssim_inter = [], []
    niqe_20m_li, niqe_20m_or, niqe_20m_inter = [], [], []

    uqi_li, uqi_inter = [], []

    for i, data in enumerate(eval_dataloader):
        ms, ms_rgb, ms_up, ms_interp = data
        ms = ms.to(device)
        ms_rgb = ms_rgb.to(device)
        ms_up = ms_up.to(device)

        ms_fake = netG(ms, ms_rgb)

        ms_fake = ms_fake.detach().cpu()
        ms_up = ms_up.detach().cpu()
        ms_interp = ms_interp.detach().cpu()
        ms_rgb = ms_rgb.detach().cpu()

        _, _, H, W = ms_fake.shape
        ms_fake = center_crop(ms_fake, int(H*0.9))[:,4:]
        ms_interp = center_crop(ms_interp, int(H*0.9))[:,4:]
        ms_up = center_crop(ms_up, int(H*0.9))[:,4:]

        psnr_results = psnr(ms_fake, ms_up)
        psnr_results_bi = psnr(ms_interp, ms_up)
        psnr_li.append(psnr_results.item())
        psnr_inter.append(psnr_results_bi.item())

        psnr_bands_res = psnr_bands(ms_fake.detach().cpu(), ms_up.detach().cpu())
        psnr_bands_res_bi = psnr_bands(ms_interp.detach().cpu(), ms_up.detach().cpu())
        psnr_li_batch.append(psnr_bands_res.numpy())
        psnr_inter_batch.append(psnr_bands_res_bi.numpy())

        rmse_results = rmse(recover_bandwise_20m(ms_fake, norm_data=norm), recover_bandwise_20m(ms_up, norm_data=norm))
        rmse_results_bi = rmse(recover_bandwise_20m(ms_interp, norm_data=norm), recover_bandwise_20m(ms_up, norm_data=norm))
        rmse_li.append(rmse_results.item())
        rmse_inter.append(rmse_results_bi.item())

        rmse_bands_res = rmse_bands(recover_bandwise_20m(ms_fake, norm_data=norm), recover_bandwise_20m(ms_up, norm_data=norm))
        rmse_bands_res_bi = rmse_bands(recover_bandwise_20m(ms_interp, norm_data=norm), recover_bandwise_20m(ms_up, norm_data=norm))
        rmse_li_batch.append(rmse_bands_res.numpy())
        rmse_inter_batch.append(rmse_bands_res_bi.numpy())

        ssim_results = ssim_module(ms_fake, ms_up)
        ssim_results_bi = ssim_module(ms_interp, ms_up)
        ssim_li.append(ssim_results.item())
        ssim_inter.append(ssim_results_bi.item())

        niqe_20m_res = niqe_20m(ms_fake[:,:3].flip(1)*255).mean().item()
        niqe_20m_res_bi = niqe_20m(ms_up[:,:3].flip(1)*255).mean().item()
        niqe_20m_res_at = niqe_20m(ms_interp[:,:3].flip(1)*255).mean().item()

        niqe_20m_li.append(niqe_20m_res)
        niqe_20m_inter.append(niqe_20m_res_bi)
        niqe_20m_or.append(niqe_20m_res_at)

        cc_res = cc_batch(ms_up[:,:], ms_fake[:,:])
        cc_res_bi = cc_batch(ms_up[:,:], ms_interp[:,:])
        cc_li_batch.append(cc_res)
        cc_inter_batch.append(cc_res_bi)

        sam_li.append(sam_batch(ms_up[:,:3], ms_fake[:,:3]))
        sam_inter.append(sam_batch(ms_interp[:,:3], ms_fake[:,:3]))

        uqi_res = uqi_batch(ms_up[:,:].detach().cpu(), ms_fake[:,:].detach().cpu())
        uqi_res_bi = uqi_batch(ms_up[:,:].detach().cpu(), ms_interp[:,:].detach().cpu())
        uqi_li.append(uqi_res)
        uqi_inter.append(uqi_res_bi)

    print(f'Final Evaluations \n'
            f'PSNR(our/inter): ({np.array(psnr_li).mean()}/{np.array(psnr_inter).mean()}) \n'
            f'RMSE(our/inter): ({np.array(rmse_li).mean()}/{np.array(rmse_inter).mean()}) \n'
            f'SSIM(our/inter): ({np.array(ssim_li).mean()}/{np.array(ssim_inter).mean()}) \n'
            f'NIQE_20m(our/inter): ({np.array(niqe_20m_li).mean()}/{np.array(niqe_20m_or).mean()}/{np.array(niqe_20m_inter).mean()}) \n'
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

    print('-----------------------------------------------------------')

    # Final evaluation of model
    # netG = netG.to(device)
    # ms_orig_up = F.interpolate(ms_orig, scale_factor=4, mode="nearest")
    # ms_fake = netG(ms_orig, ms_rgb)
    # D_fake = torch.abs(ms_fake - ms_orig_up).mean(1)
    # # savemat_plot_senven_20mnorm(ms_fake, ms_orig_up, ven_up_orig, save_dir, s, D_fake, abs_diff=True, filename=f'plot_final.png')
    # savemat_plot_senven_RGBnorm(ms_fake[:,:4], ms_orig_up[:,:4], ven_up_orig[:,:4], save_dir, s, D_fake, filename=f'plot_rgbn_final.png')
    # savemat_plot_senven_20mnorm(ms_fake[:,4:], ms_orig_up[:,4:], ven_up_orig[:,4:], save_dir, s, D_fake, filename=f'plot_20m_final.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_TA', dest='target_adapt', action='store_true')
    parser.add_argument('--evals', dest='evaluate', action='store_true')
    parser.add_argument('--checkpoint_g', type=str, default=None, help='Select generator weights by providing path')
    parser.add_argument('--checkpoint_d', type=str, default=None, help='Select discriminator weights by providing path')
    parser.add_argument('--data_path', type=str, default=None, help='Select data clips path')
    parser.add_argument('--batch', type=int, default=None, help='path to checkpoint')
    parser.add_argument('--nepoch', type=int, default=None, help='Num of epoch for TA')
    parser.add_argument('--artif', dest='do_artif', action='store_true')
    parser.set_defaults(feature=False)

    args = parser.parse_args()

    if args.batch == None:
        batch_size = config.TRAIN.batch_size
    else:
        batch_size = args.batch
        print(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.nepoch==None:
        n_epoch_init = 150 #config.TRAIN.n_epoch_init
    else:
        n_epoch_init = args.nepoch
    # create folders to save result images and trained models
    save_dir = "samples_test/"
    checkpoint_dir = "../weights/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    print(args.do_artif)

    test(args)
