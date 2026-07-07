import numpy as np
import torch
import torch.nn as nn
from stylegan.torch_utils import misc
from stylegan.torch_utils import persistence
from stylegan.torch_utils.ops import conv2d_resample
from stylegan.torch_utils.ops import upfirdn2d
from stylegan.torch_utils.ops import bias_act
from stylegan.torch_utils.ops import fma
from torch.nn import functional as F
from stylegan.training.networks import *
from networks.pixel_shuffle import PixelShufflePack
from networks.model import _ResidualResidualDenseBlock, ConstantInput
from networks import CALayer
from networks.aan import PA


class GLEANStyleGANv2(torch.nn.Module):
    def __init__(self, 
                 in_size, 
                 out_size, 
                 n_channels=4,
                 rrdb_channels=64, 
                 num_rrdbs=23, 
                 style_channels=512,
                 num_mlps=8,
                 rgb_channels=3, 
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1]
    ):
        super().__init__()
        if in_size >= out_size:
            raise ValueError('in_size must be smaller than out_size, but got '
                             f'{in_size} and {out_size}.')

        self.in_size = in_size
        self.style_channels = style_channels
        self.register_buffer('blur_kernel', upfirdn2d.setup_filter(blur_kernel))
        self.generator = SynthesisNetwork(style_channels, out_size, n_channels, channel_base=16384)
        self.generator.requires_grad_(False)        
        channels = self.generator.channels_dict
        
        # print(channels, 'channel_dict')

        # encoder
        num_styles = int(np.log2(out_size)) * 2 - 2
        self.encoder_res = [2**i for i in range(int(np.log2(in_size)), 1, -1)]
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    n_channels, rrdb_channels, num_blocks=num_rrdbs),
                nn.Conv2d(
                    rrdb_channels, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        )

        for res in self.encoder_res:
            in_channels = channels[res]
            if res > 4:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.encoder.append(block)

        # encoder 2
        self.encoder2_res = [2**i for i in range(int(np.log2(out_size)), 1, -1)]
        self.encoder2 = nn.ModuleList()
        self.encoder2.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    rgb_channels, rrdb_channels, num_blocks=9),
                nn.Conv2d(
                    rrdb_channels, channels[out_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        )
        
        for res in self.encoder2_res:
            in_channels = channels[res]
            if res > 4:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.encoder2.append(block)
            
        # additional modules for StyleGANv2
        self.const_inp = ConstantInput(channels[4])
        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for res in self.encoder2_res[::-1]:
            num_channels = channels[res]
            if res not in self.encoder_res:
                self.fusion_out.append(
                    nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))
                self.fusion_skip.append(
                    nn.Conv2d(num_channels + 4, 4, 3, 1, 1, bias=True))
            else:
                self.fusion_out.append(
                    nn.Conv2d(num_channels * 3, num_channels, 3, 1, 1, bias=True))
                self.fusion_skip.append(
                    nn.Conv2d((num_channels * 2) + 4, 4, 3, 1, 1, bias=True))

        # decoder
        decoder_res = [
            2**i
            for i in range(int(np.log2(in_size)), int(np.log2(out_size) + 1))
        ]
        
        # print('channels', channels)
        
        self.decoder = nn.ModuleList()
        for res in decoder_res:
            # print(res, 'decoder')
            if res == in_size:
                in_channels = channels[res]
            else:
                in_channels = 2 * channels[res]

            if res < out_size:
                out_channels = channels[res * 2]
                self.decoder.append(
                    PixelShufflePack(
                        in_channels, out_channels, 2, upsample_kernel=3))
            else:
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, 64, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(64, n_channels, 3, 1, 1)))

    def forward(self, lq, rgb):

        h, w = lq.shape[2:]
        if h != self.in_size or w != self.in_size:
            raise AssertionError(
                f'Spatial resolution must equal in_size ({self.in_size}).'
                f' Got ({h}, {w}).')
        
        # encoder
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]

        latent = encoder_features[0].view(lq.size(0), -1, self.style_channels)
        # print(latent.shape, encoder_features[0].shape)
        encoder_features = encoder_features[1:]

        # encoder RGB
        feat2 = rgb
        encoder2_features = []
        for block in self.encoder2:
            feat2 = block(feat2)
            encoder2_features.append(feat2)
        encoder2_features = encoder2_features[::-1]
        latent2 = encoder2_features[0].view(lq.size(0), -1, self.style_channels)
        encoder2_features = encoder2_features[1:]

        # 4x4 stage
        block = getattr(self.generator, f'b{4}')
        out = self.const_inp(latent)
        out = block.conv1(out, latent[:, 0])
        skip = block.torgb(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher res
        generator_features = []
        for res in self.encoder2_res[:-1][::-1]:
            # print('b', res)
            # feature fusion by channel-wise concatenation
            block = getattr(self.generator, f'b{res}')
            # print(vars(block))
            if out.size(2) <= self.in_size:
                fusion_index = (_index - 1) // 2
                feat = encoder_features[fusion_index]
                feat2 = encoder2_features[fusion_index]

                out = torch.cat([out, feat, feat2], dim=1)
                # print(out.shape, feat.shape, feat2.shape, "fusion out")
                out = self.fusion_out[fusion_index](out)

                skip = torch.cat([skip, feat, feat2], dim=1)
                # print(skip.shape, feat.shape, feat2.shape, "fusion skip")
                skip = self.fusion_skip[fusion_index](skip)
            else:
                fusion_index = (_index - 1) // 2
                feat2 = encoder2_features[fusion_index]

                out = torch.cat([out, feat2], dim=1)
                out = self.fusion_out[fusion_index](out)

                skip = torch.cat([skip, feat2], dim=1)
                skip = self.fusion_skip[fusion_index](skip)

            # original StyleGAN operations
            out = block.conv0(out, latent[:, _index])
            out = block.conv1(out, latent[:, _index + 1])
            img = upfirdn2d.upsample2d(skip, self.blur_kernel)
            skip = block.torgb(out, latent[:, _index + 2])   
            skip = img + skip
            skip = skip.to(torch.float32)

            # store features for decoder
            if out.size(2) > self.in_size:
                generator_features.append(out)

            _index += 2

        # decoder
        hr = encoder_features[-1]
        for i, block in enumerate(self.decoder):
            if i > 0:
                hr = torch.cat([hr, generator_features[i - 1]], dim=1)
            hr = block(hr)

        return hr


class SenGLEANStyleGANv2(torch.nn.Module):
    def __init__(self, 
                 in_size, 
                 out_size, 
                 n_channels=4,
                 rrdb_channels=64, 
                 num_rrdbs=16, 
                 style_channels=512,
                 num_mlps=8,
                 rgb_channels=4, 
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1]
    ):
        super().__init__()
        if in_size >= out_size:
            raise ValueError('in_size must be smaller than out_size, but got '
                             f'{in_size} and {out_size}.')

        self.in_size = in_size
        self.out_size = out_size
        self.style_channels = style_channels
        self.register_buffer('blur_kernel', upfirdn2d.setup_filter(blur_kernel))
        self.generator = SynthesisNetwork(style_channels, out_size, n_channels + rgb_channels, channel_base=16384)
        # self.generator.requires_grad_(False)        
        channels = self.generator.channels_dict
        
        print(channels)
        # encoder
        num_styles = int(np.log2(out_size)) * 2 - 2
        self.encoder_res = [2**i for i in range(int(np.log2(in_size)), 1, -1)]
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    n_channels, rrdb_channels, num_blocks=num_rrdbs),
                nn.Conv2d(
                    rrdb_channels, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        )

        for res in self.encoder_res:
            in_channels = channels[res]
            if res > 4:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.encoder.append(block)

        # encoder 2
        self.encoder2_res = [2**i for i in range(int(np.log2(out_size//2)), 1, -1)]
        self.encoder2 = nn.ModuleList()
        self.encoder2.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    rgb_channels, rrdb_channels, num_blocks=num_rrdbs),
                nn.Conv2d(
                    rrdb_channels, channels[int(out_size//2)], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        )
        
        self.out_res = [2**i for i in range(int(np.log2(out_size)), 1, -1)]
        
        for res in self.encoder2_res:
            in_channels = channels[res]
            if res > 4:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.encoder2.append(block)
        
        self.latent_fusion = nn.Linear(2*(num_styles * style_channels), (num_styles * style_channels))
            
        # additional modules for StyleGANv2
        self.const_inp = ConstantInput(channels[4])
        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for res in self.out_res[::-1]:
            num_channels = channels[res]
            if (res not in self.encoder_res) and (res in self.encoder2_res):
                # print(res, 'encoder1', num_channels * 2, channels[res])
                # self.fusion_out.append(
                #     nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))
                self.fusion_out.append(
                    Merge(num_channels * 2, num_channels))
                self.fusion_skip.append(
                    nn.Conv2d(num_channels + 8, 8, 3, 1, 1, bias=True))
            elif (res not in self.encoder2_res) and (res in self.out_res):
                # print(res, 'out')
                # self.fusion_out.append(
                #     nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=True))
                self.fusion_out.append(
                    Merge(num_channels, num_channels))
                self.fusion_skip.append(
                    nn.Conv2d(8, 8, 3, 1, 1, bias=True))
            else:
                # print(res, 'encoder2', num_channels * 3)
                # self.fusion_out.append(
                #     nn.Conv2d(num_channels * 3, num_channels, 3, 1, 1, bias=True))
                self.fusion_out.append(
                    Merge(num_channels * 3, num_channels))
                self.fusion_skip.append(
                    nn.Conv2d((num_channels * 2) + 8, 8, 3, 1, 1, bias=True))

        # decoder
        decoder_res = [
            2**i
            for i in range(int(np.log2(in_size)), int(np.log2(out_size) + 1))
        ]
        
        # print('channels', channels)
        self.decoder_fusion = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for res in decoder_res:
            if res == in_size:
                in_channels = channels[res]
            elif res == (out_size//2):
                in_channels = 3 * channels[res]
            else:
                in_channels = 2 * channels[res]

            if res < out_size:
                out_channels = channels[res * 2]
                # self.decoder.append(
                #     PixelShufflePack(
                #         in_channels, out_channels, 2, upsample_kernel=3)
                #     )
                self.decoder.append(
                    UpsamplingBlock(in_channels, out_channels, 2)
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, 64, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(64, n_channels + rgb_channels, 3, 1, 1)))

    def forward(self, lq, rgb):

        h, w = lq.shape[2:]
        if h != self.in_size or w != self.in_size:
            raise AssertionError(
                f'Spatial resolution must equal in_size ({self.in_size}).'
                f' Got ({h}, {w}).')
        
        # encoder
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]

        latent1 = encoder_features[0]
        encoder_features = encoder_features[1:]

        # encoder RGB
        feat2 = rgb
        encoder2_features = []
        for block in self.encoder2:
            feat2 = block(feat2)
            encoder2_features.append(feat2)
        encoder2_features = encoder2_features[::-1]
        latent2 = encoder2_features[0]
        latent = self.latent_fusion(torch.cat([latent2, latent1], dim=1))
        latent = latent.view(lq.size(0), -1, self.style_channels)
        
        encoder2_features = encoder2_features[1:]

        # 4x4 stage
        block = getattr(self.generator, f'b{4}')
        out = self.const_inp(latent)
        out = block.conv1(out, latent[:, 0])
        skip = block.torgb(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher res
        generator_features = []
        for res in self.out_res[:-1][::-1]:
            # print('b', res, out.shape)
            # feature fusion by channel-wise concatenation
            block = getattr(self.generator, f'b{res}')
            if out.size(2) <= (self.in_size):
                fusion_index = (_index - 1) // 2
                feat = encoder_features[fusion_index]
                feat2 = encoder2_features[fusion_index]

                out = torch.cat([out, feat, feat2], dim=1)
                # print(out.shape, feat.shape, feat2.shape, "fusion out")
                out = self.fusion_out[fusion_index](out)

                skip = torch.cat([skip, feat, feat2], dim=1)
                # print(skip.shape, feat.shape, feat2.shape, "fusion skip")
                skip = self.fusion_skip[fusion_index](skip)
            elif out.size(2) == self.out_size:
                fusion_index = (_index - 1) // 2
                out = self.fusion_out[fusion_index](out)
                skip = self.fusion_skip[fusion_index](skip)
            else:
                fusion_index = (_index - 1) // 2
                feat2 = encoder2_features[fusion_index]

                out = torch.cat([out, feat2], dim=1)
                out = self.fusion_out[fusion_index](out)

                skip = torch.cat([skip, feat2], dim=1)
                skip = self.fusion_skip[fusion_index](skip)

            # original StyleGAN operations
            out = block.conv0(out, latent[:, _index])
            out = block.conv1(out, latent[:, _index + 1])
            img = upfirdn2d.upsample2d(skip, self.blur_kernel)
            skip = block.torgb(out, latent[:, _index + 2])   
            skip = img + skip
            skip = skip.to(torch.float32)

            # store features for decoder
            if out.size(2) > self.in_size:
                generator_features.append(out)

            _index += 2

        # decoder
        hr = encoder_features[-1]
        hr_rgb = encoder2_features[-1]
        for i, block in enumerate(self.decoder):
            if i==1:
                hr = torch.cat([hr_rgb, hr, generator_features[i - 1]], dim=1)
            elif i > 0:
                hr = torch.cat([hr, generator_features[i - 1]], dim=1)
            hr = block(hr)

        return hr
    
    
class LightSenGLEANStyleGANv2(torch.nn.Module):
    def __init__(self, 
                 in_size, 
                 out_size, 
                 n_channels=4,
                 rrdb_channels=64, 
                 num_rrdbs=16, 
                 style_channels=512,
                 num_mlps=8,
                 rgb_channels=4, 
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1]
    ):
        super().__init__()
        if in_size >= out_size:
            raise ValueError('in_size must be smaller than out_size, but got '
                             f'{in_size} and {out_size}.')

        self.in_size = in_size
        self.out_size = out_size
        self.style_channels = style_channels
        self.register_buffer('blur_kernel', upfirdn2d.setup_filter(blur_kernel))
        self.generator = LightSynthesisNetwork(style_channels, in_size, out_size, n_channels + rgb_channels, channel_base=16384)
        # self.generator.requires_grad_(False)        
        channels = self.generator.channels_dict
        
        print(channels)
        # encoder
        num_styles = (int(np.log2(out_size)) - int(np.log2(in_size))) * 3
        self.encoder_res = [2**i for i in range(int(np.log2(in_size)), 1, -1)]
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    n_channels, rrdb_channels, num_blocks=num_rrdbs),
                nn.Conv2d(
                    rrdb_channels, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        )

        # Encoder 1 downscaling layers removed
        # for res in self.encoder_res:
        #     in_channels = channels[res]
        #     if res > 4:
        #         out_channels = channels[res // 2]
        #         block = nn.Sequential(
        #             nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
        #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #             nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
        #             nn.LeakyReLU(negative_slope=0.2, inplace=True))
        #     else:
        #         block = nn.Sequential(
        #             nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
        #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #             nn.Flatten(),
        #             nn.Linear(16 * in_channels, num_styles * style_channels))
        #     self.encoder.append(block)

        # encoder 2
        self.encoder2_res = [2**i for i in range(int(np.log2(out_size//2)), int(np.log2(in_size))-1, -1)]
        self.encoder2 = nn.ModuleList()
        self.encoder2.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    rgb_channels, rrdb_channels, num_blocks=num_rrdbs),
                nn.Conv2d(
                    rrdb_channels, channels[int(out_size//2)], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        )
        
        self.out_res = [2**i for i in range(int(np.log2(out_size)), int(np.log2(in_size))-1, -1)]
        
        for res in self.encoder2_res[:-1]:
            print('encoder2', res)
            in_channels = channels[res]
            if res > 4:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.encoder2.append(block)
        
        # self.latent_fusion = nn.Linear(2*(num_styles * style_channels), (num_styles * style_channels))
        self.parameter = nn.Parameter(torch.rand((num_styles, style_channels)), requires_grad=True)
        # additional modules for StyleGANv2
        # self.const_inp = ConstantInput(channels[4])
        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for res in self.out_res[::-1]:
            num_channels = channels[res]
            if (res not in self.encoder_res) and (res in self.encoder2_res):
                self.fusion_out.append(
                    Merge(num_channels * 2, num_channels))
                self.fusion_skip.append(
                    nn.Conv2d(num_channels + 8, 8, 3, 1, 1, bias=True))
            elif (res not in self.encoder2_res) and (res in self.out_res):
                self.fusion_out.append(
                    Merge(num_channels, num_channels))
                self.fusion_skip.append(
                    nn.Conv2d(8, 8, 3, 1, 1, bias=True))
            else:
                self.fusion_out.append(
                    Merge(num_channels * 2, num_channels))
                self.fusion_skip.append(
                    nn.Conv2d((num_channels * 2), 8, 3, 1, 1, bias=True))

        # decoder
        decoder_res = [
            2**i
            for i in range(int(np.log2(in_size)), int(np.log2(out_size) + 1))
        ]
        
        # print('channels', channels)
        self.decoder_fusion = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for res in decoder_res:
            if res == in_size:
                in_channels = channels[res]
            elif res == (out_size//2):
                in_channels = 3 * channels[res]
            else:
                in_channels = 2 * channels[res]

            if res < out_size:
                out_channels = channels[res * 2]
                self.decoder.append(
                    UpsamplingBlock(in_channels, out_channels, 2)
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, 64, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(64, n_channels + rgb_channels, 3, 1, 1)))

    def forward(self, lq, rgb):

        h, w = lq.shape[2:]
        if h != self.in_size or w != self.in_size:
            raise AssertionError(
                f'Spatial resolution must equal in_size ({self.in_size}).'
                f' Got ({h}, {w}).')
        
        # encoder
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]

        # latent1 = encoder_features[0]
        # encoder_features = encoder_features[1:]

        # encoder RGB
        feat2 = rgb
        encoder2_features = []
        for block in self.encoder2:
            feat2 = block(feat2)
            encoder2_features.append(feat2)
        encoder2_features = encoder2_features[::-1]
        
        # latent2 = encoder2_features[0]
        # latent = self.latent_fusion(torch.cat([latent2, latent1], dim=1))
        # latent = latent.view(lq.size(0), -1, self.style_channels)
        latent = self.parameter.repeat(lq.size(0), 1, 1)
        # encoder2_features = encoder2_features[1:]

        # 4x4 stage
        # block = getattr(self.generator, f'b{4}')
        # out = self.const_inp(latent)
        # out = block.conv1(out, latent[:, 0])
        # skip = block.torgb(out, latent[:, 1])

        _index = 1
        out = encoder_features[0]
        out_rgbn = encoder2_features[0]

        # 8x8 ---> higher res
        generator_features = []
        for res in self.out_res[:-1][::-1]:
            # print('b', res, out.shape)
            
            # feature fusion by channel-wise concatenation
            block = getattr(self.generator, f'b{res}')
            fusion_index = (_index - 1) // 2
            
            if out.size(2) <= (self.in_size):
                # feat = encoder_features[fusion_index]
                feat2 = encoder2_features[fusion_index]

                out = torch.cat([out, feat2], dim=1)
                # print(out.shape, feat.shape, feat2.shape, "fusion out")
                out = self.fusion_out[fusion_index](out)

                skip = torch.cat([out, feat2], dim=1)
                # print(skip.shape, feat.shape, feat2.shape, "fusion skip")
                skip = self.fusion_skip[fusion_index](skip)
            elif out.size(2) == self.out_size:
                out = self.fusion_out[fusion_index](out)
                skip = self.fusion_skip[fusion_index](skip)
                
            else:
                feat2 = encoder2_features[fusion_index]

                out = torch.cat([out, feat2], dim=1)
                out = self.fusion_out[fusion_index](out)

                skip = torch.cat([skip, feat2], dim=1)
                skip = self.fusion_skip[fusion_index](skip)

            # original StyleGAN operations
            out = block.conv0(out, latent[:, _index])
            out = block.conv1(out, latent[:, _index + 1])
            img = upfirdn2d.upsample2d(skip, self.blur_kernel)
            skip = block.torgb(out, latent[:, _index + 2])   
            skip = img + skip
            skip = skip.to(torch.float32)

            # store features for decoder
            if out.size(2) > self.in_size:
                generator_features.append(out)

            _index += 2

        # decoder
        hr = encoder_features[-1]
        hr_rgb = encoder2_features[-1]
        for i, block in enumerate(self.decoder):
            if i==1:
                hr = torch.cat([hr_rgb, hr, generator_features[i - 1]], dim=1)
            elif i > 0:
                hr = torch.cat([hr, generator_features[i - 1]], dim=1)
            hr = block(hr)

        return hr


class LightGLEANStyleGANv2(torch.nn.Module):
    def __init__(self, 
                 in_size, 
                 out_size, 
                 n_channels=4,
                 rrdb_channels=64, 
                 num_rrdbs=23, 
                 style_channels=512,
                 num_mlps=8,
                 rgb_channels=3, 
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1]
    ):
        super().__init__()
        if in_size >= out_size:
            raise ValueError('in_size must be smaller than out_size, but got '
                             f'{in_size} and {out_size}.')

        self.in_size = in_size
        self.style_channels = style_channels
        self.register_buffer('blur_kernel', upfirdn2d.setup_filter(blur_kernel))
        self.generator = LightSynthesisNetwork(style_channels, in_size, out_size, n_channels, channel_base=16384)
        self.generator.requires_grad_(False)
        channels = self.generator.channels_dict

        # encoder
        num_styles = (int(np.log2(out_size)) - int(np.log2(in_size))) * 3
        self.encoder_res = [2**i for i in range(int(np.log2(in_size)), 1, -1)]
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    n_channels, rrdb_channels, num_blocks=num_rrdbs),
                nn.Conv2d(
                    rrdb_channels, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        )

        # for res in self.encoder_res:
        #     in_channels = channels[res]
        #     if res > 4:
        #         out_channels = channels[res // 2]
        #         block = nn.Sequential(
        #             nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
        #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #             nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
        #             nn.LeakyReLU(negative_slope=0.2, inplace=True))
        #     else:
        #         block = nn.Sequential(
        #             nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
        #             nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #             nn.Flatten(),
        #             nn.Linear(16 * in_channels, num_styles * style_channels))
        #     self.encoder.append(block)

        # encoder 2
        self.encoder2_res = [2**i for i in range(int(np.log2(out_size)), int(np.log2(in_size))-1, -1)]
        self.encoder2 = nn.ModuleList()
        
        self.encoder2.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    rgb_channels, rrdb_channels, num_blocks=9),
                nn.Conv2d(
                    rrdb_channels, channels[out_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        )
        
        # print(channels, self.generator.block_resolutions, num_styles)
        for res in self.encoder2_res[:-1]:
            print(res)
            in_channels = channels[res]
            if res > 4:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.encoder2.append(block)
            
        # additional modules for StyleGANv2
        self.parameter = nn.Parameter(torch.rand((num_styles, style_channels)), requires_grad=True)
        # self.const_inp = ConstantInput(channels[4])
        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for res in self.encoder2_res[::-1]:
            num_channels = channels[res]
            if res not in self.encoder_res:
                self.fusion_out.append(
                    nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))
                self.fusion_skip.append(
                    nn.Conv2d(num_channels + 4, 4, 3, 1, 1, bias=True))
            else:
                self.fusion_out.append(
                    nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))
                self.fusion_skip.append(
                    nn.Conv2d(num_channels*2, 4, 3, 1, 1, bias=True))

        # decoder
        decoder_res = [
            2**i
            for i in range(int(np.log2(in_size)), int(np.log2(out_size) + 1))
        ]
        # print('channels', channels)
        
        self.decoder = nn.ModuleList()
        for res in decoder_res:
            # print(res, 'decoder')
            if res == in_size:
                in_channels = channels[res]
            else:
                in_channels = 2 * channels[res]

            if res < out_size:
                out_channels = channels[res * 2]
                self.decoder.append(
                    PixelShufflePack(
                        in_channels, out_channels, 2, upsample_kernel=3))
            else:
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, 64, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(64, n_channels, 3, 1, 1)))

    def forward(self, lq, rgb):

        h, w = lq.shape[2:]
        if h != self.in_size or w != self.in_size:
            raise AssertionError(
                f'Spatial resolution must equal in_size ({self.in_size}).'
                f' Got ({h}, {w}).')
        
        # encoder
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]
        # print(len(encoder_features))

        latent = self.parameter.repeat(lq.size(0), 1, 1)
        # encoder_features = encoder_features[1:]

        # encoder RGB
        feat2 = rgb
        encoder2_features = []
        for block in self.encoder2:
            feat2 = block(feat2)
            encoder2_features.append(feat2)
        encoder2_features = encoder2_features[::-1]
        # print(len(encoder2_features), latent.shape, 'latent')
        # encoder2_features = encoder2_features[1:]

        # 4x4 stage
        # block = getattr(self.generator, f'b{4}')
        # out = self.const_inp(latent)
        # out = block.conv1(out, latent[:, 0])
        # skip = block.torgb(out, latent[:, 1])

        _index = 1
        out = encoder_features[0]

        # 8x8 ---> higher res
        generator_features = []
        for res in self.encoder2_res[:-1][::-1]:
            block = getattr(self.generator, f'b{res}')
            fusion_index = (_index - 1) // 2
            # print(res, 'forward', fusion_index, _index)
            
            if out.size(2) <= self.in_size:
                # feat = encoder_features[fusion_index]
                feat2 = encoder2_features[fusion_index]

                out = torch.cat([out, feat2], dim=1)
                # print(out.shape, feat.shape, feat2.shape, "fusion out")
                out = self.fusion_out[fusion_index](out)

                skip = torch.cat([out, feat2], dim=1)
                # print(skip.shape, feat.shape, feat2.shape, "fusion skip")
                skip = self.fusion_skip[fusion_index](skip)
            else:
                feat2 = encoder2_features[fusion_index]

                out = torch.cat([out, feat2], dim=1)
                out = self.fusion_out[fusion_index](out)

                skip = torch.cat([skip, feat2], dim=1)
                skip = self.fusion_skip[fusion_index](skip)

            # original StyleGAN operations
            out = block.conv0(out, latent[:, _index])
            out = block.conv1(out, latent[:, _index + 1])
            img = upfirdn2d.upsample2d(skip, self.blur_kernel)
            skip = block.torgb(out, latent[:, _index + 2])   
            skip = img + skip
            skip = skip.to(torch.float32)

            # store features for decoder
            if out.size(2) > self.in_size:
                generator_features.append(out)

            _index += 2

        # decoder
        hr = encoder_features[-1]
        for i, block in enumerate(self.decoder):
            if i > 0:
                hr = torch.cat([hr, generator_features[i - 1]], dim=1)
            hr = block(hr)

        return hr


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        
        self.scale = scale
        self.block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, (3, 3), (1, 1), (1, 1)),
                    PA(in_channels),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(in_channels, in_channels, (3, 3), (1, 1), (1, 1)),
                    nn.LeakyReLU(0.2, True))

        if scale==4:
            self.block2 = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, (3, 3), (1, 1), (1, 1)),
                    PA(in_channels),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(in_channels, in_channels, (3, 3), (1, 1), (1, 1)),
                    nn.LeakyReLU(0.2, True))
        
        self.conv_out = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1))
    
    def forward(self, x):
        
        x = self.block(F.interpolate(x, scale_factor=2, mode="nearest"))
        if self.scale == 4:
            x = self.block2(F.interpolate(x, scale_factor=2, mode="nearest"))
        x = self.conv_out(x)
        return x
    
    
class Merge(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.ca_layer = CALayer(in_channel)
        self.conv_out = nn.Conv2d(in_channel, out_channel, (3, 3), (1, 1), (1, 1))
        
    def forward(self, x):
        
        x = self.ca_layer(x)
        x = self.conv_out(x)
        return x
        

class RRDBFeatureExtractor(nn.Module):
    """Feature extractor composed of Residual-in-Residual Dense Blocks (RRDBs).

    It is equivalent to ESRGAN with the upsampling module removed.

    Args:
        in_channels (int): Channel number of inputs. Default: 3.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Default: 23.
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 in_channels=3,
                 mid_channels=64,
                 num_blocks=23,
                 growth_channels=32):

        super().__init__()
        
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # Feature extraction backbone network.
        trunk = []
        for _ in range(num_blocks):
            trunk.append(_ResidualResidualDenseBlock(mid_channels, growth_channels))
        self.body = nn.Sequential(*trunk)
        
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        feat = self.conv_first(x)
        return feat + self.conv_body(self.body(feat))
