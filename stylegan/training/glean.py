import numpy as np
import torch
import torch.nn as nn
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

from training.networks import *
from training.pixel_shuffle import PixelShufflePack

class GLEANStyleGANv2(torch.nn.Module):
    def __init__(self, 
                 in_size, 
                 out_size, 
                 num_channels=4, 
                 rrdb_channels=64, 
                 num_rrdbs=23, 
                 style_channels=512,
                 num_mlps=8,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1]
                 ):
        super().__init__()
        if in_size >= out_size:
            raise ValueError('in_size must be smaller than out_size, but got '
                             f'{in_size} and {out_size}.')

        self.in_size = in_size
        self.style_channels = style_channels
        self.generator = SynthesisNetwork(style_channels, out_size, num_channels)
        self.generator.requires_grad_(False)        
        channels = self.generator.channels_dict

        # encoder
        num_styles = int(np.log2(out_size)) * 2 - 2
        self.encoder_res = [2**i for i in range(int(np.log2(in_size)), 1, -1)]
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    num_channels, rrdb_channels, num_blocks=num_rrdbs),
                nn.Conv2d(
                    rrdb_channels, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        )

        self.encoder2_res = [2**i for i in range(int(np.log2(out_size)), 1, -1)]
        self.encoder2 = nn.ModuleList()
        self.encoder2.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    num_channels, rrdb_channels, num_blocks=num_rrdbs),
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
        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for res in self.encoder2_res[::-1]:
            num_channels = channels[res]
            if res not in self.encoder_res:
                self.fusion_out.append(
                    nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))
                self.fusion_skip.append(
                    nn.Conv2d(num_channels + 3, 3, 3, 1, 1, bias=True))
            else:
                self.fusion_out.append(
                    nn.Conv2d(num_channels * 3, num_channels, 3, 1, 1, bias=True))
                self.fusion_skip.append(
                    nn.Conv2d((num_channels * 2) + 3, 3, 3, 1, 1, bias=True))

        # decoder
        decoder_res = [
            2**i
            for i in range(int(np.log2(in_size)), int(np.log2(out_size) + 1))
        ]
        self.decoder = nn.ModuleList()
        for res in decoder_res:
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
                        nn.Conv2d(64, num_channels, 3, 1, 1)))

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
        encoder_features = encoder_features[1:]

        encoder2_features = []
        for block in self.encoder:
            feat = block(rgb)
            encoder2_features.append(rgb)
        encoder2_features = encoder2_features[::-1]
        latent2 = encoder2_features[0].view(lq.size(0), -1, self.style_channels)
        encoder2_features = encoder2_features[1:]

        # 4x4 stage
        block = getattr(self.generator, f'b{4}')
        out = self.generator.constant_input(latent)
        out = block.conv1(out, latent[:, 0])
        skip = block.torgb(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher res
        generator_features = []
        for res in self.encoder2_res[::-1]:
            # feature fusion by channel-wise concatenation
            block = getattr(self.generator, f'b{res}')
            if out.size(2) <= self.in_size:
                fusion_index = (_index - 1) // 2
                feat = encoder_features[fusion_index]
                feat2 = encoder2_features[fusion_index]

                out = torch.cat([out, feat, feat2], dim=1)
                out = self.fusion_out[fusion_index](out)

                skip = torch.cat([skip, feat, feat2], dim=1)
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
            skip = block.torgb(out, latent[:, _index + 2])   # skip is added out 
            out = out + skip
            out = out.to(torch.float32)

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
