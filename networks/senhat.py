import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.glean import RRDBFeatureExtractor, Merge, UpsamplingBlock
from networks.hat.hat_arch import HAT


# HAT size presets. All variants run at a 64x64 working resolution for the first
# implementation, so ``window_size`` is chosen to divide 64. A 128x128 working
# resolution is intentionally left as a separate future study, not coupled to size.
HAT_VARIANTS = {
    'tiny':  dict(embed_dim=60,  depths=(4, 4),          num_heads=(4, 4),          window_size=16),
    'small': dict(embed_dim=96,  depths=(6, 6, 6),       num_heads=(6, 6, 6),       window_size=16),
    'base':  dict(embed_dim=144, depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6), window_size=16),
}

# HAT attention hyper-parameters shared across variants (official HAT defaults).
_HAT_COMMON = dict(compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, mlp_ratio=2.)


def _stylegan_channels_dict(out_size, channel_base=16384, channel_max=512):
    """Reproduce StyleGAN's ``SynthesisNetwork.channels_dict`` exactly.

    SenGLEANStyleGANv2 builds its generator with ``channel_base=16384`` (and the
    default ``channel_max=512``), so the encoder/decoder channel widths are keyed
    off this table. We replicate it here (without instantiating StyleGAN) so the
    decoder interface is byte-for-byte identical. For out_size=256 this yields
    ``{4:512, 8:512, 16:512, 32:512, 64:256, 128:128, 256:64}``.
    """
    log2 = int(np.log2(out_size))
    block_resolutions = [2 ** i for i in range(2, log2 + 1)]
    return {res: min(channel_base // res, channel_max) for res in block_resolutions}


class SenHAT(torch.nn.Module):
    """SenGLEAN with the StyleGAN generative latent bank replaced by a HAT backbone.

    Drop-in replacement for ``SenGLEANStyleGANv2``: identical constructor signature
    (plus ``hat_variant``) and identical ``forward(lq, rgb)`` contract returning an
    ``(n_channels + rgb_channels)``-channel image at ``out_size``.

    The two encoders and the decoder are preserved unchanged. The StyleGAN synthesis
    (const input + per-scale style-block conv0/conv1/torgb + latent path) is replaced
    by: a fused input projection -> official HAT deep-feature body at ``in_size`` ->
    HAT's native PixelShuffle reconstruction path, tapped after each 2x stage and
    fused (via the existing ``Merge``/CALayer channel-attention) to produce exactly
    the ``generator_features`` tensors the decoder expects.
    """

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
                 blur_kernel=[1, 3, 3, 1],
                 hat_variant='small'):
        super().__init__()
        if in_size >= out_size:
            raise ValueError('in_size must be smaller than out_size, but got '
                             f'{in_size} and {out_size}.')
        if hat_variant not in HAT_VARIANTS:
            raise ValueError(f'Unknown hat_variant {hat_variant!r}; '
                             f'expected one of {list(HAT_VARIANTS)}.')
        # This first implementation is designed around the 4x SenGLEAN pipeline
        # (in_size -> out_size//2 -> out_size). The tapped-upsample interface and
        # the decoder both assume this factor.
        if out_size // in_size != 4 or out_size % in_size != 0:
            raise ValueError('SenHAT currently supports only a 4x setup '
                             f'(out_size == 4 * in_size); got in_size={in_size}, out_size={out_size}.')

        self.in_size = in_size
        self.out_size = out_size
        self.style_channels = style_channels

        channels = _stylegan_channels_dict(out_size)
        self.channels = channels
        num_styles = int(np.log2(out_size)) * 2 - 2

        # ---------------- encoder E_A (Sentinel-2 20 m set, in_size) -------------
        # Kept verbatim from SenGLEANStyleGANv2. The res==4 Linear "latent head" is
        # retained so feature extraction is byte-identical; its latent output is now
        # unused (isolated, to be pruned only after HAT integration is verified).
        self.encoder_res = [2 ** i for i in range(int(np.log2(in_size)), 1, -1)]
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

        # ---------------- encoder E_B (Sentinel-2 10 m set, out_size//2) ---------
        self.encoder2_res = [2 ** i for i in range(int(np.log2(out_size // 2)), 1, -1)]
        self.encoder2 = nn.ModuleList()
        self.encoder2.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    rgb_channels, rrdb_channels, num_blocks=num_rrdbs),
                nn.Conv2d(
                    rrdb_channels, channels[int(out_size // 2)], 3, 1, 1, bias=True),
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

        # ---------------- HAT generative bank (replaces StyleGAN synthesis) ------
        cfg = HAT_VARIANTS[hat_variant]
        self.hat_variant = hat_variant
        # Official HAT, unchanged. in_chans is only used by hat.conv_first/conv_last,
        # which we bypass (we feed fused features and tap the upsample path), but it
        # must be a valid value. upsampler='pixelshuffle' builds the native
        # conv_before_upsample -> Upsample(4) reconstruction path we tap into.
        self.hat = HAT(
            img_size=in_size,
            patch_size=1,
            in_chans=n_channels + rgb_channels,
            embed_dim=cfg['embed_dim'],
            depths=cfg['depths'],
            num_heads=cfg['num_heads'],
            window_size=cfg['window_size'],
            upscale=out_size // in_size,
            img_range=1.,
            upsampler='pixelshuffle',
            **_HAT_COMMON,
        )
        self._hat_num_feat = 64  # HAT hard-codes num_feat=64 in its reconstruction path

        # Input projection (adapter): fuse a MULTI-SCALE encoder pyramid and project
        # to HAT's embed_dim. Replaces hat.conv_first, operating on FUSED feature maps
        # rather than raw imagery. Merge == CALayer(channel attention) + conv.
        # Option 1 (multi-resolution encoder guidance): reuse E_A and E_B feature maps
        # at these resolutions, resize the sub-in_size ones to in_size, concatenate and
        # channel-attention-fuse. This restores the per-scale encoder injection the
        # original SenGLEAN fusion loop provided (whose removal the paper's ablation
        # ties to a large 20 m RMSE regression). Feature maps are reused, not recomputed.
        self.pyramid_res = [in_size // (2 ** k) for k in range(4)]  # e.g. [64, 32, 16, 8]
        pyramid_in = 2 * sum(channels[r] for r in self.pyramid_res)  # E_A + E_B at each scale
        self.hat_input_fuse = Merge(pyramid_in, cfg['embed_dim'])

        # Output projections (adapters) + fusion, producing generator_features that
        # exactly match channels[out_size//2] @ out_size//2 and channels[out_size] @ out_size.
        self.hat_proj_mid = nn.Conv2d(self._hat_num_feat, channels[out_size // 2], 3, 1, 1)
        self.fusion_mid = Merge(channels[out_size // 2] + channels[out_size // 2], channels[out_size // 2])
        self.hat_proj_top = nn.Conv2d(self._hat_num_feat, channels[out_size], 3, 1, 1)
        # Option 2: the top (out_size) tap also receives encoder guidance -- E_B's
        # mid-resolution map upsampled -- so the final reconstruction stage is not
        # encoder-blind, matching the original design.
        self.fusion_top = Merge(channels[out_size] + channels[out_size // 2], channels[out_size])

        # ---------------- decoder (unchanged) -----------------------------------
        decoder_res = [
            2 ** i
            for i in range(int(np.log2(in_size)), int(np.log2(out_size) + 1))
        ]
        self.decoder = nn.ModuleList()
        for res in decoder_res:
            if res == in_size:
                in_channels = channels[res]
            elif res == (out_size // 2):
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

        # encoder E_A
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]
        encoder_features = encoder_features[1:]  # drop unused latent head output

        # encoder E_B
        feat2 = rgb
        encoder2_features = []
        for block in self.encoder2:
            feat2 = block(feat2)
            encoder2_features.append(feat2)
        encoder2_features = encoder2_features[::-1]
        encoder2_features = encoder2_features[1:]  # drop unused latent head output

        # index encoder feature maps by spatial resolution (reuse; do not recompute)
        ea_by_res = {f.shape[-1]: f for f in encoder_features}
        eb_by_res = {f.shape[-1]: f for f in encoder2_features}
        e_b_mid = encoder2_features[-1]      # channels[out_size//2] @ out_size//2

        # ----- Option 1: multi-scale encoder pyramid -> HAT input -----
        pyramid = []
        for r in self.pyramid_res:
            for feats in (ea_by_res, eb_by_res):
                fmap = feats[r]
                if r != self.in_size:
                    fmap = F.interpolate(fmap, size=(self.in_size, self.in_size),
                                         mode='bilinear', align_corners=False)
                pyramid.append(fmap)
        x = self.hat_input_fuse(torch.cat(pyramid, dim=1))  # embed_dim @ in_size
        # official HAT deep-feature path (unchanged), with HAT's residual conv_after_body
        x = self.hat.conv_after_body(self.hat.forward_features(x)) + x
        y = self.hat.conv_before_upsample(x)                             # num_feat @ in_size

        # HAT's native PixelShuffle reconstruction, tapped after each 2x stage.
        # (Upsample subclasses nn.Sequential, which cannot be slice-indexed, so we
        # iterate its children and capture the output after every PixelShuffle: for
        # a 4x factor that yields num_feat maps at out_size//2 and out_size.)
        upsample_stages = []
        y_stage = y
        for module in self.hat.upsample:
            y_stage = module(y_stage)
            if isinstance(module, nn.PixelShuffle):
                upsample_stages.append(y_stage)
        y_mid, y_top = upsample_stages[0], upsample_stages[1]
        g_mid = self.fusion_mid(torch.cat([self.hat_proj_mid(y_mid), e_b_mid], dim=1))
        # Option 2: inject E_B mid-res guidance (upsampled) into the top tap.
        e_b_mid_up = F.interpolate(e_b_mid, size=(self.out_size, self.out_size),
                                   mode='bilinear', align_corners=False)
        g_top = self.fusion_top(torch.cat([self.hat_proj_top(y_top), e_b_mid_up], dim=1))
        generator_features = [g_mid, g_top]

        # decoder (unchanged)
        hr = encoder_features[-1]
        hr_rgb = encoder2_features[-1]
        for i, block in enumerate(self.decoder):
            if i == 1:
                hr = torch.cat([hr_rgb, hr, generator_features[i - 1]], dim=1)
            elif i > 0:
                hr = torch.cat([hr, generator_features[i - 1]], dim=1)
            hr = block(hr)

        return hr
