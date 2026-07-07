from networks.model_test import common

import torch.nn as nn

url = {
    'r16f64': 'https://cv.snu.ac.kr/research/EDSR/models/mdsr_baseline-a00cab12.pt',
    'r80f64': 'https://cv.snu.ac.kr/research/EDSR/models/mdsr-4a78bedf.pt'
}

def make_model(args, parent=False):
    return MDSR(args)

class MDSR(nn.Module):
    def __init__(self, n_channels=4, n_resblocks=16, n_feats=64, conv=common.default_conv):
        super(MDSR, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        self.scale = [2]
        self.scale_idx = 0
        self.url = url['r{}f{}'.format(n_resblocks, n_feats)]

        m_head = [conv(n_channels, n_feats, kernel_size)]

        self.pre_process = nn.ModuleList([
            nn.Sequential(
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act)
            ) for _ in self.scale
        ])

        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.upsample = nn.ModuleList([
            common.Upsampler(conv, s, n_feats, act=False) for s in self.scale
        ])

        m_tail = [conv(n_feats, n_channels, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        x = self.pre_process[self.scale_idx](x)

        res = self.body(x)
        res += x

        x = self.upsample[self.scale_idx](res)
        x = self.tail(x)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

