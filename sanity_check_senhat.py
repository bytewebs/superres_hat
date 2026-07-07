"""Minimal single forward/backward sanity check for the SenHAT generator.

Intended to be run on an HPC login/compute node BEFORE submitting a full
training job, to confirm the model builds and that gradients flow end-to-end
with the configured `hat_variant`. It uses synthetic tensors only -- no dataset,
no network, and no iterative training.

Usage:
    python sanity_check_senhat.py                # uses config.NETWORK.HAT_VARIANT
    python sanity_check_senhat.py --variant base --batch 2
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn

from config import config
from networks.senhat import SenHAT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default=config.NETWORK.HAT_VARIANT,
                    choices=['tiny', 'small', 'base'])
    ap.add_argument('--rrdbs', type=int, default=config.NETWORK.NUM_RRDB)
    ap.add_argument('--batch', type=int, default=2)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  variant={args.variant}  rrdbs={args.rrdbs}  batch={args.batch}")

    torch.manual_seed(0)
    netG = SenHAT(64, 256, 4, 64, num_rrdbs=args.rrdbs, rgb_channels=4,
                  hat_variant=args.variant).to(device)
    print(f"SenHAT params: {sum(p.numel() for p in netG.parameters())/1e6:.2f}M")

    b = args.batch
    ms = torch.rand(b, 4, 64, 64, device=device)          # E_A input (20 m set)
    ms_rgb_up = torch.rand(b, 4, 128, 128, device=device)  # E_B input (10 m set)
    target = torch.rand(b, 8, 256, 256, device=device)     # 8-channel 5 m target

    # forward
    netG.train()
    out = netG(ms, ms_rgb_up)
    assert out.shape == (b, 8, 256, 256), f"unexpected output shape {tuple(out.shape)}"
    print(f"forward OK: output {tuple(out.shape)}")

    # backward (plain L1 -- this is only a gradient-flow check, not the training loss)
    loss = nn.L1Loss()(out, target)
    loss.backward()
    assert torch.isfinite(loss), "non-finite loss"
    n_with_grad = sum(1 for p in netG.parameters()
                      if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all())
    n_train = sum(1 for p in netG.parameters() if p.requires_grad)
    gnorm = sum(p.grad.norm().item() for p in netG.parameters() if p.grad is not None)
    print(f"backward OK: loss={loss.item():.5f}  finite-grad tensors={n_with_grad}/{n_train}  "
          f"grad-norm sum={gnorm:.2f}")
    print("SANITY CHECK PASSED")


if __name__ == '__main__':
    main()
