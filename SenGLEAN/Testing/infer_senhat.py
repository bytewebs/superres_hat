#!/usr/bin/env python3
"""Run SenHAT inference on a real Sentinel-2 8-band GeoTIFF.

Input raster band order (10 m grid, 20 m bands already resampled to that grid):
    [B2, B3, B4, B8,  B4, B5, B6, B8a]

Example (Narval):
    cd /scratch/aayushk/superresolution/superres_hat/SenGLEAN/Training
    # run from repo root so imports resolve:
    cd /scratch/aayushk/superresolution/superres_hat
    python SenGLEAN/Testing/infer_senhat.py \\
        --checkpoint_g /scratch/aayushk/superresolution/weights/gens_7_389.pt \\
        --data_path /path/to/your_sentinel2_clip.tif \\
        --out_path /scratch/aayushk/superresolution/samples/senhat_389_enhanced.tif
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from SenGLEAN.Testing.inference_utils import generate_raster_senglean


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SenHAT raster inference')
    parser.add_argument('--checkpoint_g', type=str, required=True,
                        help='Path to gens_7_389.pt (or resume_7.pt)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='8-band Sentinel-2 GeoTIFF path')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Output GeoTIFF path (default: <input>_senhat_enhanced.tif)')
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint_g):
        raise FileNotFoundError(
            f'Checkpoint not found: {args.checkpoint_g}\n'
            'Did you mean /scratch/aayushk/superresolution/weights/gens_7_389.pt ?')
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(f'Input raster not found: {args.data_path}')

    out_fn = args.out_path
    if out_fn is None:
        out_fn = args.data_path[:-4] + '_senhat_enhanced.tif' if args.data_path.endswith('.tif') \
            else args.data_path + '_senhat_enhanced.tif'

    print(f'checkpoint : {args.checkpoint_g}')
    print(f'input      : {args.data_path}')
    print(f'output     : {out_fn}')

    # scale=2: output at 2x the input 10 m grid (= 5 m). Matches SenHAT 128→256 path.
    generate_raster_senglean(
        out_fn,
        args.checkpoint_g,
        raster_file=args.data_path,
        scale=2,
        mscale=2,
        pad=5,
        model='senhat',
    )
    print(f'Done. Wrote {out_fn}')


if __name__ == '__main__':
    main()
