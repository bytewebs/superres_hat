import os
import sys
from pathlib import Path
print(str(Path('.').absolute()))
sys.path.insert(0, str(Path('.').absolute()))
sys.path.append(str(Path('.').absolute().parent.parent))

import subprocess, sys
from inference_utils import generate_raster_senglean

def execute(cmd):
    subprocess.check_call(cmd, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_TA', dest='target_adapt', action='store_true')
    parser.add_argument('--checkpoint_g', type=str, default=None, help='Select generator weights by providing path')
    parser.add_argument('--checkpoint_d', type=str, default=None, help='Select discriminator weights by providing path')
    parser.add_argument('--data_path', type=str, default=None, help='Select data clips path')
    parser.add_argument('--batch', type=int, default=4, help='path to checkpoint')
    parser.add_argument('--nepoch', type=int, default=150, help='Num of epoch for TA')
    parser.set_defaults(feature=False)

    args = parser.parse_args()

    if args.target_adapt:
        cmd = f'python eval_light_glean_all.py --checkpoint_g {args.checkpoint_g} --checkpoint_d {args.checkpoint_d} --do_TA --batch {args.batch} --nepoch {args.nepoch} --data_path {args.data_path}'
    else:
        cmd = f'python eval_light_glean_all.py --checkpoint_g {args.checkpoint_g} --checkpoint_d {args.checkpoint_d} --batch {args.batch} --nepoch {args.nepoch} --data_path {args.data_path}'

    execute(cmd)

    if args.target_adapt:
        chpt = f'{args.checkpoint_g[:-3]}_TA.pt'
    else:
        chpt = f'{args.checkpoint_g}'
    out_fn = f'{args.data_path[:-4]}_lightsenglean_enhanced.tif'
    generate_raster_senglean(out_fn, chpt, raster_file=args.data_path, scale=2, mscale=2, pad=5, model='light_senglean')

