## ESRGAN_PA Evaluation

For evaluation phase you have to provide a raster which would be divided into patches of specified sizes and evaluation would be performed by degrading the 10m bands to 20m resolution and comparing the model outputs with the original ones.

### Usage

```python
python eval_esrgan_pa.py  --checkpoint_g ./weights/gens_esrgan_pa.pt --checkpoint_d ./weights/discs_rgbn.pt --data_path ../sentinel2_AOI5_clip.tif --model esrgan_pa --do_TA --nepoch 100 --patch_size 128 --evals
```

> **_NOTE:_**  For wandb logging enter the account username and authentication key in the section provided in ***train.py***

### Arguments

```python
usage: eval_esrgan_pa.py [-h] [--do_TA] [--evals]
                         [--checkpoint_g CHECKPOINT_G]
                         [--checkpoint_d CHECKPOINT_D] [--data_path DATA_PATH]
                         [--model MODEL] [--batch BATCH]
                         [--patch_size PATCH_SIZE] [--nepoch NEPOCH]

optional arguments:
  -h, --help            show this help message and exit
  --do_TA               Boolean use if wanting to do target adaptation 
  --checkpoint_g CHECKPOINT_G
                        Select weights for generator by providing path
  --checkpoint_d CHECKPOINT_D
                        Select weights for discriminator by providing path
  --data_path DATA_PATH
                        Select data raster path
  --batch BATCH         Batch size for target adaptation
  --patch_size PATCH_SIZE
                        Specify patch size to be used [Default: 64]
  --nepoch NEPOCH       Num of epoch for TA
```

[]
