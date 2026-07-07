## LightRefGLEAN Evaluation (20m Bands)

For evaluation phase you have to provide a raster which would be divided into patches of specified sizes and evaluation would be performed by degrading the 20m bands to 40m resolution and comparing the model outputs with the original ones.

### Usage

```python
python eval_light_glean.py --checkpoint_g ./weights/gens_light_glean.pt --checkpoint_d ./weights/discs_20m.pt --batch 4 --evals --do_TA --nepoch 150 --data_path ../sentinel2_AOI5_clip.tif
```

> **_NOTE:_**  For wandb logging enter the account username and authentication key in the section provided in ***train.py***

### Arguments

```python
usage: eval_light_glean.py [-h] [--do_TA] [--evals]
                           [--checkpoint_g CHECKPOINT_G]
                           [--checkpoint_d CHECKPOINT_D]
                           [--data_path DATA_PATH] [--batch BATCH]
                           [--nepoch NEPOCH] [--artif]

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
  --nepoch NEPOCH       Num of epoch for TA
  --artif               Use artifact loss or not
```

[]
