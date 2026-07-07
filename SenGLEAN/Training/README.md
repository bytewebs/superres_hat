## SenGLEAN Training

We are using [Sen2Venus](https://doi.org/10.5281/zenodo.6514159), which needs to be extracted and all the TIF files must be stored in two seperate directory, one for training and another for testing

### Usage

```python
python train.py --name SenGLEAN_training --resume --checkpoint ./weights/gens_glean.pt --disc_spec  --do_usm --batch 4 --logging
```

> **_NOTE:_**  For wandb logging enter the account username and authentication key in the section provided in ***train.py***

### Arguments

```python
usage: train.py [-h] [--logging] [--disc_spec] [--do_usm] [--name NAME]
                [--resume] [--checkpoint CHECKPOINT] [--batch BATCH]

optional arguments:
  -h, --help            show this help message and exit
  --logging             do wandb logging or not
  --disc_spec           do disciminator assited spectral loss or not
  --do_usm              perform USM sharpening based loss or not
  --name NAME           w&b runtime name in UI
  --resume              load pretrained weights or not
  --checkpoint CHECKPOINT
                        path to generator checkpoint
  --batch BATCH         batch-size for training 
```
