from easydict import EasyDict as edict
import json

config = edict()
config.NETWORK = edict()
config.NETWORK.G_CHANNELS = 64
config.NETWORK.GROWTH_CHANNELS = 32
config.NETWORK.NUM_RRDB = 23
config.NETWORK.D_CHANNELS = 64

config.TRAIN = edict()
config.TRAIN.batch_size = 12 # [16] use 8 if your GPU memory is small
config.TRAIN.lr_init = 5e-5
config.TRAIN.lr_adv = 2e-5
config.TRAIN.beta1 = 0.9
config.TRAIN.use_usm = False

## initialize G
config.TRAIN.n_epoch_init = 500
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)
config.TRAIN.vgg_features = ['features.2', 'features.7', 'features.16', 'features.25', 'features.34']
config.TRAIN.vgg_weights = [0.1, 0.1, 0.5, 1, 1]

## train set location
config.TRAIN.hr_img_path = '../Geoeye/pan/'
config.TRAIN.lr_img_path = '../Geoeye/ms/'
config.TRAIN.unet_path = '../checkpoints/optical2pan/15_net_G.pth'
config.TRAIN.data_path = '../sentinel_dataset/'
config.TRAIN.usm_sharpening = True

config.TRAIN.DATA = edict()
config.TRAIN.DATA.blur_prob = 0.05
config.TRAIN.DATA.gaussian_noise_prob = 0.5
config.TRAIN.DATA.kernel_list = ['aniso', 'iso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
config.TRAIN.DATA.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]

config.VALID = edict()
## test set location
config.VALID.data_path = '../sentinel_test/'
config.VALID.lr_img_path = 'DIV2K/DIV2K_valid_LR_bicubic/X4/'
config.VALID.batch_size = 8

config.TEST = edict()
config.TEST.checkpoint_path = '/content/SRGAN/models/gen_2.pt'
config.TEST.batch_size = 16
config.TEST.hr_img_path = '/content/pan/'
config.TEST.lr_img_path = '/content/clips/'
config.TEST.hr_img_path = '../sentinel_test/'
# '/content/drive/MyDrive/ALSACE'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
