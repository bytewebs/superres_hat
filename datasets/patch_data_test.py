import torch, glob
import numpy as np
from osgeo import gdal
from torch.utils.data import Dataset
from utils import random_resize_torch
from datasets.tiff_mat_conversion import tiff_to_mat_conversion
from datasets.interpolator_tools import interp23tap


class TrainData(Dataset):
    def __init__(self, pan_path, ms_path, s):
        self.pan_path = pan_path
        self.ms_path = ms_path
        self.train_pan_imgs = glob.glob(pan_path + '*')
        self.max_value = 2 ** s.nbits
        self.s = s

    def __getitem__(self, index):
        pan_path = self.train_pan_imgs[index]
        fl = pan_path.split('/')[-1]
        n = len(fl.split('_'))
        ms_path = glob.glob(self.ms_path + '*' + fl.split('_', n-2)[-1])[0]
        temp = tiff_to_mat_conversion(ms_path, pan_path)
        temp_ms = temp['I_MS_LR']/self.max_value   # Normalise
        # Testing
        temp_ms_cp = random_resize_torch(temp_ms, resize_scales=[1, 2])
        
        ms_patch = torch.from_numpy(temp_ms_cp)       # To torch convert
        pan_patch = torch.from_numpy(temp['I_PAN']/self.max_value)
        ms_patch_ups = interp23tap(np.moveaxis(temp_ms, 0, -1), self.s.ratio)
        # print(ms_patch_ups.shape, np.moveaxis(ms_patch_ups, -1, 0).shape)
        ms_patch_ups = torch.from_numpy(np.moveaxis(ms_patch_ups, -1, 0))
        return ms_patch.float(), pan_patch.float(), ms_patch_ups.float()

    def __len__(self):
        return len(self.train_pan_imgs)


def process_path(patch):
    b24 = np.array(patch[1:4])
    b8 = np.expand_dims(np.array(patch[7]), 0)
    b = np.concatenate((b24, b8), 0)
    return b



class TestData(Dataset):
    def __init__(self, ms_path, s):
        self.ms_path = ms_path
        self.ms_files = glob.glob(self.ms_path + '/*')
        self.max_value = 2 ** s.nbits
        self.s = s

    def __getitem__(self, index):
        ms_path = self.ms_files[index]
        ms = gdal.Open(ms_path)
        temp_ms = ms.ReadAsArray()
        temp_ms = process_path(temp_ms)

        temp_ms = temp_ms/self.max_value   # Normalise
        ms_patch = torch.from_numpy(temp_ms)       # To torch convert
        ms_patch_ups = interp23tap(np.moveaxis(temp_ms, 0, -1), self.s.ratio)
        ms_patch_ups = torch.from_numpy(np.moveaxis(ms_patch_ups, -1, 0))
        return ms_patch.float(), ms_patch_ups.float()

    def __len__(self):
        return len(self.ms_files)
    