import torch, glob
import numpy as np
from osgeo import gdal
from torch.utils.data import Dataset
from utils import random_resize_torch
from skimage.transform import resize
from datasets.tiff_mat_conversion import tiff_to_mat_conversion
from datasets.interpolator_tools import interp23tap
from utils import normalise_bandwise_RGB


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
        temp_ms_cp = temp_ms.copy()
        # temp_ms_cp = random_resize_torch(temp_ms, resize_scales=[1, 2])
        
        ms_patch = torch.from_numpy(temp_ms_cp)       # To torch convert
        pan_patch = torch.from_numpy(temp['I_PAN']/self.max_value)
        pan_patch = torch.unsqueeze(pan_patch, dim=0)
        ms_patch_ups = interp23tap(np.moveaxis(temp_ms, 0, -1), self.s.ratio)
        # print(ms_patch_ups.shape, np.moveaxis(ms_patch_ups, -1, 0).shape)
        ms_patch_ups = torch.from_numpy(np.moveaxis(ms_patch_ups, -1, 0))
        
        return ms_patch[:].float(), pan_patch[:].float(), ms_patch_ups[:].float()

    def __len__(self):
        return len(self.train_pan_imgs)


def process_path(patch):
    b24 = np.array(patch[1:4])
    b8 = np.expand_dims(np.array(patch[7]), 0)
    b = np.concatenate((b24, b8), 0)
    return b


def process_path(patch):
    b24 = np.array(patch[:])
    # b8 = np.expand_dims(np.array(patch[7]), 0)
    b = b24 #np.concatenate((b24, b8), 0)
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
        ms_patch_ups = interp23tap(np.moveaxis(temp_ms, 0, -1), self.s.ratio//2)
        ms_patch_ups = torch.from_numpy(np.moveaxis(ms_patch_ups, -1, 0))
        return ms_patch[:].float(), ms_patch_ups[:].float()

    def __len__(self):
        return len(self.ms_files)
    

class TestData_TA(Dataset):
    def __init__(self, ms_path, s):
        self.ms_path = ms_path
        self.ms_files = glob.glob(self.ms_path + '/*')
        self.max_value = 2 ** s.nbits
        self.s = s
        self.scale = 4
        self.ms_norms = []
        self.ms_data = []

    def __getitem__(self, index):
        ms_path = self.ms_files[index]
        ms = gdal.Open(ms_path)
        temp_ms = ms.ReadAsArray()
        temp_ms = process_path(temp_ms)

        temp_ms = temp_ms/self.max_value   # Normalise
        _, H, W = temp_ms.shape
        ms_patch_ups = torch.from_numpy(temp_ms)       # To torch convert
        ms_patch = resize(np.moveaxis(temp_ms, 0, -1), (H//(self.scale/2), W//(self.scale/2)))
        ms_patch_bic = resize(ms_patch, (H, W), order=0)
        ms_patch = torch.from_numpy(np.moveaxis(ms_patch, -1, 0))
        ms_patch_bic = torch.from_numpy(np.moveaxis(ms_patch_bic, -1, 0))
        return ms_patch[:].float(), ms_patch_ups[:].float(), ms_patch_bic[:].float()

    def __len__(self):
        return len(self.ms_files)
    
    
class TestData_single(Dataset):
    def __init__(self, ms_path, s, patch_size=64):
        self.ms_path = ms_path
        self.s = s
        self.ms_norms = []
        self.ms_data = []
        self.patch_size = patch_size
        self.prefetch_data()

    def __getitem__(self, index):
        temp_ms = torch.tensor(self.ms_data[index, :4])
        # temp_ms = torch.from_numpy(temp_ms)
        temp_ms = normalise_bandwise_RGB(temp_ms, self.ms_norms)   # Normalise
        ms_patch = torch.clip(temp_ms, 0, 1)
        # ms_patch = torch.from_numpy(temp_ms)                       # To torch convert
        ms_patch_ups = interp23tap(np.moveaxis(np.array(ms_patch), 0, -1), self.s.ratio//2)
        ms_patch_ups = torch.from_numpy(np.moveaxis(ms_patch_ups, -1, 0))
        return ms_patch[:].float(), ms_patch_ups[:].float()
    
    def prefetch_data(self):
        ds = gdal.Open(self.ms_path)
        arr = ds.ReadAsArray()
        ds = None
        for i in range(7):
            i, j = np.percentile(arr[i], [2, 98])
            self.ms_norms.append([max(0, i-100), j+100])
        
        _, n, m = arr.shape
        for i in range((n//self.patch_size)+1):
            for j in range((m//self.patch_size)+1):
                strt_i = (i*self.patch_size)
                end_i = ((i+1)*self.patch_size)
                if i==(n//self.patch_size):
                    strt_i = (i*self.patch_size) - (end_i - n)
                    end_i = n
                    
                strt_j = (j*self.patch_size)
                end_j = ((j+1)*self.patch_size)
                if j==(m//self.patch_size):
                    strt_j = (j*self.patch_size) - (end_j - m)
                    end_j = m
                self.ms_data.append(arr[:, strt_i:end_i, strt_j:end_j])
        
        self.ms_data = np.array(self.ms_data)

    def __len__(self):
        return len(self.ms_data)


class TestData_TA_single(Dataset):
    def __init__(self, ms_path, s, patch_size=64):
        self.ms_path = ms_path
        self.s = s
        self.scale = 4
        self.ms_norms = []
        self.ms_data = []
        self.patch_size = patch_size
        self.prefetch_data()
        print(self.ms_norms)

    def __getitem__(self, index):
        # temp_ms = np.array(self.ms_data[index, :4])
        # temp_ms = torch.from_numpy(temp_ms)
        temp_ms = torch.tensor(self.ms_data[index, :4])
        temp_ms = normalise_bandwise_RGB(temp_ms, self.ms_norms)   # Normalise
        _, H, W = temp_ms.shape
        ms_patch_ups = torch.clip(temp_ms, 0, 1)
        temp_ms = np.array(temp_ms)
        # ms_patch_ups = torch.from_numpy(temp_ms)       # To torch convert
        ms_patch = resize(np.moveaxis(temp_ms, 0, -1), (H//(self.scale/2), W//(self.scale/2)))
        ms_patch_bic = resize(ms_patch, (H, W), order=2)
        ms_patch = torch.from_numpy(np.moveaxis(ms_patch, -1, 0))
        ms_patch_bic = torch.from_numpy(np.moveaxis(ms_patch_bic, -1, 0))
        return ms_patch[:].float(), ms_patch_ups[:].float(), ms_patch_bic[:].float()
    
    def prefetch_data(self):
        ds = gdal.Open(self.ms_path)
        arr = ds.ReadAsArray()
        ds = None
        for i in range(7):
            i, j = np.percentile(arr[i], [2, 98])
            self.ms_norms.append([max(0, i-100), j+100])
        
        _, n, m = arr.shape
        for i in range((n//self.patch_size)+1):
            for j in range((m//self.patch_size)+1):
                strt_i = (i*self.patch_size)
                end_i = ((i+1)*self.patch_size)
                if i==(n//self.patch_size):
                    strt_i = (i*self.patch_size) - (end_i - n)
                    end_i = n
                    
                strt_j = (j*self.patch_size)
                end_j = ((j+1)*self.patch_size)
                if j==(m//self.patch_size):
                    strt_j = (j*self.patch_size) - (end_j - m)
                    end_j = m
                self.ms_data.append(arr[:, strt_i:end_i, strt_j:end_j])
        
        self.ms_data = np.array(self.ms_data)

    def __len__(self):
        return len(self.ms_data)
