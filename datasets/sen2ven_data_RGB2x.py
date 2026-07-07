import random
import torch, glob
import numpy as np
from torch.utils.data import Dataset
from datasets.interpolator_tools import interp23tap
from utils import random_crop_torch, random_flip, normalise_quantile2, normalise_quantile
from scipy.ndimage.filters import gaussian_filter
from datasets.img_process_utils import deform_single


class TrainData(Dataset):
    def __init__(self, data_path, s):
        self.data_path = data_path
        self.venus_data = glob.glob(f'{data_path}/*05m*8.pt')  # MuS2
        self.max_value = 2 ** s.nbits
        self.s = s

    def __getitem__(self, index):
        venus_fl = self.venus_data[index]
        s2_fl = venus_fl.replace('05m', '10m')
        venus = torch.load(venus_fl)
        if isinstance(venus, np.ndarray):
            venus = torch.from_numpy(venus.astype('int32'))
            
        num = venus.shape[0]
        chs = random.randint(0, num-1)
        
        data_type = random.choices(['sen2ven', 'sen', 'ven'], [0.7, 0.2, 0.1])[0]
        if data_type == 'sen2ven':
            sentinel = torch.load(s2_fl)
            if isinstance(sentinel, np.ndarray):
                sentinel = torch.from_numpy(sentinel.astype('int32'))
            sentinel, Q = normalise_quantile2(sentinel, chs)
            sentinel = torch.clip(sentinel, 0, 1)
            venus, _ = normalise_quantile(venus[chs], Q)
            venus = torch.clip(venus, 0, 1)
        elif data_type == 'sen':
            venus = torch.load(s2_fl)
            if isinstance(venus, np.ndarray):
                venus = torch.from_numpy(venus.astype('int32'))
            venus, Q = normalise_quantile2(venus, chs)
            venus = torch.clip(venus, 0, 1)
            sentinel = deform_single(venus, 2)
        else:
            venus, Q = normalise_quantile2(venus, chs)
            venus = torch.clip(venus, 0, 1)
            sentinel = deform_single(venus, 2)
        
        venus, sentinel = random_crop_torch(venus, sentinel, 128, 2)    
        venus_up = interp23tap(np.moveaxis(np.array(venus), 0, -1), self.s.ratio//2)
        venus_up = torch.from_numpy(np.moveaxis(venus_up, -1, 0))

        sentinel_up = interp23tap(np.moveaxis(np.array(sentinel), 0, -1), self.s.ratio//2)
        sentinel_up = torch.from_numpy(np.moveaxis(sentinel_up, -1, 0))

        # Random flip
        sentinel, venus, venus_up, sentinel_up = random_flip(sentinel, venus, venus_up, sentinel_up)
        
        return sentinel[:3].float(), venus[:3].float(), venus[:3].float(), sentinel_up[:3].float()

    def __len__(self):
        return len(self.venus_data)


class ValidData(Dataset):
    def __init__(self, data_path, s):
        self.data_path = data_path
        self.venus_data = glob.glob(f'{data_path}/*05m*8.pt')
        self.max_value = 2 ** s.nbits
        self.s = s
        # self.transform = transforms.Compose([transforms.RandomCrop(64)])

    def __getitem__(self, index):
        venus_fl = self.venus_data[index]
        s2_fl = venus_fl.replace('05m', '10m')
        venus = torch.load(venus_fl)
        sentinel = torch.load(s2_fl)
        num = venus.shape[0]
        chs = random.randint(0, num-1)

        sentinel, Q = normalise_quantile(sentinel[chs])
        sentinel = torch.clip(sentinel, 0, 1)
        venus, _ = normalise_quantile(venus[chs], Q)
        venus = torch.clip(venus, 0, 1)

        venus, sentinel = random_crop_torch(venus, sentinel, 128, 2)

        venus_up = interp23tap(np.moveaxis(np.array(venus), 0, -1), self.s.ratio//2)
        venus_up = torch.from_numpy(np.moveaxis(venus_up, -1, 0))

        sentinel_up = interp23tap(np.moveaxis(np.array(sentinel), 0, -1), self.s.ratio//2)
        sentinel_up = torch.from_numpy(np.moveaxis(sentinel_up, -1, 0))

        # Random flip
        sentinel, venus, venus_up, sentinel_up = random_flip(sentinel, venus, venus_up, sentinel_up)
        
        return sentinel[:3].float(), venus[:3].float(), venus[:3].float(), sentinel_up[:3].float()

    def __len__(self):
        return len(self.venus_data)


class TestData(Dataset):
    def __init__(self, data_path, s):
        self.data_path = data_path
        self.venus_data = glob.glob(f'{data_path}/*05m*8.pt')
        self.max_value = 2 ** s.nbits
        self.s = s
        # self.transform = transforms.Compose([transforms.RandomCrop(64)])

    def __getitem__(self, index):
        venus_fl = self.venus_data[index]
        s2_fl = venus_fl.replace('05m', '10m')
        venus = torch.load(venus_fl)
        sentinel = torch.load(s2_fl)
        num = venus.shape[0]
        chs = random.randint(0, num-1)
        sentinel, Q = normalise_quantile(sentinel[chs])
        # /self.max_value
        sentinel = torch.clip(sentinel, 0, 1)
        venus, _ = normalise_quantile(venus[chs], Q)
        # /self.max_value
        venus = torch.clip(venus, 0, 1)

        venus, sentinel = random_crop_torch(venus, sentinel, 128, 2)

        venus_up = interp23tap(np.moveaxis(np.array(venus), 0, -1), self.s.ratio//2)
        venus_up = torch.from_numpy(np.moveaxis(venus_up, -1, 0))

        sentinel_up = interp23tap(np.moveaxis(np.array(sentinel), 0, -1), self.s.ratio//2)
        sentinel_up = torch.from_numpy(np.moveaxis(sentinel_up, -1, 0))
        
        return sentinel[:3].float(), sentinel_up[:3].float()

    def __len__(self):
        return len(self.venus_data)


class TestData_TA(Dataset):
    def __init__(self, data_path, s):
        self.data_path = data_path
        self.venus_data = glob.glob(f'{data_path}/*05m*8.pt')
        self.max_value = 2 ** s.nbits
        self.s = s
        # self.transform = transforms.Compose([transforms.RandomCrop(64)])

    def __getitem__(self, index):
        venus_fl = self.venus_data[index]
        s2_fl = venus_fl.replace('05m', '10m')
        venus = torch.load(venus_fl)
        sentinel = torch.load(s2_fl)
        num = venus.shape[0]
        chs = random.randint(0, num-1)
        sentinel, Q = normalise_quantile(sentinel[chs])
        # /self.max_value
        sentinel = torch.clip(sentinel, 0, 1)
        venus, _ = normalise_quantile(venus[chs], Q)
        # /self.max_value
        venus = torch.clip(venus, 0, 1)

        venus, sentinel = random_crop_torch(venus, sentinel, 128, 2)

        venus_up = interp23tap(np.moveaxis(np.array(venus), 0, -1), self.s.ratio//2)
        venus_up = torch.from_numpy(np.moveaxis(venus_up, -1, 0))

        sentinel_up = interp23tap(np.moveaxis(np.array(sentinel), 0, -1), self.s.ratio//2)
        sentinel_up = torch.from_numpy(np.moveaxis(sentinel_up, -1, 0))

        return sentinel[:3].float(), venus[:3].float(), sentinel_up[:3].float()

    def __len__(self):
        return len(self.venus_data)
