import random
import torch, glob
import numpy as np
from torch.utils.data import Dataset
from datasets.interpolator_tools import interp23tap
from utils import random_crop_torch, random_flip, normalise_bandwise_RGB, NORM_DATA
from datasets.img_process_utils import deform_single


class TrainData(Dataset):
    def __init__(self, data_path, s):
        self.data_path = data_path
        self.venus_data = glob.glob(f'{data_path}/*_05m*8.pt')  # MuS2
        l2 = glob.glob(f'{data_path}/MuS2*.pt')
        self.venus_data = [x for x in self.venus_data if x not in l2]
        self.max_value = 2 ** s.nbits
        self.s = s
        self.ms_norms = NORM_DATA

    def __getitem__(self, index):
        venus_fl = self.venus_data[index]
        s2_fl = venus_fl.replace('05m', '10m')
        # print(s2_fl)
        venus = torch.load(venus_fl)
        if isinstance(venus, np.ndarray):
            venus = torch.from_numpy(venus.astype('int32'))
            
        num = venus.shape[0]
        chs = random.randint(0, num-1)
        # venus = venus[chs]/self.max_value
        
        data_type = random.choices(['sen2ven', 'sen', 'ven'], [0.7, 0.2, 0.1])[0]
        if data_type == 'sen2ven':
            sentinel = torch.load(s2_fl)
            if isinstance(sentinel, np.ndarray):
                sentinel = torch.from_numpy(sentinel.astype('int32'))
            sentinel = normalise_bandwise_RGB(sentinel[chs])
            # /self.max_value
            sentinel = torch.clip(sentinel, 0, 1)
            venus = normalise_bandwise_RGB(venus[chs])
            venus = torch.clip(venus, 0, 1)
        elif data_type == 'sen':
            venus = torch.load(s2_fl)
            if isinstance(venus, np.ndarray):
                venus = torch.from_numpy(venus.astype('int32'))
            venus = normalise_bandwise_RGB(venus[chs])
            # /self.max_value
            venus = torch.clip(venus, 0, 1)
            sentinel = deform_single(venus, 2)
        else:
            venus = normalise_bandwise_RGB(venus[chs])
            venus = torch.clip(venus, 0, 1)
            sentinel = deform_single(venus, 2)
            
        
        venus, sentinel = random_crop_torch(venus, sentinel, 128, 2)    
        venus_up = interp23tap(np.moveaxis(np.array(venus), 0, -1), self.s.ratio//2)
        venus_up = torch.from_numpy(np.moveaxis(venus_up, -1, 0))

        sentinel_up = interp23tap(np.moveaxis(np.array(sentinel), 0, -1), self.s.ratio//2)
        sentinel_up = torch.from_numpy(np.moveaxis(sentinel_up, -1, 0))

        # Random flip
        sentinel, venus, venus_up, sentinel_up = random_flip(sentinel, venus, venus_up, sentinel_up)
        
        return sentinel[:].float(), venus[:].float(), venus[:].float(), sentinel_up[:].float()

    def __len__(self):
        return len(self.venus_data)


class ValidData(Dataset):
    def __init__(self, data_path, s):
        self.data_path = data_path
        self.venus_data = glob.glob(f'{data_path}/*_05m*8.pt')
        self.max_value = 2 ** s.nbits
        self.s = s
        self.ms_norms = NORM_DATA
        # self.transform = transforms.Compose([transforms.RandomCrop(64)])

    def __getitem__(self, index):
        venus_fl = self.venus_data[index]
        s2_fl = venus_fl.replace('05m', '10m')
        venus = torch.load(venus_fl)
        sentinel = torch.load(s2_fl)
        num = venus.shape[0]
        chs = random.randint(0, num-1)

        sentinel = normalise_bandwise_RGB(sentinel[chs])
        sentinel = torch.clip(sentinel, 0, 1)
        venus = normalise_bandwise_RGB(venus[chs])
        venus = torch.clip(venus, 0, 1)

        venus, sentinel = random_crop_torch(venus, sentinel, 128, 2)

        venus_up = interp23tap(np.moveaxis(np.array(venus), 0, -1), self.s.ratio//2)
        venus_up = torch.from_numpy(np.moveaxis(venus_up, -1, 0))

        sentinel_up = interp23tap(np.moveaxis(np.array(sentinel), 0, -1), self.s.ratio//2)
        sentinel_up = torch.from_numpy(np.moveaxis(sentinel_up, -1, 0))

        # Random flip
        sentinel, venus, venus_up, sentinel_up = random_flip(sentinel, venus, venus_up, sentinel_up)
        
        return sentinel[:].float(), venus[:].float(), venus[:].float(), sentinel_up[:].float()

    def __len__(self):
        return len(self.venus_data)


class TestData(Dataset):
    def __init__(self, data_path, s):
        self.data_path = data_path
        self.venus_data = glob.glob(f'{data_path}/*_05m*8.pt')
        self.max_value = 2 ** s.nbits
        self.s = s
        self.ms_norms = NORM_DATA
        # self.transform = transforms.Compose([transforms.RandomCrop(64)])

    def __getitem__(self, index):
        venus_fl = self.venus_data[index]
        s2_fl = venus_fl.replace('05m', '10m')
        venus = torch.load(venus_fl)
        sentinel = torch.load(s2_fl)
        num = venus.shape[0]
        chs = random.randint(0, num-1)
        sentinel = normalise_bandwise_RGB(sentinel[chs])
        # /self.max_value
        sentinel = torch.clip(sentinel, 0, 1)
        venus = normalise_bandwise_RGB(venus[chs])
        # /self.max_value
        venus = torch.clip(venus, 0, 1)

        venus, sentinel = random_crop_torch(venus, sentinel, 128, 2)

        venus_up = interp23tap(np.moveaxis(np.array(venus), 0, -1), self.s.ratio//2)
        venus_up = torch.from_numpy(np.moveaxis(venus_up, -1, 0))

        sentinel_up = interp23tap(np.moveaxis(np.array(sentinel), 0, -1), self.s.ratio//2)
        sentinel_up = torch.from_numpy(np.moveaxis(sentinel_up, -1, 0))
        
        return sentinel[:].float(), sentinel_up[:].float()

    def __len__(self):
        return len(self.venus_data)


class TestData_TA(Dataset):
    def __init__(self, data_path, s):
        self.data_path = data_path
        self.venus_data = glob.glob(f'{data_path}/*_05m*8.pt')
        self.max_value = 2 ** s.nbits
        self.s = s
        self.ms_norms = NORM_DATA
        # self.transform = transforms.Compose([transforms.RandomCrop(64)])

    def __getitem__(self, index):
        venus_fl = self.venus_data[index]
        s2_fl = venus_fl.replace('05m', '10m')
        venus = torch.load(venus_fl)
        sentinel = torch.load(s2_fl)
        num = venus.shape[0]
        chs = random.randint(0, num-1)
        sentinel = normalise_bandwise_RGB(sentinel[chs])
        # /self.max_value
        sentinel = torch.clip(sentinel, 0, 1)
        venus = normalise_bandwise_RGB(venus[chs])
        # /self.max_value
        venus = torch.clip(venus, 0, 1)

        venus, sentinel = random_crop_torch(venus, sentinel, 128, 2)

        venus_up = interp23tap(np.moveaxis(np.array(venus), 0, -1), self.s.ratio//2)
        venus_up = torch.from_numpy(np.moveaxis(venus_up, -1, 0))

        sentinel_up = interp23tap(np.moveaxis(np.array(sentinel), 0, -1), self.s.ratio//2)
        sentinel_up = torch.from_numpy(np.moveaxis(sentinel_up, -1, 0))

        return sentinel[:].float(), venus[:].float(), sentinel_up[:].float()

    def __len__(self):
        return len(self.venus_data)
