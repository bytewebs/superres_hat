import random, os
import torch, glob
import numpy as np
from torch.utils.data import Dataset
from datasets.interpolator_tools import interp23tap
from utils import random_crop_torch, random_flip2, random_flip
from datasets.img_process_utils import deform_single


class TrainData(Dataset):
    def __init__(self, data_path, s):
        self.data_path = data_path
        self.venus_data = glob.glob(f'{data_path}/*05m*8a.pt')  # MuS2
        self.max_value = 2 ** s.nbits
        self.s = s

    def __getitem__(self, index):
        venus_fl = self.venus_data[index]
        ven_rgb_fl = venus_fl.replace('05m_b4b5b6b8a.pt', '05m_b2b3b4b8.pt')
        venus = torch.load(venus_fl)
        if isinstance(venus, np.ndarray):
            venus = torch.from_numpy(venus.astype('int32'))
            
        num = venus.shape[0]
        chs = random.randint(0, num-1)
        venus = venus[chs]/self.max_value
        venus = torch.clip(venus, 0, 1)
        
        venus_rgb = torch.load(ven_rgb_fl)[:,:3]
        venus_rgb = venus_rgb[chs]/self.max_value
        venus_rgb = torch.clip(venus_rgb, 0, 1)
        
        # venus, venus_rgb = random_crop_torch(venus, venus_rgb, 128, 1)
        
        # data_type = random.choices(['sen', 'ven'], [0.2, 0.8])[0]
        # if data_type == 'sen':
        #     sentinel = torch.load(s2_fl)
        #     sentinel_rgb = torch.load(s2_rgb_fl)
        #     # print(sentinel_rgb.shape, s2_rgb_fl)
        #     if isinstance(sentinel, np.ndarray):
        #         sentinel = torch.from_numpy(sentinel.astype('int32'))
        #     sentinel = sentinel[chs]/self.max_value
        #     sentinel = torch.clip(sentinel, 0, 1)     # ()
        #     sentinel_rgb = sentinel_rgb[chs]/self.max_value
        #     sentinel_rgb = torch.clip(sentinel_rgb, 0, 1)
        #     # print(sentinel.shape, venus.shape, sentinel_rgb.shape)
            
        # elif data_type == 'ven':
        #     sentinel_rgb = torch.load(ven_rgb_fl)[:,:3]
        #     sentinel_rgb = sentinel_rgb[chs]/self.max_value
        #     sentinel_rgb = torch.clip(sentinel_rgb, 0, 1)
            
        #     sentinel = deform_single(venus, 4)
        #     # sentinel_rgb = deform_single(venus_rgb, 2)
            # print(sentinel.shape, venus.shape, sentinel_rgb.shape, venus_rgb.shape)
                
        # Random flip
        venus, venus_rgb = random_flip2(venus, venus_rgb)
        
        return venus.float(), venus_rgb.float()
    
    def __len__(self):
        return len(self.venus_data)


class ValidData(Dataset):
    def __init__(self, data_path, s):
        self.data_path = data_path
        self.venus_data = glob.glob(f'{data_path}/*05m*8a.pt')  # MuS2
        self.max_value = 2 ** s.nbits
        self.s = s

    def __getitem__(self, index):
        venus_fl = self.venus_data[index]
        ven_rgb_fl = venus_fl.replace('05m_b4b5b6b8a.pt', '05m_b2b3b4b8.pt')
        venus = torch.load(venus_fl)
        if isinstance(venus, np.ndarray):
            venus = torch.from_numpy(venus.astype('int32'))
            
        num = venus.shape[0]
        chs = random.randint(0, num-1)
        venus = venus[chs]/self.max_value
        venus = torch.clip(venus, 0, 1)
        
        venus_rgb = torch.load(ven_rgb_fl)[:,:3]
        venus_rgb = venus_rgb[chs]/self.max_value
        venus_rgb = torch.clip(venus_rgb, 0, 1)
        
        # Random flip
        venus, venus_rgb = random_flip2(venus, venus_rgb)
        
        return venus.float(), venus_rgb.float()
    
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
        venus = venus[chs]/self.max_value
        venus = torch.clip(venus, 0, 1)
        sentinel = sentinel[chs]/self.max_value
        sentinel = torch.clip(sentinel, 0, 1)

        venus, sentinel = random_crop_torch(venus, sentinel, 128, 2)

        venus_up = interp23tap(np.moveaxis(np.array(venus), 0, -1), self.s.ratio//2)
        venus_up = torch.from_numpy(np.moveaxis(venus_up, -1, 0))

        sentinel_up = interp23tap(np.moveaxis(np.array(sentinel), 0, -1), self.s.ratio)
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
        venus = venus[chs]/self.max_value
        venus = torch.clip(venus, 0, 1)
        sentinel = sentinel[chs]/self.max_value
        sentinel = torch.clip(sentinel, 0, 1)

        venus, sentinel = random_crop_torch(venus, sentinel, 128, 2)

        venus_up = interp23tap(np.moveaxis(np.array(venus), 0, -1), self.s.ratio//2)
        venus_up = torch.from_numpy(np.moveaxis(venus_up, -1, 0))

        sentinel_up = interp23tap(np.moveaxis(np.array(sentinel), 0, -1), self.s.ratio)
        sentinel_up = torch.from_numpy(np.moveaxis(sentinel_up, -1, 0))

        return sentinel[:3].float(), venus_up[:3].float(), sentinel_up[:3].float()

    def __len__(self):
        return len(self.venus_data)
