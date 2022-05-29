# from io import BytesIO
# from PIL import Image
from torch.utils.data import Dataset
import numpy as np
# import torch
# import os
import glob as glb
from lidar_utils import *
from torch.utils.data import DataLoader

class KITTI(Dataset):

    def __init__(self, path = '', split = 'train', resolution=None, transform=None):
        self.transform = transform
        self.return_remission = True # (config.data.channels == 2)
        self.random_roll = True #config.data.random_roll
        self.full_list = glb.glob('/root/PU-NET/datas/Lidar/*.bin')
        # if split == "train":
        #     self.full_list = list(filter(lambda file: '0000_sync' not in file and '0001_sync' not in file, full_list))
        # else:
        #     self.full_list = list(filter(lambda file: '0000_sync' in file or '0001_sync' in file, full_list))
        self.length = len(self.full_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        filename = self.full_list[idx]
        if self.return_remission:
            real, intensity = point_cloud_to_range_image(filename, False, self.return_remission)
        else:
            real = point_cloud_to_range_image(filename, False, self.return_remission)
        #Make negatives 0
        real = np.where(real<0, 0, real) + 0.0001
        #Apply log
        real = ((np.log2(real+1)) / 6)
        #Make negatives 0
        real = np.clip(real, 0, 1)
        random_roll = np.random.randint(1024)

        if self.random_roll:
            real = np.roll(real, random_roll, axis = 1)
        real = np.expand_dims(real, axis = 0)

        if self.return_remission:
            intensity = np.clip(intensity, 0, 1.0)
            if self.random_roll:
                intensity = np.roll(intensity, random_roll, axis = 1)
            intensity = np.expand_dims(intensity, axis = 0)
            real = np.concatenate((real, intensity), axis = 0)

        # get point cloud img
        range_im = real[0] # 64*1024
        pts_im = range_image_to_point_cloud_image(range_im) # 64*1024*3
        features = np.zeros((64,16,64,3)) + 0.042
        pts_im = pts_im.reshape((64,16,64,3))
        pts_im = np.concatenate((pts_im,features),axis = 3)
        # split the point images 
        # pts_ims = 
        return pts_im# , 0

if __name__ == "__main__":
    dataloader = KITTI()
    eval_loader = DataLoader(dataloader, batch_size=8, shuffle=False, pin_memory=True, num_workers=1)

    for itr, batch in enumerate(eval_loader):
        print(np.shape(batch))

    
