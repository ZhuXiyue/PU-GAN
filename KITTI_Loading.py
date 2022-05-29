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
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]    # get xyz
        np.save('raw',points)
        if self.return_remission:
            ori_points, real, intensity = point_cloud_to_range_image(filename, False, self.return_remission)
        else:
            real = point_cloud_to_range_image(filename, False, self.return_remission)
        # print(np.shape(ori_points))
        # print(np.shape(real))
        #Make negatives 0
        # real = np.where(real<0, 0, real) + 0.0001
        #Apply log
        # real = ((np.log2(real+1)) / 6)
        #Make negatives 0
        # real = np.clip(real, 0, 1)
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
        pts_im = ori_points#range_image_to_point_cloud_image(range_im) # 64*1024*3
        np.save("sample.npy",pts_im.reshape((64*1024,3)))
        np.save("sample_ori.npy",range_image_to_point_cloud(range_im))

        # split the point images and add features
        features = np.zeros((64,16,64,3)) + 0.042
        pts_im = pts_im.reshape((64,16,64,3))
        pts_im = np.concatenate((pts_im,features),axis = 3) # (64,16,64,6)
        ## sampling
        input_data = pts_im[0:64:4,:,:,:] #(16,16,64,6)
        pts_im = pts_im.transpose((1,0,2,3))# 16 64, 64,6
        input_data = input_data.transpose((1,0,2,3))# (16,16,64,6)
        
        pts = pts_im.reshape(16,4096,6)
        input_data = input_data.reshape(16,1024,6)
        
        
        # normalize
        data_npoint = pts.shape[1]

        centroid = np.mean(pts[..., :3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((pts[..., :3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        # radius = furthest_distance[:, 0] # not very sure?

        radius = np.ones((16))
        pts[..., :3] -= centroid
        pts[..., :3] /= np.expand_dims(furthest_distance, axis=-1)
        input_data[..., :3] -= centroid
        input_data[..., :3] /= np.expand_dims(furthest_distance, axis=-1)

        # pts_ims = 
        return input_data,pts, radius # , 0

if __name__ == "__main__":
    dataloader = KITTI()
    eval_loader = DataLoader(dataloader, batch_size=8, shuffle=False, pin_memory=True, num_workers=1)

    for itr, batch in enumerate(eval_loader):
        print(np.shape(batch[0]))
        print(np.shape(batch[1]))
        print(np.shape(batch[2]))

    
