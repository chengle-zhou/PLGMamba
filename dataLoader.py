import random
import math
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import cv2
import os
import scipy.io as sio
import h5py

# from utils import ssrImgMerge

"""
此脚本用于图像（e.g., HSI, RGB, etc.）空间超分辨数据对（HR-LR）的构建，支持x2, x4, x8等倍率超分
Chengle Zhou, Sun Yat-sen University, 2024-07-06, chengle_zhou@foxmail.com
"""


class TrainHSRDataset(Dataset):
    """将HSI进行有重叠的分割，形成图像块，并从这些图像块随机固定选择若干个图像块进行训练"""
    def __init__(self, dataroot, cropsize=256, testarea=(1511, 1767, 'row'), overlapixel=128, rdm_sel_num=1024, upscale=4, arg=True):
        super(TrainHSRDataset, self).__init__()
        self.dataroot = dataroot
        self.cropsize = cropsize
        self.sampling = cropsize - overlapixel
        self.rdm_sel_num = rdm_sel_num
        self.upscale = upscale
        self.arg = arg
        with h5py.File(dataroot, 'r') as file:
            data = file['data'][:].transpose(2, 1, 0).astype(np.float32)
        if testarea[-1] == 'row':
            self.trdata = np.delete(data, np.arange(testarea[0], testarea[1]), axis=0)
        else:
            self.trdata = np.delete(data, np.arange(testarea[0], testarea[1]), axis=1)
        [h_tr_bef, w_tr_bef, self.c] = self.trdata.shape
        print('Training image size: {} x {} x {}'.format(h_tr_bef, w_tr_bef, self.c))
        h_tr_rem, w_tr_rem = (h_tr_bef - cropsize) % self.sampling, (w_tr_bef - cropsize) % self.sampling
        h_tr_pad = self.sampling - h_tr_rem if h_tr_rem != 0 else h_tr_rem
        w_tr_pad = self.sampling - w_tr_rem if w_tr_rem != 0 else w_tr_rem
        self.trdata = np.pad(self.trdata, ((0, h_tr_pad), (0, w_tr_pad), (0, 0)), 'reflect')
        [h_tr, w_tr, _] = self.trdata.shape
        self.n_h = int(((h_tr - cropsize) - 0) / self.sampling + 1)
        self.n_w = int(((w_tr - cropsize) - 0) / self.sampling + 1)
        self.block_pos = np.arange(0, self.n_h*self.n_w).reshape(self.n_h, self.n_w)
        self.random_integers = np.random.randint(0, self.n_h * self.n_w, size=self.rdm_sel_num)
        print('Block number: {} = {}(row) x {}(col)'.format(len(self.block_pos.reshape(-1)), self.n_h, self.n_w))
        print('Selection number: {} | size: {} x {}'.format(self.rdm_sel_num, cropsize, cropsize))

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        item = self.random_integers[idx]
        rows = item // self.n_w
        cols = item - rows * self.n_w
        x_r = int(0 + (rows + 1 - 1) * self.sampling)
        y_c = int(0 + (cols + 1 - 1) * self.sampling)
        hr_img = self.trdata[x_r:x_r+self.cropsize, y_c:y_c+self.cropsize, :]
        lr_img = cv2.resize(hr_img, (0, 0), fx=1./self.upscale, fy=1./self.upscale, interpolation=cv2.INTER_CUBIC)
        hr_img = hr_img.transpose(2, 1, 0)
        lr_img = lr_img.transpose(2, 1, 0)
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            hr_img = self.arguement(hr_img, rotTimes, vFlip, hFlip)
            lr_img = self.arguement(lr_img, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(lr_img), np.ascontiguousarray(hr_img)

    def __len__(self):
        return self.rdm_sel_num


class TestHSRDataset(Dataset):
    def __init__(self, dataroot, cropsize=256, testarea=(1511, 1767, 'row'), overlapixel=24, upscale=4, arg=True):
        super(TestHSRDataset, self).__init__()
        self.dataroot = dataroot
        self.cropsize = cropsize
        self.overlapixel = overlapixel
        self.sampling = cropsize - overlapixel
        self.upscale = upscale
        self.arg = arg
        with h5py.File(dataroot, 'r') as file:
            data = file['data'][:].transpose(2, 1, 0).astype(np.float32)  # HSI
        if testarea[-1] == 'row':
            self.tedata = data[testarea[0]:testarea[1], :, :]
        else:
            self.tedata = data[:, testarea[0]:testarea[1], :]
        [self.h_te_bef, self.w_te_bef, self.c] = self.tedata.shape
        print('Testing image size: {} x {} x {}'.format(self.h_te_bef, self.w_te_bef, self.c))
        h_te_rem, w_te_rem = (self.h_te_bef - cropsize) % self.sampling, (self.w_te_bef - cropsize) % self.sampling
        self.h_te_pad = self.sampling - h_te_rem if h_te_rem != 0 else h_te_rem
        self.w_te_pad = self.sampling - w_te_rem if w_te_rem != 0 else w_te_rem
        self.tedata = np.pad(self.tedata, ((0, self.h_te_pad), (0, self.w_te_pad), (0, 0)), 'reflect')
        [self.h_te, self.w_te, _] = self.tedata.shape
        self.n_h = int(((self.h_te - cropsize) - 0) / self.sampling + 1)
        self.n_w = int(((self.w_te - cropsize) - 0) / self.sampling + 1)
        self.block_pos = np.arange(0, self.n_h*self.n_w).reshape(self.n_h, self.n_w)
        print('Block number: {} = {}(row) x {}(col) | size: {} x {}'.format(len(self.block_pos.reshape(-1)),
                                                                            self.n_h, self.n_w, cropsize, cropsize))


    def __getitem__(self, item):
        rows = item // self.n_w  # index (1,2,3,...,x*y-1) to coordinate (x,y)
        cols = item - rows * self.n_w
        x_r = int(0 + (rows + 1 - 1) * self.sampling)
        y_c = int(0 + (cols + 1 - 1) * self.sampling)
        hr_img = self.tedata[x_r:x_r+self.cropsize, y_c:y_c+self.cropsize, :]
        lr_img = cv2.resize(hr_img, (0, 0), fx=1./self.upscale, fy=1./self.upscale, interpolation=cv2.INTER_CUBIC)
        hr_img = hr_img.transpose(2, 1, 0)
        lr_img = lr_img.transpose(2, 1, 0)
        return np.ascontiguousarray(lr_img), np.ascontiguousarray(hr_img)

    def __len__(self):
        return self.n_h*self.n_w
