# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 01:25:39 2023

@author: gianl
"""

from ATMANpy.types.Frames import BGRFrame
from ATMANpy.types.Streams import NCStream

import numpy as np
import torch
import torchvision.transforms.v2 as tf
from torchvision.io import read_video
from torch.utils.data import Dataset, IterableDataset, get_worker_info

class BallData(Dataset):
    
    @staticmethod
    def from_arrays(objArr, labArr):
        ds = BallData(None, None)
        ds.objArr = torch.Tensor(objArr)
        ds.labArr = torch.Tensor(labArr)
        ds.num_samples = len(labArr)
        ds.num_balls = len(labArr[labArr == 1])
        ds.num_nonballs = len(labArr[labArr == 0])
        return ds
    
    def __init__(self, cap, cntSF, d = 20, z = 3):
        ### TODO: fix in labelled data the situation where there are two ball contours, but we could only mark one as ball
        if cap is None: ### a way to initialize an empty dataset
            self.objArr = None
            self.labArr = None
            self.num_samples = 0
            self.num_balls = 0
            self.num_nonballs = 0
            return None
        
        frames = cntSF.get_frames()
        cntStm = NCStream(cap, frames).attach(cntSF).decap()
        
        ballArrs = []
        nonballArrs = []
        for frame in frames:
            myFrm = cntStm.get(frame, put = False)
            
            ballCnts = cntSF[frame]['svm_labelled_ballCnts']
            if ballCnts is not None:
                for cnt in ballCnts:
                    objReg = cnt.enclosing_region(margin = 0).resize(to_height = d, to_width = d)
                    zoomcropFrm = myFrm.crop(objReg).zoom(z)
                    zoomcropFrm = zoomcropFrm.pad(to_height = z * d, to_width = z * d) ### in case we are at the edge of image
                    assert zoomcropFrm.array.shape == (z * d, z * d, 3)
                    ballArrs.append(zoomcropFrm.array)
                    
            nonballCnts = cntSF[frame]['svm_labelled_nonballCnts']
            if nonballCnts is not None:
                for cnt in nonballCnts:
                    objReg = cnt.enclosing_region(margin = 0).resize(to_height = d, to_width = d)
                    zoomcropFrm = myFrm.crop(objReg).zoom(z)
                    zoomcropFrm = zoomcropFrm.pad(to_height = z * d, to_width = z * d) ### in case we are at the edge of image
                    assert zoomcropFrm.array.shape == (z * d, z * d, 3)
                    nonballArrs.append(zoomcropFrm.array)
           
        num_balls = len(ballArrs)
        num_nonballs = len(nonballArrs)
        print(f'Found {num_balls} balls and {num_nonballs} non-balls in data.')
        
        labArr = np.concatenate([np.ones(num_balls), np.zeros(num_nonballs)])
        objArr = np.stack([*ballArrs, *nonballArrs]) ### shape: N, H, W, C 
        objArr = np.moveaxis(objArr, -1, 1) ### shape: N, C, H, W
        
        objArr = objArr / 255 ### normalize
        
        self.objArr = torch.Tensor(objArr)
        self.labArr = torch.Tensor(labArr)
        self.num_samples = len(labArr)
        self.num_balls = num_balls
        self.num_nonballs = num_nonballs
            
    def __getitem__(self, idx):
        return self.objArr[idx], self.labArr[idx]
    
    def __len__(self):
        return self.num_samples
    
    def __repr__(self):
        return f'BallData({self.num_samples} samples, of which {self.num_balls} balls and {self.num_nonballs} non-balls)'
    
    def train_test_split(self, share_train = 0.8, seed = 1234):
        num_train_samples = int(share_train * self.num_samples)
        
        objArr_shuffle, labArr_shuffle = self.objArr[:], self.labArr[:]
        
        rng = np.random.default_rng(seed = seed)
        objArr_shuffle = rng.permutation(objArr_shuffle)
        labArr_shuffle = rng.permutation(labArr_shuffle)
        
        objArr_train, labArr_train = objArr_shuffle[:num_train_samples], labArr_shuffle[:num_train_samples]
        objArr_test, labArr_test = objArr_shuffle[num_train_samples:], labArr_shuffle[num_train_samples:]
        
        ds_train = BallData.from_arrays(objArr_train, labArr_train)
        ds_test = BallData.from_arrays(objArr_test, labArr_test)
        
        return ds_train, ds_test
        
        


