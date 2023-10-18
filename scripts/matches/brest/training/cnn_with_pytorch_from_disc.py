# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:49:11 2023

TODO:
    - How to do split (should be deterministic) into training and testing
    - Transform label_frame into a certainty vector 
    - Transform clips (normalize at 0.5)

@author: gianl
"""

import os
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms.v2 as tf
from torchvision.io import read_video
from torch.utils.data import Dataset, DataLoader
from itertools import product

class ClipDataset(Dataset):
    def __init__(self, vidpath, player, 
                 vid_indices = None, d_crop_size = 112, t_crop_size = 32):
        player_dir = f'{player}_player_labels_sequential'
        player_path = os.path.join(vidpath, player_dir)
        
        clip_lengths = {}
        label_frames = {}
        
        ### filter by video indices -> this is good for training and testing split!
        if vid_indices is None:
            videos = os.listdir(player_path)
        else:
            videos = [f for i, f in enumerate(os.listdir(player_path)) if i in vid_indices]
        
        for vid in videos:
            vid_path = os.path.join(player_path, vid)
            
            clip_names = [file for file in os.listdir(vid_path) if file.startswith('clip')]
            clip_lengths[vid] = len(clip_names)
            
            label_file = pd.read_csv(os.path.join(vid_path, 'labels.csv'), index_col = [0, 1])
            label_frames[vid] = label_file
            
        self.player_path = player_path
        self.clip_lengths = clip_lengths
        self.label_frames = label_frames
        
        self.d_crop_size = d_crop_size
        self.t_crop_size = t_crop_size
          
    def idx_map(self, idx):
        ### with respect to clips, the idea is to be able to construct a sequential
        ### index so we can use a single index for __getitem__, instead of having to 
        ### index ClipDataset()[vid, idx] like so. So we construct a map.
        vids, clip_lens = list(self.clip_lengths.keys()), list(self.clip_lengths.values())
        
        cum_idxs = np.cumsum(clip_lens)
        diff_idxs = cum_idxs - idx
        
        if all(diff_idxs <= 0):  ### in this case, the clip does not exist! 
            return None, None    ### equivalent to idx >= len(self)
        
        anchor = np.argmax(diff_idxs > 0) ### first index that is larger than given index
        anchor_idxs = [0, *cum_idxs[:-1]]
        anchor_idx = anchor_idxs[anchor]
        
        return vids[anchor], idx - anchor_idx
      
    def set_relabel_maps(self, hit_side_map = None, hit_spin_map = None, hit_type_map = None, 
                         serve_side_map = None):
        
        hit_side_labels = ['forehand', 'backhand']
        hit_spin_labels = ['topspin', 'slice', 'flat']
        hit_type_labels = ['groundstroke', 'volley', 'overhead']
        serve_side_labels = ['left', 'right']
        
        if hit_side_map is None:
            hit_side_map = {x: x for x in hit_side_labels}
        if hit_spin_map is None:
            hit_spin_map = {x: x for x in hit_spin_labels}
        if hit_type_map is None:
            hit_type_map = {x: x for x in hit_type_labels}
        if serve_side_map is None:
            serve_side_map = {x: x for x in serve_side_labels}
        
        self.hit_relabel_maps = {'side': hit_side_map, 'spin': hit_spin_map, 'type': hit_type_map}
        self.serve_relabel_maps = {'side': serve_side_map}
        
    def set_proximity_function(self, prox_func):
        self.prox_func = prox_func
        
    def get_all_labels(self):
        ### Create One Hot Positions for Label Categories
        hit_combos = []
        for cat, cat_map in self.hit_relabel_maps.items():
            cat_values = list(set(cat_map.values())) ### get unique values in category AFTER mapping
            hit_combos.append(cat_values)
        hit_combos = list(product(*hit_combos))
        
        serve_combos = []
        for cat, cat_map in self.serve_relabel_maps.items():
            cat_values = list(set(cat_map.values())) ### get unique values in category AFTER mapping
            serve_combos.append(cat_values)
        serve_combos = list(product(*serve_combos))
        
        all_labels = [('other',), *hit_combos, *serve_combos] ### other
        return all_labels
        
    def label_frame_to_vector(self, label_frame):
        all_labels = self.get_all_labels()
        
        ### Prepare Hit and Serve Frames
        hit_frame = label_frame[label_frame['hit_event'] == 'impact'][['hit_side', 'hit_spin', 'hit_type']]
        serve_frame = label_frame[label_frame['serve_event'] == 'impact'][['serve_side']]
        
        hit_frame.columns = [c.split('_')[1] for c in hit_frame]
        serve_frame.columns = [c.split('_')[1] for c in serve_frame]
        
        hit_frame = hit_frame.replace(self.hit_relabel_maps)
        serve_frame = serve_frame.replace(self.serve_relabel_maps)
        
        ### Prepare Output -> feature vector of length = len(all_labels)
        out_vector = [0 for _ in all_labels]
        if not hit_frame.empty:
            cat_tuple = tuple(hit_frame.iloc[0].values)
            idx = np.argmax([el == cat_tuple for el in all_labels]) ### this index corresponds to our category: it will potentially be assigned a non-zero value!
            
            ### Calculate Proximity to Center
            central_index = np.mean(label_frame.index) ### since the index can begin at non-zero value
            impact_indices = hit_frame.index 
            proximity = max([self.prox_func(impact_index, central_index) for impact_index in impact_indices])
            
            out_vector[0] = 1 - proximity
            out_vector[idx] = proximity
            
        elif not serve_frame.empty:
            cat_tuple = tuple(serve_frame.iloc[0].values)
            idx = np.argmax([el == cat_tuple for el in all_labels]) ### this index corresponds to our category: it will potentially be assigned a non-zero value!
            
            ### Calculate Proximity to Center
            central_index = np.mean(label_frame.index)
            impact_indices = serve_frame.index 
            proximity = max([self.prox_func(impact_index, central_index) for impact_index in impact_indices])
            
            out_vector[0] = 1 - proximity
            out_vector[idx] = proximity
            
        else:
            out_vector[0] = 1
        
        return torch.Tensor(out_vector)
      
    def __len__(self):
        clip_lens = list(self.clip_lengths.values())
        cum_idxs = np.cumsum(clip_lens)
        return cum_idxs[-1]
    
    def __getitem__(self, idx):
        vid, clip_number = self.idx_map(idx)
        
        if vid is None:
            raise IndexError(f'{idx} out of range in dataset of length {len(self)}')
        
        clip_name = f'clip_{clip_number}.avi'
        
        label_frame = self.label_frames[vid].loc[clip_number]
        
        clip_path = os.path.join(self.player_path, vid, clip_name)
        clip, audio, metadata = read_video(clip_path)
        
        ### for 3d convolutions, we need C x T x H x W
        ### right now, we have T x H x W x C
        clip = clip.movedim(-1, 0) 
        
        ### crop to desired size (112 emulates the 2+1 D paper)
        clip = tf.CenterCrop(self.d_crop_size)(clip)
        
        ### crop to desired length (16 or 32 emulate the 2+1 D paper)
        T = clip.shape[1]
        if T > self.t_crop_size:
            margin = (T - self.t_crop_size) / 2
            kill_left = math.floor(margin)
            kill_right = math.ceil(margin)    
            clip = clip[:, kill_left:-kill_right, :, :]
        
            label_frame = label_frame.iloc[kill_left:-kill_right]
        
        ### rescale and normalize the clip
        clip = clip / 255
        ### TODO: Normalize -> find out global averages across channels!
        
        return clip, self.label_frame_to_vector(label_frame)
        
    
### TODO: Batch Normalization is applied to all convolutional layers
### TODO: ReLus, where? Seems like after every single convolution?!
### TODO: The whole thing has to be a ResNet -> how to do? residual blocks are brackets!
### TODO: Data augmentation -> how?
### TODO: Train Test Split -> how?
### TODO: Convolutional Striding. conv1 -> (1, 2, 2), conv_3_1, conv_4_1, conv_5_1 -> (2, 2, 2)
### TODO: Average Pooling -> what dimensions? In paper, 16 x 112 x 112 becomes 2 x 7 x 7, but what do we have?

class TwoPlusOneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d, t, 
                 stride_d = 1, stride_t = 1):
        super(TwoPlusOneBlock, self).__init__()

        ### Calculate the number of temporal kernels M, using 
            # temporal length of kernel t, 
            # height and width of kernel d
            # input channels N_in
            # spatial output channels N_out 
        # (see tran et al 2018)
        numerator = t * d ** 2 * in_channels * out_channels
        denominator = (d ** 2 * in_channels + t * out_channels)
        M_i = numerator // denominator
        
        ### Calculate padding
        pad_d = (d - 1) // 2  ### note that since d and t are odd, 
        pad_t = (t - 1) // 2  ### floor division only enforces an integer pad
        
        self.conv_spatial = nn.Conv3d(in_channels, M_i, (1, d, d), 
                                      stride = (1, stride_d, stride_d),
                                      padding = (0, pad_d, pad_d))
        self.bn1 = nn.BatchNorm3d(M_i)
        self.relu = nn.ReLU()
        self.conv_temporal = nn.Conv3d(M_i, out_channels, (t, 1, 1),
                                       stride = (stride_t, 1, 1),
                                       padding = (pad_t, 0, 0))
        self.bn2 = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        y = self.relu(self.bn1(self.conv_spatial(x)))
        return self.relu(self.bn2(self.conv_temporal(y)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d, t, 
                 stride_d = 1, stride_t = 1):
        super(ResidualBlock, self).__init__()
        self.twoplusone1 = TwoPlusOneBlock(in_channels, out_channels, d, t,
                                           stride_d = stride_d, stride_t = stride_t) 
        self.twoplusone2 = TwoPlusOneBlock(out_channels, out_channels, d, t,
                                           stride_d = 1, stride_t = 1) 
        if stride_d == stride_t == 1:
            self.downsample_id = nn.Identity()
        else:
            self.downsample_id = nn.Conv3d(in_channels, out_channels, 1, 
                                           stride = (stride_t, stride_d, stride_d))

    def forward(self, x):
        f_x = self.twoplusone2(self.twoplusone1(x))
        down_x = self.downsample_id(x)
        return down_x + f_x
    
class ClipModel(nn.Module):
    
    def __init__(self, num_labels = 14, T = 32, D = 112):
        super(ClipModel, self).__init__()
        
        self.conv1 = TwoPlusOneBlock(3, 64, 7, 3, 
                                     stride_d = 2, stride_t = 1)
        
        self.conv2_1 = ResidualBlock(64, 64, 3, 3, 
                                     stride_d = 1, stride_t = 1)
        self.conv2_2 = ResidualBlock(64, 64, 3, 3,
                                     stride_d = 1, stride_t = 1)
        
        self.conv3_1 = ResidualBlock(64, 128, 3, 3, 
                                     stride_d = 2, stride_t = 2)
        self.conv3_2 = ResidualBlock(128, 128, 3, 3,
                                     stride_d = 1, stride_t = 1)
        
        self.conv4_1 = ResidualBlock(128, 256, 3, 3, 
                                     stride_d = 2, stride_t = 2)
        self.conv4_2 = ResidualBlock(256, 256, 3, 3,
                                     stride_d = 1, stride_t = 1)
        
        self.conv5_1 = ResidualBlock(256, 512, 3, 3, 
                                     stride_d = 2, stride_t = 2)
        self.conv5_2 = ResidualBlock(512, 512, 3, 3,
                                     stride_d = 1, stride_t = 1)
  
        avgpool_t = math.ceil(T / (2 ** 3)) ### 3 times stride_t == 2
        avgpool_d = math.ceil(D / (2 ** 4)) ### 4 times stride_d == 2
    
        self.global_avgpool = nn.AvgPool3d((avgpool_t, avgpool_d, avgpool_d))
        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(512, num_labels)
        self.softmax = nn.Softmax(dim = 1) ### dim = 1 --> the label dimension
        
    def forward(self, x, verbose = False):
        y1 = self.conv1(x)
        y2 = self.conv2_2(self.conv2_1(y1))
        y3 = self.conv3_2(self.conv3_1(y2))
        y4 = self.conv4_2(self.conv4_1(y3))
        y5 = self.conv5_2(self.conv5_1(y4))
        
        ypool = self.global_avgpool(y5)
        z = self.softmax(self.fc_out(self.flatten(ypool)))
        
        if verbose:
            print(f'Shape of Input:             {x.shape}')
            print(f'  ... after conv1:          {y1.shape}')
            print(f'  ... after conv2:          {y2.shape}')
            print(f'  ... after conv3:          {y3.shape}')
            print(f'  ... after conv4:          {y4.shape}')
            print(f'  ... after conv5:          {y5.shape}')
            print(f'  ... after global avgpool: {ypool.shape}')
            print(f'  ... of Output:            {z.shape}')
        
        return z

VIDPATH = r'C:\Users\gianl\Documents\GitHub\ATMAN\video_files\labelled_data'
PLAYER = 'bot'

D_CROP_SIZE = 112
T_CROP_SIZE = 16

def PROX_FUNC(impact_index, central_index):
    LO = 0.05
    HI = 0.95
    
    out = 1 - ((impact_index - central_index) / central_index) ** 2
    
    if out < LO:
        return 0
    elif HI <= out < HI:
        return out
    else:
        return 1
    
data = ClipDataset(VIDPATH, PLAYER, 
                   d_crop_size = D_CROP_SIZE, 
                   t_crop_size = T_CROP_SIZE)
data.set_proximity_function(PROX_FUNC)    
data.set_relabel_maps(hit_spin_map = {'topspin': 'topspin', 'slice': 'slice_or_flat', 'flat': 'slice_or_flat'},
                      serve_side_map = {'left': 'any', 'right': 'any'}) 

dataloader = DataLoader(data, batch_size = 32, shuffle = True)

num_labels = len(data.get_all_labels())    
model = ClipModel(num_labels = num_labels, T = T_CROP_SIZE, D = D_CROP_SIZE)

inputs, labels = next(iter(dataloader))

out = model(inputs, verbose = True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)



def train_one_epoch(epoch_index):
    running_loss = 0
    last_loss = 0
    
    for i, data in enumerate(dataloader):
        inputs, labels = data
        optimizer.zero_grad()
        
        outputs = model(inputs)











