# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:49:37 2023

@author: gianl
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms.v2 as tf
from torchvision.io import read_video
from torch.utils.data import Dataset

from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream

from itertools import product

def clip_to_stream(clip):
    clipArr = clip.numpy()
    if not clipArr.dtype == 'uint8':
        clipArr = (clipArr * 255).astype('uint8')
    clipArr = np.moveaxis(clipArr, 0, -1) ### right now, shape is C, T, H, W --> we want T, H, W, C
    clipArr = np.flip(clipArr, -1) ### additionally, flip RGB to BGR
    clipFrms = BGRFrames(clipArr)
    clipStm = DecapStream(clipFrms, start_frame = 0, stop_frame = len(clipArr))
    return clipStm

def label_frame_to_series(label_frame, data):
    label_frame.index = label_frame.index.droplevel('clip')
    return pd.Series(data.label_frame_to_vector(label_frame).numpy())

class ClipDataset(Dataset):
    def __init__(self, data_path, dataset_folder, 
                 vid_indices = None, zero_out_clips = False,
                 d_crop_size = 112, t_crop_size = 32, ### crop is done first
                 d_scale_size = 112): ### rescaling is done afterwards
        dataset_path = os.path.join(data_path, dataset_folder)
        
        clip_lengths = {}
        clip_indices = {}
        label_frames = {}
        
        ### filter by video indices -> this is good for training and testing split!
        if vid_indices is None:
            videos = os.listdir(dataset_path)
        else:
            videos = [f for i, f in enumerate(os.listdir(dataset_path)) if i in vid_indices]
        
        for vid in videos:
            vid_path = os.path.join(dataset_path, vid)
            
            clip_names = [file for file in os.listdir(vid_path) if file.startswith('clip')]
            clip_lengths[vid] = len(clip_names)
            clip_indices[vid] = sorted([int(f.split('.')[0].split('_')[1]) for f in clip_names])
            
            label_file = pd.read_csv(os.path.join(vid_path, 'labels.csv'), index_col = [0, 1])
            label_frames[vid] = label_file
            
        ### write init arguments
        self.data_path = data_path
        self.dataset_folder = dataset_folder
        self.vid_indices = vid_indices
        self.zero_out_clips = zero_out_clips
        self.d_crop_size = d_crop_size
        self.t_crop_size = t_crop_size
        self.d_scale_size = d_scale_size
        
        ### write more useful stuff
        self.dataset_path = dataset_path
        self.clip_lengths = clip_lengths
        self.clip_indices = clip_indices
        self.label_frames = label_frames
        
        ### these have to be set by set_relabel_maps!
        self.hit_relabel_maps = None
        self.serve_relabel_maps = None
        ### this has to be set by set_prox_func
        self.prox_func = None
        
    def train_test_split(self, share_train = 0.8):
        lengths = [len(frame) for frame in self.label_frames.values()]
        total_length = sum(lengths)
        shares = np.cumsum([l/total_length for l in lengths])
        shares_dev = np.abs(shares - share_train)
        cutoff_idx = np.argmin(shares_dev) + 1
        
        train_idxs = list(range(0, cutoff_idx ))
        test_idxs = list(range(cutoff_idx, len(lengths)))
        
        train_data = ClipDataset(self.data_path, self.dataset_folder, 
                                 vid_indices = train_idxs, 
                                 zero_out_clips = self.zero_out_clips,
                                 d_crop_size = self.d_crop_size,
                                 t_crop_size = self.t_crop_size,
                                 d_scale_size = self.d_scale_size)
        test_data = ClipDataset(self.data_path, self.dataset_folder,  
                                 vid_indices = test_idxs, 
                                 zero_out_clips = self.zero_out_clips,
                                 d_crop_size = self.d_crop_size,
                                 t_crop_size = self.t_crop_size,
                                 d_scale_size = self.d_scale_size)
        
        train_data.hit_relabel_maps = self.hit_relabel_maps
        train_data.serve_relabel_maps = self.serve_relabel_maps
        train_data.prox_func = self.prox_func
        
        test_data.hit_relabel_maps = self.hit_relabel_maps
        test_data.serve_relabel_maps = self.serve_relabel_maps
        test_data.prox_func = self.prox_func
            
        return train_data, test_data
        
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
        
        vid_name = vids[anchor]
        
        ### return the proper clip number, instead of the running idx
        # this would be irrelevant if clip numbers are sequential,
        # but in some cases they are not (e.g. for overfit testing)
        clip_idx = idx - anchor_idx
        clip_number = self.clip_indices[vid_name][clip_idx]
       
        return vid_name, clip_number
      
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
        hit_combos = sorted(product(*hit_combos))
        
        serve_combos = []
        for cat, cat_map in self.serve_relabel_maps.items():
            cat_values = list(set(cat_map.values())) ### get unique values in category AFTER mapping
            serve_combos.append(cat_values)
        serve_combos = sorted(product(*serve_combos))
        
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
        vid_path = os.path.join(self.dataset_path, vid)

        if vid is None:
            raise IndexError(f'{idx} out of range in dataset of length {len(self)}')
        
        clip_name = f'clip_{clip_number}.avi'
        
        label_frame = self.label_frames[vid].loc[clip_number]
        
        clip_path = os.path.join(vid_path, clip_name)
        clip, audio, metadata = read_video(clip_path, pts_unit = 'sec')
        
        ### for 3d convolutions, we need C x T x H x W
        ### right now, we have T x H x W x C
        clip = clip.movedim(-1, 0) 
        
        # H, W = clip.shape[2], clip.shape[3]
        # if not (H == W >= self.d_crop_size >= self.d_scale_size): ## irregular image!
            # print(f'Irregular image dimensions -- H: {H}, W: {W}, crop: {self.d_crop_size}, scale: {self.d_scale_size}')
            # print(f'{vid}, {clip_number}')
            # clip_to_stream(clip).show(ms_between_frames = 100)
        ### Note: It can happen that H != W and either W or H < d_crop_size
        ### What happens in this case is tf.CenterCrop will pad at the sides, 
        ### which I think is undramatic and should be ok.
        ### However, you can use the above logic to be alerted to this kind of padding.
        
        ### crop and scale to desired size (112 emulates the 2+1 D paper)
        clip = tf.CenterCrop(self.d_crop_size)(clip)
        clip = tf.Resize(self.d_scale_size, antialias = True)(clip)
        
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
        ### Normalize maybe even with average background color!!!
        
        ### option to zero out data: to test data-independent baseline
        if self.zero_out_clips:
            clip = torch.zeros(clip.shape, dtype = clip.dtype)
        
        return clip, self.label_frame_to_vector(label_frame)
        
    
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
                                      padding = (0, pad_d, pad_d),
                                      bias = False) ### we are doing batchnorm right after -> bias not needed!
        self.bn1 = nn.BatchNorm3d(M_i)
        self.relu = nn.ReLU()
        self.conv_temporal = nn.Conv3d(M_i, out_channels, (t, 1, 1),
                                       stride = (stride_t, 1, 1),
                                       padding = (pad_t, 0, 0),
                                       bias = False) ### we are doing batchnorm right after -> bias not needed!
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
    
class OrigModel(nn.Module):
    
    def __init__(self, num_labels = 14, T = 32, D = 112):
        super(OrigModel, self).__init__()
        
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
        # self.softmax = nn.Softmax(dim = 1) ### dim = 1 --> the label dimension
        ### do not calculate softmax -> CrossEntropyLoss softmaxes by itself.
        
    def forward(self, x, verbose = False):
        y1 = self.conv1(x)
        y2 = self.conv2_2(self.conv2_1(y1))
        y3 = self.conv3_2(self.conv3_1(y2))
        y4 = self.conv4_2(self.conv4_1(y3))
        y5 = self.conv5_2(self.conv5_1(y4))
        
        ypool = self.global_avgpool(y5)
        # z = self.softmax(self.fc_out(self.flatten(ypool)))
        ### do not calculate softmax -> CrossEntropyLoss softmaxes by itself.
        z = self.fc_out(self.flatten(ypool))
        
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
    
class SmallModel(nn.Module):
    
    def __init__(self, num_labels = 14, T = 32, D = 112, share_other = 0.9):
        super(SmallModel, self).__init__()
        
        self.conv1 = TwoPlusOneBlock(3, 64, 3, 3, 
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
        
        avgpool_t = math.ceil(T / (2 ** 2)) ### 2 times stride_t == 2
        avgpool_d = math.ceil(D / (2 ** 3)) ### 3 times stride_d == 2
    
        self.global_avgpool = nn.AvgPool3d((avgpool_t, avgpool_d, avgpool_d))
        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(256, num_labels)
        
        ### re-initialize fc_out biases
        # idea: initially, the output of the FC layer with Softmax has to (roughly) match
        # the empirical share of hits and serves and others, etc.
        # to do this, we set the first bias (for "other") to zero.
        # all other biases are the same -> topspin backhand groundstroke same as serve
        # given share_other, what do the other biases need to equal?
        bias_event = np.log(1 - share_other) - np.log(share_other) - np.log(num_labels - 1)
        bias_other = 0
        biases = [bias_other, *[bias_event] * (num_labels - 1)]
        biases = nn.Parameter(torch.Tensor(biases))
        self.fc_out.bias = biases
        
        # self.softmax = nn.Softmax(dim = 1) ### dim = 1 --> the label dimension
        ### do not calculate softmax -> CrossEntropyLoss softmaxes by itself.
        
    def forward(self, x, verbose = False):
        if verbose: print(f'Shape of Input:             {x.shape}')
        
        y1 = self.conv1(x)
        if verbose: print(f'  ... after conv1:          {y1.shape}')
        
        y2 = self.conv2_2(self.conv2_1(y1))
        if verbose: print(f'  ... after conv2:          {y2.shape}')
            
        y3 = self.conv3_2(self.conv3_1(y2))
        if verbose: print(f'  ... after conv3:          {y3.shape}')
            
        y4 = self.conv4_2(self.conv4_1(y3))
        if verbose: print(f'  ... after conv4:          {y4.shape}')
        
        ypool = self.global_avgpool(y4)
        if verbose: print(f'  ... after global avgpool: {ypool.shape}')
        
        # z = self.softmax(self.fc_out(self.flatten(ypool)))
        ### do not calculate softmax -> CrossEntropyLoss softmaxes by itself.
        z = self.fc_out(self.flatten(ypool))
        if verbose: print(f'  ... of Output:            {z.shape}')
        
        return z

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
