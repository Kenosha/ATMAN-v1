# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:49:37 2023

@author: gianl
"""

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.v2 as tf
from torchvision.io import read_video
from torch.utils.data import Dataset

from itertools import product

def label_frame_to_series(label_frame, data):
    label_frame.index = label_frame.index.droplevel('clip')
    return pd.Series(data.label_frame_to_vector(label_frame).numpy())

class ClipDataset(Dataset):
    def __init__(self, data_path, dataset_folder, 
                 vid_indices = None, zero_out_clips = False,
                 d_crop_size = 112, t_crop_size = 32, ### crop is done first
                 d_scale_size = 112, ### rescaling is done afterwards
                 flip_prob = 0.5,
                 ): 
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
        self.flip_prob = flip_prob
        
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
      
    def set_relabel_maps(self, 
                         hit_side_map = None, hit_hand_map = None, hit_spin_map = None, hit_type_map = None, 
                         serve_side_map = None, serve_hand_map = None):
        
        hit_side_labels = ['forehand', 'backhand']
        hit_hand_labels = ['left', 'right']
        hit_spin_labels = ['topspin', 'slice', 'flat']
        hit_type_labels = ['groundstroke', 'volley', 'overhead']
        
        serve_side_labels = ['left', 'right']
        serve_hand_labels = ['left', 'right']
        
        if hit_side_map is None:
            hit_side_map = {x: x for x in hit_side_labels}
        if hit_hand_map is None:
            hit_hand_map = {x: x for x in hit_hand_labels}
        if hit_spin_map is None:
            hit_spin_map = {x: x for x in hit_spin_labels}
        if hit_type_map is None:
            hit_type_map = {x: x for x in hit_type_labels}
            
        if serve_side_map is None:
            serve_side_map = {x: x for x in serve_side_labels}
        if serve_hand_map is None:
            serve_hand_map = {x: x for x in serve_hand_labels}

        self.hit_relabel_maps = {'side': hit_side_map, 'hand': hit_hand_map, 'spin': hit_spin_map, 'type': hit_type_map}
        self.serve_relabel_maps = {'side': serve_side_map, 'hand': serve_hand_map}
        
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
        hit_frame = label_frame[label_frame['event'] == 'hit'][['side', 'hand', 'spin', 'type']]
        serve_frame = label_frame[label_frame['event'] == 'serve'][['side', 'hand']]
        
        # hit_frame.columns = [c.split('_')[1] for c in hit_frame]
        # serve_frame.columns = [c.split('_')[1] for c in serve_frame]
        
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
        
        label_frame = self.label_frames[vid].loc[clip_number].copy()
        
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
        
        hflip = random.random() < self.flip_prob
        if hflip:
            clip = tf.functional.horizontal_flip(clip)
            label_frame['hand'] = label_frame['hand'].replace({'left': 'right', 'right': 'left'})
        
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

