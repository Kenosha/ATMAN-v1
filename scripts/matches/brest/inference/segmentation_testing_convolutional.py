# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 22:07:58 2023

@author: gianl
"""

import math
import os
import dill as pickle
import torch
import torch.nn as nn
import torchvision.transforms.v2 as tf
import numpy as np
import pandas as pd

import ATMANpy.funcs.util_funcs as ut

from ATMANpy.models.TemporalCNN import SmallModel, SmallFullConvModel, clip_to_stream, aggregate_probs_and_labels

from ATMANpy.types.Lines import Region
from ATMANpy.types.Features import BoolFeatures, ConstFeatures
from ATMANpy.types.Streams import Stream, DecapStream
from ATMANpy.types.MultiFrames import BGRFrames

LOG_PATH = ut.get_folder_struct('cnn')['runs']
RES_PATH = ut.get_folder_struct('cnn')['weights']
PRM_PATH = ut.get_folder_struct('cnn')['params']
MODEL_TO_READ = 'bot_player_labels_foremotion32_small_fullconv_model_foremotion_20231103_151149'

DEVICE = 'cpu'

prm_read_where = os.path.join(PRM_PATH, MODEL_TO_READ + '.pkl')
res_read_where = os.path.join(RES_PATH, MODEL_TO_READ)

with open(prm_read_where, 'rb') as fp:
    params = pickle.load(fp)
    print('Read parameters: ')
    print(params)
    
model = SmallFullConvModel(num_labels = len(params['LABELS']), 
                   T = params['T_CROP_SIZE'], D = params['D_SCALE_SIZE'],
                   share_other = 0.92)
model.to(DEVICE)
model.load_state_dict(torch.load(res_read_where))
model.train(False)

### Read in Video Stream

vid = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
sto = 'segmentation_and_labelling_impactonly'

cap = ut.get_video(vid)
segBF = BoolFeatures.from_store(vid, sto, featnameList = ['isCourt']).squeeze() 
segBFs = segBF.filter(segBF).split()


seg = segBFs[1]

fs, ls = seg.first()[0], seg.last()[0]

foreStm = Stream.from_store(f'from_{fs}_to_{ls}.avi', 'foregroundMask', vid, sto,
                            cap_color_space = 'bool', target_color_space = 'mask')
moveStm = Stream.from_store(f'from_{fs}_to_{ls}.avi', 'fine_motMask', vid, sto,
                            cap_color_space = 'bool', target_color_space = 'mask')
foreFrms = foreStm.decap().frames
moveFrms = moveStm.decap().frames
maskFrms = foreFrms | moveFrms
del moveFrms
del foreFrms

vidFrms = BGRFrames.from_video(cap, fs, ls)
masked_vidFrms = vidFrms.mask(maskFrms)
del vidFrms
del maskFrms

masked_vidStm = DecapStream(masked_vidFrms, start_frame = fs, stop_frame = ls)

regCF = ConstFeatures.from_store(vid, sto, featnameList = ['botserveReg', 'topserveReg'], featTypeList = [Region, Region], frameList = [0])
botROI, topROI = regCF['botserveReg'].feature, regCF['topserveReg'].feature

bot_crop_vidFrms = masked_vidFrms.crop(botROI)
bot_crop_vidStms = DecapStream(bot_crop_vidFrms, start_frame = fs, stop_frame = ls)
# bot_crop_vidStms.show()

### Get some parameters

H, W = int(botROI.height()), int(botROI.width())

T = params['T_CROP_SIZE']
D = params['D_CROP_SIZE']
D_SC = params['D_SCALE_SIZE']

scale_factor = D_SC / D
frames_in_batch = 1000
start_frame = 5000

stride_d, stride_t = model.stride_d, model.stride_t
avgpool_d, avgpool_t = model.avgpool_d, model.avgpool_t

t_after_conv4 = math.ceil(frames_in_batch / stride_t)
d_after_conv4 = math.ceil(W * scale_factor / stride_d)

t_after_pool = t_after_conv4 - avgpool_t + 1 ### simply the kernel size formula (N - K + 2P) / S + 1, but S = 1 and P = 0
d_after_pool = d_after_conv4 - avgpool_d + 1 ### simply the kernel size formula (N - K + 2P) / S + 1, but S = 1 and P = 0

rightmost_pixel = (d_after_pool - 1) * stride_d + D_SC
last_frame = (t_after_pool - 1) * stride_t + T 

##### The input array is padded to rightmost_pixel and last_frame. 
# To be as clean as possible, we can make it so that the last_frame always falls on 
# the actual last frame that we feed into the model for inference, so that no padding is needed. 
# to this end, (frames_in_batch - T) has to be a multiple of stride_t.

### Convolutional Sliding Windows Inference 
vidFrms = bot_crop_vidFrms[start_frame:start_frame + frames_in_batch].zoom(scale_factor)
vidArr = vidFrms.array / 255 ### right now, we have T x H x W x C
vidArr = np.moveaxis(vidArr, -1, 0) ### for 3d convolutions, we need C x T x H x W
vidArr = np.flip(vidArr, axis = 0).copy() ### flip channels -> BGRFrames need to become RGB

clips = torch.Tensor(vidArr[None, :, :, :, :])
# clips = tf.CenterCrop(D)(clips)
# clips = tf.Resize(D_SC, antialias = True)(clips)

clip_to_stream(clips[0]).show(ms_between_frames = 25)

with torch.inference_mode():
    out = model(clips, verbose = True)   
    probs = nn.Softmax(dim = 1)(out)
    print(probs.shape)

category_names = [l[0] for l in params['LABELS']]
unique_categories = sorted(set(category_names))
agg_probs = []
for cat in unique_categories:
    cat_idxs = [idx for idx, c in enumerate(category_names) if c == cat]
    agg_prob = probs[:, cat_idxs].sum(axis = 1)
    agg_probs.append(agg_prob)
    
agg_probs = torch.stack(agg_probs, -1)

agg_probs = agg_probs.flatten(0, 1) ### TODO: Check if this works --> fold batches into time. 
agg_probs.shape  ### Dimension now: T, H, W, Labels
agg_probs = agg_probs[:, 0, :, :] ### since the ROI is as tall as the expected model input, H dimension will always be 1

serve_idx = unique_categories.index('any')
serve_probs = agg_probs[:, :, serve_idx]
serve_probs.shape ### T, W

T_index = range(T//2, frames_in_batch - T//2 + 1, stride_t) ### first clip begins at frame 0, but is centered at T/2 = 8, and so on.
W_index = range(0, d_after_pool)

outFrm = pd.DataFrame(serve_probs.numpy(), index = T_index, columns = W_index)





