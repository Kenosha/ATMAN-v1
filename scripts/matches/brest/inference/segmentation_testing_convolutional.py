# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 22:07:58 2023

### Convolutional Sliding Windows Inference 

@author: gianl
"""

import os
import dill as pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms.v2 as tf
from torch.utils.data import get_worker_info, IterableDataset, DataLoader


import ATMANpy.funcs.util_funcs as ut

from ATMANpy.models.TemporalCNN.nets import SmallModel, SmallFullConvModel
from ATMANpy.models.TemporalCNN.utils import clip_to_stream, frames_to_stream, infer_with_tbatch, make_cat_frame, plot_timing, split_and_group
from ATMANpy.models.TemporalCNN.data import BaselineInferenceDataset, get_masked_video_frames, preprocess_cropped_frames

from ATMANpy.types.Features import BoolFeature, ConstFeature, StreamFeature, StreamFeatures
from ATMANpy.types.Streams import Stream, DecapStream
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Frames import BGRFrame

from ATMANpy.types.Lines import Raster, Line, Region
from ATMANpy.types.Points import Point2D


#### HELPER FUNCTIONS ###

def get_model_output_shape(model, T_in, H_in, W_in):
    print('Calculating output shape (T x H x W) of array fed to model...')

    ### Collect some Model Parameters
    T_train, H_train, W_train = model.T, model.D, model.D
    stride_d, stride_t = model.stride_d, model.stride_t
    avgpool_d, avgpool_t = model.avgpool_d, model.avgpool_t
    
    print(f'    Input ({T_in}, {H_in}, {W_in}) shrunk to ({T_in}, {H_train}, {W_train})')
    
    ### Calculate Shape after Last Convolutional Layer
    t_after_conv4 = math.ceil(T_in / stride_t)
    h_after_conv4 = math.ceil(H_train / stride_d)
    w_after_conv4 = math.ceil(W_train / stride_d)
    
    ### Calculate Shape after AvgPool
    ### simply the kernel size formula (N - K + 2P) / S + 1, but S = 1 and P = 0
    t_after_pool = t_after_conv4 - avgpool_t + 1
    h_after_pool = h_after_conv4 - avgpool_d + 1 
    w_after_pool = w_after_conv4 - avgpool_d + 1
    
    ### Calculate Other Useful Stuff
    last_frame = (t_after_pool - 1) * stride_t + T_train
    uppermost_pixel = (h_after_pool - 1) * stride_d + H_train
    rightmost_pixel = (w_after_pool - 1) * stride_d + W_train
    
    out = t_after_pool, h_after_pool, w_after_pool
    print(f'    Out shape: {out}')
    
    ##### The input array is padded to rightmost_pixel and last_frame. 
    # To be as clean as possible, we can make it so that the last_frame always falls on 
    # the actual last frame that we feed into the model for inference, so that no padding is needed. 
    # to this end, (frames_in_batch - T) has to be a multiple of stride_t. 
    # Since T is a multiple of stride_t (16 and 4 resp.), frames_in_batch has to be a multiple of stride_t
    print(f'    Last frame convolved over: {last_frame}')
    print(f'    Uppermost pixel convolved over: {uppermost_pixel}')
    print(f'    Rightmost pixel convolved over: {rightmost_pixel}')
    return out


# models = {'top': 'top_player_labels_foremotion32_small_fullconv_model_foremotion_20231105_220049',
#           'bot': 'bot_player_labels_foremotion32_small_fullconv_model_foremotion_20231105_191602'}
models = {'top': 'top_player_labels_foremotion32_small_fullconv_model_foremotion_20231105_220049',
          'bot': 'bot_foremotion_T32_S8_randcrop5_dim120_small_fullconv_model_foremotion_20231118_002935'}

PLAYER = 'bot'
SERVEREG = 'large'
LOG_PATH = ut.get_folder_struct('cnn')['runs']
RES_PATH = ut.get_folder_struct('cnn')['weights']
PRM_PATH = ut.get_folder_struct('cnn')['params']
MODEL_TO_READ = models[PLAYER]

prm_read_where = os.path.join(PRM_PATH, MODEL_TO_READ + '.pkl')
res_read_where = os.path.join(RES_PATH, MODEL_TO_READ)
grp_save_where = os.path.join(LOG_PATH, MODEL_TO_READ, f'timing_charts_{SERVEREG}')

try:
    os.mkdir(grp_save_where)
except FileExistsError:
    pass

vid = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
sto = 'segmentation_and_labelling_impactonly'

DEVICE = 'cuda'

with open(prm_read_where, 'rb') as fp:
    params = pickle.load(fp)
    print('Read parameters: ')
    print(params)
    
model = SmallFullConvModel(labels = params['LABELS'], 
                           T = params['T_CROP_SIZE'], D = params['D_SCALE_SIZE'],
                           stride_d1 = 2, stride_d2 = 1, stride_d3 = 2, stride_d4 = 2, ### TODO: initialize these as parameters
                           stride_t1 = 1, stride_t2 = 1, stride_t3 = 1, stride_t4 = 2,
                           share_other = 0.92)
model.to(DEVICE)
model.load_state_dict(torch.load(res_read_where))
model.train(False)

### Read in Data
# a) video stream
cap = ut.get_video(vid)

# b) features
courtBF = BoolFeature.from_store(vid, sto, featname = 'isCourt') 
regCF = ConstFeature.from_store(vid, sto, featname = f'{PLAYER}serve_{SERVEREG}Reg', featType = Region, frameList = [0])
evtSF = StreamFeatures.from_store(vid, sto, featnameList = [f'labelled_{PLAYER}_event_Flag', f'labelled_{PLAYER}_side_Flag'])

labelled_courtBF = courtBF.filter(courtBF).split()[:4].concatenate() ### only have labels for the first four segments...
baselineROI = regCF.feature
baselineROI = Region(lineList=[Line(leftPt=Point2D(coords=(279, 123), value=None), rightPt=Point2D(coords=(279, 243), value=None), name='Left'), Line(leftPt=Point2D(coords=(279, 123), value=None), rightPt=Point2D(coords=(399, 123), value=None), name='Top'), Line(leftPt=Point2D(coords=(399, 123), value=None), rightPt=Point2D(coords=(399, 243), value=None), name='Right'), Line(leftPt=Point2D(coords=(279, 243), value=None), rightPt=Point2D(coords=(399, 243), value=None), name='Bottom')], name='Region')

# c) Get some parameters
T = params['T_CROP_SIZE']
D = params['D_CROP_SIZE']
D_SC = params['D_SCALE_SIZE']
scale_factor = D_SC / D
stride_t, stride_d = model.stride_t, model.stride_d
frames_in_sample = 16

# d) some ground truth info
is_serve = evtSF[f'labelled_{PLAYER}_event_Flag'] == 'serve'
is_hit = evtSF[f'labelled_{PLAYER}_event_Flag'] == 'hit'
is_fh = (evtSF[f'labelled_{PLAYER}_side_Flag'] == 'forehand') & is_hit
is_bh = (evtSF[f'labelled_{PLAYER}_side_Flag'] == 'backhand') & is_hit

labelled_serves = is_serve.filter(is_serve).get_frames()
labelled_hits = is_hit.filter(is_hit).get_frames()
labelled_forehands = is_fh.filter(is_fh).get_frames()
labelled_backhands = is_bh.filter(is_bh).get_frames()

labDict = {'serve': labelled_serves,
           'forehand': labelled_forehands,
           'backhand': labelled_backhands}


#%% INFERENCE

get_model_output_shape(model, frames_in_sample, baselineROI.height(), baselineROI.width())

ds = BaselineInferenceDataset(cap, vid, sto, labelled_courtBF, baselineROI,
                              frames_in_sample = frames_in_sample, D = D, D_SC = D_SC,
                              stride_t = stride_t, T = T)

inf_loader = DataLoader(ds, batch_size = 1, num_workers = 0, pin_memory = True)

prbArr, labArr = infer_with_tbatch(model, inf_loader, device = DEVICE)
prbArr, labArr = prbArr.numpy(), labArr.numpy()



#%% ANALYSIS 

#############################
##### Check Inference #######
#############################
SHOW_MARGIN = 10
THRESHOLD = 0.2
shape_x, shape_y = 3, 3
serves_in_graph = shape_x * shape_y
COLORS = {'serve': 'darkblue', 'forehand': 'firebrick', 'backhand': 'orange'}

### b) Aggregate Inference
category_names = [l[0] for l in params['LABELS']]
catFrm = make_cat_frame(category_names, prbArr, labArr.tolist())
evtFrm = catFrm.drop('other', axis = 1)
    
### c) Check Recall: Recognition of serves that actually are serves!
rec_save_where = os.path.join(grp_save_where, 'recall')
try:
    os.mkdir(rec_save_where)
except FileExistsError:
    pass

serve_trues = split_and_group(labelled_serves, offset = 1)
plot_timing(serve_trues, evtFrm, f'serve', labDict, rec_save_where)

### c) Check Precision: Truth of a serve that is predicted as a serve.
prc_save_where = os.path.join(grp_save_where, 'precision')
try:
    os.mkdir(prc_save_where)
except FileExistsError:
    pass

tFrm = catFrm[catFrm['serve'] > THRESHOLD]
serve_preds = split_and_group(tFrm.index.tolist(), offset = stride_t)
plot_timing(serve_preds, evtFrm, f'serve', labDict, prc_save_where)

### TODO: make_cat_frame calculates maximum for each label independently! 
# find first location of highest probability of event (1 - other), then check what is happening in this quadrant. 
# maybe plot these quadrants visually, too?! this way, we can check where serves occur (and thereby reduce our map.)
# also: why is backhand so prominent at serve???! what the hell. must be a bug. check superadditive charts (with indipendent (= wrong) label calc.)

#### DONT NEED THESE --> recognition of hits should be centered at the player, not at the baseline!
# ### d) Check recognition of forehands
# # Check Recall: Recognition of forehand that actually are forehand!
# serve_trues = split_and_group(labelled_forehands, offset = 1)
# plot_timing(serve_trues, evtFrm, 'recall_forehand', labDict)

# # Check Precision: Truth of a forehand that is predicted as a forehand.
# tFrm = catFrm[catFrm['forehand'] > THRESHOLD]
# serve_preds = split_and_group(tFrm.index.tolist(), offset = stride_t)
# plot_timing(serve_preds, evtFrm, 'precision_forehand', labDict)

# ### e) Check recognition of backhands
# # Check Recall: Recognition of backhand that actually are backhand!
# serve_trues = split_and_group(labelled_backhands, offset = 1)
# plot_timing(serve_trues, evtFrm, 'recall_backhand', labDict)

# # Check Precision: Truth of a backhand that is predicted as a backhand.
# tFrm = catFrm[catFrm['backhand'] > THRESHOLD]
# serve_preds = split_and_group(tFrm.index.tolist(), offset = stride_t)
# plot_timing(serve_preds, evtFrm, 'precision_backhand', labDict)


#%% Testing

frame = 14135
idx = np.argmin(np.abs(labArr - frame))

### 
a, b = next(iter(inf_loader))

for subArr, idxs in inf_loader:
    if torch.min(idxs) < frame < torch.max(idxs):
        break
    
clip_to_stream(subArr[0]).show()
batch_idx = np.argmax(idxs == frame + 0.5)

softmax = nn.Softmax(dim = 1)
subArr[:, :, batch_idx].shape
with torch.inference_mode():
    devArr = subArr.to(DEVICE)
    res = model(devArr)
    prb = softmax(res)
    
prb[0, -1, batch_idx]


### get frames
chkArr = prbArr[:, idx]
central_frame = labArr[idx]
offset = (T - 1) / 2
start_frame, stop_frame = int(central_frame - offset), int(central_frame + offset)
myFrms = BGRFrames.from_video(cap, start_frame, stop_frame)
# myFrms.put(baselineROI)

frames_to_stream(myFrms).show(ms_between_frames = 100)

### get probability values
catDict = {}
category_names = [l[0] for l in params['LABELS']]
unique_categories = sorted(set(category_names))
for cat in unique_categories:
    cat_idxs = [idx for idx, c in enumerate(category_names) if c == cat]

    catArr = chkArr[cat_idxs]
    catArr = catArr.sum(axis = 0) ### sum over contained labels to aggregate

    catDict[cat] = catArr

(catDict['backhand'] * 100).round()
(catDict['any'] * 100).round()

someReg = Region(lineList=[Line(leftPt=Point2D(coords=(323, 151), value=None), rightPt=Point2D(coords=(323, 215), value=None), name='Left'), Line(leftPt=Point2D(coords=(323, 151), value=None), rightPt=Point2D(coords=(356, 151), value=None), name='Top'), Line(leftPt=Point2D(coords=(356, 151), value=None), rightPt=Point2D(coords=(356, 215), value=None), name='Right'), Line(leftPt=Point2D(coords=(323, 215), value=None), rightPt=Point2D(coords=(356, 215), value=None), name='Bottom')], name='Region')
someReg = someReg.resize(to_height = D, to_width = D, how = 'any') ### shrink excess height or width --> the rest we do with padding! 

myRst = Raster(baselineROI, stride = int(stride_d / scale_factor), size = D)
myFrms[8].put(baselineROI).put(myRst[6, 0]).show()
myFrms[8].put(baselineROI).put(someReg).show()

fslsDict = {0: (0, 4925),
            1: (5426, 23225),
            2: (23726, 36925),
            3: (37426, 47225)}
seg = 1
fs, ls = fslsDict[seg]
masked_vidFrms = get_masked_video_frames(cap, vid, sto, fs = fs, ls = ls)
crop_vidFrms = masked_vidFrms.crop(myRst[6, 0])
crop_vidFrms = crop_vidFrms[start_frame - fs: stop_frame - fs + 1]
vidArr = preprocess_cropped_frames(crop_vidFrms, D, D_SC) ### C, T, H, W

softmax = nn.Softmax(dim = 1)
with torch.inference_mode():
    vidTns = torch.Tensor(vidArr[None, :, :, :, :]).to(DEVICE)
    zroTns = torch.zeros(vidTns.shape).to(DEVICE)
    res = model(vidTns)
    res0 = model(zroTns)

clip_to_stream(vidTns[0].cpu()).show(ms_between_frames = 100)

out = softmax(res).squeeze()
print(out)  



# vidArr.shape
# vidArr.sum()

# ds[idx]

# baselineROI

# baselineROI


# frames_to_stream(myFrms).show(ms_between_frames = 100)    

    

#%% Check Inference on Players

# plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayer', 'botPlayer'])

# seg.merge(plySF).split()



# #%%% OTHER

# Stream(cap, 9810, 9830).show(ms_between_frames = 100, frame_counter = True)
# Stream(cap, 19745, 19775).show(ms_between_frames = 100, frame_counter = True)
# Stream(cap, 21750, 21850).show(ms_between_frames = 100, frame_counter = True)

# srv_frm = 9820
# srv_frm = 19765
# frames_to_show = list(range(srv_frm - 20, srv_frm + 20))
# frames_to_show = [f for f in frames_to_show if f in srvFrm.index] 
# chkFrm = outFrm.loc[frames_to_show]



