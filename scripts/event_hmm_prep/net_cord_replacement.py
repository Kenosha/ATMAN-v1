# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 02:11:39 2022

@author: gianl
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 00:57:32 2022

@author: gianl
"""



import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.funcs.util_funcs as atu
from ATMANpy.types.Points import Point, SafePoint
from ATMANpy.types.Lines import Line, Lines, CourtLines, Region
from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame
from ATMANpy.types.Contours import Contour

from ATMANpy.funcs.draw_funcs import draw

# image operation
import cv2 as cv
import numpy as np

import pandas as pd

import os
from random import randrange

from copy import deepcopy

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################


### STORENAME ###
STORE_NAME = '10to100_ok2'

###### GET SEQUENCES #####
from ATMANpy.funcs.feature_funcs import read_play_sequences, load_stored_features
seqFrm = read_play_sequences(VIDEO_ID, STORE_NAME, keep_which = 'last')

#### GET STREAM FEATURES ####
stream_features = load_stored_features(VIDEO_ID, STORE_NAME)


### GET LABELS ###

vidPath = atu.get_folder_struct('ball_event_hmm')[VIDEO_ID]
labPath = os.path.join(vidPath, 'labelled_sequences')

frmList = []
for file in os.listdir(labPath):
    frm = pd.read_csv(os.path.join(labPath, file), index_col = 0)
    frmList.append(frm)
    
labFrm = pd.concat(frmList)
labFrm = labFrm.groupby('Index', group_keys = False).apply(lambda df: df.sort_values('FrameNum'))
labFrm = labFrm[labFrm.Event != 'NOHIT']

def show_index(idx):
    labels = labFrm[labFrm.Index == idx]
    start, stop = labels.StartFrame.unique()[0], labels.StopFrame.unique()[0]
    
    from ATMANpy.funcs.cv_funcs import show_video
    show_video(cap, start_frame = start, stop_frame = stop)

#### Distinguish between NET and CORD!
# -> this way, NET will always end the point!!!
to_reassess = labFrm[labFrm.Event == 'NET'].Index.unique()

okList = []
cordList = []
for idx in to_reassess:
    show_index(idx)
    
    val = input("Press 1 if NET is always NET. Press 2 if NET should be CORD at least once")

    if val == '1':
        okList.append(idx)
    elif val == '2':
        cordList.append(idx)


idx = cordList[0]
# show_index(idx)
idx_filt = labFrm.Index == idx
labFrm.loc[idx_filt, 'Event'] = labFrm.loc[idx_filt, 'Event'].replace('NET', 'CORD')

idx = cordList[1]
# show_index(idx)
idx_filt = labFrm.Index == idx
labFrm.loc[idx_filt, 'Event'] = labFrm.loc[idx_filt, 'Event'].replace('NET', 'CORD')

idx = cordList[2]
# show_index(idx)
idx_filt = labFrm.Index == idx
labFrm.loc[idx_filt, 'Event'] = labFrm.loc[idx_filt, 'Event'].replace('NET', 'CORD')

idx = cordList[3]
show_index(idx)
idx_filt = labFrm.Index == idx
labFrm.loc[idx_filt, 'Event'] = labFrm.loc[idx_filt, 'Event'].replace('NET', 'CORD')

idx = cordList[4]
show_index(idx)
idx_filt = labFrm.Index == idx
labFrm.loc[idx_filt, 'Event'] = labFrm.loc[idx_filt, 'Event'].replace('NET', 'CORD')

idx = cordList[5]
# show_index(idx)
idx_filt = labFrm.Index == idx
net_filt = labFrm.Event == 'NET'
print(labFrm[idx_filt & net_filt])

cord_frames = [79695]
crd_filt = labFrm.FrameNum.isin(cord_frames)
labFrm.loc[crd_filt, 'Event'] = labFrm.loc[crd_filt, 'Event'].replace('NET', 'CORD')

idx = cordList[6]
show_index(idx)
idx_filt = labFrm.Index == idx
labFrm.loc[idx_filt, 'Event'] = labFrm.loc[idx_filt, 'Event'].replace('NET', 'CORD')

idx = cordList[7]
show_index(idx)
idx_filt = labFrm.Index == idx
labFrm.loc[idx_filt, 'Event'] = labFrm.loc[idx_filt, 'Event'].replace('NET', 'CORD')

idx = cordList[8]
show_index(idx)
idx_filt = labFrm.Index == idx
net_filt = labFrm.Event == 'NET'
print(labFrm[idx_filt & net_filt])

idx = cordList[9]
show_index(idx)


labFrm.to_csv(os.path.join(vidPath, 'labelled_sequences_cord_correction', 'events.csv'))

