# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 19:55:05 2022

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
from ATMANpy.funcs.feature_funcs import read_play_sequences, load_stored_features, split_joint_features
seqFrm = read_play_sequences(VIDEO_ID, STORE_NAME, keep_which = 'last')

#### GET STREAM FEATURES ####
stream_features = load_stored_features(VIDEO_ID, STORE_NAME)
split_sf = split_joint_features(stream_features)

### GET LABELS ###

vidPath = atu.get_folder_struct('ball_event_hmm')[VIDEO_ID]
labPath = os.path.join(vidPath, 'labelled_sequences_cord_correction')

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


####### Fixing another issue
lab_idx = 8
labFrm[labFrm.Index == lab_idx]
show_index(8)
### -> this is a play sequence was split in two by ballboy segmentation.
### thus, we labelled it as if it was one sequence 
### (since at the time of labelling we did not have ballboy segmentation).
#### -> sequence needs to be split ahead of fill_in_directions!!!

sf_idx1 = 8
sf_idx2 = 9

fst_serve_frames = sorted(split_sf[sf_idx1].keys())
sec_serve_frames = sorted(split_sf[sf_idx2].keys())

labFrm.loc[labFrm.Index == lab_idx, 'StartFrame'] = fst_serve_frames[0]
labFrm.loc[labFrm.Index == lab_idx, 'StopFrame'] = fst_serve_frames[-1]

### -> the second serve will be labelled as RELABEL automatically by the join_features_and_label script
### so we don't do anything here.

labFrm.to_csv(os.path.join(vidPath, 'labelled_sequences_segment_correction', 'events.csv'))







