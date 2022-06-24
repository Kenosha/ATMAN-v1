# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:27:14 2022

Idea is to detect ballboys well enough to be able to interrupt the play sequence if they move.

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

STORE_NAME = '10to100_faster'

#%% Reading these features back in

ACCEPTABLE_LENGTH_OF_SEGMENT = 25

from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features
from ATMANpy.funcs.cv_funcs import replay
from ATMANpy.funcs.high_level_funcs import smart_interpolate_ball_contours

# featnameList = ['ball', 'playerTop', 'playerBot', 'H']
stream_features_original = load_stored_features(VIDEO_ID, STORE_NAME, 
                                                featnameList = None, keep_which = 'last')

### add a detected flag
for k, featDict in stream_features_original.items():
    if featDict['ball'] is not None:
        featDict['ballDetected'] = True
    else:
        featDict['ballDetected'] = False
    
origList = split_joint_features(stream_features_original, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)

print(f'Before splitting ballboy clutter: {len(origList)} segments')

#%% Split features according to play sequences created earlier

###### GET SEQUENCES #####
from ATMANpy.funcs.feature_funcs import read_play_sequences
seqFrm = read_play_sequences(VIDEO_ID, STORE_NAME, keep_which = 'last')


from ATMANpy.funcs.feature_funcs import filter_features_by_play_sequence_frame
stream_features_decluttered = filter_features_by_play_sequence_frame(stream_features_original, seqFrm)        
declList = split_joint_features(stream_features_decluttered, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)

print(f'After splitting ballboy clutter: {len(declList)} segments')


#%% Testing Decluttering versus Original Frames

    
def stream_features_add_splitbool(orig_sf, split_sfs):
    
    """
    takes an original stream feature that has been split, 
    and compares it with a list of split features. 
    It then marks each frame with a boolean denoting whether it found a correspondence 
    in the split frames.
    This boolean can then be used to brighten or darken frames in the replay, 
    indicating that a frame was kept or left our respectively.
    """

    out_sf = deepcopy(orig_sf)
    
    all_keys = [frmNum for sf in split_sfs for frmNum in sf]
    
    for k, featDict in out_sf.items():
        if k in all_keys:
            featDict['isPlay'] = True
        else:
            featDict['isPlay'] = False
            
    return out_sf
    
def sequence_bools(boolList):
    b, bs = boolList[0], boolList[1:]
    bool_seq = [b]
    for myBool in boolList:
        if myBool != bool_seq[-1]:
            bool_seq.append(myBool)
    return bool_seq


compList = [stream_features_add_splitbool(sf, declList) for sf in origList]

def make_addstr(sf, idx = None):

    ### number of del frames ###    
    boolList = [fD['isPlay'] for fD in sf.values()]
    number_of_frames_deleted = len(boolList) - sum(boolList)
    delStr = f'Del. Frms.: {number_of_frames_deleted}'

    ### number of segments
    distinct_sequences = sequence_bools(boolList)
    seqStr = f'Seqs.: {sum(distinct_sequences)}'

    if not idx is None:
        addStr = str(idx) + ': ' + delStr + '  ' + seqStr
    else:
        addStr = delStr + '  ' + seqStr
    
    return addStr

for idx, sf in enumerate(compList):
    addStr = make_addstr(sf)
    drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'isPlay']
    drawFunc = draw(drawList, add_text = addStr)
    replay(cap, sf, drawFunc = drawFunc)
        


#%% Problematic Cases: Analysis.

def show_sf(sf, splits, ms_between_frames = 25):
    print(f'Number of segments: {len(splits)}')
    
    addStr = make_addstr(orig_sf)
    drawList = ['frmNum', 'courtLines', 'ball', 'playerTop', 'playerBot', 'isPlay']
    drawFunc = draw(drawList, add_text = addStr)
    replay(cap, orig_sf, drawFunc = drawFunc, ms_between_frames = ms_between_frames)
    
    
def get_matching_split_sfs(orig_sf, split_sfs):
    outList = []
    
    ks = list(orig_sf.keys())
    start_frame, end_frame = min(ks), max(ks)
    
    for idx, sf in enumerate(split_sfs):
        ks_split = list(sf.keys())
        split_start, split_end = min(ks_split), max(ks_split)
        
        if split_end < start_frame: ### all sfs before our sf of interest
            continue
        elif split_start > end_frame:
            continue
        else:
            outList.append(sf)
    
    return outList
    

drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'isPlay']
drawFunc = draw(drawList)

### 1) 
sfNum = 8
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
show_sf(orig_sf, splits)

### -> ISSUE: segment with ballkids not entirely deleted.
problSeq = splits[1]
print(len(problSeq))

        
        