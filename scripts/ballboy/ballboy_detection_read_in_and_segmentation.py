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

STORE_NAME = '10to100_segmentation_fixes'

#%% Reading these features back in

ACCEPTABLE_LENGTH_OF_SEGMENT = 25

from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features
from ATMANpy.funcs.cv_funcs import replay
from ATMANpy.funcs.high_level_funcs import smart_interpolate_ball_contours

stream_features_with_movement = load_stored_features(VIDEO_ID, STORE_NAME, featnameList = None, keep_which = 'last')

### add a detected flag
for k, featDict in stream_features_with_movement.items():
    if featDict['ball'] is not None:
        featDict['ballDetected'] = True
    else:
        featDict['ballDetected'] = False
    
moveList = split_joint_features(stream_features_with_movement, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)

print(f'Read in {len(moveList)} segments')


#%% Serious Run

verbose = False

split_sfs = []
for idx, move_sf in enumerate(moveList):
    
    # ### b) SPLIT THE FEATURE ACCORDING TO CLUTTER MOVEMENT
    # from ATMANpy.funcs.sf_postprocess_funcs import detect_movement_contours
    # move_sf = detect_movement_contours(corr_sf) 
    #### -> already done in parallel!
    
    if verbose:
        drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours']
        drawFunc = draw(drawList)
        replay(cap, move_sf, drawFunc = drawFunc)
    
    from ATMANpy.funcs.sf_postprocess_funcs import calculate_contiguous_movement
    cont_sf = calculate_contiguous_movement(move_sf)
    
    if verbose:
        drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours', 'movInt']
        drawFunc = draw(drawList)
        replay(cap, cont_sf, drawFunc = drawFunc)
    
    from ATMANpy.funcs.sf_postprocess_funcs import drop_movement_frames
    drop_sf = drop_movement_frames(cont_sf)
    
    sfList = split_joint_features(drop_sf, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)
            
    split_sfs.extend(sfList)
    
    ### c) TEST THE RESULTS
    frames_before = len(move_sf)
    frames_after = sum([len(sf) for sf in sfList])
    frames_killed = frames_before - frames_after 
    
    print(f'Feature Number: {idx}')
    print(f'  Number of segments created: {len(sfList)}')
    print(f'  Number of frames killed: {frames_killed}')


from ATMANpy.funcs.feature_funcs import save_contiguous_play_sequences
new_seqFrm = save_contiguous_play_sequences(split_sfs, VIDEO_ID, STORE_NAME)
    
print(f'Segmented into {len(new_seqFrm)} segments')

###

# drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours', 'movInt']
# drawFunc = draw(drawList)

# replay(cap, split_sfs[130], drawFunc = drawFunc)









