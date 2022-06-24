# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:53:55 2022

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
STORE_NAME = '10to100_problematic'

###### GET SEQUENCES #####
from ATMANpy.funcs.feature_funcs import read_play_sequences
seqFrm = read_play_sequences(VIDEO_ID, STORE_NAME, keep_which = 'last')

######## GET SCHEMA ########
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)



from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features


probFrm = read_play_sequences(VIDEO_ID, STORE_NAME)

stream_features_loaded = load_stored_features(VIDEO_ID, STORE_NAME)
loadList = split_joint_features(stream_features_loaded)

assert len(probFrm) == len(loadList)

from ATMANpy.funcs.draw_funcs import draw
drawList = ['playerTop', 'playerBot', 'courtLines', 'ball']
drawFunc = draw(drawList)

#### Give a better categorization of these!
from ATMANpy.funcs.cv_funcs import replay
okList = []
ballList = [] ### ball issue -> not found (due to pan, zoom)
cutList = [] ### scene cut due to movement
longList = [] ### sometimes, the scene is too long. this might not be a problem, if we have a 'noise' state in HMM after the scene is judged to be over, of varying length. 
otherList = [] ### anything else?

for idx, sf in enumerate(loadList):
    
    if sf == {}:
        continue
    
    # ### interpolate ball positions
    # sf_itpl = smart_interpolate_ball_contours(sf, verbose = False)
    
    replay(cap, sf, drawFunc = drawFunc, ms_between_frames = 70)
    
    messageStr = """
    Press
        1 if the replay is ok. 
        2 if ball is problematic, 
        3 if scene has been cut, 
        4 if the scene is too long at the end,
        5 if other.
    """
    
    val = input(messageStr)

    if val == '1':
        okList.append(idx)
    elif val == '2':
        ballList.append(idx)
    elif val == '3':
        cutList.append(idx)
    elif val == '4':
        longList.append(idx)
    elif val == '5':
        otherList.append(idx)


print(f'Ball issues: {len(ballList)}')
print(f'Scene cut fragments: {len(cutList)}')
print(f'Scene too long: {len(longList)}')
print(f'Other issues: {len(otherList)}')

# cutList = [c for c in range(len(loadList)) if c not in ballList + longList + otherList + okList]


#####
### identify those which have erroneoulsy been cut
cutFrm = probFrm.loc[cutList]

### read in the original stream features, and filter for those which actually contain 
### those in cutFrm

ORIG_STORE = '10to100_faster'

origFrm = read_play_sequences(VIDEO_ID, ORIG_STORE, keep_which = 'first')
stream_features_orig = load_stored_features(VIDEO_ID, ORIG_STORE)
origList = split_joint_features(stream_features_orig)

assert len(origList) == len(origFrm)

### which original sequences are the constituents of the faulty fragments?
def find_constituent_segments(origFrm, cutFrm):
    containList = []
    for idx, row in cutFrm.iterrows():
        containSer = (origFrm.StopFrame >= row.StopFrame) & (origFrm.StartFrame <= row.StartFrame)
        containList.append(origFrm[containSer])
    return pd.concat(containList).drop_duplicates()

constFrm = find_constituent_segments(origFrm, cutFrm)

from ATMANpy.funcs.feature_funcs import filter_features_by_play_sequence_frame
stream_features_filt = filter_features_by_play_sequence_frame(stream_features_orig, constFrm)


for k, FD in stream_features_filt.items():
    try:
        stream_features_loaded[k]
    except KeyError:
        FD['isPlay'] = False
    else:
        FD['isPlay'] = True

moveList = split_joint_features(stream_features_filt)

#### ANALYSE ####

verbose = True

split_sfs = []
for idx, move_sf in enumerate(moveList):
    
    # ### b) SPLIT THE FEATURE ACCORDING TO CLUTTER MOVEMENT
    # from ATMANpy.funcs.sf_postprocess_funcs import detect_movement_contours
    # move_sf = detect_movement_contours(corr_sf) 
    #### -> already done in parallel!
    
    # if verbose:
    #     drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours']
    #     drawFunc = draw(drawList)
    #     replay(cap, move_sf, drawFunc = drawFunc)
    
    from ATMANpy.funcs.sf_postprocess_funcs import calculate_contiguous_movement
    cont_sf = calculate_contiguous_movement(move_sf)
    
    if verbose:
        drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours', 'movInt', 'isPlay']
        drawFunc = draw(drawList)
        replay(cap, cont_sf, drawFunc = drawFunc, ms_between_frames = 70)
    
    from ATMANpy.funcs.sf_postprocess_funcs import drop_movement_frames
    drop_sf = drop_movement_frames(cont_sf)
    
    sfList = split_joint_features(drop_sf, min_feature_length = 0)
            
    split_sfs.extend(sfList)
    
    ### c) TEST THE RESULTS
    frames_before = len(move_sf)
    frames_after = sum([len(sf) for sf in sfList])
    frames_killed = frames_before - frames_after 
    
    print(f'Feature Number: {idx}')
    print(f'  Number of segments created: {len(sfList)}')
    print(f'  Number of frames killed: {frames_killed}')

#### -> very strange... seems wrong.

        #### -> in the first sequence, players are not removed! 
        #### this is because they had to be redone, due to parallelization issue in case of 
        #### large sequences.


        







