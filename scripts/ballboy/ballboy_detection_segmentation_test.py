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


########## READ-IN PASSAGES OF PLAY ##########

evtDict = atu.get_folder_struct('ball_event_hmm')

try:
    readPath = evtDict[VIDEO_ID]
except KeyError:
    print('Data not found.')
    
playseqPath = os.path.join(readPath, 'play_sequences')    
start_stop_timestamp = '2022-05-26-16-02-20'
file = f'play_start_stop_{start_stop_timestamp}.csv'

seqFrm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)

######## GET SCHEMA ########
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)

#%% Reading these features back in

ACCEPTABLE_LENGTH_OF_SEGMENT = 25

from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features
from ATMANpy.funcs.cv_funcs import replay
from ATMANpy.funcs.high_level_funcs import smart_interpolate_ball_contours

# featnameList = ['ball', 'playerTop', 'playerBot', 'H']
stream_features_loaded = load_stored_features(VIDEO_ID + '_old3', featnameList = None)


### add a detected flag
for k, featDict in stream_features_loaded.items():
    if featDict['ball'] is not None:
        featDict['ballDetected'] = True
    else:
        featDict['ballDetected'] = False
    

loadList = split_joint_features(stream_features_loaded, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)

from ATMANpy.funcs.draw_funcs import draw

drawList = ['frmNum', 'ball', 'playerTop', 'playerBot']
drawFunc = draw(drawList)
replay(cap, loadList[141], drawFunc = drawFunc)


#%% Normalize Homographies

from ATMANpy.funcs.court_funcs import find_courtlines
from ATMANpy.funcs.sf_postprocess_funcs import stream_features_median_corner_regions
from ATMANpy.funcs.sf_postprocess_funcs import stream_features_correct_homography

from ATMANpy.funcs.util_funcs import compose
corrFunc = compose(stream_features_correct_homography, stream_features_median_corner_regions)
corrList = [corrFunc(sf) for sf in loadList]


### FOR TESTING ###

sf_num = 9
load_sf = loadList[sf_num]     
corr_sf = corrList[sf_num]

for k in corr_sf:
    if not np.array_equal(load_sf[k]['H'], corr_sf[k]['H']):
        myLines1 = find_courtlines(load_sf[k]['H'], schemaLines)
        myLines2 = find_courtlines(corr_sf[k]['H'], schemaLines)

        sendFrm = BGRFrame.from_video(cap, k)
        sendFrm = sendFrm.put(myLines1, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(myLines2, thickness = 2, color = (255, 0, 0))
        
        sendFrm.show()

#%% Serious Run

split_sfs = []
for idx, corr_sf in enumerate(corrList):
    
    ### b) SPLIT THE FEATURE ACCORDING TO CLUTTER MOVEMENT
    from ATMANpy.funcs.sf_postprocess_funcs import detect_movement_contours
    move_sf = detect_movement_contours(corr_sf) 
    
    drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours']
    drawFunc = draw(drawList)
    # replay(cap, move_sf, drawFunc = drawFunc)
    
    from ATMANpy.funcs.sf_postprocess_funcs import calculate_contiguous_movement
    cont_sf = calculate_contiguous_movement(move_sf)
    drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours', 'movInt']
    drawFunc = draw(drawList)
    # replay(cap, cont_sf, drawFunc = drawFunc)
    
    from ATMANpy.funcs.sf_postprocess_funcs import drop_movement_frames
    drop_sf = drop_movement_frames(cont_sf)
    
    from ATMANpy.funcs.feature_funcs import split_joint_features
    sfList = split_joint_features(drop_sf, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)
            
    split_sfs.extend(sfList)
    
    ### c) TEST THE RESULTS
    frames_before = len(corr_sf)
    frames_after = sum([len(sf) for sf in sfList])
    frames_killed = frames_before - frames_after 
    
    print(f'Feature Number: {idx}')
    print(f'  Number of segments created: {len(sfList)}')
    print(f'  Number of frames killed: {frames_killed}')


#%% Testing all of these:
    
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
    

# for sf in corrList:

sf = corrList[7]

sfList = get_matching_split_sfs(sf, split_sfs)
    
drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours', 'movInt']
drawFunc = draw(drawList, add_text = 'Original Sequence')
replay(cap, sf, drawFunc = drawFunc)

for idx, sf in enumerate(sfList):
    drawFunc = draw(drawList, add_text = f'Split Sequence: {idx}')
    replay(cap, sf, drawFunc = drawFunc)

    
len(split_sfs)

idx


#%% Split the loaded stream features if there is movement in frame.

########################
##### FOR TESTING ######
########################

### a) CHOOSE A STREAM FEATURE
sf_num = 12
corr_sf = corrList[sf_num]

### b) SPLIT THE FEATURE ACCORDING TO CLUTTER MOVEMENT
from ATMANpy.funcs.sf_postprocess_funcs import detect_movement_contours
move_sf = detect_movement_contours(corr_sf) 

drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours']
drawFunc = draw(drawList)
# replay(cap, move_sf, drawFunc = drawFunc)

from ATMANpy.funcs.sf_postprocess_funcs import calculate_contiguous_movement
cont_sf = calculate_contiguous_movement(move_sf)
drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours', 'movInt']
drawFunc = draw(drawList)
# replay(cap, cont_sf, drawFunc = drawFunc)

from ATMANpy.funcs.sf_postprocess_funcs import drop_movement_frames
drop_sf = drop_movement_frames(cont_sf)

from ATMANpy.funcs.feature_funcs import split_joint_features
ACCEPTABLE_LENGTH_OF_SEGMENT = 25
sfList = split_joint_features(drop_sf, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)
        
### c) TEST THE RESULTS
frames_before = len(corr_sf)
frames_after = sum([len(sf) for sf in sfList])
frames_killed = frames_before - frames_after 

print(f'Feature Number: {sf_num}')
print(f'  Number of segments created: {len(sfList)}')
print(f'  Number of frames killed: {frames_killed}')

for sf in sfList:
    replay(cap, sf, drawFunc = drawFunc, ms_between_frames = 40)
    
    
#### Results 

sf_num = 8
corr_sf = corrList[sf_num]
# -> we get three segments, with one being a clutter segment of length 42 ### len(sfList[1])
    

sf_num = 9
corr_sf = corrList[sf_num]
# -> weird 'breathing' clutter detected at picture's edge.
### line detection for this one is systematically bad! can not be normalized away!


drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours', 'courtLines']
drawFunc = draw(drawList)
replay(cap, move_sf, drawFunc = drawFunc, ms_between_frames = 40)
        
        
        
        