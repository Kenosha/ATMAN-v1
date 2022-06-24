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


########## READ-IN PASSAGES OF PLAY ##########
evtDict = atu.get_folder_struct('ball_event_hmm')

try:
    readPath = evtDict[VIDEO_ID]
except KeyError:
    print('Data not found.')
    
playseqPath = os.path.join(readPath, 'play_sequences')    
start_stop_timestamp = '2022-06-16-23-29-17'
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
# stream_features_uncorrected = load_stored_features(VIDEO_ID, featnameList = None, keep_which = 'first')
stream_features_corrected = load_stored_features(VIDEO_ID, featnameList = None, keep_which = 'last')

### add a detected flag
for k, featDict in stream_features_corrected.items():
    if featDict['ball'] is not None:
        featDict['ballDetected'] = True
    else:
        featDict['ballDetected'] = False
    
# loadList = split_joint_features(stream_features_uncorrected, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)
corrList = split_joint_features(stream_features_corrected, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)

print(f'Read in {len(corrList)} segments')


#%% Serious Run

verbose = False

split_sfs = []
for idx, corr_sf in enumerate(corrList):
    
    ### b) SPLIT THE FEATURE ACCORDING TO CLUTTER MOVEMENT
    from ATMANpy.funcs.sf_postprocess_funcs import detect_movement_contours
    move_sf = detect_movement_contours(corr_sf) 
    
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
    frames_before = len(corr_sf)
    frames_after = sum([len(sf) for sf in sfList])
    frames_killed = frames_before - frames_after 
    
    print(f'Feature Number: {idx}')
    print(f'  Number of segments created: {len(sfList)}')
    print(f'  Number of frames killed: {frames_killed}')


#%% IMPORTANT - SAVE THESE SPLITS!!
def save_contiguous_play_sequences(split_sfs):
    outList = []
    for sf in split_sfs:
        ks = list(sf.keys())
        start_frame, end_frame = min(ks), max(ks)
    
        outList.append((start_frame, end_frame))
        
    #### Write List to text ####
    import pandas as pd
    outFrm = pd.DataFrame(outList, columns = ['StartFrame', 'StopFrame'])
    
    import os
    ballDict = atu.get_folder_struct('ball_event_hmm')
    savePath = os.path.join(ballDict['main'], VIDEO_ID, 'play_sequences')
    
    try:
        os.mkdir(savePath)
    except FileExistsError:
        print('   Adding to existing play_sequence data')
    
    import datetime as dt
    timestamp = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    filePath = os.path.join(savePath, f'play_start_stop_{timestamp}.csv')
    
    outFrm.to_csv(filePath)   
    
save_contiguous_play_sequences(split_sfs)
    
