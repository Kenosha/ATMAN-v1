# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:45:04 2022

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
start_stop_timestamp = '2022-06-11-02-42-30'
file = f'play_start_stop_{start_stop_timestamp}.csv'

seqFrm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)


###############################################
        
######## GET SCHEMA ########
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)


#%% ############# STREAM FUNCTIONS ########

import ATMANpy.funcs.high_level_funcs as hlf
import ATMANpy.funcs.particle_funcs_vectorized as pf ### in the best case, we ONLY import from hlf

def init_featDict(current_featDict = None, stored_featDict = None):

    # features_that_need_to_exist = ['H', 'config', 'courtLines', 'netLines', 
    #                                'trackFlag', 'playerBot', 'playerTop', 
    #                                'ballContours', 'ballParticles', 'ballIdxHistory', 
    #                                'ballCloseHistory', 'ballModelHistory']
    
    features_that_need_to_exist = ['H', 'courtLines', 
                                   'playerBot', 'playerTop', 
                                   'playerBot_secondary', 'playerTop_secondary']
    
    ### use the featDict that was last available, or initialize an empty one.
    if current_featDict is not None:
        featDict = current_featDict
    else:
        featDict = {k: None for k in features_that_need_to_exist}
    
    ### add in the possibility of loading in stored values read in from disc.
    if stored_featDict is not None:
        for feat in stored_featDict:
            featDict[feat] = stored_featDict[feat]
    
    return featDict

### CACHE ######
FRAME_DIFF_LIST = [4, 6, 8]
CACHE_LENGTH = max(FRAME_DIFF_LIST) * 2 + 1 # we have a frame diff of 8 on both sides, so we need 8 * 2 + 1 = 17
CURR_FRAME_IDX = CACHE_LENGTH // 2 ### the middle frame
FRAME_DIFF_INDICES = [CURR_FRAME_IDX - f for f in reversed(FRAME_DIFF_LIST)] + [CURR_FRAME_IDX + f for f in FRAME_DIFF_LIST]
################

def init_frameDict(frmCache, 
                   curr_frame_idx = CURR_FRAME_IDX, 
                   frame_diff_indices = FRAME_DIFF_INDICES):
    
    cFrm = frmCache[curr_frame_idx]
    gFrm = cFrm.to_grey()
    hFrm = cFrm.to_HSV()
    
    diffFrames = [frmCache[idx].to_grey() for idx in frame_diff_indices]
    gFrm_before = frmCache[curr_frame_idx - 1].to_grey()
    
    frameDict = {'cFrm': cFrm, 'gFrm': gFrm, 'hFrm': hFrm, 
                 'diffFrames': diffFrames, 'gFrm_before': gFrm_before}
    
    return frameDict
    
def get_court_lines(frmDict, featDict):
    
    from ATMANpy.funcs.court_funcs import find_courtlines, find_netlines
    
    gFrm = frmDict['gFrm']
    H = featDict['H']
    
    courtLines = find_courtlines(H, schemaLines)
    netLines = find_netlines(gFrm, courtLines)       
    
    featDict['courtLines'] = courtLines
    featDict['netLines'] = netLines
    
    return featDict

def get_court_features(frmDict, featDict):
    
    
    hFrm = frmDict['hFrm']
    
    H = featDict['H']
    courtLines = featDict['courtLines']
    
    playerTop = featDict['playerTop']
    playerBot = featDict['playerBot']
    
    playerTop_secondary = featDict['playerTop_secondary'] 
    playerBot_secondary = featDict['playerBot_secondary'] 
    
    outDict = featDict.copy() 
    
    ### Player ROIs, static
    roiMask_top, roiMask_bot = hlf.get_adaptive_ROIs(playerTop, playerBot, H, hFrm)
    outDict['roiMask_bot'], outDict['roiMask_top'] = roiMask_bot, roiMask_top
    
    ### Player ROIs, static
    roiMask_top_secondary, roiMask_bot_secondary = hlf.get_adaptive_ROIs(playerTop_secondary, playerBot_secondary, H, hFrm)
    outDict['roiMask_bot_secondary'], outDict['roiMask_top_secondary'] = roiMask_bot_secondary, roiMask_top_secondary  

    ### Players with static algorithm
    playerTop, playerBot = hlf.find_both_players_smart(hFrm, courtLines, H, remove_shadow = True,
                                                       iterative_burn_in = False,
                                                       playerTop_prev = playerTop, playerBot_prev = playerBot,
                                                       roiMask_top = roiMask_top, roiMask_bot = roiMask_bot)
    outDict['playerTop'], outDict['playerBot'] = playerTop, playerBot

    
    ### Players with iterative algoritgm
    playerTop_secondary, playerBot_secondary = hlf.find_both_players_smart(hFrm, courtLines, H, remove_shadow = True,
                                                       iterative_burn_in = True,
                                                       playerTop_prev = playerTop_secondary, playerBot_prev = playerBot_secondary,
                                                       roiMask_top = roiMask_top_secondary, roiMask_bot = roiMask_bot_secondary)
    outDict['playerTop_secondary'], outDict['playerBot_secondary'] = playerTop_secondary, playerBot_secondary
    
    return outDict
  

#%% STREAM AND EXTRACT BALL FEATURES

### LOAD HOMOGRAPHIES ###
from ATMANpy.funcs.feature_funcs import load_stored_features
featnameList = ['H'] 
homographies_loaded = load_stored_features(VIDEO_ID, featnameList = featnameList)

#### BUILD STREAM FEATURES FUNCTION ####
from functools import partial
from ATMANpy.funcs.cv_funcs import get_stream_features, replay

from ATMANpy.funcs.draw_funcs import draw
drawList = ['courtLines', 'playerTop', 'playerBot', 'playerTop_secondary', 'playerBot_secondary']
drawFunc = draw(drawList)

my_stream_func = partial(get_stream_features, 
                         existing_features = homographies_loaded,
                         cache_length = CACHE_LENGTH,
                         init_feat_func = init_featDict, 
                         init_frame_func = init_frameDict,
                         court_line_func = get_court_lines, 
                         court_feature_func = get_court_features,
                         draw_func = drawFunc,
                         verbose = True)

#### CREATE THE FEATURES ###
import time
from ATMANpy.funcs.feature_funcs import save_contours, save_particles

save_where = VIDEO_ID + '_player_comp_strict'

video_segments = seqFrm[1:20]

for idx, start_stop_frames in video_segments.iterrows():
    print(f'Doing {idx}...')
    start_time = time.time()
    
    start_frame, stop_frame = start_stop_frames
    stream_features_original = my_stream_func(cap, start_frame, stop_frame, verbose = False)
    # replay(cap, stream_features_smooth, drawFunc = draw_replay,
    #        ms_between_frames = 30)
     
    save_contours(stream_features_original, 'playerTop', save_where, compression = 48)
    save_contours(stream_features_original, 'playerBot', save_where, compression = 48) 
    save_contours(stream_features_original, 'playerTop_secondary', save_where, compression = 48)
    save_contours(stream_features_original, 'playerBot_secondary', save_where, compression = 48) 
    
    end_time = time.time()
    time_elapsed = round((end_time - start_time) / 60, 1)
    print(f'...Done. Time elapsed: {time_elapsed} minutes.')


#%% Load Everything


ACCEPTABLE_LENGTH_OF_SEGMENT = 25

from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features
from ATMANpy.funcs.cv_funcs import replay
from ATMANpy.funcs.high_level_funcs import smart_interpolate_ball_contours

# featnameList = ['ball', 'playerTop', 'playerBot', 'H']
stream_features_loaded = load_stored_features(save_where, featnameList = None, keep_which = 'last')

    
origList = split_joint_features(stream_features_loaded, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)


from ATMANpy.funcs.draw_funcs import draw
drawList = ['playerTop', 'playerBot', 'playerTop_secondary', 'playerBot_secondary']
drawFunc = draw(drawList)
for sf in origList:
    replay(cap, sf, drawFunc = drawFunc, ms_between_frames = 40)



