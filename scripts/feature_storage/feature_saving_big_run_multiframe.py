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

### STORENAME ###
STORE_NAME = '10to100_lowerlikelihood'

###### GET SEQUENCES #####
from ATMANpy.funcs.feature_funcs import read_play_sequences
seqFrm = read_play_sequences(VIDEO_ID, STORE_NAME, keep_which = 'first')

######## GET SCHEMA ########
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)


#%% ############# STREAM FUNCTIONS ########

import ATMANpy.funcs.high_level_funcs as hlf
import ATMANpy.funcs.particle_funcs_vectorized as pf ### in the best case, we ONLY import from hlf
from ATMANpy.funcs.scoreboard_funcs import find_scoreboard

def init_featDict(current_featDict = None, stored_featDict = None):

    # features_that_need_to_exist = ['H', 'config', 'courtLines', 'netLines', 
    #                                'trackFlag', 'playerBot', 'playerTop', 
    #                                'ballContours', 'ballParticles', 'ballIdxHistory', 
    #                                'ballCloseHistory', 'ballModelHistory']
    
    features_that_need_to_exist = ['H', 'courtLines', 'netLines', 
                                   'playerBot', 'playerTop', 
                                   'ballContours', 'ballParticles', 'ballIdxHistory', 
                                   'ballCloseHistory', 'ballModelHistory']
    
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

def get_court_features(frmDict, featDict): ### these two quantities will be used in simulations
    
    gFrm_before = frmDict['gFrm_before']
    cFrm = frmDict['cFrm']
    gFrm = frmDict['gFrm']
    hFrm = frmDict['hFrm']
    diffFrames = frmDict['diffFrames']
    
    H = featDict['H']
    courtLines = featDict['courtLines']
    netLines = featDict['netLines']
    
    particles = featDict['ballParticles']
    playerTop = featDict['playerTop']
    playerBot = featDict['playerBot']
    ballIdxHistory = featDict['ballIdxHistory']
    ballCloseHistory = featDict['ballCloseHistory']
    ballModelHistory = featDict['ballModelHistory']
    
    outDict = featDict.copy() 
    
    ### Find Scoreboard
    SOBEL_THRESHOLD = 30
    SLOPE_THRESHOLD = 0.0001
    MERGE_LINES = False
    MIN_LINE_LENGTH = 10
    MIN_LINES = 1
    
    scoreReg = find_scoreboard(gFrm, 
                               sobel_threshold = SOBEL_THRESHOLD, 
                               slope_threshold = SLOPE_THRESHOLD,
                               merge_lines = MERGE_LINES,
                               min_line_length_after_merge = MIN_LINE_LENGTH, 
                               min_lines = MIN_LINES)
    outDict['scoreBoard'] = scoreReg
    
    # print(scoreReg)
    if not scoreReg is None:
        scoreMask = gFrm.within_region(scoreReg)
    
    ### Player ROIs
    roiMask_top, roiMask_bot = hlf.get_adaptive_ROIs(playerTop, playerBot, H, hFrm)
    
    if not scoreReg is None:
        roiMask_bot = roiMask_bot & ~scoreMask ### adjust for scoreboard
        # roiMask_top = roiMask_top & ~scoreMask ### in a general setting, both sides need to be excluded.
    
    outDict['roiMask_bot'], outDict['roiMask_top'] = roiMask_bot, roiMask_top

    ### Based on previous frame (and previous players, if they exist...)
    # a) ... if necessary, initialize particles
    if particles is None:
        particles = hlf.initialize_particles(gFrm_before, courtLines, N = 10000)

    # b) ... get particle propagation.
    propagation_models, closeBools, modelBools = hlf.decide_particle_propagation(particles, gFrm_before, 
                            topplayer = playerTop, botplayer = playerBot, 
                            closeHistories = ballCloseHistory, 
                            modelHistories = ballModelHistory)
    # c) ... display particle propagation
    # print(modelBools.sum())
    # primPts = pf.particles_to_points(particles[modelBools], particles_to_produce = 100)
    # normPts = pf.particles_to_points(particles[~modelBools], particles_to_produce = 100)

    # cFrm.put(primPts, color = (255, 0, 0)).put(normPts, color = (0, 0, 255)).show()

    ### Players with adaptive ROIs.
    playerTop, playerBot = hlf.find_both_players_smart(hFrm, courtLines, H, remove_shadow = True,
                                                       iterative_burn_in = True,
                                                       playerTop_prev = playerTop, playerBot_prev = playerBot,
                                                       roiMask_top = roiMask_top, roiMask_bot = roiMask_bot)
    outDict['playerTop'], outDict['playerBot'] = playerTop, playerBot

    
    ### find ball candidates
    ballContours = hlf.find_ball_contours(gFrm, diffFrames, playerTop, playerBot,
                                          cFrm = cFrm, hFrm = hFrm)
    outDict['ballContours'] = ballContours
    
    ### find ball particles
    particles, particleIdxs = hlf.update_particles(particles, propagation_models, ballContours)
    ballIdxHistory, ballCloseHistory, ballModelHistory = pf.update_histories(ballIdxHistory, ballCloseHistory, ballModelHistory, 
                                                                             particleIdxs, closeBools, modelBools)
    outDict['ballParticles'] = particles
    outDict['ballIdxHistory'] = ballIdxHistory
    outDict['ballCloseHistory'] = ballCloseHistory
    outDict['ballModelHistory'] = ballModelHistory
    
    return outDict


#%% Get a few multiframes --> see how much gbs they need.

from ATMANpy.types.MultiFrames import Frames

def add_cache_frames(seqFrm, cache_length):
    outFrm = seqFrm.copy()
    cache_offset = cache_length // 2
    outFrm['StartFrame'] = outFrm['StartFrame'] - cache_offset
    outFrm['StopFrame'] = outFrm['StopFrame'] + cache_offset
    return outFrm

cacheFrm = add_cache_frames(seqFrm, CACHE_LENGTH)

# segments = seqFrm.values.tolist()[:HOW_MANY]
# segments = [Frames(VIDEO_ID, start, stop, 'bgr') for start, stop in segments]

#%% BASIC STREAM FUNCTION 

### LOAD HOMOGRAPHIES ###
featnameList = ['H'] 
from ATMANpy.funcs.feature_funcs import load_stored_features
homographies_loaded = load_stored_features(VIDEO_ID, STORE_NAME,
                                           featnameList = featnameList,
                                           keep_which = 'last')



### BUILT THE DRAW FUNCTION ####
from ATMANpy.funcs.draw_funcs import draw
drawList = ['frmNum', 'playerTop', 'playerBot', 'courtLines', 'ball', 'scoreBoard']
drawFunc = draw(drawList)

#### BUILD STREAM FEATURES FUNCTION ####
from functools import partial
from ATMANpy.funcs.cv_funcs import get_stream_features_from_generator, replay
from ATMANpy.funcs.high_level_funcs import smoothe_and_choose_ball_contour

basic_stream_func = partial(get_stream_features_from_generator, 
                            existing_features = homographies_loaded,
                            cache_length = CACHE_LENGTH,
                            init_feat_func = init_featDict, 
                            init_frame_func = init_frameDict,
                            court_line_func = get_court_lines, 
                            court_feature_func = get_court_features,
                            draw_func = drawFunc,
                            verbose = False)

from ATMANpy.funcs.feature_funcs import save_contours, save_region
def my_stream_func(segFrames):
    segGen = segFrames.to_generator()
    stream_features_original = basic_stream_func(segGen)
    stream_features_smooth = smoothe_and_choose_ball_contour(stream_features_original)
     
    save_contours(stream_features_smooth, 'ball', VIDEO_ID, STORE_NAME, compression = 6)
    save_contours(stream_features_smooth, 'playerTop', VIDEO_ID, STORE_NAME, compression = 24)
    save_contours(stream_features_smooth, 'playerBot', VIDEO_ID, STORE_NAME, compression = 24)
    save_region(stream_features_smooth, 'scoreBoard', VIDEO_ID, STORE_NAME)
    
    print(f'Done with {segFrames.start_frame} to {segFrames.stop_frame}')

import time 

# for segFrms in segments:
#     time_start = time.time()
    
#     stream_features_smooth = my_stream_func(segFrms)
    
#     time_elapsed = time.time() - time_start
#     minutes_elapsed = round(time_elapsed / 60, 1)
#     print(f'Took {minutes_elapsed} minutes')


# replay(cap, stream_features_smooth, drawFunc = drawFunc)

#%% 

#### CREATE THE PARALLEL FUNCTION ###



from ATMANpy.funcs.feature_funcs import save_features_parallel

doList = cacheFrm.values.tolist()

if __name__ == '__main__':
    save_features_parallel(VIDEO_ID, doList, my_stream_func, num_workers = 8)


#%% Correct sequences in which no player or scoreboard is found 
### -> in these, the parallelization failed due to a weird bug.

# from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features

# featnameList = ['ball', 'playerTop', 'playerBot', 'scoreBoard']
# stream_features = load_stored_features(VIDEO_ID, STORE_NAME)

# # loadList = split_joint_features(stream_features_uncorrected, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)
# sfList = split_joint_features(stream_features, min_feature_length = 0)

# print(f'Read in {len(sfList)} segments')

# def check_if_all_none(sf, which = featnameList):

#     for k in sf:
#         for feat in which:
#             if sf[k][feat] is not None:
#                 return False
#             else:
#                 continue
#     return True
            
# all_none = np.array([check_if_all_none(sf) for sf in sfList])

# noneIdxs = np.argwhere(all_none).flatten()

# for idx in noneIdxs:
#     start, stop = doList[idx]

#     print(f'Redoing element {idx}, consisting of {stop - start} frames')
#     replay(cap, sfList[idx], drawFunc = drawFunc)

#     idxFrames = Frames(VIDEO_ID, start, stop, 'bgr')
#     my_stream_func(idxFrames)


    
    
    
    

