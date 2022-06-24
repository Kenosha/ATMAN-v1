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

STORE_NAME = '10to100'


########## READ-IN PASSAGES OF PLAY ##########

###### GET SEQUENCES #####
from ATMANpy.funcs.feature_funcs import read_play_sequences
seqFrm = read_play_sequences(VIDEO_ID, STORE_NAME, keep_which = 'last')


# frmList = []
# for file in os.listdir(playseqPath):
#     frm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)
#     frmList.append(frm)
# seqFrm = pd.concat(frmList, ignore_index = True)
    
# homography_timestamp = '2022-05-03-21-53-31'
# file = f'homographies_{homography_timestamp}.csv'

# homFrm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)


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


#%% STREAM AND EXTRACT BALL FEATURES

### LOAD HOMOGRAPHIES ###
from ATMANpy.funcs.feature_funcs import load_stored_features
featnameList = ['H'] 
homographies_loaded = load_stored_features(VIDEO_ID, STORE_NAME, featnameList = featnameList)

# ### PLUG HOLES IN HOMOGRAPHIES ####
# from ATMANpy.funcs.sf_postprocess_funcs import plug_hole_in_homographies
# homographies_plugged = plug_hole_in_homographies(homographies_loaded, cap)
# # need to do the same for seqFrm!
# ### DONE! Holes are now plugged in the parallel_play_identifier script


# ### NORMALIZE HOMOGRAPHIES ###
# # a) build a function to normalize the homographies where they are weird.
# from ATMANpy.funcs.sf_postprocess_funcs import stream_features_median_corner_regions
# from ATMANpy.funcs.sf_postprocess_funcs import stream_features_correct_homography
# from ATMANpy.funcs.util_funcs import compose
# corrFunc = compose(stream_features_correct_homography, stream_features_median_corner_regions) 

### -> if we don't also do the same for the segments, the whole thing is useless!

# # b) apply the function to the split features
# from ATMANpy.funcs.feature_funcs import split_joint_features, join_split_features
# homographies_correct = join_split_features([corrFunc(sf) for sf in split_joint_features(homographies_plugged)])

# assert len(homographies_plugged) == len(homographies_correct)

# # c) save the normalized homographies
# from ATMANpy.funcs.feature_funcs import save_homographies
# save_homographies(homographies_correct, VIDEO_ID) ### we save these because we normalized H

### This part is now done in the homography creation directly!



#### BUILD STREAM FEATURES FUNCTION ####
from functools import partial
from ATMANpy.funcs.cv_funcs import get_stream_features, replay
from ATMANpy.funcs.high_level_funcs import smoothe_and_choose_ball_contour
from ATMANpy.funcs.high_level_funcs import interpolate_ball_contours, smart_interpolate_ball_contours 
from ATMANpy.funcs.high_level_funcs import smoothe_and_choose_ball_contour_with_likelihood


### BUILT THE DRAW FUNCTION ####
from ATMANpy.funcs.draw_funcs import draw
drawList = ['frmNum', 'playerTop', 'playerBot', 'courtLines', 'ball', 'scoreBoard']
drawFunc = draw(drawList)

my_stream_func = partial(get_stream_features, 
                         existing_features = homographies_loaded,
                         cache_length = CACHE_LENGTH,
                         init_feat_func = init_featDict, 
                         init_frame_func = init_frameDict,
                         court_line_func = get_court_lines, 
                         court_feature_func = get_court_features,
                         draw_func = drawFunc,
                         verbose = False)

#### CREATE THE FEATURES ###
import time
from ATMANpy.funcs.feature_funcs import save_contours, save_particles, save_region

video_segments = seqFrm

for idx, start_stop_frames in video_segments.iterrows():
    print(f'Doing {idx}...')
    start_time = time.time()
    
    start_frame, stop_frame = start_stop_frames
    stream_features_original = my_stream_func(cap, start_frame, stop_frame, verbose = False)
    stream_features_smooth = smoothe_and_choose_ball_contour(stream_features_original)
    # replay(cap, stream_features_smooth, drawFunc = drawFunc,
    #         ms_between_frames = 30)
     
    save_contours(stream_features_smooth, 'ball', VIDEO_ID, STORE_NAME, compression = 6)
    save_contours(stream_features_smooth, 'playerTop', VIDEO_ID, STORE_NAME, compression = 24)
    save_contours(stream_features_smooth, 'playerBot', VIDEO_ID, STORE_NAME, compression = 24)
    save_region(stream_features_smooth, 'scoreBoard', VIDEO_ID, STORE_NAME)
    
    end_time = time.time()
    time_elapsed = round((end_time - start_time) / 60, 1)
    print(f'...Done. Time elapsed: {time_elapsed} minutes.')




