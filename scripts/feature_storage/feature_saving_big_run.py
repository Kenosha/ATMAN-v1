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
start_stop_timestamp = '2022-05-26-16-02-20'
file = f'play_start_stop_{start_stop_timestamp}.csv'

seqFrm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)

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
    
    ### Player ROIs
    roiMask_top, roiMask_bot = hlf.get_adaptive_ROIs(playerTop, playerBot, H, hFrm)
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
  

#%%### DRAW FUNCTIONS #####  
def make_sendframe(frmDict, featDict, frameNum = None):
    
    cFrm = frmDict['cFrm']
    
    sendFrm = cFrm
    
    if frameNum is not None:
        sendFrm = sendFrm.put(str(int(frameNum)), thickness = 2, color = (0, 255, 255), where = (1800, 50))

    particles = featDict['ballParticles']
    playerTop = featDict['playerTop']
    playerBot = featDict['playerBot']
    ballContours = featDict['ballContours']
    
    pPts = pf.particles_to_points(particles, particles_to_produce = 100)
    
    sendFrm = sendFrm.put(playerTop, thickness = 2, color = (0, 0, 255))
    sendFrm = sendFrm.put(playerBot, thickness = 2, color = (0, 0, 255))
    sendFrm = sendFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
    sendFrm = sendFrm.put(pPts, thickness = 1, color = (0, 255, 0))

    return sendFrm

def draw_replay(frmDict, featDict, frameNum = None):
    
    cFrm = frmDict['cFrm']

    courtLines = featDict['courtLines']
    netLines = featDict['netLines']
    
    selectedBall = featDict['ball']
    detectedBall = featDict['ballDetected']
    playerTop = featDict['playerTop']
    playerBot = featDict['playerBot']
    ballContours = featDict['ballContours']
    
    try:
        ballLikelihood = featDict['ballLikelihood']
        ballThreshold = featDict['ballThreshold']
    except:
        pass
    
    sendFrm = cFrm
    
    if frameNum is not None:
        sendFrm = sendFrm.put(str(int(frameNum)), thickness = 2, color = (0, 255, 255), where = (1800, 50))
    
    sendFrm = sendFrm.put(courtLines, thickness = 2, color = (255, 255, 0))
    sendFrm = sendFrm.put(netLines, thickness = 2, color = (100, 100, 0))
    
    sendFrm = sendFrm.put(playerTop, thickness = 2, color = (0, 0, 255))
    sendFrm = sendFrm.put(playerBot, thickness = 2, color = (0, 0, 255))
    sendFrm = sendFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
    
    try:
        sendFrm = sendFrm.put(str(100 * ballLikelihood), thickness = 2, color = (0, 255, 255), where = (1800, 700))
        sendFrm = sendFrm.put(str(100 * ballThreshold), thickness = 2, color = (0, 255, 255), where = (1800, 750))
    except:
        pass
    
    if selectedBall is not None:
        if detectedBall: ### we have identified the contour as ball!
            sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 0))
        else: ### 
            sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 255))
    
    return sendFrm


def draw_replay_onlyball(frmDict, featDict):
    
    cFrm = frmDict['cFrm']

    selectedBall = featDict['ball']
    detectedBall = featDict['ballDetected']
    ballContours = featDict['ballContours']
    ballParticles = featDict['ballParticles']
    
    try:
        ballLikelihood = featDict['ballLikelihood']
        ballThreshold = featDict['ballThreshold']
    except:
        pass
    
    sendFrm = cFrm
    
    try:
        sendFrm = sendFrm.put(str(round(1000000 * ballLikelihood, 40)), thickness = 2, color = (0, 255, 255), where = (1300, 700))
        sendFrm = sendFrm.put(str(round(1000000 * ballThreshold, 40)), thickness = 2, color = (0, 255, 255), where = (1300, 750))
    except:
        pass
    
    
    sendFrm = sendFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
    if not ballParticles is None:
        pPts = pf.particles_to_points(ballParticles, particles_to_produce = 200)
        sendFrm = sendFrm.put(pPts, thickness = 1, color = (0, 0, 255))
    
    if selectedBall is not None:
        if detectedBall: ### we have identified the contour as ball!
            sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 0))
        else: ### 
            sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 255))
    
    return sendFrm



#%% STREAM AND EXTRACT BALL FEATURES

from ATMANpy.funcs.feature_funcs import load_stored_features
from ATMANpy.funcs.feature_funcs import save_contours, save_homographies, save_particles

featnameList = ['H'] ### load homographies
homographies_loaded = load_stored_features(VIDEO_ID, featnameList = featnameList)

from functools import partial
from ATMANpy.funcs.cv_funcs import get_stream_features, replay
from ATMANpy.funcs.high_level_funcs import smoothe_and_choose_ball_contour
from ATMANpy.funcs.high_level_funcs import interpolate_ball_contours, smart_interpolate_ball_contours 
from ATMANpy.funcs.high_level_funcs import smoothe_and_choose_ball_contour_with_likelihood

import time

my_stream_func = partial(get_stream_features, 
                         existing_features = homographies_loaded,
                         cache_length = CACHE_LENGTH,
                         init_feat_func = init_featDict, 
                         init_frame_func = init_frameDict,
                         court_line_func = get_court_lines, 
                         court_feature_func = get_court_features,
                         draw_func = make_sendframe,
                         verbose = False)



#### Simulation 
video_segments = seqFrm.loc[42:]

for idx, start_stop_frames in video_segments.iterrows():
    print(f'Doing {idx}...')
    start_time = time.time()
    
    start_frame, stop_frame = start_stop_frames
    stream_features_original = my_stream_func(cap, start_frame, stop_frame, verbose = False)
    stream_features_smooth = smoothe_and_choose_ball_contour(stream_features_original)
    replay(cap, stream_features_smooth, drawFunc = draw_replay,
           ms_between_frames = 30)
     
    save_contours(stream_features_smooth, 'ball', VIDEO_ID, compression = 6)
    save_contours(stream_features_smooth, 'playerTop', VIDEO_ID, compression = 24)
    save_contours(stream_features_smooth, 'playerBot', VIDEO_ID, compression = 24)
    
    end_time = time.time()
    time_elapsed = round((end_time - start_time) / 60, 1)
    print(f'...Done. Time elapsed: {time_elapsed} minutes.')




