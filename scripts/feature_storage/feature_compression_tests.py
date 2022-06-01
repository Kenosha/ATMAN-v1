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
start_stop_timestamp = '2022-04-28-19-09-45'
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
        


#%% ############# STREAM FUNCTIONS ########

import ATMANpy.funcs.high_level_funcs as hlf
import ATMANpy.funcs.particle_funcs_vectorized as pf ### in the best case, we ONLY import from hlf

def init_featDict():
    # H = None
    # config = None
    # topplayerRegion, botplayerRegion = None, None
    # particles, idxHistories, closeHistories, modelHistories = None, None, None, None    
    
    featDict = {'H': None, 'config': None,
                'courtLines': None, 'netLines': None, 'trackFlag': None,
                'playerBot': None, 'playerTop': None,
                'ballContours': None, 'ballParticles': None, 'ballIdxHistory': None, 
                'ballCloseHistory': None, 'ballModelHistory': None}
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
    
def get_court_lines(frmDict, featDict, cached_lines = None):
    
    gFrm = frmDict['gFrm']
    H = featDict['H']
    config = featDict['config']
    
    if not cached_lines is None:
        courtLines, H, trackFlag = cached_lines
        from ATMANpy.funcs.court_funcs import find_netlines
        netLines = find_netlines(gFrm, courtLines)     
    else:
        try:
            courtLines, netLines, H, config, trackFlag = hlf.track_courtlines(gFrm, H, config,
                                                                              only_225 = True)
        except Exception as e:
            print('Exception in track_courtlines:')
            print(e)
            courtLines, netLines, H, config = None
            trackFlag = 'Not Found'
    
    featDict['courtLines'] = courtLines
    featDict['netLines'] = netLines
    featDict['H'] = H
    featDict['config'] = config
    featDict['trackFlag'] = trackFlag
    
    return featDict

def get_court_features(frmDict, featDict): ### these two quantities will be used in simulations
    
    ### handle the case if the court is not found ---> simply reset all features!
    if featDict['trackFlag'] == 'Not Found':
        outDict = init_featDict()
        outDict['trackFlag'] = 'Not Found'
        return outDict
        
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
        particles = pf.initialize_particles(gFrm_before, courtLines)

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
    topPlayer, botPlayer = hlf.find_both_players(hFrm, courtLines, netLines, H, remove_shadow = True,
                                                 roiMask_top = roiMask_top, roiMask_bot = roiMask_bot)
    outDict['playerTop'], outDict['playerBot'] = topPlayer, botPlayer

    
    ### find ball candidates
    ballContours = hlf.find_ball_contours(gFrm, diffFrames, topPlayer, botPlayer,
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
    
    if featDict['trackFlag'] == 'Not Found':
        sendFrm = sendFrm.put('No Lines!', thickness = 2, color = (0, 0, 255))
    else:
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

def draw_replay(frmDict, featDict):
    
    cFrm = frmDict['cFrm']
    
    if featDict['trackFlag'] == 'Not Found':
        sendFrm = cFrm.put('No Lines!', thickness = 2, color = (0, 0, 255))
    else:
        selectedBall = featDict['ball']
        detectedBall = featDict['ballDetected']
        playerTop = featDict['playerTop']
        playerBot = featDict['playerBot']
        ballContours = featDict['ballContours']
        
        sendFrm = cFrm.put(playerTop, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(playerBot, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
        
        if selectedBall is not None:
            if detectedBall: ### we have identified the contour as ball!
                sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 0))
            else: ### 
                sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 255))
        
    return sendFrm


def draw_replay_onlyball(frmDict, featDict):
    
    cFrm = frmDict['cFrm']
    
    if featDict['trackFlag'] == 'Not Found':
        sendFrm = cFrm.put('No Lines!', thickness = 2, color = (0, 0, 255))
    else:
        selectedBall = featDict['ball']
        detectedBall = featDict['ballDetected']
        ballContours = featDict['ballContours']
        ballParticles = featDict['ballParticles']
        
        sendFrm = cFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
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


from functools import partial
from ATMANpy.funcs.cv_funcs import get_stream_features, replay
from ATMANpy.funcs.high_level_funcs import smoothe_and_choose_ball_contour
from ATMANpy.funcs.high_level_funcs import interpolate_ball_contours, smart_interpolate_ball_contours 

my_stream_func = partial(get_stream_features, 
                        cache_length = CACHE_LENGTH,
                        init_feat_func = init_featDict, 
                        init_frame_func = init_frameDict,
                        court_line_func = get_court_lines, 
                        court_feature_func = get_court_features,
                        draw_func = make_sendframe,
                        verbose = False)



#### Simulation 
video_segments = seqFrm.head(5)
featList = []


for idx, start_stop_frames in video_segments.iterrows():
    start_frame, stop_frame = start_stop_frames
    stream_features_original = my_stream_func(cap, start_frame, stop_frame)
    stream_features_smooth = smoothe_and_choose_ball_contour(stream_features_original)
    featList.append(stream_features_smooth)
    replay(cap, stream_features_smooth, drawFunc = draw_replay,
           ms_between_frames = 30)
    

#%% Saving and Loading...




from ATMANpy.funcs.feature_funcs import save_contours, save_homographies, save_particles

for play_sequence_stream_features in featList:
        
    # save_contours(play_sequence_stream_features, 'ball', VIDEO_ID, compression = 6)
    # save_contours(play_sequence_stream_features, 'playerTop', VIDEO_ID, compression = 24)
    # save_contours(play_sequence_stream_features, 'playerBot', VIDEO_ID, compression = 24)
    # save_homographies(play_sequence_stream_features, VIDEO_ID)

    save_particles(play_sequence_stream_features, 'ballParticles', VIDEO_ID)


#%% Reading these features back in


from ATMANpy.funcs.feature_funcs import load_stored_features

# featnameList = ['ball', 'playerTop', 'playerBot', 'H']
stream_features_loaded = load_stored_features(VIDEO_ID, featnameList = None)
 
stream_features_loaded.keys()
stream_features_loaded[13584]
    
def draw_loaded(frmDict, featDict):

    cFrm = frmDict['cFrm']

    selectedBall = featDict['ball']
    playerTop = featDict['playerTop']
    playerBot = featDict['playerBot']
    ballParticles = featDict['ballParticles']
    
    sendFrm = cFrm.put(playerTop, thickness = 2, color = (0, 0, 255))
    sendFrm = sendFrm.put(playerBot, thickness = 2, color = (0, 0, 255))
    # sendFrm = sendFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
    
    if selectedBall is not None:
        sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 0))
    
    return sendFrm


stream_features_loaded.keys()
replay(cap, stream_features_loaded, start_frame = 10280, stop_frame = 10636, drawFunc=draw_loaded)


#%%##### Small Investigation
H = None
config = None

playerTop = None
playerBot = None

for frame in range(10470, 10490):
    hFrm = HSVFrame.from_video(cap, frame)
    cFrm = hFrm.to_BGR()
    gFrm = hFrm.to_grey() 
    
    courtLines, netLines, H, config, trackFlag = hlf.track_courtlines(gFrm, H, config,
                                                                      only_225 = False)
    
    if not H is None:
        roiMask_top, roiMask_bot = hlf.get_adaptive_ROIs(playerTop, playerBot, H, hFrm)
        
        
        ### Players with adaptive ROIs.
        playerTop, playerBot = hlf.find_both_players(hFrm, courtLines, netLines, H, remove_shadow = True,
                                                     roiMask_top = roiMask_top, roiMask_bot = roiMask_bot)
        
        cFrm.mask(roiMask_bot).show()
        cFrm.put(playerTop, color = (0, 0, 255), thickness = 2).put(playerBot, color = (0, 0, 255), thickness = 2).show()












