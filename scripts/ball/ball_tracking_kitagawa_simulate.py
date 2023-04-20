# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:27:15 2022

@author: gianl
"""

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.funcs.util_funcs as atu
from ATMANpy.types.Points import Point2D, SafePoint2D
from ATMANpy.types.Lines import Line, Lines, CourtLines, Region
from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame
from ATMANpy.types.Contours import Contour

import cv2 as cv
import pandas as pd
import numpy as np

############# STREAM FUNCTIONS ########

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

def init_frameDict(frmCache):
    
    cFrm = frmCache[CURR_FRAME_IDX]
    gFrm = cFrm.to_grey()
    hFrm = cFrm.to_HSV()
    
    diffFrames = [frmCache[idx].to_grey() for idx in FRAME_DIFF_INDICES]
    gFrm_before = frmCache[CURR_FRAME_IDX - 1].to_grey()
    
    frameDict = {'cFrm': cFrm, 'gFrm': gFrm, 'hFrm': hFrm, 
                 'diffFrames': diffFrames, 'gFrm_before': gFrm_before}
    
    return frameDict
    
def get_court_lines(frmDict, featDict, cached_lines = None):
    
    gFrm = frmDict['gFrm']
    H = featDict['H']
    config = featDict['config']
    
    try:
        courtLines, netLines, H, config, trackFlag = hlf.track_courtlines(gFrm, H, config)
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
    
    if featDict['trackFlag'] == 'Tracked':
        roiMask_bot, roiMask_top = featDict['roiMask_bot'], featDict['roiMask_top']
    else:
        ### Project Regions of Interest
        projROI_bot, projROI_top = hlf.schema_specific_roi_projection_func(H)
        roiMask_bot, roiMask_top = hFrm.within_region(projROI_bot), hFrm.within_region(projROI_top)
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

    ### Top Player, with updated averages
    topplayerRegion = hlf.find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_top)
    outDict['playerTop'] = topplayerRegion
    ### Bottom Player, with updated averages
    botplayerRegion = hlf.find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_bot)
    outDict['playerBot'] = botplayerRegion
    
    ### find ball candidates
    ballContours = hlf.find_ball_contours(gFrm, diffFrames, topplayerRegion, botplayerRegion,
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
def make_sendframe(frmDict, featDict):
    
    cFrm = frmDict['cFrm']
    
    if featDict['trackFlag'] == 'Not Found':
        sendFrm = cFrm.put('No Lines!', thickness = 2, color = (0, 0, 255))
    else:
        particles = featDict['ballParticles']
        playerTop = featDict['playerTop']
        playerBot = featDict['playerBot']
        ballContours = featDict['ballContours']
        
        pPts = pf.particles_to_points(particles, particles_to_produce = 100)
        
        sendFrm = cFrm.put(playerTop, thickness = 2, color = (0, 0, 255))
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
    
    

#%% Define stream feature simulators ###


##### STREAM PARAMETERS ########

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################

### CACHE ######
FRAME_DIFF_LIST = [4, 6, 8]
CACHE_LENGTH = max(FRAME_DIFF_LIST) * 2 + 1 # we have a frame diff of 8 on both sides, so we need 8 * 2 + 1 = 17
CURR_FRAME_IDX = CACHE_LENGTH // 2 ### the middle frame
FRAME_DIFF_INDICES = [CURR_FRAME_IDX - f for f in reversed(FRAME_DIFF_LIST)] + [CURR_FRAME_IDX + f for f in FRAME_DIFF_LIST]
################

from functools import partial
from ATMANpy.funcs.cv_funcs import get_stream_features, replay
from ATMANpy.funcs.high_level_funcs import smoothe_and_choose_ball_contour, smart_interpolate_ball_contours

my_stream_func = partial(get_stream_features, 
                        cache_length = CACHE_LENGTH,
                        init_feat_func = init_featDict, 
                        init_frame_func = init_frameDict,
                        court_line_func = get_court_lines, 
                        court_feature_func = get_court_features,
                        draw_func = make_sendframe,
                        verbose = True)


#### Simulation 
video_segments = [(11830, 11930)]

for start_frame, stop_frame in video_segments:
    stream_features = my_stream_func(cap, start_frame, stop_frame)
    stream_features_smooth = smoothe_and_choose_ball_contour(stream_features)
    stream_features_int = smart_interpolate_ball_contours(stream_features_smooth)
    # replay(cap, stream_features_int, 
    #        cache_frame_offset = CURR_FRAME_IDX, drawFunc = draw_replay)

#############





