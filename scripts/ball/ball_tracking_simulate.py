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
    
def get_court_lines(frmDict, featDict):
    
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

def get_court_features(frmDict, featDict,
                       N = None, close_propagation = None): ### these two quantities will be used in simulations
    
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
        particles = pf.initialize_particles(gFrm_before, courtLines, N = N)

    # b) ... get particle propagation.
    propagation_models, closeBools, modelBools = hlf.decide_particle_propagation(particles, gFrm_before, 
                            topplayer = playerTop, botplayer = playerBot, 
                            close_propagation = close_propagation,
                            closeHistories = ballCloseHistory, modelHistories = ballModelHistory)
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
        
##### STREAM PARAMETERS ########

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################

FRAME_DIFF_LIST = [4, 6, 8]
CACHE_LENGTH = max(FRAME_DIFF_LIST) * 2 + 1 # we have a frame diff of 8 on both sides, so we need 8 * 2 + 1 = 17
CURR_FRAME_IDX = CACHE_LENGTH // 2 ### the middle frame
FRAME_DIFF_INDICES = [CURR_FRAME_IDX - f for f in reversed(FRAME_DIFF_LIST)] + [CURR_FRAME_IDX + f for f in FRAME_DIFF_LIST]



START_FRAME = 11830
STOP_FRAME = 11920

################################

def get_stream_features(START_FRAME, STOP_FRAME, verbose = True,
                        N = None, close_propagation = None):

    ### init outputList
    streamFeatures = {}
    
    #### init feature dictionary
    featDict = init_featDict()
    
    #### init cache ###
    frmCache = []
    
    ### init video stream ###
    from ATMANpy.funcs.cv_funcs import video_stream
    gen = video_stream(cap, start_frame = START_FRAME - CACHE_LENGTH, stop_frame = STOP_FRAME)
    frmNum, rawArr = next(gen) ### this is needed to "prime" the generator
    ######
    
    while True:
        
        ### prepare and feed the cache ###
        # cache frames, so we can frame difference 
        # without having to re-generate the frames from the video capture
        latestFrame = BGRFrame(rawArr)    
        frmCache.append(latestFrame)
        if len(frmCache) < 17:
            frmNum, rawArr = gen.send(None)
            continue
        if len(frmCache) > 17:
            print(f'Cache FUBAR, {len(frmCache)} items') ### this happens if the cache drop at the end fails
            break
        
        ### prepare frame objects of interest
        frmDict = init_frameDict(frmCache)
        
        ### Get the features of interest on the frame
        featDict = get_court_lines(frmDict, featDict)
        featDict = get_court_features(frmDict, featDict, N = N, close_propagation = close_propagation)
        streamFeatures[frmNum] = featDict
    
        if verbose:
            ### Send Frame Back to Generator
            sendFrm = make_sendframe(frmDict, featDict)
            sendArr = sendFrm.array  
        else:
            sendArr = None
            
        try:
            frmNum, rawArr = gen.send(sendArr)
        except:
            cv.destroyAllWindows()
            break 
        
        ### drop oldest element from cache ### 
        frmCache = frmCache[1:] # drop oldest item
        
    else:
        cv.destroyAllWindows()

    return streamFeatures

##### SIMULATION ######

VERBOSE = False
M = 3 #number of simulations

#### a) Compare the close-to-player propagation models

# the standard close-to-player movement model
A_prime = np.matrix([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]) 

# the close-to-player movement model with reversed thrust
A_primeprime = np.matrix([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, -1]]) 


# compute closeness criterion/criteria --> how good are particles relative to whatever is ground truth? (for us: unique contour)
def calculate_feature_score(featDict):
    scoreDict = {}
    for frmNum, features in featDict.items():
        if len(features['ballContours']) == 1:
            cnt = features['ballContours'][0]
            squared_dists = (features['ballParticles'][:, :2] - cnt.center().coords) ** 2
            scores = squared_dists.sum(axis = 1) ** 0.5
            score = scores.mean()
            scoreDict[frmNum] = score
    return scoreDict

def mean_score(scores):
    if isinstance(scores, dict):
        return np.array(list(scores.values())).mean()
    elif isinstance(scores, list):
        return np.array(scores).mean()
    
def median_score(scores):
    if isinstance(scores, dict):
        return np.median(np.array(list(scores.values())))
    elif isinstance(scores, list):
        return np.median(np.array(scores))
    
    
### Standard
print('Testing Standard Model...')
scores_prime = []
for i in range(M):
    print(f' ... Trial Number {i}')
    featDict = get_stream_features(START_FRAME, STOP_FRAME,
                                   N = 1000, close_propagation = A_prime, 
                                   verbose = VERBOSE)
    scoreDict = calculate_feature_score(featDict)
    
    trial_score = mean_score(scoreDict)
    print(f'   -> scored {trial_score}')
    
    scores_prime.append(scoreDict)
    

### Thrust-Reversal
print('Testing Thrust-Reversal Model...')
scores_primeprime = []
for i in range(M):
    print(f' ... Trial Number {i}')
    featDict = get_stream_features(START_FRAME, STOP_FRAME,
                                   N = 1000, close_propagation = A_primeprime, 
                                   verbose = VERBOSE)
    scoreDict = calculate_feature_score(featDict)
    
    trial_score = mean_score(scoreDict)
    print(f'   -> scored {trial_score}')
    scores_primeprime.append(scoreDict)



median_score([mean_score(trial) for trial in scores_prime])
median_score([mean_score(trial) for trial in scores_primeprime])

######### REPLAY ########



def replay(cap, stream_features):
    START_FRAME = min(stream_features.keys())
    STOP_FRAME = max(stream_features.keys())
    
    ### init video stream ###
    from ATMANpy.funcs.cv_funcs import video_stream
    gen = video_stream(cap, start_frame = START_FRAME, stop_frame = STOP_FRAME)
    frmNum, rawArr = next(gen) ### this is needed to "prime" the generator
    ######
    
    while True:
        
        cFrm = BGRFrame(rawArr)
        featDict = stream_features[frmNum]
        
        ### Send Frame Back to Generator
        sendFrm = make_sendframe({'cFrm': cFrm}, featDict)
        sendArr = sendFrm.array  
        
        try:
            frmNum, rawArr = gen.send(sendArr)
        except:
            cv.destroyAllWindows()
            break 
    
  
# featDict2 = {int(k): v for k, v in featDict.items()}
# replay(cap, featDict2)
    
        





