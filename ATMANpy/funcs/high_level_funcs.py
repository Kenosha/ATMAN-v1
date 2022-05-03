# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:15:55 2022

@author: gianl
"""


import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.funcs.util_funcs as atu
from ATMANpy.types.Points import Point, SafePoint
from ATMANpy.types.Lines import Line, Lines, CourtLines, Region
from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame
from ATMANpy.types.Contours import Contour

import cv2 as cv
import pandas as pd
import numpy as np

######## GET SCHEMA ########
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)

###### DEFINE TRACK PROCEDURE ########
from ATMANpy.funcs.court_funcs import find_homography, find_homography_225_only, update_homography
from ATMANpy.funcs.court_funcs import find_courtlines, find_netlines

VIDEO_ID = 'pqxAmUWIoKc'

def track_courtlines(gFrm, prev_H, prev_config, schemaLines = schemaLines,
                     get_net = True, only_225 = True):
    if prev_H is None:
        if only_225:
            H, config = find_homography_225_only(gFrm)
        else:
            H, config = find_homography(gFrm)
        outFlag = 'Found'
    else:
        # print('Courtlines found, updating homography...')
        H, config = update_homography(gFrm, prev_H, prev_config)
        outFlag = 'Tracked'
        
    if not H is None:
        courtLines = find_courtlines(H, schemaLines)
        if get_net:
            netLines = find_netlines(gFrm, courtLines)       
    
        # outFrm = gFrm.to_BGR().put(courtLines, thickness = 3)
        # outFrm = outFrm.put(netLines, color = (255, 0, 0), thickness = 3)
    else:
        outFlag = 'Not Found'
        courtLines, netLines = None, None
        
    if get_net:
        return courtLines, netLines, H, config, outFlag
    else:
        return courtLines, H, config, outFlag
        

def cached_courtlines(frameNum, homFrm, 
                      schemaLines = schemaLines):
    
    try:
        flat_H = homFrm.loc[frameNum]
    except KeyError:
        return None
    else:
        flat_H.at[8] = 1
        H = flat_H.values.reshape((3, 3))
    
        courtLines = find_courtlines(H, schemaLines)
        outFlag = 'Cached'
            
        return courtLines, H, outFlag


######## READ IN AND PREPARE BURN-IN ######
burnDict = atu.get_folder_struct('burn_in')

try:
    readPath = burnDict[VIDEO_ID]
except KeyError:
    print('Burn-In Data not found.')
    
medFrm = HSVFrame.from_csv(readPath, name = 'median')
stdFrm = HSVFrame.from_csv(readPath, name = 'std')

##### GET THE BURNED IN MASK ########
NORMALIZED_DEVIATION_THRESHOLD = 4
BURN_IN_HSV_CHANNEL_WEIGHTS = (1, 50, 1) #interested only in hue and value

from ATMANpy.funcs.player_funcs import get_burn_in_mask
from functools import partial
match_specific_burn_in_func = partial(get_burn_in_mask, medFrm, stdFrm,
                                      normalized_deviation_threshold = NORMALIZED_DEVIATION_THRESHOLD,
                                      channel_weights = BURN_IN_HSV_CHANNEL_WEIGHTS) ### these are always the same!
#######################


########## GET REGIONS OF INTEREST ON SCHEMA ##########

WIDTH_FACTOR = 2 ### how much of the doubles alley width does a player maximally next to court
DEPTH_FACTOR = 0.2 ### how much of side length does a player maximally stand behind baseline
ADD_DEPTH_FACTOR = 1.25 ### top side depth adjustment - player figure is tall, so looks to be outside of ROI.
# we could adjust this with the camera matrix, but let's do something eyeballed here, for the moment.

from ATMANpy.funcs.player_funcs import get_initial_ROIs
ROI_lower, ROI_upper = get_initial_ROIs(schemaLines,
                                         width_factor = WIDTH_FACTOR, 
                                         depth_factor = DEPTH_FACTOR, 
                                         add_depth_factor_top = ADD_DEPTH_FACTOR)

def get_projected_ROIs(ROI_lower, ROI_upper, H):
    projROI_lower = ROI_lower.transform('apply_homography', H).transform('round_points')
    projROI_upper = ROI_upper.transform('apply_homography', H).transform('round_points')
    # cFrm.put(projROI_upper, color = (0, 0, 255), thickness = 2).put(projROI_lower, color = (0, 255, 0), thickness = 2).show()
    return projROI_lower, projROI_upper

schema_specific_roi_projection_func = partial(get_projected_ROIs, ROI_lower, ROI_upper)


#### PLAYER DETECTION ####
verbose = False
COURT_DILATION = 1.5
NET_DILATION = 2.5
ROI_DILATION = 1

from ATMANpy.funcs.court_funcs import COURT_LINE_WIDTH, WHITE_PIXEL_THRESHOLD
from ATMANpy.funcs.player_funcs import find_player_contour


def find_player(frame, courtLines, netLines, H, roiMask = None, verbose = False):
    
    ### get the burned in mask for this specific homography
    burn_foreMask = match_specific_burn_in_func(frame, H,
                                    normalized_deviation_threshold = NORMALIZED_DEVIATION_THRESHOLD, 
                                    channel_weights = BURN_IN_HSV_CHANNEL_WEIGHTS, 
                                    verbose = verbose)
    
    playerRegion = find_player_contour(frame, burn_foreMask, roiMask)
    
    return playerRegion



#%%### BALL DETECTION ####

#### READ IN THE BALL TRACKING SVM ######
import os
ballDict = atu.get_folder_struct('ball_svm')
fitPath = os.path.join(ballDict['main'], VIDEO_ID, 'fitted_svms')
svm_name = 'svm_dict_2022-04-13'

svmPath = os.path.join(fitPath, svm_name + '.pickle')

import pickle

with open(svmPath, 'rb') as f:
    svmDict = pickle.load(f)


### DEFINE BALL CANDIDATE FINDER ###
MAX_NUMBER_OF_CONTOURS = 40 # if there are too many contours, likely the camera was zooming or panning.
                            # this causes many call candidates (due to frame differencing).
                            # these spurious candidates we don't want in our training data
                            # also, these frames take ages to load due to labelling.

from ATMANpy.funcs.ball_funcs import get_motion_mask_from_frames, find_ball_candidate_mask, make_feature_frame

def find_ball_contours(gFrm, diffFrames, topplayerRegion, botplayerRegion,
                       svmDict = svmDict, cFrm = None, hFrm = None):
    
    clf, features, mean, std = svmDict['svm'], svmDict['features'], svmDict['mean'], svmDict['std']
    
    motMask = get_motion_mask_from_frames(gFrm, diffFrames)
    intMask = find_ball_candidate_mask(cFrm, motMask, topplayerRegion, botplayerRegion)
    
    cnts, _ = intMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = [Contour(c) for c in cnts if len(c) > 2] ### need at least three points for a contour
    
    if not 0 < len(cnts) < MAX_NUMBER_OF_CONTOURS:
        return []
        
    cntFrm = make_feature_frame(cnts, hFrm, gFrm, intMask).dropna()
    
    if len(cntFrm) == 0:
        return []
    
    cntFrm = (cntFrm - mean) / std

    cntFrm['isBall'] = clf.predict(cntFrm[features].values)
    
    ballIdx = list(cntFrm[cntFrm.isBall == 1].index)
    
    return [c for i, c in enumerate(cnts) if i in ballIdx]


#### PARTICLE FILTER ######
N = 1000

import ATMANpy.funcs.particle_funcs_vectorized as pf

def initialize_particles(gFrm, courtLines, N = 1000, fps = 25):
    return pf.initialize_particles(gFrm, courtLines, N = N, fps = fps)
    
def decide_particle_propagation(particles, frm, 
                                topplayer = None, botplayer = None, 
                                closeHistories = None, modelHistories = None,
                                chamfer_threshold = None,
                                normal_propagation = None, close_propagation = None, switch_probability = None):
    
    if topplayer is None or botplayer is None:
        chamferMask = BoolFrame(np.zeros((frm.rows, frm.cols), dtype = 'uint8'))
    else:
        chamferMask = pf.chamfer_distance(frm, topplayer, botplayer,
                                          threshold = chamfer_threshold) 
        ### note: motion model found based on last periods frame, which is also where particles S_in come from!
    
    propagation_models, closeBools, modelBools = pf.get_motion_model(particles, chamferMask,
                                                     closeHistories = closeHistories,
                                                     modelHistories = modelHistories,
                                                     normal_propagation = normal_propagation, 
                                                     close_propagation = close_propagation, 
                                                     switch_probability = switch_probability)    

    return propagation_models, closeBools, modelBools


def update_particles(particles, propagation_models, ballContours):
        
    mDict, wDict = pf.get_means_and_weights(particles, propagation_models, ballContours) ### means_and_weights updates last periods particles, and compares to evidence in this period!
    
    S_out, I_out, J_out = pf.resample(mDict, wDict)
    
    # print(f'Evidence Based Filtering Rate: {round(J_out.sum() / len(J_out), 3)}')

    return S_out, I_out

#%% BALL POST PROCESSING ####

def find_next_item(stream_features, 
                   startNum = None,
                   item_to_find = 'ball'):
    if startNum is None:
        startNum = min(stream_features.keys())
    
    for frmNum, featDict in stream_features.items():
        if frmNum < startNum:
            continue
        
        if featDict[item_to_find] is None:
            continue
        
        return frmNum, featDict[item_to_find]
    
    return None, None ### if no next ball is found.


def postprocess_ball_flight(stream_features):

    ### first pass (from back to front): kitagawa smoothing
    stream_features_smooth = pf.kitagawa_smooth(stream_features)
    
    ### second pass: determine if ball contour passes threshold
    for frmNum, featDict in stream_features_smooth.items():
        if not featDict['ballParticles'] is None:
            goodContour = pf.choose_likely_ball_contour(featDict['ballParticles'], 
                                                        featDict['ballContours'],
                                                        verbose = False)
        else:
            goodContour = None
        featDict['ballDetected'] = True if goodContour is not None else False
        featDict['ball'] = goodContour
        
        
    ### third pass: interpolate contours
    currCtr = None
    for frmNum, featDict in stream_features_smooth.items():    
        if featDict['ball'] is not None:
            currCtr = featDict['ball']
            last_frame_with_ball = frmNum
            continue
        
        if currCtr is None:
            continue
        
        if featDict['ball'] is None:
            ### determine the next found ball position
            next_frame_with_ball, nextCtr = find_next_item(stream_features_smooth, 
                                                           startNum = frmNum, 
                                                           item_to_find = 'ball')
            
            if not nextCtr is None:
                time_from_ball = frmNum - last_frame_with_ball
                time_to_ball = next_frame_with_ball - frmNum
            
                itplBall = pf.interpolate_ball_features(currCtr, nextCtr, 
                                                        frames_before = time_from_ball,
                                                        frames_after = time_to_ball)
        
                featDict['ball'] = itplBall
                featDict['ballDetected'] = False
            else:
                featDict['ball'] = None
                featDict['ballDetected'] = False
            
    return stream_features_smooth

