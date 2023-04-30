# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:15:55 2022

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

from copy import deepcopy

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
    if prev_H is None or prev_config is None:
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

from functools import partial

from ATMANpy.funcs.player_funcs import get_burn_in_mask
match_specific_burn_in_func = partial(get_burn_in_mask, medFrm, stdFrm,
                                      normalized_deviation_threshold = NORMALIZED_DEVIATION_THRESHOLD,
                                      channel_weights = BURN_IN_HSV_CHANNEL_WEIGHTS) ### these are always the same!

MINIMUM_PROPORTION = 0.0125
MINIMUM_STEP_GAIN = 0.002
from ATMANpy.funcs.player_funcs import get_burn_in_mask_iterative
match_specific_iterative_burn_in_func = partial(get_burn_in_mask_iterative, medFrm, stdFrm,
                                                minimum_proportion = MINIMUM_PROPORTION,
                                                minimum_step_gain = MINIMUM_STEP_GAIN) ### these are always the same!



H_THRESHOLD = 5
S_THRESHOLD = 10
V_THRESHOLD_LOWER = 0.75
V_THRESHOLD_UPPER = 0.97
from ATMANpy.funcs.player_funcs import get_shadow_mask
match_specific_shadow_func = partial(get_shadow_mask, medFrm, 
                                     h_threshold = H_THRESHOLD, s_threshold = S_THRESHOLD, 
                                     v_threshold_lower = V_THRESHOLD_LOWER, 
                                     v_threshold_upper = V_THRESHOLD_UPPER)

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

def get_projected_ROI(ROI, H):
    return ROI.transform('apply_homography', H).transform('round_points')

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
from ATMANpy.funcs.player_funcs import find_player_contour, find_player_contour_with_overlap



def find_player(frame, courtLines, H, roiMask = None, remove_shadow = True,
                verbose = False):
    
    ### get the burned in mask for this specific homography
    burn_foreMask = match_specific_burn_in_func(frame, H, verbose = verbose)
    
    if remove_shadow:
        burn_shadowMask = match_specific_shadow_func(frame, H, verbose = verbose)
    else:
        burn_shadowMask = None
        
    playerRegion = find_player_contour(frame, burn_foreMask, roiMask, 
                                       shadowMask = burn_shadowMask)
    
    return playerRegion


def adaptive_player_ROI(playerContour, initialROI, H, hFrm,
                        player_tracking_scale_factor = 1.2,
                        min_bounding_height = 100,
                        min_bounding_width = 40,
                        halfcourtLine = None, keep_below_or_above = None):
    
    if playerContour is None:
        initial_projROI = get_projected_ROI(initialROI, H)
    
    if playerContour is None:
        roiMask = hFrm.within_region(initial_projROI)
    else:
        enclReg = playerContour.enclosing_region()
        updated_projROI = enclReg.scale_equal(player_tracking_scale_factor)
        updated_projROI = updated_projROI.enlarge(to_min_height = min_bounding_height, to_min_width = min_bounding_width)
        roiMask = hFrm.within_region(updated_projROI)
        
    # if not halfcourtROI is None:
    #     ### to prevent adaptive ROIs to spill over into the other player's court
    #     ### we can impose a half court mask
    #     proj_halfcourtROI = get_projected_ROI(halfcourtROI, H)
    #     halfcourtMask = hFrm.within_region(proj_halfcourtROI)
    #     roiMask = roiMask & halfcourtMask
        
    if not halfcourtLine is None:
        ### to prevent adaptive ROIs to spill over into the other player's court
        ### we can impose a half court mask
        if not keep_below_or_above in ['below', 'above']:
            raise Exception('When specifying halfcourt boundary in player selection, specify whether to keep everything above or below.')
        else:
            proj_halfcourtLine = halfcourtLine.apply_homography(H).round_points()
            halfMask = getattr(hFrm, keep_below_or_above)(proj_halfcourtLine)
            roiMask = roiMask & halfMask
        
    return roiMask


INITIAL_WIDTH_FACTOR = 1
INITIAL_DEPTH_FACTOR = 1
ADDITIONAL_DEPTH_FACTOR = 0.3

from ATMANpy.funcs.player_funcs import get_service_ROIs
initial_ROI_lower, initial_ROI_upper = get_service_ROIs(schemaLines,
                                                        width_factor = INITIAL_WIDTH_FACTOR, 
                                                        depth_factor = INITIAL_DEPTH_FACTOR, 
                                                        add_depth_factor_bottom = ADDITIONAL_DEPTH_FACTOR)

from ATMANpy.funcs.player_funcs import get_half_court_Lines
# halfcourt_lower, halfcourt_upper = get_half_court_ROIs(schemaLines)
halfLine_lower, halfLine_upper = get_half_court_Lines(schemaLines)


PLAYER_TRACKING_SCALE_FACTOR = 1.5
BOT_MIN_BOUNDING_ROI_HEIGHT = 200 ### bounding region has to be 40 pixels wide at least...
BOT_MIN_BOUNDING_ROI_WIDTH  = 80 ## ... and 100 pixels tall
TOP_MIN_BOUNDING_ROI_HEIGHT = 150 ## reduce values a bit for top because player is farther away.
TOP_MIN_BOUNDING_ROI_WIDTH  = 60
    
def get_adaptive_ROIs(topPlayer, botPlayer, H, hFrm):
    top_ROI = adaptive_player_ROI(topPlayer, initial_ROI_upper, H, hFrm,
                                  player_tracking_scale_factor = PLAYER_TRACKING_SCALE_FACTOR,
                                  min_bounding_height = TOP_MIN_BOUNDING_ROI_HEIGHT,
                                  min_bounding_width = TOP_MIN_BOUNDING_ROI_WIDTH,
                                  halfcourtLine = halfLine_upper, keep_below_or_above = 'above')
    
    bot_ROI = adaptive_player_ROI(botPlayer, initial_ROI_lower, H, hFrm,
                                  player_tracking_scale_factor = PLAYER_TRACKING_SCALE_FACTOR,
                                  min_bounding_height = BOT_MIN_BOUNDING_ROI_HEIGHT,
                                  min_bounding_width = BOT_MIN_BOUNDING_ROI_WIDTH,
                                  halfcourtLine = halfLine_lower, keep_below_or_above = 'below')
    
    return top_ROI, bot_ROI

def find_both_players(frame, courtLines, H, remove_shadow = True,
                      roiMask_top = None, roiMask_bot = None,
                      verbose = False):
    
    ### get the burned in mask for this specific homography
    burn_foreMask = match_specific_burn_in_func(frame, H, verbose = verbose)
    
    if remove_shadow:
        burn_shadowMask = match_specific_shadow_func(frame, H, verbose = verbose)
    else:
        burn_shadowMask = None
        
    topPlayer = find_player_contour(frame, burn_foreMask, roiMask_top, 
                                    shadowMask = burn_shadowMask)
    
    botPlayer = find_player_contour(frame, burn_foreMask, roiMask_bot, 
                                    shadowMask = burn_shadowMask)

    return topPlayer, botPlayer


def find_both_players_smart(frame, courtLines, H, remove_shadow = True,
                            playerTop_prev = None, playerBot_prev = None,
                            roiMask_top = None, roiMask_bot = None,
                            iterative_burn_in = False,
                            verbose = False):
    
    ### get the burned in mask for this specific homography
    if iterative_burn_in:
        burn_foreMask = match_specific_iterative_burn_in_func(frame, H, verbose = verbose)
    else:
        burn_foreMask = match_specific_burn_in_func(frame, H, verbose = verbose)
    
    if remove_shadow:
        burn_shadowMask = match_specific_shadow_func(frame, H, verbose = verbose)
    else:
        burn_shadowMask = None
        
    topPlayer = find_player_contour_with_overlap(frame, burn_foreMask, roiMask_top, playerTop_prev,
                                                 shadowMask = burn_shadowMask)
    
    botPlayer = find_player_contour_with_overlap(frame, burn_foreMask, roiMask_bot, playerBot_prev,
                                                 shadowMask = burn_shadowMask)

    return topPlayer, botPlayer



def detendrilize_player(frame, playerCnt_bef, playerCnt_aft, kernel = (5, 5)):

    if playerCnt_bef is None:
        return playerCnt_aft
    
    playerMask_bef = frame.within_contour(playerCnt_bef)
    playerMask_aft = frame.within_contour(playerCnt_aft)
    
    diffBool = playerMask_aft & ~playerMask_bef

    tendrilBool = diffBool.open(kernel) ### whatever survives this opening is a tendril, or an 
                                        ### unnatural increase in the surface of the player.
                                        ### SADLY, THE OPENING STEP IS QUITE INEFFECTIVE.....
                                        ### NO MATTER THE KERNEL SIZE, IT DOESN'T REMOVE MUCH BEYOND NOISE.
                                        
    newMask = playerMask_aft & ~tendrilBool
    
    newContours = newMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    assert len(newContours) == 1
    return Contour(newContours[0])



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

def initialize_particles(gFrm, courtLines, N = N, fps = 25):
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

def find_last_item(stream_features, 
                   startNum = None,
                   item_to_find = 'ball'):
    if startNum is None:
        startNum = max(stream_features.keys())
    
    for frmNum in sorted(list(stream_features.keys()), reverse = True):
                    
        if frmNum > startNum:
            continue
        
        featDict = stream_features[frmNum]      
               
        if featDict[item_to_find] is None:
            continue
        
        return frmNum, featDict[item_to_find]
    
    return None, None ### if no next ball is found.

def find_next_N_items(stream_features, N = 3,
                      startNum = None, 
                      item_to_find = 'ball'):
    frmNum, item = find_next_item(stream_features, 
                                  startNum = startNum,
                                  item_to_find = item_to_find)
    
    if not frmNum is None:
        outList = [item]
        item_count = 1
        frame_count = frmNum
        while item_count < N:
            item_count += 1
            frame_count += 1
            
            try: 
                add_item = stream_features[frame_count][item_to_find]
            except KeyError:
                add_item = None
            outList.append(add_item)
        
        return frmNum, outList
    else:
        return frmNum, [None] * N
    
def find_last_N_items(stream_features, N = 3,
                      startNum = None, 
                      item_to_find = 'ball'):
    frmNum, item = find_last_item(stream_features, 
                                  startNum = startNum,
                                  item_to_find = item_to_find)
    
    if not frmNum is None:
        outList = [item]
        item_count = 1
        frame_count = frmNum
        while item_count < N:
            item_count += 1
            frame_count -= 1
            
            try: 
                add_item = stream_features[frame_count][item_to_find]
            except KeyError:
                add_item = None
            outList.append(add_item)
        
        return frmNum, outList
    else:
        return frmNum, [None] * N
    


def calculate_speeds_and_acceleration(cnt1, cnt2, cnt3):
    ### given three subsequent contours, calculate their speed and acceleration per frame
    try:
        speed1 = cnt2.center() - cnt1.center()
    except:
        speed1 = None
    
    try:
        speed2 = cnt3.center() - cnt2.center()
    except:
        speed2 = None
    
    try:
        accel = speed2 - speed1
    except:
        accel = None
        
    return speed1, speed2, accel

def smoothe_and_choose_ball_contour(stream_features):
    
    LIKELIHOOD_THRESHOLD = 10 ** (-15)
    
    ### first pass (from back to front): kitagawa smoothing
    stream_features_smooth = pf.kitagawa_smooth(stream_features)
    
    ### second pass: determine if ball contour passes threshold
    for frmNum, featDict in stream_features_smooth.items():
        if not featDict['ballParticles'] is None:
            goodContour = pf.choose_likely_ball_contour(featDict['ballParticles'], 
                                                        featDict['ballContours'],
                                                        verbose = False,
                                                        return_likelihood = False,
                                                        l_threshold = LIKELIHOOD_THRESHOLD)
        else:
            goodContour = None
        featDict['ballDetected'] = True if goodContour is not None else False
        featDict['ball'] = goodContour
        
    return stream_features_smooth
        
def smoothe_and_choose_ball_contour_with_likelihood(stream_features):
    
    LIKELIHOOD_THRESHOLD = 10 ** (-10)
    
    ### first pass (from back to front): kitagawa smoothing
    stream_features_smooth = pf.kitagawa_smooth(stream_features)
    
    ### second pass: determine if ball contour passes threshold
    for frmNum, featDict in stream_features_smooth.items():
        if not featDict['ballParticles'] is None:
            goodContour, max_likelihood, threshold = pf.choose_likely_ball_contour(featDict['ballParticles'], 
                                                                                  featDict['ballContours'],
                                                                                  verbose = False,
                                                                                  return_likelihood = True,
                                                                                  l_threshold = LIKELIHOOD_THRESHOLD)
        else:
            goodContour, max_likelihood, threshold = None, None, None
        
        featDict['ballDetected'] = True if goodContour is not None else False
        featDict['ball'] = goodContour
        featDict['ballLikelihood'] = max_likelihood
        featDict['ballThreshold'] = threshold
        
    return stream_features_smooth

def interpolate_ball_contours(stream_features):
    stream_features_interpolated = deepcopy(stream_features)
    
    ### third pass: interpolate contours
    currCtr = None
    for frmNum, featDict in stream_features_interpolated.items():    
        if featDict['ball'] is not None:
            currCtr = featDict['ball']
            last_frame_with_ball = frmNum
            continue
        
        if currCtr is None:
            continue
        
        if featDict['ball'] is None:
            ### determine the next found ball position
            next_frame_with_ball, nextCtr = find_next_item(stream_features_interpolated, 
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
            
    return stream_features_interpolated


def smart_interpolate_ball_contours(stream_features, verbose = False):
    stream_features_interpolated = deepcopy(stream_features)
    
    for frmNum, featDict in stream_features_interpolated.items():    
        
        
        if featDict['ball'] is None:
            
            ### determine the last found ball position
            last_frame_with_ball, last_three_ctrs = find_last_N_items(stream_features_interpolated, N = 3,
                                                                      startNum = frmNum, 
                                                                      item_to_find = 'ball')
            
            currCtr, currCtr_1, currCtr_2 = last_three_ctrs
                
            if currCtr is None:
                continue
                
            ### determine the next found ball position
            next_frame_with_ball, next_three_ctrs = find_next_N_items(stream_features_interpolated, N = 3,
                                                                      startNum = frmNum, 
                                                                      item_to_find = 'ball')
            
            nextCtr, nextCtr_1, nextCtr_2 = next_three_ctrs
            
            ### find speed and acceleration after the next ball
            if not nextCtr is None:
                
                _, currSpeed, currAccel = calculate_speeds_and_acceleration(currCtr_2, currCtr_1, currCtr)
                
                nextSpeed, _, nextAccel = calculate_speeds_and_acceleration(nextCtr, nextCtr_1, nextCtr_2)
                
                time_from_ball = frmNum - last_frame_with_ball
                time_to_ball = next_frame_with_ball - frmNum
                
                if currSpeed is not None and nextSpeed is not None:
                    itplBall = pf.interpolate_ball_features_smart_circle(currCtr, nextCtr,
                                                                         speed_before = currSpeed, 
                                                                         speed_after = nextSpeed,
                                                                         accel_before = currAccel,
                                                                         accel_after = nextAccel,
                                                                         frames_before = time_from_ball,
                                                                         frames_after = time_to_ball,
                                                                         verbose = verbose)
                else:
                    itplBall = pf.interpolate_ball_features_circle(currCtr, nextCtr, 
                                                                   frames_before = time_from_ball,
                                                                   frames_after = time_to_ball)
        
                featDict['ball'] = itplBall
                # featDict['ballDetected'] = False
            else:
                featDict['ball'] = None
                # featDict['ballDetected'] = False

            
    return stream_features_interpolated



