# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:27:15 2022

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

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################

#### READ IN THE BALL TRACKING SVM ######
import os
ballDict = atu.get_folder_struct('ball_svm')
fitPath = os.path.join(ballDict['main'], VIDEO_ID, 'fitted_svms')
svm_name = 'svm_dict_2022-04-13'

svmPath = os.path.join(fitPath, svm_name + '.pickle')

import pickle

with open(svmPath, 'rb') as f:
    svmDict = pickle.load(f)
    
    
# clf.predict([[1] * 8]) ## test if it works

######## GET SCHEMA ########
from ATMANpy.funcs.high_level_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)


###### DEFINE TRACK PROCEDURE ########
from ATMANpy.funcs.high_level_funcs import find_homography_225_only, update_homography
from ATMANpy.funcs.high_level_funcs import find_courtlines, find_netlines

def track_courtlines(gFrm, prev_H, prev_config, schemaLines = schemaLines):
    if prev_H is None:
        res = find_homography_225_only(gFrm)
        outFlag = 'Found'
    else:
        # print('Courtlines found, updating homography...')
        res = update_homography(gFrm, prev_H, prev_config)
        outFlag = 'Tracked'
        
    if not res is None:
        H, config = res
        courtLines = find_courtlines(H, schemaLines)
        netLines = find_netlines(gFrm, courtLines)       
    
        # outFrm = gFrm.to_BGR().put(courtLines, thickness = 3)
        # outFrm = outFrm.put(netLines, color = (255, 0, 0), thickness = 3)
    else:
        outFlag = 'Not Found'
        courtLines, netLines = None, None
        H, config = None, None
        
    return courtLines, netLines, H, config, outFlag

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

# schemaFrm.to_BGR().put(ROI_upper, color = (0, 0, 255), thickness = 2).put(ROI_lower, color = (0, 255, 0), thickness = 2).show()
##############

#### now, we get a frame and an h
verbose = False
COURT_DILATION = 1.5
NET_DILATION = 2.5
ROI_DILATION = 1

from ATMANpy.funcs.high_level_funcs import COURT_LINE_WIDTH, WHITE_PIXEL_THRESHOLD
from ATMANpy.funcs.player_funcs import find_player_contour


def find_player(frame, courtLines, netLines, H, roiMask = None, verbose = False):
    
    ### get the burned in mask for this specific homography
    burn_foreMask = match_specific_burn_in_func(frame, H,
                                    normalized_deviation_threshold = NORMALIZED_DEVIATION_THRESHOLD, 
                                    channel_weights = BURN_IN_HSV_CHANNEL_WEIGHTS, 
                                    verbose = verbose)
    
    playerRegion = find_player_contour(frame, burn_foreMask, roiMask)
    
    return playerRegion


################ ################

######### FIND BALL CANDIDATES ######

MAX_NUMBER_OF_CONTOURS = 40 # if there are too many contours, likely the camera was zooming or panning.
                            # this causes many call candidates (due to frame differencing).
                            # these spurious candidates we don't want in our training data
                            # also, these frames take ages to load due to labelling.

from ATMANpy.funcs.ball_funcs import get_motion_mask_from_frames, find_ball_candidate_mask, make_feature_frame

def find_ball_contours(gFrm, diffFrames, svmDict, topplayerRegion, botplayerRegion,
                       cFrm = None, hFrm = None):
    
    clf, features, mean, std = svmDict['svm'], svmDict['features'], svmDict['mean'], svmDict['std']
    
    motMask = get_motion_mask_from_frames(gFrm, diffFrames)
    intMask = find_ball_candidate_mask(cFrm, motMask, topplayerRegion, botplayerRegion)
    
    cnts, _ = intMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = [Contour(c) for c in cnts if len(c) > 2] ### need at least three points for a contour
    
    if not 0 < len(cnts) < MAX_NUMBER_OF_CONTOURS:
        return None
        
    cntFrm = make_feature_frame(cnts, hFrm, gFrm, intMask).dropna()
    cntFrm = (cntFrm - mean) / std

    cntFrm['isBall'] = clf.predict(cntFrm[features].values)
    
    ballIdx = list(cntFrm[cntFrm.isBall == 1].index)
    
    return [c for i, c in enumerate(cnts) if i in ballIdx]

#############################
START_FRAME = 11840
STOP_FRAME = 11880


H = None
config = None
    
#### init cache ###
FRAME_DIFF_LIST = [4, 6, 8]
CACHE_LENGTH = max(FRAME_DIFF_LIST) * 2 + 1 # we have a frame diff of 8 on both sides, so we need 8 * 2 + 1 = 17
CURR_FRAME_IDX = CACHE_LENGTH // 2 ### the middle frame
FRAME_DIFF_INDICES = [CURR_FRAME_IDX - f for f in reversed(FRAME_DIFF_LIST)] + [CURR_FRAME_IDX + f for f in FRAME_DIFF_LIST]
frmCache = []

### init video stream ###
from ATMANpy.funcs.cv_funcs import video_stream
gen = video_stream(cap, start_frame = START_FRAME - CACHE_LENGTH, stop_frame = STOP_FRAME)
frmNum, rawArr = next(gen) ### this is needed to "prime" the generator
######

frmDict = {}

while True:
    latestFrame = BGRFrame(rawArr)    
    ### cache frames, so we can frame difference 
    ### without having to re-generate the frames from the video capture
    frmCache.append(latestFrame)
    if len(frmCache) < 17:
        continue
    if len(frmCache) > 17:
        print(f'Cache FUBAR, {len(frmCache)} items') ### this happens if the cache drop at the end fails
        break
    
    cFrm = frmCache[CURR_FRAME_IDX]
    gFrm = cFrm.to_grey()
    hFrm = cFrm.to_HSV()
    
    diffFrames = [frmCache[idx].to_grey() for idx in FRAME_DIFF_INDICES]
    
    try:
        courtLines, netLines, H, config, trackFlag = track_courtlines(gFrm, H, config)
    except Exception as e:
        print('Exception in track_courtlines:')
        print(e)
        courtLines = None
    
    if trackFlag == 'Not Found':
        sendFrm = cFrm.put('No Lines!', thickness = 2, color = (0, 0, 255))
        

    elif trackFlag == 'Found':
        
        ### Project Regions of Interest
        projROI_lower, projROI_upper = schema_specific_roi_projection_func(H)
        roiMask_lower, roiMask_upper = hFrm.within_region(projROI_lower), hFrm.within_region(projROI_upper)

        ### Top Player, with updated averages
        topplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_upper)
        
        ### Bottom Player, with updated averages
        botplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_lower)
        
        ### find ball
        ballContours = find_ball_contours(gFrm, diffFrames, svmDict, topplayerRegion, botplayerRegion,
                                          cFrm = cFrm, hFrm = hFrm)
        
        sendFrm = cFrm.put(topplayerRegion, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(botplayerRegion, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
        
        ### save frames in dict. Not needed usually, but for particle filter testing
        frmDict[frmNum] = [gFrm, topplayerRegion, botplayerRegion, ballContours]
        ###
        
    elif trackFlag == 'Tracked':
        ### If the court was tracked, we don't change the projected ROI, 
        #   and we don't change the color averages and standard deviations.
        
        topplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_upper)
        botplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_lower)
        ballContours = find_ball_contours(gFrm, diffFrames, svmDict, topplayerRegion, botplayerRegion,
                                          cFrm = cFrm, hFrm = hFrm)
        
        sendFrm = cFrm.put(topplayerRegion, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(botplayerRegion, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
        
        ### save frames in dict. Not needed usually, but for particle filter testing
        frmDict[frmNum] = [gFrm, topplayerRegion, botplayerRegion, ballContours]
        ###
    
    sendArr = sendFrm.array  
    try:
        frmNum, rawArr = gen.send(sendArr)
    except:
        cv.destroyAllWindows()
        break 
    
    ### cache drop  
    frmCache = frmCache[1:] # drop oldest item
    
else:
    cv.destroyAllWindows()


######### Particle Filter

sendFrm.show()

### Parameters ###

N = 1000

import ATMANpy.funcs.particle_funcs as pf


chamferThresh = 50
def decide_particle_propagation(particles, frm, 
                                topplayer = None, botplayer = None, 
                                closeHistories = None, modelHistories = None):
    
    if topplayer is None or botplayer is None:
        chamferMask = BoolFrame(np.zeros((frm.rows, frm.cols), dtype = 'uint8'))
    else:
        chamferMask = pf.chamfer_distance(frm, topplayer, botplayer,
                                          threshold = chamferThresh) 
        ### note: motion model found based on last periods frame, which is also where particles S_in come from!
    
    propagation_models, closeBools, modelBools = pf.get_motion_model(particles, chamferMask,
                                                     closeHistories = closeHistories,
                                                     modelHistories = modelHistories)    

    return propagation_models, closeBools, modelBools

def zero_weight_for_external_particles(mDict, wDict, rows = 1080, cols = 1920):
    ### quick fix for particle dispersion issue
    newweights = []
    for m, w in zip(mDict[0], wDict[0]):
        x, y = np.asarray(m[:2]).squeeze()
        if (0 < x < cols) and (0 < y < rows):
            newweights.append(w)
        else:
            newweights.append(0)
    wDict[0] = newweights
    return wDict

def update_particles(particles, propagation_models, ballContours):
        
    mDict, wDict = pf.get_means_and_weights(particles, propagation_models, ballContours) ### means_and_weights updates last periods particles, and compares to evidence in this period!
    wDict = zero_weight_for_external_particles(mDict, wDict)
    
    S_out, I_out, J_out = pf.resample(mDict, wDict)
    
    # print(f'Evidence Based Filtering Rate: {round(J_out.sum() / len(J_out), 3)}')

    return S_out, I_out



###

START_FRAME = 11853

frm1, tp1, bt1, ct1 = frmDict[START_FRAME]
frm2, tp2, bt2, ct2 = frmDict[START_FRAME + 1]
frm3, tp3, bt3, ct3 = frmDict[START_FRAME + 2]
frm4, tp4, bt4, ct4 = frmDict[START_FRAME + 3]

### init

topplayerRegion, botplayerRegion = None, None
particles, idxHistories, closeHistories, modelHistories = None, None, None, None


def cycle(frmNum, particles = None, 
          idxHistories = None, closeHistories = None, modelHistories = None):
    frm_bef, tp_bef, bt_bef, ct_bef = frmDict[frmNum - 1]
    frm_aft, tp_aft, bt_aft, ct_aft = frmDict[frmNum]
    
    if particles is None:
        particles = pf.initialize_particles(frm_bef, courtLines, N = N)    
        
    ##
    propagation_models, closeBools, modelBools = decide_particle_propagation(particles, frm_bef, 
                            topplayer = tp_bef, botplayer = bt_bef, 
                            closeHistories = closeHistories, modelHistories = modelHistories)
    print(f'Propagated with switched model: {modelBools.sum()}')
    
    
    ### find ball particles
    particles, particleIdxs = update_particles(particles, propagation_models, ct_aft)
    idxHistories, closeHistories, modelHistories = pf.update_histories(idxHistories, closeHistories, modelHistories, 
                                                        particleIdxs, closeBools, modelBools)
    
    pPts = pf.particles_to_points(particles, particles_to_produce = 100)
    
    showFrm = frm_aft.to_BGR().put(pPts, thickness = 1, color = (0, 255, 0))
    showFrm.show()

    return particles, idxHistories, closeHistories, modelHistories


frmNum = START_FRAME
particles, idxHistories, closeHistories, modelHistories = cycle(frmNum)

frmNum = START_FRAME + 1
particles, idxHistories, closeHistories, modelHistories = cycle(frmNum, particles = particles,
                                                                idxHistories = idxHistories, closeHistories = closeHistories, modelHistories = modelHistories)



# S = particles
# As = propagation_models
# ballContours = ct_aft











