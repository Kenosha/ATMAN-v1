# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 17:12:13 2022

@author: gianl
"""

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.utils as atu
from ATMANpy.Points import Point, SafePoint
from ATMANpy.Lines import Line, Lines, CourtLines, Region
from ATMANpy.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame
from ATMANpy.Contours import Contour

# image operation
import cv2 as cv
import numpy as np

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################

######## GET SCHEMA ########
from ATMANpy.high_level_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)


###### DEFINE TRACK PROCEDURE ########
from ATMANpy.high_level_funcs import find_homography_225_only, update_homography
from ATMANpy.high_level_funcs import find_courtlines, find_netlines

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

from ATMANpy.player_funcs import get_burn_in_mask
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

from ATMANpy.player_funcs import get_initial_ROIs
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

from ATMANpy.high_level_funcs import COURT_LINE_WIDTH, WHITE_PIXEL_THRESHOLD
from ATMANpy.player_funcs import find_player_contour


def find_player(frame, courtLines, netLines, H, roiMask = None, verbose = False):
    
    ### get the burned in mask for this specific homography
    burn_foreMask = match_specific_burn_in_func(frame, H,
                                    normalized_deviation_threshold = NORMALIZED_DEVIATION_THRESHOLD, 
                                    channel_weights = BURN_IN_HSV_CHANNEL_WEIGHTS, 
                                    verbose = verbose)
    
    playerContour = find_player_contour(frame, burn_foreMask, roiMask)
    
    return playerContour

#%% Motion Segmentation


def get_motion_mask(frameNum, diffList = [4, 6, 8]):
    myFrame = GreyFrame.from_video(cap, frameNum)
    
    frameList = [frameNum - d for d in diffList] + [frameNum + d for d in diffList]
    gFrames = [GreyFrame.from_video(cap, fnum) for fnum in frameList]
    
    def isFG(myFrame, diffFrame):
        THRESH = 10
        return myFrame - diffFrame > THRESH
    
    outFrame = BoolFrame(myFrame.zero().array + 1)
    for diffFrame in gFrames:
        outFrame = outFrame & isFG(myFrame, diffFrame)
   
    return outFrame

frameNum = 11019

cFrm = BGRFrame.from_video(cap, frameNum)
hFrm = HSVFrame.from_video(cap, frameNum)
gFrm = GreyFrame.from_video(cap, frameNum)

### find players ###

### Find Court ###
courtLines, netLines, H, config, trackFlag = track_courtlines(gFrm, None, None)

### Project Regions of Interest
projROI_lower, projROI_upper = schema_specific_roi_projection_func(H)
roiMask_lower, roiMask_upper = hFrm.within_region(projROI_lower), hFrm.within_region(projROI_upper)

### Top Player, with updated averages
topplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_upper)

### Bottom Player, with updated averages
botplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_lower)



motMask = get_motion_mask(frameNum)

def find_ball_candidate_mask(cFrm, motMask, topPlayerContour, botPlayerContour):
    # cFrm.mask(motMask).show()
    
    def cntMask(frm, cnt):
        ### Dilated player mask
        dil_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 10))
        return frm.within_contour(cnt).dilate(dil_kernel)
    playerMask = cntMask(cFrm, topPlayerContour) | cntMask(cFrm, botPlayerContour)
    # cFrm.mask(playerMask).show()
    
    ### Opened Motion Mask
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    motMask = motMask.open(open_kernel)
    # cFrm.mask(motMask).show()
    
    ### Player and Motion Mask
    intMask = ~playerMask & motMask
    # cFrm.mask(intMask).show()
    
    return intMask
    
### finding alpha-criterion ###

intMask = find_ball_candidate_mask(cFrm, motMask, topplayerRegion, botplayerRegion)
cnts, _ = intMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = Contour(cnts[4])

def find_alpha_and_std_and_semiaxes(cnt, gFrm, maskFrm):
    PIXEL_MARGIN = 3
    ZOOM_FACTOR = 10
    
    cntRegion = cnt.enclosing_region(margin = PIXEL_MARGIN)
    
    foreBlob_large = gFrm.crop(cntRegion).zoom(ZOOM_FACTOR)
    foreBool_large = maskFrm.crop(cntRegion).zoom(ZOOM_FACTOR)
    
    cnts_large, _ = foreBool_large.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    if len(cnts_large) == 1:
        cnt_large = Contour(cnts_large[0])
    else:
        cnt_large = sorted([Contour(c) for c in cnts_large], key = lambda c: c.area())[-1] #take largest
    
    from ATMANpy.ball_funcs import get_ellipse, get_ellipse_mask, get_sobel_grads, get_normal_grads, calculate_alpha
    
    centerCoords, semiAxes, theta = get_ellipse(cnt_large)
    ellipseMask = get_ellipse_mask(foreBlob_large, centerCoords, semiAxes, theta)
    
    sobGrads = get_sobel_grads(foreBlob_large)
    normGrads = get_normal_grads(foreBlob_large, centerCoords)
    
    alpha, std = calculate_alpha(sobGrads, normGrads, ellipseMask)
    return alpha, std, semiAxes ### we also return the semiAxes, since they will be params


import pandas as pd
def make_feature_frame(cnts):
    
    cntSers = []
    
    for cnt_index, cnt_points in enumerate(cnts):
        cnt = Contour(cnt_points)
        
        ### COLLECT SVM FEATURES
        cntDict = {}
            
        # Alpha (and maybe add std?) and semiaxes
        try:
            alpha, std, semiAxes = find_alpha_and_std_and_semiaxes(cnt, gFrm, intMask)
            semiMaj, semiMin = semiAxes
        except:
            alpha, std, semiMaj, semiMin = np.nan, np.nan, np.nan, np.nan #occasionally, a contour has no ellipse that can be fitted.
        finally:
            cntDict['Alpha'] = alpha
            cntDict['AlphaSTD'] = std
            cntDict['SemiMaj'] = semiMaj
            cntDict['SemiMin'] = semiMin
        
        # Blob Center Location
        centPt = cnt.center()
        cntDict['Cx'] = centPt.x
        cntDict['Cy'] = centPt.y
        
        # HSV Channels
        mean, std = hFrm.mean_std(mask = hFrm.within_contour(cnt))
        h, s, v = mean
        cntDict['H'] = h
        cntDict['S'] = s
        cntDict['V'] = v
        
        cntSer = pd.Series(cntDict, name = cnt_index)
        cntSers.append(cntSer)
        
    cntFrm = pd.concat(cntSers, axis = 1).T
    return cntFrm

cntFrm = make_feature_frame(cnts)



