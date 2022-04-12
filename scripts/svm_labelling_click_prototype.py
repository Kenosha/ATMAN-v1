# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 00:35:26 2022

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

import pandas as pd

from random import randrange

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################

total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)

#%% Finding Lines and Players

######## GET SCHEMA ########
from ATMANpy.high_level_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)

##### TRACKING COURTLINES ####
from ATMANpy.high_level_funcs import find_homography_225_only
from ATMANpy.high_level_funcs import find_courtlines, find_netlines

def find_court(gFrm, schemaLines = schemaLines):
    try:
        res = find_homography_225_only(gFrm)
    except:
        return None
    
    if not res is None:
        H, config = res
        courtLines = find_courtlines(H, schemaLines)
        netLines = find_netlines(gFrm, courtLines)   
        return courtLines, netLines, H, config
    else:
        return None

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

########## GET PLAYER REGIONS OF INTEREST ON SCHEMA ##########
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

###### SPECIFY FIND PLAYER FUNCTION #####
verbose = False
COURT_DILATION = 1.5
NET_DILATION = 2.5
ROI_DILATION = 1

from ATMANpy.player_funcs import find_player_contour
def find_player(frame, courtLines, netLines, H, roiMask = None, verbose = False):
    
    ### get the burned in mask for this specific homography
    burn_foreMask = match_specific_burn_in_func(frame, H,
                                    normalized_deviation_threshold = NORMALIZED_DEVIATION_THRESHOLD, 
                                    channel_weights = BURN_IN_HSV_CHANNEL_WEIGHTS, 
                                    verbose = verbose)
    
    playerContour = find_player_contour(frame, burn_foreMask, roiMask, verbose = verbose)
    
    return playerContour

#%%% 

MAX_NUMBER_OF_CONTOURS = 40 # if there are too many contours, likely the camera was zooming or panning.
                            # this causes many call candidates (due to frame differencing).
                            # these spurious candidates we don't want in our training data
                            # also, these frames take ages to load due to labelling.

from ATMANpy.ball_funcs import get_motion_mask, find_ball_candidate_mask, make_feature_frame


clickPt = None
def select_point(event, x, y, flags, param):
	# grab references to the global variables
	global clickPt
	# if the left mouse button was clicked, record the (x, y) coordinates
	if event == cv.EVENT_LBUTTONDOWN:
		clickPt = (x, y)
        
def label_ball_candidate(clickPt, contourFrm: pd.DataFrame, max_range_threshold = 100):
    outFrm = contourFrm.copy()
    outFrm['isBall'] = 0

    if clickPt is None:
        print('No click detected, no ball.')
    else:
        dFrm = (contourFrm[['Cx', 'Cy']] - clickPt) ** 2
        distance = (dFrm['Cx'] + dFrm['Cy']) ** 0.5
        if distance.min() < max_range_threshold:
            identified_ball = distance.argmin()
            outFrm.loc[identified_ball, 'isBall'] = 1
            print(f'Contour Nr.{identified_ball} identified as ball.')
        else:
            print('Click was too far away, no ball.')
            
    return outFrm

# frameNum = randrange(0, total_frames - 1)
# cFrm = BGRFrame.from_video(cap, frameNum)

clickList = []
frameList = []
while True:
    
    cv.destroyAllWindows()
    frameNum = randrange(0, total_frames - 1)
    gFrm = GreyFrame.from_video(cap, frameNum)
     
    ####### FIND COURT #########
    res = find_court(gFrm)
    if res is None:
        continue
    else:
        courtLines, netLines, H, config = res
    
    cFrm = BGRFrame.from_video(cap, frameNum)
    hFrm = HSVFrame.from_video(cap, frameNum) 
    
    ### Project Regions of Interest
    projROI_lower, projROI_upper = schema_specific_roi_projection_func(H)
    roiMask_lower, roiMask_upper = hFrm.within_region(projROI_lower), hFrm.within_region(projROI_upper)
    
    ### Top Player, with updated averages
    topplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_upper)
    
    ### Bottom Player, with updated averages
    botplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_lower)
    
    print('Found Court and Players...')
    
    ######### FIND BALL CANDIDATES ######q
    motMask = get_motion_mask(cap, frameNum)
    intMask = find_ball_candidate_mask(cFrm, motMask, topplayerRegion, botplayerRegion)
    
    cnts, _ = intMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = [Contour(c) for c in cnts if len(c) > 2] ### need at least three points for a contour
    
    if len(cnts) == 0:
        continue
    if len(cnts) > MAX_NUMBER_OF_CONTOURS:
        continue
    
    cntFrm = make_feature_frame(cnts, hFrm, gFrm, intMask)

    #### PREPARE FRAME TO SHOW ####
    showFrm = cFrm
    
    ### put players (if their mask fails, then the measurement of ball candidates is invalid!)
    showFrm = showFrm.put(topplayerRegion, color = (255, 0, 0)).put(botplayerRegion, color = (255, 0, 0))
    
    ### put contours
    showFrm = showFrm.put(cnts, thickness = 1, color = (0, 0, 255))
    
    ### put contour index on showFrm
    for idx, cnt in enumerate(cnts):
        offset_margin = 20
        try:
            cX, cY = cnt.center().coords      
        except ZeroDivisionError:
            continue
        showFrm = showFrm.put(str(idx), where = (cX - offset_margin, cY - offset_margin))
    
    
    ######### CLICK STUFF ########
    # reset the mouse click location
    clickPt = None
    # set up the window
    cv.namedWindow("image")
    cv.setMouseCallback("image", select_point)
    # display the image and wait for a keypress
    cv.imshow("image", showFrm.array)
    key = cv.waitKey(0) & 0xFF
    
    if key == ord('q'): ## press q to quit
        cv.destroyAllWindows()
        break
    
    if key == ord(' '): ## press confirm click and go to next picture
        clickList.append(clickPt)
        print(f'Logged {clickPt}')
        outFrm = label_ball_candidate(clickPt, cntFrm)
        
        outFrm = outFrm.reset_index().rename({'index': 'idx_within_image'}, axis = 1)
        outFrm['frameNum'] = frameNum
        
        frameList.append(outFrm)
    
    if key == ord('x'): ## press x to indicate there is nothing to click here
        pass
        # frameNum = randrange(0, total_frames - 1)
        # cFrm = BGRFrame.from_video(cap, frameNum)
    

##### CONCATENATE AND SAVE ######

import os
burnDict = atu.get_folder_struct('ball_svm')
savePath = os.path.join(burnDict['main'], VIDEO_ID)

try:
    os.mkdir(savePath)
except FileExistsError:
    print('   Adding to existing ball_svm data')

import datetime as dt
timestamp = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

filePath = os.path.join(savePath, f'labelled_balls_{timestamp}.csv')

# frameList2 = [frm.reset_index().rename({'index': 'idx_within_image'}, axis = 1) for frm in frameList]
bigFrm = pd.concat(frameList, ignore_index = True)    

bigFrm.to_csv(filePath)   
    

    
    