# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:27:14 2022

Idea is to detect ballboys well enough to be able to interrupt the play sequence if they move.

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

from copy import deepcopy

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

######## GET SCHEMA ########
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)

#%% Reading these features back in

from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features
from ATMANpy.funcs.cv_funcs import replay
from ATMANpy.funcs.high_level_funcs import smart_interpolate_ball_contours

# featnameList = ['ball', 'playerTop', 'playerBot', 'H']
stream_features_loaded = load_stored_features(VIDEO_ID, featnameList = None)


### add a detected flag
for k, featDict in stream_features_loaded.items():
    if featDict['ball'] is not None:
        featDict['ballDetected'] = True
    else:
        featDict['ballDetected'] = False
    

loadList = split_joint_features(stream_features_loaded)

def draw_loaded(frmDict, featDict, frameNum = None):

    cFrm = frmDict['cFrm']

    selectedBall = featDict['ball']
    playerTop = featDict['playerTop']
    playerBot = featDict['playerBot']
    
    ballDetected = featDict['ballDetected']
    # ballParticles = featDict['ballParticles']
    
    sendFrm = cFrm
    
    if frameNum is not None:
        sendFrm = sendFrm.put(str(int(frameNum)), thickness = 2, color = (0, 255, 255), where = (1800, 50))
    
    sendFrm = sendFrm.put(playerTop, thickness = 2, color = (0, 0, 255))
    sendFrm = sendFrm.put(playerBot, thickness = 2, color = (0, 0, 255))
    # sendFrm = sendFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
    
    if selectedBall is not None:
        if ballDetected:
            sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 0))
        else: ### ball is interpolated!
            sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 255))
        
    return sendFrm

#%% Normalize Homographies

from ATMANpy.funcs.court_funcs import find_courtlines
from ATMANpy.funcs.sf_postprocess_funcs import stream_features_median_corner_regions
from ATMANpy.funcs.sf_postprocess_funcs import stream_features_correct_homography

load_sf = loadList[2]     
corr_sf = stream_features_correct_homography(stream_features_median_corner_regions(load_sf))

for k in corr_sf:
    if not np.array_equal(load_sf[k]['H'], corr_sf[k]['H']):
        myLines1 = find_courtlines(load_sf[k]['H'], schemaLines)
        myLines2 = find_courtlines(corr_sf[k]['H'], schemaLines)

        sendFrm = BGRFrame.from_video(cap, k)
        sendFrm = sendFrm.put(myLines1, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(myLines2, thickness = 2, color = (255, 0, 0))
        
        sendFrm.show()

#%% Split the loaded stream features if there is movement in frame.


def get_frames_warped(sf):    
    frmType = GreyFrame
    
    frmList = []
    for frmNum in sf:
        myFrm = frmType.from_video(cap, frmNum)
    
        H = sf[frmNum]['H']
        myFrm = myFrm.warp(np.linalg.inv(H), cols = schemaFrm.cols, rows = schemaFrm.rows)
    
        frmList.append(myFrm.array)
    
    frames = np.stack(frmList, axis = 0)
    # frames = np.swapaxes(np.swapaxes(frames, 2, 1), 1, 0) ### move last axis to front.
    
    return frames


def roll_safe(frames, offset):
    ### rolling without rolling over -> we shift the frames by offset, 
    ### but we do not reinsert at the other end
    ### instead, we repeat the first or the last frame.
    offsetSlice = - offset ## if we roll forward by 6, then we want
                                # i) the first frame, six times
                                # ii) all other frames except the last six.
    if offset == 0:
        return frames
    if offset > 0:
        return np.concatenate([np.stack([frames[0] for i in range(offset)]), frames[:offsetSlice]])
    else:
        return np.concatenate([frames[offsetSlice:], np.stack([frames[-1] for i in range(offsetSlice)])])


def diff_frames(frames, offsetList):
    
    min_offset = min(offsetList)
    max_offset = max(offsetList)
    
    outList = []
    for offset in offsetList:
        shift_frames = roll_safe(frames, offset)
        diffFrm = cv.absdiff(frames, shift_frames)
        outList.append(diffFrm)
    return np.stack(outList, axis = -1)
     

def detect_movement_contours(sf, verbose = False):
        
    out_sf = deepcopy(sf)
    
    ###### PARAMETERS ######
    OFFSETLIST = [-8, -6, -4, 4, 6, 8]
    MOVEMENT_INTENSITY_THRESHOLD = 20 
    
    ### PREP DIFFERENCE FRAME ###
    frames = get_frames_warped(out_sf)   
    diffs = diff_frames(frames, OFFSETLIST)           

    joint_diff = (diffs > MOVEMENT_INTENSITY_THRESHOLD).all(axis = 3)  
    ####

    from ATMANpy.funcs.court_funcs import find_courtlines
    from ATMANpy.funcs.high_level_funcs import match_specific_burn_in_func    
        
    for idx, frmNum in enumerate(out_sf):
        featDict = out_sf[frmNum]
        
        cFrm = BGRFrame.from_video(cap, frmNum)
        hFrm = HSVFrame.from_video(cap, frmNum)
        
        H = featDict['H']
        courtLines = find_courtlines(H, schemaLines)
        playerTop = featDict['playerTop']
        playerBot = featDict['playerBot']
        ball = featDict['ball']
        
        motionMask_warped = BoolFrame((joint_diff[idx] * 255).astype('uint8'))
        motionMask = motionMask_warped.warp(H, cols = cFrm.cols, rows = cFrm.rows)
        
        courtlineMask = motionMask.zero().put(courtLines, thickness = 10) > 0
        
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
        # courtlineMask.dilate(kernel).show()
        
        def cntMask(frm, cnt, kernelsize = (20, 10)):
            ### Dilated player mask
            dil_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernelsize)
            return frm.within_contour(cnt).dilate(dil_kernel)
        
        if not playerTop is None:
            topMask = cntMask(cFrm, playerTop, kernelsize = (23, 11))
        else:
            topMask = BoolFrame(cFrm.to_grey().zero().array)
            # print('Lost Top Player Tracking!')
            
        if not playerBot is None:
            botMask = cntMask(cFrm, playerBot, kernelsize = (27, 13))
        else:
            botMask = BoolFrame(cFrm.to_grey().zero().array)
            # print('Lost Bottom Player Tracking!')
        
        if not ball is None:
            ballMask = cntMask(cFrm, ball, kernelsize = (7, 7))
        else:
            ballMask = BoolFrame(cFrm.to_grey().zero().array)
            # print('Lost Ball Tracking!')
        
        # ballMask.show()
        
        motionMask2 = motionMask & ~courtlineMask & ~topMask & ~botMask & ~ballMask
        # if verbose: motionMask2.show()
        
        # ### Adding a blur stage!
        # motionMask2 = motionMask2.gaussian_blur((7, 7))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        motionMask2 = motionMask2.open(kernel) >= 255
        # if verbose: motionMask2.show()
        
        ### Adding a closing stage
        ### large kernel, so that we join blobs belonging to the same ball kid.
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (60, 60))
        motionMask2 = motionMask2.close(kernel)
        # if verbose: motionMask2.show()
        
        burnMask = match_specific_burn_in_func(hFrm, H)
        
        #####
        finalMask = motionMask2 & burnMask
        # if verbose: finalMask.show()
        
        cts, _ = finalMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        myCts = [Contour(ct) for ct in cts]
        
        featDict['movContours'] = myCts
        
    return out_sf


move_sf = detect_movement_contours(corr_sf) 


def draw_mov_contours(frmDict, featDict, frameNum = None):

    cFrm = frmDict['cFrm']

    selectedBall = featDict['ball']
    playerTop = featDict['playerTop']
    playerBot = featDict['playerBot']
    
    ballDetected = featDict['ballDetected']
    # ballParticles = featDict['ballParticles']
    
    movContours = featDict['movContours']
    
    sendFrm = cFrm
    
    if frameNum is not None:
        sendFrm = sendFrm.put(str(int(frameNum)), thickness = 2, color = (0, 255, 255), where = (1800, 50))
    
    sendFrm = sendFrm.put(playerTop, thickness = 2, color = (0, 0, 255))
    sendFrm = sendFrm.put(playerBot, thickness = 2, color = (0, 0, 255))
    # sendFrm = sendFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
    
    sendFrm = sendFrm.put(movContours, thickness = 2, color = (255, 0, 0))
    
    if selectedBall is not None:
        if ballDetected:
            sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 0))
        else: ### ball is interpolated!
            sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 255))
        
    return sendFrm


replay(cap, move_sf, drawFunc = draw_mov_contours)


def calculate_contiguous_movement(sf):

    out_sf = deepcopy(sf)    

    ### PARAMETERS ###
    AREA_THRESHOLD = 600

    ### INIT ###
    current_movement_index = 0
    
    for frmNum, featDict in out_sf.items():
        
        cFrm = BGRFrame.from_video(cap, frmNum)
        H = featDict['H']
        courtLines = find_courtlines(H, schemaLines)
        
        myCts = featDict['movContours']
        myCts = [ct for ct in myCts if ct.area() > AREA_THRESHOLD]
        # myCts.sort(key = lambda el: el.area())
        # try:
        #     largestCnt = myCts[-1].area()
        # except:
        #     largestCnt = 'None'

        if myCts != []:
            current_movement_index += 1
        else:
            current_movement_index = 0
            
        featDict['curr_movInt'] = current_movement_index 
    
    return out_sf
    

def draw_mov_contours_with_mov_int(frmDict, featDict, frameNum = None):

    cFrm = frmDict['cFrm']

    selectedBall = featDict['ball']
    playerTop = featDict['playerTop']
    playerBot = featDict['playerBot']
    
    ballDetected = featDict['ballDetected']
    # ballParticles = featDict['ballParticles']
    
    movContours = featDict['movContours']
    movInt = featDict['curr_movInt']
    
    sendFrm = cFrm
    
    if frameNum is not None:
        sendFrm = sendFrm.put(str(int(frameNum)), thickness = 2, color = (0, 255, 255), where = (1800, 50))
    
    sendFrm = sendFrm.put(playerTop, thickness = 2, color = (0, 0, 255))
    sendFrm = sendFrm.put(playerBot, thickness = 2, color = (0, 0, 255))
    # sendFrm = sendFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
    
    sendFrm = sendFrm.put(movContours, thickness = 2, color = (255, 0, 0))
    sendFrm = sendFrm.put(f'Consec. Count: {str(movInt)}', thickness = 2)
    
    if selectedBall is not None:
        if ballDetected:
            sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 0))
        else: ### ball is interpolated!
            sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 255))
        
    return sendFrm    

cont_sf = calculate_contiguous_movement(move_sf)
replay(cap, cont_sf, drawFunc = draw_mov_contours_with_mov_int, ms_between_frames = 120)

def drop_movement_frames(sf, verbose = False):
    
    ### PARAMETERS ###
    MINIMUM_LENGTH_OF_CONTINUOUS_MOVEMENT = 10
    KEEP_FIRST_AND_LAST_X_FRAMES = 3
        
    ### INIT ###
    prior_movement_index = 0
    frames_to_toss = [] ### the overall list of lists, containing all continuous segments of extraneous movement on court.
    current_frames_to_toss = [] ### a list containing a single contiguous segment of extraneous movement.
    
    for frmNum, featDict in sf.items():
        
        current_movement_index = featDict['curr_movInt']
        
        ### SELECT FRAME FOR REMOVAL ###
        if current_movement_index > 0:
            current_frames_to_toss.append(frmNum)
            prior_movement_index = current_movement_index
        else:
            if prior_movement_index > MINIMUM_LENGTH_OF_CONTINUOUS_MOVEMENT:
            
                start_idx = KEEP_FIRST_AND_LAST_X_FRAMES
                stop_idx = - KEEP_FIRST_AND_LAST_X_FRAMES
                current_frames_to_toss = current_frames_to_toss[start_idx:stop_idx]
            
                print(current_frames_to_toss)
                frames_to_toss.append(current_frames_to_toss)
            
            current_frames_to_toss = []
            prior_movement_index = 0
            
                
    flat_frames_to_toss = [el for ls in frames_to_toss for el in ls]
    
    return {frmNum: featDict for frmNum, featDict in sf.items() if frmNum not in flat_frames_to_toss}
          

drop_sf = drop_movement_frames(cont_sf)

for k in cont_sf:
    print(cont_sf[k]['curr_movInt'])
    
    
for k in drop_sf:
    print(drop_sf[k]['curr_movInt'])
        
def separate_out_contiguous_segments(sf):
    
    ### PARAMETERS ###
    ACCEPTABLE_LENGTH_OF_SEGMENT = 25 ### a segment below X frames will be ignored.
    
    outList = [] ### the list of stream features which are contiguous (do not skip frames.)
    
    keys = sorted(sf.keys())
    k_prev, ks = keys[0], keys[1:]
    contiguous_sf = {k_prev: sf[k_prev]}
    
    for k in ks:
        if k == k_prev + 1:
            contiguous_sf[k] = sf[k]
        elif k > k_prev + 1:
            
            ### determine whether the segment should be kept!!!
            if len(contiguous_sf) >= ACCEPTABLE_LENGTH_OF_SEGMENT:
                outList.append(contiguous_sf)
            
            contiguous_sf = {k: sf[k]}
        
        k_prev = k
    
    return outList

sfList = separate_out_contiguous_segments(drop_sf)

print(f'Number of segments detected: {len(sfList)}')
for mySF in sfList:
    print(f'    Length of Segment: {len(mySF)}')
    replay(cap, mySF, drawFunc = draw_mov_contours_with_mov_int, ms_between_frames = 120)   

    
        
        
        
        
        