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

#%% Select example sequences where splitting is necessary

# testList = []
# for idx, sf in enumerate(loadList[:10]):
    
#     if sf == {}:
#         continue
    
#     ### interpolate ball positions
#     sf_itpl = smart_interpolate_ball_contours(sf, verbose = False)
    
#     replay(cap, sf_itpl, drawFunc = draw_loaded, ms_between_frames = 30)
#     val = input("Press 1 if the replay can be used to test ballboy detection, 2 else.")

#     if val == '1':
#         testList.append(idx)
        
testList = [0, 1, 2, 7, 8, 9]
        
#%% Case 0

sf = loadList[0]      
     
start_frame = 10550
stop_frame = 10625

replay(cap, sf, drawFunc = draw_loaded, ms_between_frames = 30, 
       start_frame = start_frame, stop_frame = stop_frame)
    
### get frames:

def get_frames(frmType, start, stop, offset = 0):
    outDict = {}
    for frmNum in range(start - offset, stop - offset):
        outDict[frmNum] = frmType.from_video(cap, frmNum)
    return outDict

def get_frames2(frmType, start, stop, offsetList = None):
    if offsetList is None:
        offsetList = [0]
    
    min_offset = min(offsetList)
    max_offset = max(offsetList)
     
    frmList = []
    for frmNum in range(start + min_offset, stop + max_offset):
        myFrm = frmType.from_video(cap, frmNum)
        frmList.append(myFrm.array)
    
    frames = np.stack(frmList, axis = 0)
    # frames = np.swapaxes(np.swapaxes(frames, 2, 1), 1, 0) ### move last axis to front.
    
    return frames

def get_frames_warped(frmType, start, stop, sf = None, offsetList = None):
    if offsetList is None:
        offsetList = [0]
    
    min_offset = min(offsetList)
    max_offset = max(offsetList)
     
    frmList = []
    for frmNum in range(start + min_offset, stop + max_offset):
        myFrm = frmType.from_video(cap, frmNum)
        if sf is not None:
            H = sf[frmNum]['H']
            myFrm = myFrm.warp(np.linalg.inv(H), cols = schemaFrm.cols, rows = schemaFrm.rows)
        
        frmList.append(myFrm.array)
    
    frames = np.stack(frmList, axis = 0)
    # frames = np.swapaxes(np.swapaxes(frames, 2, 1), 1, 0) ### move last axis to front.
    
    return frames

def diff_frames(ref_frame, frames, offsetList):
    
    min_offset = min(offsetList)
    max_offset = max(offsetList)
    
    outList = []
    for offset in offsetList:
        shiftFrm = np.roll(frames, offset, axis = 0)[-min_offset:-max_offset]
        assert shiftFrm.shape == ref_frame.shape
        diffFrm = cv.absdiff(ref_frame, shiftFrm)
        outList.append(diffFrm)
    return np.stack(outList, axis = -1)

    

offsetList = [-8, -6, -4, 4, 6, 8]
# 

# ### non warped ###
# frames = get_frames2(GreyFrame, start_frame, stop_frame, offsetList = offsetList)
# ref_frame = get_frames2(GreyFrame, start_frame, stop_frame, offsetList = None)
# diffs = diff_frames(ref_frame, frames, offsetList)

# diffs.shape
# THRESHOLD = 10
# joint_diff = (diffs > THRESHOLD).all(axis = 3)
# joint_diff.shape

# for idx, frmNum in enumerate(range(start_frame, stop_frame)):
#     cFrm = BGRFrame.from_video(cap, frmNum)

#     motionMask = BoolFrame((joint_diff[idx] * 255).astype('uint8'))
    
#     cFrm.mask(motionMask).show()
        
#     # motionMask.show()
        
        
#### warped version ###
frames = get_frames_warped(GreyFrame, start_frame, stop_frame, sf = sf, offsetList = offsetList)
ref_frame = get_frames_warped(GreyFrame, start_frame, stop_frame, sf = sf, offsetList = None)
diffs = diff_frames(ref_frame, frames, offsetList)   
       
THRESHOLD = 50 
joint_diff = (diffs > THRESHOLD).all(axis = 3)        
    
from ATMANpy.funcs.court_funcs import find_courtlines
from ATMANpy.funcs.high_level_funcs import match_specific_burn_in_func    

for idx, frmNum in enumerate(range(start_frame, stop_frame)):
    cFrm = BGRFrame.from_video(cap, frmNum)
    hFrm = HSVFrame.from_video(cap, frmNum)

    H = sf[frmNum]['H']
    courtLines = find_courtlines(H, schemaLines)
    playerTop = sf[frmNum]['playerTop']
    playerBot = sf[frmNum]['playerBot']
    
    motionMask_warped = BoolFrame((joint_diff[idx] * 255).astype('uint8'))
    motionMask = motionMask_warped.warp(H, cols = cFrm.cols, rows = cFrm.rows)
    
    courtlineMask = motionMask.zero().put(courtLines, thickness = 10) > 0
    
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
    # courtlineMask.dilate(kernel).show()
    
    def cntMask(frm, cnt):
        ### Dilated player mask
        dil_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 10))
        return frm.within_contour(cnt).dilate(dil_kernel)
    
    if not playerTop is None:
        topMask = cntMask(cFrm, playerTop)
    else:
        topMask = BoolFrame(cFrm.to_grey().zero().array)
        print('Lost Top Player Tracking!')
        
    if not playerBot is None:
        botMask = cntMask(cFrm, playerBot)
    else:
        botMask = BoolFrame(cFrm.to_grey().zero().array)
        print('Lost Bottom Player Tracking!')
    
    motionMask2 = motionMask & ~courtlineMask & ~topMask & ~botMask
    
    ### Adding a blur stage!
    motionMask2 = motionMask2.gaussian_blur((7, 7))
    
    ### Adding a closing stage
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 18))
    motionMask2 = motionMask2.close(kernel)
    
    burnMask = match_specific_burn_in_func(hFrm, H)
    
    #####
    finalMask = motionMask2 & burnMask
    
    cts, _ = finalMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    AREA_THRESHOLD = 2000
    myCts = [Contour(ct) for ct in cts]
    myCts = [ct for ct in myCts if ct.area() > AREA_THRESHOLD]
    
    # cFrm.mask(finalMask).show()
    
    cFrm.put(myCts, thickness = 2).show()
    
    # cFrm.mask(motionMask2).show()
    # cFrm.mask(burnMask).show()    
    
    # motionMask.show()
    
#%% Case 1:
    
### hard to imagine that we'll crack this case without extremely good ROIs... the
# only giveaway is the ball kids raising their hands...

sf = loadList[1]      
     
start_frame = 11050
stop_frame = 11120

replay(cap, sf, drawFunc = draw_loaded, ms_between_frames = 30, 
       start_frame = start_frame, stop_frame = stop_frame)
    
### get frames:

    

offsetList = [-8, -6, -4, 4, 6, 8]
# 

# ### non warped ###
# frames = get_frames2(GreyFrame, start_frame, stop_frame, offsetList = offsetList)
# ref_frame = get_frames2(GreyFrame, start_frame, stop_frame, offsetList = None)
# diffs = diff_frames(ref_frame, frames, offsetList)

# diffs.shape
# THRESHOLD = 10
# joint_diff = (diffs > THRESHOLD).all(axis = 3)
# joint_diff.shape

# for idx, frmNum in enumerate(range(start_frame, stop_frame)):
#     cFrm = BGRFrame.from_video(cap, frmNum)

#     motionMask = BoolFrame((joint_diff[idx] * 255).astype('uint8'))
    
#     cFrm.mask(motionMask).show()
        
#     # motionMask.show()
        
        
#### warped version ###
frames = get_frames_warped(GreyFrame, start_frame, stop_frame, sf = sf, offsetList = offsetList)
ref_frame = get_frames_warped(GreyFrame, start_frame, stop_frame, sf = sf, offsetList = None)
diffs = diff_frames(ref_frame, frames, offsetList)   
       
THRESHOLD = 50 
joint_diff = (diffs > THRESHOLD).all(axis = 3)        
    
from ATMANpy.funcs.court_funcs import find_courtlines
from ATMANpy.funcs.high_level_funcs import match_specific_burn_in_func    

for idx, frmNum in enumerate(range(start_frame, stop_frame)):
    cFrm = BGRFrame.from_video(cap, frmNum)
    hFrm = HSVFrame.from_video(cap, frmNum)

    H = sf[frmNum]['H']
    courtLines = find_courtlines(H, schemaLines)
    playerTop = sf[frmNum]['playerTop']
    playerBot = sf[frmNum]['playerBot']
    
    motionMask_warped = BoolFrame((joint_diff[idx] * 255).astype('uint8'))
    motionMask = motionMask_warped.warp(H, cols = cFrm.cols, rows = cFrm.rows)
    
    courtlineMask = motionMask.zero().put(courtLines, thickness = 10) > 0
    
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
    # courtlineMask.dilate(kernel).show()
    
    def cntMask(frm, cnt):
        ### Dilated player mask
        dil_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 10))
        return frm.within_contour(cnt).dilate(dil_kernel)
    
    if not playerTop is None:
        topMask = cntMask(cFrm, playerTop)
    else:
        topMask = BoolFrame(cFrm.to_grey().zero().array)
        print('Lost Top Player Tracking!')
        
    if not playerBot is None:
        botMask = cntMask(cFrm, playerBot)
    else:
        botMask = BoolFrame(cFrm.to_grey().zero().array)
        print('Lost Bottom Player Tracking!')
    
    motionMask2 = motionMask & ~courtlineMask & ~topMask & ~botMask
    
    ### Adding a blur stage!
    motionMask2 = motionMask2.gaussian_blur((7, 7))
    
    ### Adding a closing stage
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 18))
    motionMask2 = motionMask2.close(kernel)
    
    burnMask = match_specific_burn_in_func(hFrm, H)
    
    #####
    finalMask = motionMask2 & burnMask
    
    cts, _ = finalMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    AREA_THRESHOLD = 2000
    myCts = [Contour(ct) for ct in cts]
    myCts = [ct for ct in myCts if ct.area() > AREA_THRESHOLD]
    
    # cFrm.mask(finalMask).show()
    
    cFrm.put(myCts, thickness = 2).show()
    
    # cFrm.mask(motionMask2).show()
    # cFrm.mask(burnMask).show()    
    
#%%  Case 2

sf = loadList[2]      
     
start_frame = 11510
stop_frame = 11640

replay(cap, sf, drawFunc = draw_loaded, ms_between_frames = 30, 
       start_frame = start_frame, stop_frame = stop_frame)
    
### get frames:

    
offsetList = [-8, -6, -4, 4, 6, 8]
if start_frame is None:
    start_frame = min(sf.keys()) - min(offsetList)
if stop_frame is None:
    stop_frame = max(sf.keys()) - max(offsetList)
    
    
# 

# ### non warped ###
# frames = get_frames2(GreyFrame, start_frame, stop_frame, offsetList = offsetList)
# ref_frame = get_frames2(GreyFrame, start_frame, stop_frame, offsetList = None)
# diffs = diff_frames(ref_frame, frames, offsetList)

# diffs.shape
# THRESHOLD = 10
# joint_diff = (diffs > THRESHOLD).all(axis = 3)
# joint_diff.shape

# for idx, frmNum in enumerate(range(start_frame, stop_frame)):
#     cFrm = BGRFrame.from_video(cap, frmNum)

#     motionMask = BoolFrame((joint_diff[idx] * 255).astype('uint8'))
    
#     cFrm.mask(motionMask).show()
        
#     # motionMask.show()
        
        
#### warped version ###
frames = get_frames_warped(GreyFrame, start_frame, stop_frame, sf = sf, offsetList = offsetList)
ref_frame = get_frames_warped(GreyFrame, start_frame, stop_frame, sf = sf, offsetList = None)
diffs = diff_frames(ref_frame, frames, offsetList)   
       
THRESHOLD = 50 
joint_diff = (diffs > THRESHOLD).all(axis = 3)        
    
from ATMANpy.funcs.court_funcs import find_courtlines
from ATMANpy.funcs.high_level_funcs import match_specific_burn_in_func    

current_movement_index = 0
for idx, frmNum in enumerate(range(start_frame, stop_frame)):
    cFrm = BGRFrame.from_video(cap, frmNum)
    hFrm = HSVFrame.from_video(cap, frmNum)
    
    
    cFrm = cFrm.put(str(int(frmNum)), thickness = 2, color = (0, 255, 255), where = (1800, 50))
    
    H = sf[frmNum]['H']
    courtLines = find_courtlines(H, schemaLines)
    playerTop = sf[frmNum]['playerTop']
    playerBot = sf[frmNum]['playerBot']
    
    motionMask_warped = BoolFrame((joint_diff[idx] * 255).astype('uint8'))
    motionMask = motionMask_warped.warp(H, cols = cFrm.cols, rows = cFrm.rows)
    
    courtlineMask = motionMask.zero().put(courtLines, thickness = 10) > 0
    
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
    # courtlineMask.dilate(kernel).show()
    
    def cntMask(frm, cnt):
        ### Dilated player mask
        dil_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 10))
        return frm.within_contour(cnt).dilate(dil_kernel)
    
    if not playerTop is None:
        topMask = cntMask(cFrm, playerTop)
    else:
        topMask = BoolFrame(cFrm.to_grey().zero().array)
        print('Lost Top Player Tracking!')
        
    if not playerBot is None:
        botMask = cntMask(cFrm, playerBot)
    else:
        botMask = BoolFrame(cFrm.to_grey().zero().array)
        print('Lost Bottom Player Tracking!')
    
    motionMask2 = motionMask & ~courtlineMask & ~topMask & ~botMask
    
    ### Adding a blur stage!
    motionMask2 = motionMask2.gaussian_blur((7, 7))
    
    ### Adding a closing stage
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 18))
    motionMask2 = motionMask2.close(kernel)
    
    burnMask = match_specific_burn_in_func(hFrm, H)
    
    #####
    finalMask = motionMask2 & burnMask
    
    cts, _ = finalMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    AREA_THRESHOLD = 2000
    myCts = [Contour(ct) for ct in cts]
    myCts = [ct for ct in myCts if ct.area() > AREA_THRESHOLD]
    
    # cFrm.mask(finalMask).show()
    if myCts != []:
        current_movement_index += 1
    else:
        current_movement_index = 0
        
    
    
    cFrm.put(myCts, thickness = 2).put(str(current_movement_index)).show()
    # cFrm.put(courtLines).show()

    # cFrm.mask(motionMask2).show()
    # cFrm.mask(burnMask).show()    
    
        
#%%  Case 7

sf = loadList[8]      
     
start_frame = 15480
stop_frame = 15650

replay(cap, sf, drawFunc = draw_loaded, ms_between_frames = 30, 
       start_frame = start_frame, stop_frame = stop_frame)
    
### get frames:

    
offsetList = [-8, -6, -4, 4, 6, 8]
if start_frame is None:
    start_frame = min(sf.keys()) - min(offsetList)
if stop_frame is None:
    stop_frame = max(sf.keys()) - max(offsetList)
    
    
# 

# ### non warped ###
# frames = get_frames2(GreyFrame, start_frame, stop_frame, offsetList = offsetList)
# ref_frame = get_frames2(GreyFrame, start_frame, stop_frame, offsetList = None)
# diffs = diff_frames(ref_frame, frames, offsetList)

# diffs.shape
# THRESHOLD = 10
# joint_diff = (diffs > THRESHOLD).all(axis = 3)
# joint_diff.shape

# for idx, frmNum in enumerate(range(start_frame, stop_frame)):
#     cFrm = BGRFrame.from_video(cap, frmNum)

#     motionMask = BoolFrame((joint_diff[idx] * 255).astype('uint8'))
    
#     cFrm.mask(motionMask).show()
        
#     # motionMask.show()
        
        
#### warped version ###
frames = get_frames_warped(GreyFrame, start_frame, stop_frame, sf = sf, offsetList = offsetList)
ref_frame = get_frames_warped(GreyFrame, start_frame, stop_frame, sf = sf, offsetList = None)
diffs = diff_frames(ref_frame, frames, offsetList)   
       
THRESHOLD = 50 
joint_diff = (diffs > THRESHOLD).all(axis = 3)        
    
from ATMANpy.funcs.court_funcs import find_courtlines
from ATMANpy.funcs.high_level_funcs import match_specific_burn_in_func    

current_movement_index = 0
for idx, frmNum in enumerate(range(start_frame, stop_frame)):
    cFrm = BGRFrame.from_video(cap, frmNum)
    hFrm = HSVFrame.from_video(cap, frmNum)
    
    
    cFrm = cFrm.put(str(int(frmNum)), thickness = 2, color = (0, 255, 255), where = (1800, 50))
    
    H = sf[frmNum]['H']
    courtLines = find_courtlines(H, schemaLines)
    playerTop = sf[frmNum]['playerTop']
    playerBot = sf[frmNum]['playerBot']
    
    motionMask_warped = BoolFrame((joint_diff[idx] * 255).astype('uint8'))
    motionMask = motionMask_warped.warp(H, cols = cFrm.cols, rows = cFrm.rows)
    
    courtlineMask = motionMask.zero().put(courtLines, thickness = 10) > 0
    
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
    # courtlineMask.dilate(kernel).show()
    
    def cntMask(frm, cnt):
        ### Dilated player mask
        dil_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 10))
        return frm.within_contour(cnt).dilate(dil_kernel)
    
    if not playerTop is None:
        topMask = cntMask(cFrm, playerTop)
    else:
        topMask = BoolFrame(cFrm.to_grey().zero().array)
        print('Lost Top Player Tracking!')
        
    if not playerBot is None:
        botMask = cntMask(cFrm, playerBot)
    else:
        botMask = BoolFrame(cFrm.to_grey().zero().array)
        print('Lost Bottom Player Tracking!')
    
    motionMask2 = motionMask & ~courtlineMask & ~topMask & ~botMask
    
    ### Adding a blur stage!
    motionMask2 = motionMask2.gaussian_blur((7, 7))
    
    ### Adding a closing stage
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 18))
    motionMask2 = motionMask2.close(kernel)
    
    burnMask = match_specific_burn_in_func(hFrm, H)
    
    #####
    finalMask = motionMask2 & burnMask
    
    cts, _ = finalMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    AREA_THRESHOLD = 2000
    myCts = [Contour(ct) for ct in cts]
    myCts = [ct for ct in myCts if ct.area() > AREA_THRESHOLD]
    
    # cFrm.mask(finalMask).show()
    if myCts != []:
        current_movement_index += 1
    else:
        current_movement_index = 0
        
    
    
    cFrm.put(myCts, thickness = 2).put(str(current_movement_index)).show()
    # cFrm.put(courtLines).show()

    # cFrm.mask(motionMask2).show()
    # cFrm.mask(burnMask).show()    
    
    