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
start_stop_timestamp = '2022-05-26-16-02-20'
file = f'play_start_stop_{start_stop_timestamp}.csv'

seqFrm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)


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

# sf = loadList[0]
okList = []
postprocessList = []
for sf in loadList:
    
    if sf == {}:
        continue
    
    ### interpolate ball positions
    sf_itpl = smart_interpolate_ball_contours(sf, verbose = True)
    
    replay(cap, sf_itpl, drawFunc = draw_loaded, ms_between_frames = 70)
    val = input("Press 1 if the replay is ok. 2 for postprocessing. 3 to discard: ")

    if val == 1:
        okList.append(sf)
    elif val == 2:
        postprocessList.append(sf)
        
        
    
        
#%% Check Interpolation

# from copy import deepcopy

# mySF = deepcopy(loadList[0])
# mySF_itpl = smart_interpolate_ball_contours(mySF)
   
# ### check weird interpolations manually.
# startFrm = 10430  
# stopFrm = 10453
# mySF_test = {k: v for k, v in mySF.items() if startFrm <= k <= stopFrm}
# mySF_test_itpl = {k: v for k, v in mySF_itpl.items() if startFrm <= k <= stopFrm}

# _ = smart_interpolate_ball_contours(mySF_test, verbose = True)


# for frmNum, featDict in mySF_test_itpl.items():
    
    
    
#     cFrm = BGRFrame.from_video(cap, frmNum)
    
#     drawFrm = draw_loaded({'cFrm': cFrm}, featDict, frameNum = frmNum)
    
#     drawFrm.show()
    


# frmNum = 10585

# next_frame_with_ball, next_three_ctrs = find_next_N_items(mySF_test, N = 3,
#                                                                   startNum = frmNum, 
#                                                                   item_to_find = 'ball')

# # assert next_frame_with_ball == 14266
# nextCtr, nextCtr_1, nextCtr_2 = next_three_ctrs


# last_frame_with_ball, last_three_ctrs = find_last_N_items(mySF_test_itpl, N = 3,
#                                                                   startNum = frmNum - 1, 
#                                                                   item_to_find = 'ball')

# currCtr, currCtr_1, currCtr_2 = last_three_ctrs

# ### find speed and acceleration after the next ball
# if not nextCtr is None:
    
#     nextSpeed, _, nextAccel = calculate_speeds_and_acceleration(nextCtr, nextCtr_1, nextCtr_2)

#     _, currSpeed, currAccel = calculate_speeds_and_acceleration(currCtr_2, currCtr_1, currCtr)
    
#     time_from_ball = frmNum - last_frame_with_ball
#     time_to_ball = next_frame_with_ball - frmNum
    
#     if currSpeed is not None and nextSpeed is not None:
#         itplBall = pf.interpolate_ball_features_smart(currCtr, nextCtr,
#                                                       speed_before = currSpeed, 
#                                                       speed_after = nextSpeed,
#                                                       accel_before = currAccel,
#                                                       accel_after = nextAccel,
#                                                       frames_before = time_from_ball,
#                                                       frames_after = time_to_ball,
#                                                       verbose = True)
#     else:
#         itplBall = pf.interpolate_ball_features(currCtr, nextCtr, 
#                                                 frames_before = time_from_ball,
#                                                 frames_after = time_to_ball)




# def drawEllipse(frame, ellipseTup):
#     center_coordinates, axesLength, angle = ellipseTup
#     # elFrm = BGRFrame(cv.ellipse(frame.array, center_coordinates, axesLength,
#     #                              angle, 0, 360, (0, 255, 0)))
#     elFrm = BGRFrame(cv.ellipse(frame.array, ellipseTup , (0, 255, 0)))
#     elFrm.show()
       
        
# cFrm.put(ell_cnt_after).show()


