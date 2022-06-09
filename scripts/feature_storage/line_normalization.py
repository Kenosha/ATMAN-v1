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

#%% Line Normalization

# Idea: Construct a moving average median line.
# If most lines are very close to this line, except a few, then correct the few lines.
# This way, we can correct for stuttering court lines.

from copy import deepcopy
sf = loadList[4]

# replay(cap, sf, drawFunc = draw_loaded)

### Fast Mask ###
from ATMANpy.funcs.court_funcs import find_courtlines

def get_court_masks(sf, thickness = 2):
    blankFrm = GreyFrame(np.zeros((1080, 1920), dtype = 'uint8'))
    arrList = []
    for frame, featDict in sf.items():
        myLines = find_courtlines(featDict['H'], schemaLines)
        lFrm = blankFrm.put(myLines, thickness = thickness)
        arrList.append(lFrm.array.astype('bool'))

    return np.stack(arrList)



def stream_features_add_quantile_lines(sf, courtMasks, 
                                       how_many_unilateral = 8, 
                                       quantile = 0.25,
                                       thickness = 2):
    blankFrm = GreyFrame(np.zeros((1080, 1920), dtype = 'uint8'))
    out_sf = deepcopy(sf)
    
    for idx, frame in enumerate(out_sf):
        featDict = out_sf[frame]
        myLines = find_courtlines(featDict['H'], schemaLines)
        
        ### calculate quantile court mask
        left_slice = max(0, idx - how_many_unilateral)
        right_slice = min(idx + how_many_unilateral + 1, courtMasks.shape[0])
        
        frameMasks = courtMasks[left_slice:right_slice]
        
        quantMask = np.sum(frameMasks, axis = 0) >= (quantile * frameMasks.shape[0])

        quantMask = BoolFrame((quantMask * 255).astype('uint8'))

        featDict['quantMask'] = quantMask
        
        ### calculate overlap
        refMask = blankFrm.put(myLines, thickness = thickness) > 0
        overlap = (refMask & quantMask).sum(axes = (0, 1)) / refMask.sum(axes = (0, 1)) 
        featDict['lineNorm'] = overlap
        
    return out_sf


def draw_lines_and_norm(frmDict, featDict, frameNum = None):

    cFrm = frmDict['cFrm']
    
    H = featDict['H']
    myLines = find_courtlines(H, schemaLines)
    lineNorm = featDict['lineNorm']
    
    sendFrm = cFrm
    
    if frameNum is not None:
        sendFrm = sendFrm.put(str(int(frameNum)), thickness = 2, color = (0, 255, 255), where = (1800, 50))
    
    sendFrm = sendFrm.put(str(int(lineNorm * 100)), thickness = 2, color = (0, 255, 255))
        
    if lineNorm > 0.8:
        sendFrm = sendFrm.put(myLines, thickness = 2, color = (0, 255, 0))
    elif lineNorm > 0.6:
        sendFrm = sendFrm.put(myLines, thickness = 2, color = (0, 255, 255))
    else:
        sendFrm = sendFrm.put(myLines, thickness = 2, color = (0, 0, 255))
    
    return sendFrm


cMasks = get_court_masks(sf, thickness = 3)

sf2 = stream_features_add_quantile_lines(sf, cMasks, thickness = 3, quantile = 0.25)    

replay(cap, sf2, drawFunc = draw_lines_and_norm)

sf2.keys()
sf2[13575]['quantMask'].show()


for frame, featDict in sf2.items():
    cFrm = BGRFrame.from_video(cap, frame)
    
    if featDict['lineNorm'] < 0.7:
        sendFrm = draw_lines_and_norm({'cFrm': cFrm}, featDict, frameNum = frame)
        
        sendFrm.show()



from ATMANpy.funcs.court_funcs import find_courtlines
def get_surrounding_courtLines(frameNum, sf, how_many_unilateral = 8):
    outList = []
    for frame in range(frameNum - how_many_unilateral, frameNum + how_many_unilateral + 1):
        try:
            myFeat = sf[frame]
        except KeyError:
            pass
        else:
            myLines = find_courtlines(myFeat['H'], schemaLines)
            outList.append(myLines)

    return outList

frameNum = 10401
myLines = get_surrounding_courtLines(frameNum, sf)

def get_median_line_mask(frameNum, myLines):
    blankFrm = GreyFrame.from_video(cap, frameNum).zero()

    calcList = []
    for l in myLines:
        lineFrm = blankFrm.put(l, thickness = 2)
        calcList.append(lineFrm.array.astype('bool'))
    
    numFrames = len(calcList)
    calcArr = np.stack(calcList, axis = 0)
    
    medArr = np.sum(calcArr, axis = 0) > numFrames / 2 ### this line generalizes to a quantile!
    medArr = (medArr * 255).astype('uint8')
    
    return BoolFrame(medArr)


medMask = get_median_line_mask(frameNum, myLines)
medMask.show()    

def calculate_overlap(frameNum, medMask):
    blankFrm = GreyFrame.from_video(cap, frameNum).zero()
    refCourt = find_courtlines(sf[frameNum]['H'], schemaLines)

    refMask = blankFrm.put(refCourt, thickness = 2) > 0

    return (refMask & medMask).sum(axes = (0, 1)) / medMask.sum(axes = (0, 1)) 


def stream_features_normalize_lines(sf):
    from copy import deepcopy
    out_sf = deepcopy(sf)

    for frameNum in sf:
        surrLines = get_surrounding_courtLines(frameNum, sf)
        
        medMask = get_median_line_mask(frameNum, surrLines) 
        
        closeness_to_median = calculate_overlap(frameNum, medMask)

        out_sf[frameNum]['lineNorm'] = closeness_to_median
        
out_sf = stream_features_normalize_lines(sf)


#### Check Lines and LineNorm

def draw_lines_and_norm(frmDict, featDict, frameNum = None):

    cFrm = frmDict['cFrm']
    
    H = featDict['H']
    myLines = find_courtlines(H, schemaLines)
    lineNorm = featDict['lineNorm']
    
    sendFrm = cFrm
    
    if frameNum is not None:
        sendFrm = sendFrm.put(str(int(frameNum)), thickness = 2, color = (0, 255, 255), where = (1800, 50))
    
    sendFrm = sendFrm.put(str(int(lineNorm * 100)), thickness = 2, color = (0, 255, 255))
        
    if lineNorm > 0.8:
        sendFrm = sendFrm.put(myLines, thickness = 2, color = (0, 255, 0))
    elif lineNorm > 0.6:
        sendFrm = sendFrm.put(myLines, thickness = 2, color = (0, 255, 255))
    else:
        sendFrm = sendFrm.put(myLines, thickness = 2, color = (0, 0, 255))
    
    return sendFrm

replay(cap, out_sf, drawFunc = draw_lines_and_norm)

for frame, featDict in out_sf.items():
    cFrm = BGRFrame.from_video(cap, frame)
    
    if featDict['lineNorm'] < 0.6:
        sendFrm = draw_lines_and_norm({'cFrm': cFrm}, featDict, frameNum = frame)
        
        sendFrm.show()
    
    


# out_sf[frameNum]


    
#     lineFrm = blankFrm.put(myLines)


# gFrm.zero().put(myLines).show()


# cache = []
# for frmNum, featDict in sf.items():
#     while 
    
    

# for k in sf:
#     print(out_sf[k]['lineNorm'])

















