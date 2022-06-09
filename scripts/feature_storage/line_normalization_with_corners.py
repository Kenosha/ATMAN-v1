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

def get_point_arrays(sf):
    tlList, trList, blList, brList = [], [], [], []
    
    for frame, featDict in sf.items():
        myLines = find_courtlines(featDict['H'], schemaLines)
        tl, tr, bl, br = myLines.get_court_region().to_points()

        tlList.append(tl.coords)
        trList.append(tr.coords)
        blList.append(bl.coords)
        brList.append(br.coords)

    return np.array(tlList), np.array(trList), np.array(blList), np.array(brList)

tlArr, trArr, blArr, brArr = get_point_arrays(sf)

def get_point_bounds(pointArr, 
                     lower_q = 0.2, upper_q = 0.8, acceptable_multiple = 3,
                     min_box_diameter = 4):
    
    qs = [lower_q, 0.5, upper_q]
    loPt, mdPt, hiPt = np.round(np.quantile(pointArr, qs, axis = 0)).astype('int')
    loPt, mdPt, hiPt = Point(loPt), Point(mdPt), Point(hiPt)

    ### interquantile distance
    iq_up, iq_lo = hiPt - mdPt, mdPt - loPt
    
    ### lower bind our interquantile distances, so we don't have tiny boxes.
    iq_up = iq_up.bind_below(min_box_diameter // 2)
    iq_lo = iq_lo.bind_below(min_box_diameter // 2)

    lower_bound = mdPt - iq_lo * acceptable_multiple - 1 ### subtract/add one because checking whether
    upper_bound = mdPt + iq_up * acceptable_multiple + 1 ### a point is inside region is checked STRONGLY.

    return lower_bound, upper_bound
    # return Point(lower_bound), Point(upper_bound)

    

# def slice_and_median
def stream_features_median_corner_regions(sf, tlArr, trArr, blArr, brArr,
                                          how_many_unilateral = 8):
    assert tlArr.shape[0] == trArr.shape[0] == blArr.shape[0] == brArr.shape[0]
    numFrames = tlArr.shape[0]
    
    out_sf = deepcopy(sf)
    
    for idx, frame in enumerate(out_sf):
        featDict = out_sf[frame]
        
        myLines = find_courtlines(featDict['H'], schemaLines)
        tl, tr, bl, br = myLines.get_court_region().to_points()

        ### calculate quantile court mask
        left_slice = max(0, idx - how_many_unilateral)
        right_slice = min(idx + how_many_unilateral + 1, numFrames)

        tlSlice = tlArr[left_slice:right_slice]
        trSlice = trArr[left_slice:right_slice]
        blSlice = blArr[left_slice:right_slice]
        brSlice = brArr[left_slice:right_slice]
        
        tl_lo, tl_hi = get_point_bounds(tlSlice)
        tr_lo, tr_hi = get_point_bounds(trSlice)
        bl_lo, bl_hi = get_point_bounds(blSlice)
        br_lo, br_hi = get_point_bounds(brSlice)
        
        featDict['tlReg'] = Region.from_points(tl_lo, tl_hi)
        featDict['trReg'] = Region.from_points(tr_lo, tr_hi)
        featDict['blReg'] = Region.from_points(bl_lo, bl_hi)
        featDict['brReg'] = Region.from_points(br_lo, br_hi)
        
        featDict['within_tlReg'] = tl.within(featDict['tlReg'])
        featDict['within_trReg'] = tr.within(featDict['trReg'])
        featDict['within_blReg'] = bl.within(featDict['blReg'])
        featDict['within_brReg'] = br.within(featDict['brReg'])
    
    return out_sf

sf2 = stream_features_median_corner_regions(sf, tlArr, trArr, blArr, brArr)


def draw_point_regions(frmDict, featDict, frameNum = None):

    cFrm = frmDict['cFrm']
    
    H = featDict['H']
    myLines = find_courtlines(H, schemaLines)
    tlReg = featDict['tlReg']
    trReg = featDict['trReg']
    blReg = featDict['blReg']
    brReg = featDict['brReg']
    
    within_tlReg = featDict['within_tlReg']
    within_trReg = featDict['within_trReg']
    within_blReg = featDict['within_blReg']
    within_brReg = featDict['within_brReg']
    
    colFunc = lambda withinBool: (0, 255, 0) if withinBool else (0, 0, 255)
    
    sendFrm = cFrm
    
    if frameNum is not None:
        sendFrm = sendFrm.put(str(int(frameNum)), thickness = 2, color = (0, 255, 255), where = (1800, 50))
    
    sendFrm = sendFrm.put(myLines, thickness = 2, color = (255, 0, 0))
    sendFrm = sendFrm.put(tlReg, colFunc(within_tlReg))
    sendFrm = sendFrm.put(trReg, colFunc(within_trReg))
    sendFrm = sendFrm.put(blReg, colFunc(within_blReg))
    sendFrm = sendFrm.put(brReg, colFunc(within_brReg))
    
    return sendFrm

replay(cap, sf2, drawFunc = draw_point_regions)
    
def stream_features_correct_homography(sf):
    out_sf = deepcopy(sf)
    
    for idx, frame in enumerate(out_sf):
        featDict = out_sf[frame]
        
        within_tlReg = featDict['within_tlReg']
        within_trReg = featDict['within_trReg']
        within_blReg = featDict['within_blReg']
        within_brReg = featDict['within_brReg']
        
        #### check if maybe all points are within - then, we don't have to correct anything.
        if within_tlReg and within_trReg and within_blReg and within_brReg:
            continue
        
        myLines = find_courtlines(featDict['H'], schemaLines)
        tlPt, trPt, blPt, brPt = myLines.get_court_region().to_points()
        
        tlReg = featDict['tlReg']
        trReg = featDict['trReg']
        blReg = featDict['blReg']
        brReg = featDict['brReg']
        
        tlPt_med = tlReg.centroid.round_point()
        trPt_med = trReg.centroid.round_point()
        blPt_med = blReg.centroid.round_point()
        brPt_med = brReg.centroid.round_point()
        
        ### replace faulty points with the median point
        newPts = []
        
        # Top Left
        if not within_tlReg:
            newPts.append(tlPt_med)
        else:
            newPts.append(tlPt)
            
        ### Top Right
        if not within_trReg:
            newPts.append(trPt_med)
        else:
            newPts.append(trPt)   
        
        ### Bottom Left
        if not within_blReg:
            newPts.append(blPt_med)
        else:
            newPts.append(blPt)
        
        ### Bottom Right
        if not within_brReg:
            newPts.append(brPt_med)
        else:
            newPts.append(brPt)
        
        corrected_courtLines = Region.from_points(*newPts)
        corrected_H = schemaRegion.get_homography(corrected_courtLines)
    
        featDict['H'] = corrected_H
        
    return out_sf
    

sf3 = stream_features_correct_homography(sf2)
replay(cap, sf3, drawFunc = draw_point_regions)

for frame, featDict in sf3.items():
    cFrm = BGRFrame.from_video(cap, frame)
    
    if not (featDict['within_tlReg'] and featDict['within_trReg'] and featDict['within_blReg'] and featDict['within_brReg']):
        sendFrm = draw_point_regions({'cFrm': cFrm}, featDict, frameNum = frame)
        
        sendFrm.show()
    
    


# out_sf[frameNum]


    
#     lineFrm = blankFrm.put(myLines)


# gFrm.zero().put(myLines).show()


# cache = []
# for frmNum, featDict in sf.items():
#     while 
    
    

# for k in sf:
#     print(out_sf[k]['lineNorm'])

















