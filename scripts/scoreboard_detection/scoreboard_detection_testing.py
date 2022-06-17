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

from ATMANpy.funcs.draw_funcs import draw

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
start_stop_timestamp = '2022-06-16-22-46-58'
file = f'play_start_stop_{start_stop_timestamp}.csv'

seqFrm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)

######## GET SCHEMA ########
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)

#%% Reading these features back in

ACCEPTABLE_LENGTH_OF_SEGMENT = 25

from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features, join_split_features
from ATMANpy.funcs.cv_funcs import replay
from ATMANpy.funcs.high_level_funcs import smart_interpolate_ball_contours

# featnameList = ['ball', 'playerTop', 'playerBot', 'H']
# stream_features_uncorrected = load_stored_features(VIDEO_ID, featnameList = None, keep_which = 'first')
stream_features_corrected = load_stored_features(VIDEO_ID, featnameList = None, keep_which = 'last')

### add a detected flag
for k, featDict in stream_features_corrected.items():
    if featDict['ball'] is not None:
        featDict['ballDetected'] = True
    else:
        featDict['ballDetected'] = False
    
# loadList = split_joint_features(stream_features_uncorrected, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)
corrList = split_joint_features(stream_features_corrected, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)
homographies_correct = join_split_features(corrList)

print(f'Read in {len(corrList)} segments')

#%% Stream Feature Functions

def init_featDict(current_featDict = None, stored_featDict = None):

    # features_that_need_to_exist = ['H', 'config', 'courtLines', 'netLines', 
    #                                'trackFlag', 'playerBot', 'playerTop', 
    #                                'ballContours', 'ballParticles', 'ballIdxHistory', 
    #                                'ballCloseHistory', 'ballModelHistory']
    
    features_that_need_to_exist = ['H']
    
    ### use the featDict that was last available, or initialize an empty one.
    if current_featDict is not None:
        featDict = current_featDict
    else:
        featDict = {k: None for k in features_that_need_to_exist}
    
    ### add in the possibility of loading in stored values read in from disc.
    if stored_featDict is not None:
        for feat in stored_featDict:
            featDict[feat] = stored_featDict[feat]
    
    return featDict

def init_frameDict(frmCache):
    
    cFrm = frmCache[0]
    gFrm = cFrm.to_grey()
    hFrm = cFrm.to_HSV()
    
    frameDict = {'cFrm': cFrm, 'gFrm': gFrm, 'hFrm': hFrm}
    
    return frameDict
    
def get_court_lines(frmDict, featDict):
    
    from ATMANpy.funcs.court_funcs import find_courtlines, find_netlines
    
    gFrm = frmDict['gFrm']
    H = featDict['H']
    
    courtLines = find_courtlines(H, schemaLines)
    netLines = find_netlines(gFrm, courtLines)       
    
    featDict['courtLines'] = courtLines
    featDict['netLines'] = netLines
    
    return featDict

def get_court_features(frmDict, featDict): ### these two quantities will be used in simulations
    
    SOBEL_THRESHOLD = 30
    SLOPE_THRESHOLD = 0.0001
    MERGE_LINES = False
    MIN_LINE_LENGTH = 10
    MIN_LINES = 1

    gFrm = frmDict['gFrm']
    
    H = featDict['H']
    courtLines = featDict['courtLines']
    netLines = featDict['netLines']
    
    outDict = featDict.copy() 
    
    from ATMANpy.funcs.scoreboard_funcs import find_scoreboard
    outDict['scoreBoard'] = find_scoreboard(gFrm, 
                                            sobel_threshold = SOBEL_THRESHOLD, 
                                            slope_threshold = SLOPE_THRESHOLD,
                                            merge_lines = MERGE_LINES,
                                            min_line_length_after_merge = MIN_LINE_LENGTH, 
                                            min_lines = MIN_LINES)
    
    return outDict
    
drawList = ['scoreBoard']
drawFunc = draw(['frameNum', 'scoreBoard'])


#%% #### BUILD STREAM FEATURES FUNCTION ####
from functools import partial
from ATMANpy.funcs.cv_funcs import get_stream_features

my_stream_func = partial(get_stream_features, 
                         existing_features = homographies_correct,
                         cache_length = 1,
                         init_feat_func = init_featDict, 
                         init_frame_func = init_frameDict,
                         court_line_func = get_court_lines, 
                         court_feature_func = get_court_features,
                         draw_func = drawFunc,
                         verbose = False)

#%%### CREATE THE FEATURES ###
import time

video_segments = seqFrm

for idx, start_stop_frames in video_segments.iterrows():
    print(f'Doing {idx}...')
    start_time = time.time()
    
    start_frame, stop_frame = start_stop_frames
    stream_features_original = my_stream_func(cap, start_frame, stop_frame, verbose = True)

#%% Miscellaneous Tests

frmNum = 11290

gFrm = GreyFrame.from_video(cap, frmNum)


def construct_scoreboard_prior(cols, rows):
    x_share = 0.33
    y_share = 0.2

    x_offset = int(x_share * cols)
    y_offset = int(y_share * rows)
    
    bot_left = Point((5, rows))
    top_right = bot_left.right(x_offset).up(y_offset)
    
    return Region.from_points(bot_left, top_right)

scoreROI = construct_scoreboard_prior(gFrm.cols, gFrm.rows)

SOBEL_THRESHOLD = 30
SLOPE_THRESHOLD = 0.0001

myFrm = gFrm.crop(scoreROI)
blur_myFrm = myFrm.gaussian_blur((3, 3))

# (gFrm.sobel_x() > SOBEL_THRESHOLD).show()
# (gFrm.sobel_y() > SOBEL_THRESHOLD).show()

# vertBool = blur_myFrm.sobel_x() > SOBEL_THRESHOLD
horiBool = blur_myFrm.sobel_y() > SOBEL_THRESHOLD

# vertBool.show()
horiBool.show()

# vertLns = vertBool.prob_hough(minLineLength = 5, maxLineGap = 10)
horiLns = horiBool.prob_hough(minLineLength = 100, maxLineGap = 10)

# myFrm.to_BGR().put(vertLns).show() ### strange...
# myFrm.to_BGR().put(lns2).show() ### ok! 

straightLns = [ln for ln in horiLns if -SLOPE_THRESHOLD < ln.slope() < SLOPE_THRESHOLD]
# myFrm.to_BGR().put(straightLns).show()

# adjust for score region cropping
tlPt = scoreROI.to_points()[0]
straightLns = [ln.right(tlPt.x).down(tlPt.y) for ln in straightLns]

gFrm.to_BGR().put(straightLns).show()
