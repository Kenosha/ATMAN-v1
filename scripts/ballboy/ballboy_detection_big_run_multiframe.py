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


### STORENAME ###
STORE_NAME = '10to100_segmentation_fixes'

###### GET SEQUENCES #####
from ATMANpy.funcs.feature_funcs import read_play_sequences
seqFrm = read_play_sequences(VIDEO_ID, STORE_NAME, keep_which = 'last')

######## GET SCHEMA ########
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)

#%% Reading these features back in

ACCEPTABLE_LENGTH_OF_SEGMENT = 0

from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features
from ATMANpy.funcs.cv_funcs import replay

stream_features = load_stored_features(VIDEO_ID, STORE_NAME, featnameList = None, keep_which = 'last')

### add a detected flag
for k, featDict in stream_features.items():
    if featDict['ball'] is not None:
        featDict['ballDetected'] = True
    else:
        featDict['ballDetected'] = False
    
# loadList = split_joint_features(stream_features_uncorrected, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)
corrList = split_joint_features(stream_features, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)

print(f'Read in {len(corrList)} segments')

# drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'scoreBoard']
# drawFunc = draw(drawList)

# seqFrm[seqFrm.StartFrame > 53000]

# replay(cap, corrList[70], drawFunc = drawFunc)


#%% Multiprocessed Run

from ATMANpy.funcs.sf_postprocess_funcs import detect_movement_contours
# from ATMANpy.funcs.sf_postprocess_funcs import calculate_contiguous_movement
from ATMANpy.funcs.feature_funcs import save_contours
def split_func(sf, bgrFrms):
    move_sf = detect_movement_contours(sf, cap = bgrFrms, save_all = False)
    save_contours(move_sf, 'movContours', VIDEO_ID, STORE_NAME, compression = 24)

from ATMANpy.funcs.feature_funcs import save_features_parallel

if __name__ == '__main__':
    save_features_parallel(VIDEO_ID, corrList, split_func, num_workers = 8)
