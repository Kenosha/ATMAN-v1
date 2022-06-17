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

ACCEPTABLE_LENGTH_OF_SEGMENT = 25

from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features
from ATMANpy.funcs.cv_funcs import replay
from ATMANpy.funcs.high_level_funcs import smart_interpolate_ball_contours

# featnameList = ['ball', 'playerTop', 'playerBot', 'H']
stream_features_uncorrected = load_stored_features(VIDEO_ID, featnameList = None, keep_which = 'first')
stream_features_corrected = load_stored_features(VIDEO_ID, featnameList = None, keep_which = 'last')

### add a detected flag
for k, featDict in stream_features_corrected.items():
    if featDict['ball'] is not None:
        featDict['ballDetected'] = True
    else:
        featDict['ballDetected'] = False
    
loadList = split_joint_features(stream_features_uncorrected, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)
corrList = split_joint_features(stream_features_corrected, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)

print(f'Read in {len(corrList)} segments')

from ATMANpy.funcs.draw_funcs import draw

# drawList = ['frmNum', 'ball', 'playerTop', 'playerBot']
# drawFunc = draw(drawList)
# replay(cap, loadList[0], drawFunc = drawFunc)


#%% Normalize Homographies
from ATMANpy.funcs.court_funcs import find_courtlines
### FOR TESTING ###

sf_num = 8
load_sf = loadList[sf_num]     
corr_sf = corrList[sf_num]

for k in corr_sf:

    if not np.array_equal(load_sf[k]['H'], corr_sf[k]['H']):
        print(k)
        myLines1 = find_courtlines(load_sf[k]['H'], schemaLines)
        myLines2 = find_courtlines(corr_sf[k]['H'], schemaLines)

        sendFrm = BGRFrame.from_video(cap, k)
        sendFrm = sendFrm.put(myLines1, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(myLines2, thickness = 2, color = (255, 0, 0))
        
        sendFrm.show()


replay(cap, load_sf, drawFunc=draw(['courtLines']))
replay(cap, corr_sf, drawFunc=draw(['courtLines']))


        
        