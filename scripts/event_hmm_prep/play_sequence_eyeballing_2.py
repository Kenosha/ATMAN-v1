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



#%% Compare two stores

from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features
from ATMANpy.funcs.feature_funcs import filter_features_by_play_sequence_frame

# STORE_NAME = '10to100_faster'

# featnameList = ['ball', 'playerTop', 'playerBot', 'H']
stream_features = load_stored_features(VIDEO_ID, STORE_NAME, featnameList = None)
stream_features = filter_features_by_play_sequence_frame(stream_features, seqFrm)

# STORE_NAME = '10to100_lowerlikelihood'

# # featnameList = ['ball', 'playerTop', 'playerBot', 'H']
# stream_features_comp = load_stored_features(VIDEO_ID, STORE_NAME, featnameList = None)
# stream_features_comp = filter_features_by_play_sequence_frame(stream_features_comp, seqFrm)

# ### first -> check if the ball is detected more often, and by how much.

# def count_feature(sf, featStr):
#     featCounter = 0
#     for k in sf:
#         if sf[k][featStr] is not None:
#             featCounter += 1
    
#     return featCounter / len(sf)

# count_feature(stream_features, 'ball')
# count_feature(stream_features_comp, 'ball')
# ### -> almost no difference...

# ### second -> add the comp. ball to the original stream features, to check if these are any better.

# stream_features_combined = deepcopy(stream_features)

# for k in stream_features_combined:
#     stream_features_combined[k]['ball_secondary'] = stream_features_comp[k]['ball']



#%% Reading these features back in

from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features
from ATMANpy.funcs.cv_funcs import replay
from ATMANpy.funcs.high_level_funcs import smart_interpolate_ball_contours

# # featnameList = ['ball', 'playerTop', 'playerBot', 'H']
# stream_features_loaded = load_stored_features(VIDEO_ID, STORE_NAME, featnameList = None)


# ### add a detected flag
# for k, featDict in stream_features_loaded.items():
#     if featDict['ball'] is not None:
#         featDict['ballDetected'] = True
#     else:
#         featDict['ballDetected'] = False
    
# ### filter for sequences
# from ATMANpy.funcs.feature_funcs import filter_features_by_play_sequence_frame
# stream_features_filtered = filter_features_by_play_sequence_frame(stream_features_loaded, seqFrm)
    
loadList = split_joint_features(stream_features)
len(loadList)

from ATMANpy.funcs.draw_funcs import draw
drawList = ['playerTop', 'playerBot', 'courtLines', 'ball']
drawFunc = draw(drawList)

# sf = loadList[0]
okList = []
postprocessList = []
for sf in loadList:
    
    if sf == {}:
        continue
    
    # ### interpolate ball positions
    # sf_itpl = smart_interpolate_ball_contours(sf, verbose = False)
    
    replay(cap, sf, drawFunc = drawFunc, ms_between_frames = 70)
    val = input("Press 1 if the replay is ok. 2 for postprocessing. 3 to discard: ")

    if val == '1':
        okList.append(sf)
    elif val == '2':
        postprocessList.append(sf)
        

#%% Write these to disc
        
len(okList)
len(postprocessList)

### -> post process SF are mostly due to:
    # a) camera zooming or panning -> messes with ball recognition (candidates or particle filter?)
    # b) something seen as moving, causes play sequence to be broken up.
    
STORE_NAME = '10to100_ok2'

from ATMANpy.funcs.feature_funcs import save_contours, save_homographies, save_contiguous_play_sequences

save_contiguous_play_sequences(okList, VIDEO_ID, STORE_NAME)

for sf in okList:
    save_contours(sf, 'playerTop', VIDEO_ID, STORE_NAME, compression = 24)
    save_contours(sf, 'playerBot', VIDEO_ID, STORE_NAME, compression = 24)
    save_contours(sf, 'ball', VIDEO_ID, STORE_NAME, compression = 6)
    save_homographies(sf, VIDEO_ID, STORE_NAME)


STORE_NAME = '10to100_problematic2'

from ATMANpy.funcs.feature_funcs import save_contours, save_homographies, save_contiguous_play_sequences

save_contiguous_play_sequences(postprocessList, VIDEO_ID, STORE_NAME)

for sf in postprocessList:
    save_contours(sf, 'playerTop', VIDEO_ID, STORE_NAME, compression = 24)
    save_contours(sf, 'playerBot', VIDEO_ID, STORE_NAME, compression = 24)
    save_contours(sf, 'ball', VIDEO_ID, STORE_NAME, compression = 6)
    save_homographies(sf, VIDEO_ID, STORE_NAME)

