# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:16:32 2023

@author: gianl
"""


import os
import time

import cv2 as cv
import numpy as np
import pandas as pd

### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video
from ATMANpy.funcs.cv_funcs import join_video_captures

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines, NetLines

import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

START_FRAME = 0
STOP_FRAME = 100000
VIDEO_ID = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
STORE_NAME = 'svm_run'
NUM_WORKERS = 24
CHUNKSIZE = 1000

cap = get_video(VIDEO_ID)

width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
NATURAL_SCALE = width / 1920
ADD_SCALE = 1 
SCALE_FACTOR = NATURAL_SCALE * ADD_SCALE


if __name__ == '__main__':

    frameList = list(range(START_FRAME, STOP_FRAME + 1))
    HSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['H'], frameList = frameList).squeeze()
    H = HSF.feature
    courtLines = sa.find_courtlines(H)
    courtlineSF = ConstFeature('courtLines', courtLines, frameList)
    
    ######
    gStm = Stream(cap, START_FRAME, STOP_FRAME, target_color_space = 'grey').attach(courtlineSF)
    
    WIDTH_OF_LINE_AT_FULL_HD = 8
    WIDTH_OF_LINE = WIDTH_OF_LINE_AT_FULL_HD * SCALE_FACTOR
    params = {'width_of_line': WIDTH_OF_LINE, 
              'int_diff_thresh': 10,
              'white_thresh': 150,
              'zoom_factor': 1.2,
              'dil_kernel': (2, 2)}
    
    matchSF = gStm.extract_mp(ex.court_match_rates, NUM_WORKERS, **params)
    
    matchSF.to_dataframe().plot() ### check this, to see where to set thresholds!!
    
    FP_THRESHOLD = None
    FN_THRESHOLD = 0.3
    
    courtBF = matchSF.apply(sa.threshold_false_positives_negatives, 
                            fp_threshold = FP_THRESHOLD, fn_threshold = FN_THRESHOLD).to_bool()
    courtBF.save(VIDEO_ID, STORE_NAME)
