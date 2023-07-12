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
RUN_LENGTH = 10000
VIDEO_ID = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
STORE_NAME = 'test'
NUM_WORKERS = 24
CHUNKSIZE = 1000

stop_frame = START_FRAME + RUN_LENGTH
cap = get_video(VIDEO_ID)

width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
NATURAL_SCALE = width / 1920
ADD_SCALE = 1 
SCALE_FACTOR = NATURAL_SCALE * ADD_SCALE


if __name__ == '__main__':
    ### TODO: Use roiMask created by court_roi.py
    
    ### Read constant Homography and Projection Matrix
    HPSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['H', 'P'], frameList = list(range(START_FRAME, stop_frame + 1)))

    ### Read median and standard deviation background
    fileName = f'from_{START_FRAME}_to_{stop_frame}.npy'
    medStm = StillStream.from_store(fileName, 'median_backFrm', VIDEO_ID, STORE_NAME)
    stdStm = StillStream.from_store(fileName, 'std_backFrm', VIDEO_ID, STORE_NAME)
    
    ### Compare Frame and Background, find overlap metric
    myStm = Stream(cap, START_FRAME, stop_frame)
    mulStm = MulStream.from_streams(myStm, medStm, stdStm)
    matchSF = mulStm.extract(ex.frame_comparison, std_multiple = 3) 
    matchBF = matchSF.apply(sa.exceeds_threshold, threshold = 0.75).to_bool().rename('isCourt')
    ### in principle, segmentation like so works, 
    # but camera cut-aways are not a good way of segmenting anymore, 
    # since the camera rarely (or never) cuts away... 
    # the only thing that happens is a blend-in graphic after games, for example.
    
    matchBF.save(VIDEO_ID, STORE_NAME)
