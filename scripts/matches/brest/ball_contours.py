# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 12:28:19 2023

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

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region, Regions
from ATMANpy.types.Contours import Contour

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

START_FRAME = 0
STOP_FRAME = 100000
VIDEO_ID = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
STORE_NAME = 'svm_run'
NUM_WORKERS = 24
CHUNKSIZE = 5000

cap = get_video(VIDEO_ID)

width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
NATURAL_SCALE = width / 1920
ADD_SCALE = 1 
SCALE_FACTOR = NATURAL_SCALE * ADD_SCALE


if __name__ == '__main__':
    frameList = list(range(START_FRAME, STOP_FRAME + 1))
    
    ### Read constant Homography and Projection Matrix
    HPSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['H', 'P'], frameList = list(range(START_FRAME, STOP_FRAME + 1)))
    courtSF = HPSF['H'].apply(sa.find_courtlines)

    ### Read Segmentation Booleans
    # courtBF = BoolFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['isCourt']).squeeze()    
    ballkidBF = BoolFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['ballkidBool']).squeeze()   
    playBF = ballkidBF.neg().conform(frameList).fillna(False).rename('isPlay')
    
    ### Read Players
    playerSFs = StreamFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['topPlayer', 'botPlayer']).split()
    
    ### Read Fine Motion Mask 
    burnStms = StreamList.from_store('foregroundMask', VIDEO_ID, STORE_NAME, cap_color_space = 'bool', target_color_space = 'mask') 
    burnStms = burnStms.conform(playerSFs).attach(playerSFs).rename('motMask')
    
    motiStms = StreamList.from_store('fine_motMask', VIDEO_ID, STORE_NAME, cap_color_space = 'bool', target_color_space = 'mask') 
    motiStms = motiStms.conform(playerSFs).attach(playerSFs).rename('motMask')
    
    ### Filter out Players from fine motion mask
    params = {'dil_kernel': (5, 3)}
    filtSFs = motiStms.extract_chunked_mp(ex.filter_player_movement, NUM_WORKERS, CHUNKSIZE, **params)
    
    ### Get contours 
    filtStms = motiStms.from_SFList(filtSFs, None)
    
    params = {'ROI': None, 
              'min_area_of_contours': 0, 
              'max_number_of_contours': None}
    cntSFs = filtStms.extract_chunked_mp(ex.get_contours_from_mask, NUM_WORKERS, CHUNKSIZE, **params)

    cntSFs.save(VIDEO_ID, STORE_NAME, compression = 24)
