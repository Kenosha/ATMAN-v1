# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 01:12:48 2023

Building the burn-in, shadow, and motion masks.

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
STOP_FRAME = 10000
VIDEO_ID = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
STORE_NAME = 'test'
NUM_WORKERS = 24
CHUNKSIZE = 5000

cap = get_video(VIDEO_ID)

width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
NATURAL_SCALE = width / 1920
ADD_SCALE = 1 
SCALE_FACTOR = NATURAL_SCALE * ADD_SCALE
LOSSLESS = False ### there is a weird saving issue when creating the lossless files...

if __name__ == '__main__':
    frameList = list(range(START_FRAME, STOP_FRAME + 1))
    ### Read constant Homography and Projection Matrix
    HPSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['H', 'P'], frameList = frameList)
    courtSF = HPSF['H'].apply(sa.find_courtlines)

    ### Read Segmentation Boolean
    courtBF = BoolFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['isCourt']).squeeze()    
    courtSFs = courtSF.filter(courtBF).split()
    HSFs = HPSF['H'].filter(courtBF).split()

    ### Read median and standard deviation background
    fileName = f'from_{START_FRAME}_to_{STOP_FRAME}.npy'
    fileName = 'from_0_to_10000.npy'
    medStm = StillStream.from_store(fileName, 'median_backFrm', VIDEO_ID, STORE_NAME)
    stdStm = StillStream.from_store(fileName, 'std_backFrm', VIDEO_ID, STORE_NAME)
    
    ### Prepare HSV MulStreams
    hsvStms = StreamList.from_SFList(HSFs, cap, target_color_space = 'hsv')
    medStms = StreamList.from_SFList(HSFs, medStm.stillFrm, target_color_space = 'hsv')
    stdStms = StreamList.from_SFList(HSFs, stdStm.stillFrm, target_color_space = 'hsv')
    
    mulStms = StreamList.combine(hsvStms, medStms, stdStms)
    
    ### Set Up Court ROI
    courtroiSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['courtroiReg'], 
                                          featTypeList = [Region], frameList = frameList).squeeze()
    courtROI = courtroiSF.feature
    
    ### Burn-In Mask
    params =  {'ROI': courtROI,
               'h_threshold': 16, 
               'v_threshold': 16, 
               'gaussian_kernel': (round(21 * SCALE_FACTOR), round(21 * SCALE_FACTOR)), 
               'closing_kernel': (round(10 * SCALE_FACTOR), round(18 * SCALE_FACTOR)),
               'verbose': False}
    burnSFs = mulStms.extract_chunked_mp(ex.get_burn_in_mask_simple, NUM_WORKERS, CHUNKSIZE, **params)
    burnSFs.save(VIDEO_ID, STORE_NAME, lossless = LOSSLESS)
    
    ### Shadow Mask
    mulStms = StreamList.combine(hsvStms, medStms)
    params = {'ROI': courtROI,
              'h_threshold': 5,
              's_threshold': None,
              'v_threshold_lower': 0.75,
              'v_threshold_upper': 0.97,
              'opening_kernel': (round(15 * SCALE_FACTOR), round(15 * SCALE_FACTOR)),
              'closing_kernel': (round(15 * SCALE_FACTOR), round(15 * SCALE_FACTOR)),
              'back_frames_are_warped': False}
    shadSFs = mulStms.extract_chunked_mp(ex.get_shadow_mask, NUM_WORKERS, CHUNKSIZE, **params)
    shadSFs.save(VIDEO_ID, STORE_NAME, lossless = LOSSLESS)
    
    ### Motion Mask
    gCtms = StreamList.from_SFList(HSFs, cap, target_color_space = 'grey', 
                                   cache_size = 8, force_fill_cache = True)
    # a) a fine grained mask, for ball detection
    params = {'ROI': courtROI,
              'intensity_threshold': 10,
              'relevant_indices': [4, 6, 8],
              'open_kernel': None,
              'close_kernel': (3, 3)}
    motSFs = gCtms.extract_chunked_mp(ex.find_motion_mask, NUM_WORKERS, CHUNKSIZE * 10, **params)
    motSFs = motSFs.rename('fine_motMask')
    motSFs.save(VIDEO_ID, STORE_NAME, lossless = LOSSLESS)
    
    # b) a rougher mask, for ballkid detection
    params = {'ROI': courtROI,
              'intensity_threshold': 10,
              'relevant_indices': [4, 6, 8],
              'close_kernel': (17, 17),
              'open_kernel': (4, 4)}
    motSFs = gCtms.extract_chunked_mp(ex.find_motion_mask, NUM_WORKERS, CHUNKSIZE * 10, **params)
    motSFs = motSFs.rename('rough_motMask')
    motSFs.save(VIDEO_ID, STORE_NAME, lossless = LOSSLESS)
    
    
    