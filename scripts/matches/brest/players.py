# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 18:03:27 2023

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

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
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

if __name__ == '__main__':
    frameList = list(range(START_FRAME, STOP_FRAME + 1))
    
    ### Read constant Homography and Projection Matrix
    HPSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['H', 'P'], frameList = list(range(START_FRAME, STOP_FRAME + 1)))
    courtSF = HPSF['H'].apply(sa.find_courtlines)

    ### Read Segmentation Boolean
    courtBF = BoolFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['isCourt']).squeeze()    
    
    ### Add empty player feature
    mySF = HPSF.to_SF().add('topPlayer', None).add('botPlayer', None).drop('P')
    mySFs = mySF.filter(courtBF).split()
    
    burnStms = StreamList.from_store('foregroundMask', VIDEO_ID, STORE_NAME, cap_color_space = 'bool', target_color_space = 'mask')
    shadStms = StreamList.from_store('shadowMask', VIDEO_ID, STORE_NAME, cap_color_space = 'bool', target_color_space = 'mask') 
    
    greyCtms = StreamList.from_SFList(mySFs, cap, 
                                      cache_size = 1, force_fill_cache = True,
                                      cap_color_space = 'bgr', target_color_space = 'grey')
    
    ### Set Up ROIs
    # get courtROI to focus on the court itself
    courtroiSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['courtroiReg'], 
                                          featTypeList = [Region], frameList = frameList).squeeze()
    courtROI = courtroiSF.feature

    # set up player ROIs using projected court regions
    params = {'width_factor': 1, 'depth_factor': 0.4, 
              'add_depth_factor_bottom': 0.5, 'add_depth_factor_top': 0.5}
    initial_ROI_lower, initial_ROI_upper = get_service_ROIs(schemaLines, **params)
    halfcourtLine_lower, halfcourtLine_upper = get_half_court_Lines(schemaLines)
    # schemaFrm.put(initial_ROI_lower).show()
    
    ### Run Player Extraction
    params = {'player_tracking_scale_factor': 1.5,
              'bot_min_bounding_height': 200 * SCALE_FACTOR,
              'bot_min_bounding_width': 80 * SCALE_FACTOR,
              'top_min_bounding_height': 150 * SCALE_FACTOR,
              'top_min_bounding_width': 60 * SCALE_FACTOR,
              'initial_ROI_upper': initial_ROI_upper,
              'initial_ROI_lower': initial_ROI_lower,
              'halfcourtLine_upper': halfcourtLine_upper,
              'halfcourtLine_lower': halfcourtLine_lower}
    
    mulStms = StreamList.combine(greyCtms, burnStms, shadStms)
    plSFs = mulStms.extract_chunked_mp(ex.find_both_players_smart, NUM_WORKERS, CHUNKSIZE, **params)
    
    ### 
    plStms = StreamList.from_SFList(plSFs, cap)    
    plStms[0].show()
    
    plStms[0][0].show()

    plStms[0].show()


