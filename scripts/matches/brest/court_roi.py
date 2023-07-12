# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:45:37 2023

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

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, StreamList, MulStream, RandStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region

from ATMANpy.funcs.player_funcs import get_service_ROIs
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

    ### Make folders for feature storage
    featPath = ut.get_folder_struct('features_stored')['main']
    feat_vidPath = os.path.join(featPath, VIDEO_ID)
    try:
        os.mkdir(feat_vidPath)
    except FileExistsError:
        pass
    
    storePath = os.path.join(feat_vidPath, STORE_NAME)
    try:
        os.mkdir(storePath)
    except FileExistsError:
        pass
    
    randStm = RandStream(cap, START_FRAME, stop_frame, number_of_frames = 1000,
                         cap_color_space = 'bgr', target_color_space = 'grey',
                         scale_factor = ADD_SCALE)
    
    ### Set up a region of interest to focus attention on court. 
    # -> region of interest will be saved as a StillFrame of Mask type
    some_frame = 300
    someFrm = randStm[some_frame]
    someFrm.show()
    
    p1, p2, p3, p4 = Point2D((200, 0)), Point2D((440, 0)), Point2D((0, 295)), Point2D((width, 295))
    ROI = Region.from_points(p1, p2, p3, p4)
    someFrm.put(ROI).show()
    
    savePath = ff.get_and_make_save_path('courtroiReg', VIDEO_ID, STORE_NAME)
    fileName = f'from_{START_FRAME}_to_{stop_frame}.npy'
    fullPath = os.path.join(savePath, fileName)
    roiArr = ff.extract_region_values(ROI)
    np.save(fullPath, roiArr)
    
    ### Set up two regions of interest for where players might stand when serving and returning.
    get_service_ROIs(schemaLines)
    
    
    roiMask.save_like_feature(VIDEO_ID, STORE_NAME, start_frame = START_FRAME, stop_frame = stop_frame)
    
    
    
    
    
    
    