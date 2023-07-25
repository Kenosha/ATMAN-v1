# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 21:05:27 2023

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

    ### Read Segmentation Boolean
    courtBF = BoolFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['isCourt']).squeeze()    
    
    ### Read ROIs
    courtroiSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['courtroiReg'], 
                                          featTypeList = [Region], frameList = frameList).squeeze()
    courtROI = courtroiSF.feature
    
    kidroiSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['ballkidROIs'], 
                                         featTypeList = [Regions], frameList = frameList).squeeze()
    kidROIs = kidroiSF.feature
    
    ### Label the ROIs depending on position
    namedROIs = []
    cntPnts = kidROIs.centroids()
    
    court = courtSF.feature
    ordered_labelDict = {'Mid Left': court['Net Line'].leftPt,
                         'Mid Right': court['Net Line'].rightPt,
                         'Top Left': court['Top Baseline'].leftPt,
                         'Top Right': court['Top Baseline'].rightPt,
                         'Bot Left': court['Bottom Baseline'].leftPt,
                         'Bot Right': court['Bottom Baseline'].rightPt}
    for label, myPt in ordered_labelDict.items():
        nearest_idx = np.argmin([cntPt.distance_to(myPt) for cntPt in cntPnts])
        nearest_centroid = cntPnts[nearest_idx]
        nearest_region = [roi for roi in kidROIs if roi.centroid() == nearest_centroid][0]
        namedROIs.append(nearest_region.rename(label))
        cntPnts.pop(nearest_idx)
        
    namedRois = Regions(namedROIs)
        
    ### Read Relevant Masks
    burnStms = StreamList.from_store('foregroundMask', VIDEO_ID, STORE_NAME, cap_color_space = 'bool', target_color_space = 'mask')
    shadStms = StreamList.from_store('shadowMask', VIDEO_ID, STORE_NAME, cap_color_space = 'bool', target_color_space = 'mask') 
    motiStms = StreamList.from_store('rough_motMask', VIDEO_ID, STORE_NAME, cap_color_space = 'bool', target_color_space = 'mask') 
    
    ### Find motion contours
    params = {'min_area_of_contours': 10,
              'ROI': None, ### 6x slower when specifying a ROI!
              'max_number_of_contours': None}
    cntSFs = motiStms.rename('intMask').extract_chunked_mp(ex.get_contours_from_mask, NUM_WORKERS, CHUNKSIZE, **params)
    
    ### Find ROI Intersection Proportions
    ballkidBFs = cntSFs.apply_whole(sf.find_ballkid_sequences, namedROIs)
    ballkidBFs.save(VIDEO_ID, STORE_NAME)
    
    
    ### TODO: Stop play sequences a bit later 
        # -> often, ballkid movememnt starts at or slightly before the second bounce
    ### TODO: Resegmentation by hand, in cases where theres a problem: Needs an interface, etc.
    
    
    