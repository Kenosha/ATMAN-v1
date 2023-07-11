# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:28:10 2023

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

from ATMANpy.types.Streams import Stream, CachedStream, DecapStream, NCStream, NCDecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import Frame, GreyFrame, BGRFrame, HSVFrame, MaskFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

START_FRAME = 0
STOP_FRAME = 100
VIDEO_ID = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
STORE_NAME = 'test'
NUM_WORKERS = 24
CHUNKSIZE = 1000
VERBOSE = False

cap = get_video(VIDEO_ID)

width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
NATURAL_SCALE = width / 1920
ADD_SCALE = 1 
SCALE_FACTOR = NATURAL_SCALE * ADD_SCALE

if __name__ == '__main__':
    frameList = list(range(START_FRAME, STOP_FRAME + 1))
    ### Read constant Homography and Projection Matrix
    HPSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['H', 'P'], frameList = frameList)
    courtSF = HPSF['H'].apply(sa.find_courtlines)

    ### Read median and standard deviation background
    fileName = 'from_0_to_10000.npy'
    medStm = StillStream.from_store(fileName, 'median_backFrm', VIDEO_ID, STORE_NAME, target_color_space = 'hsv').conform(START_FRAME, STOP_FRAME)
    stdStm = StillStream.from_store(fileName, 'std_backFrm', VIDEO_ID, STORE_NAME, target_color_space = 'hsv').conform(START_FRAME, STOP_FRAME)
    
    hsvStm = Stream.from_SF(HPSF['H'], cap, target_color_space = 'hsv')
    
    mulStm = MulStream([hsvStm, medStm, stdStm])
    
    ### get courtROI to focus on the court itself
    courtroiSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['courtroiReg'], 
                                          featTypeList = [Region], frameList = frameList).squeeze()
    courtROI = courtroiSF.feature
    
    ### set up player ROIs using projected court regions
    params = {'width_factor': 1, 'depth_factor': 0.4, 
              'add_depth_factor_bottom': 0.5, 'add_depth_factor_top': 0.5}
    initial_ROI_lower, initial_ROI_upper = get_service_ROIs(schemaLines, **params)
    halfcourtLine_lower, halfcourtLine_upper = get_half_court_Lines(schemaLines)
    if VERBOSE:
        schemaFrm.put(initial_ROI_lower).show()
        
        proj_lower = initial_ROI_lower.transform('apply_homography', HPSF['H'].feature).transform('round_points')
        proj_upper = initial_ROI_upper.transform('apply_homography', HPSF['H'].feature).transform('round_points')
        
        hsvStm[0].put(proj_lower).show()
        hsvStm[0].put(proj_upper).show()
    
    ### Sort out Parameters
    player_params = {'player_tracking_scale_factor': 1.5,
              'bot_min_bounding_height': 200 * SCALE_FACTOR,
              'bot_min_bounding_width': 80 * SCALE_FACTOR,
              'top_min_bounding_height': 150 * SCALE_FACTOR,
              'top_min_bounding_width': 60 * SCALE_FACTOR,
              'initial_ROI_upper': initial_ROI_upper,
              'initial_ROI_lower': initial_ROI_lower,
              'halfcourtLine_upper': halfcourtLine_upper,
              'halfcourtLine_lower': halfcourtLine_lower}
    
    burnDict = {}
    for minimum_proportion in [0.00]: ### this seems to be non-binding...
        for minimum_step_gain in [10 ** -5, 10 ** -4.5, 10 ** -4, 10 ** -3.5, 10 ** -3]:
            key = f'p{minimum_proportion}_s{minimum_step_gain}'
            
            burn_params = {'ROI': courtROI,
                           'minimum_proportion': minimum_proportion, 
                           'minimum_step_gain': minimum_step_gain, 
                           'gaussian_kernel': (round(21 * SCALE_FACTOR), round(21 * SCALE_FACTOR)), 
                           'closing_kernel': (round(10 * SCALE_FACTOR), round(18 * SCALE_FACTOR)),
                           'back_frames_are_warped': False,
                           'normalize': False,
                           'verbose': False}
            
            ### Burn-In Mask
            burnSF = mulStm.extract_mp(ex.get_burn_in_mask_iterative, NUM_WORKERS, **burn_params)
            burnStm = DecapStream.from_frameSF(burnSF)
    
            burnDict[key] = burnStm
            
            burnStm.show()
    
    sumDict = {frm: myStm.frames.sum((0, 1, 2)) / 255 for frm, myStm in burnDict.items()}
    
    ###
    frmNum = 0
    
    hFrm = hsvStm[frmNum]
    median_backFrm = medStm.stillFrm
    std_backFrm = stdStm.stillFrm
    locals().update(burn_params)
    minimum_step_gain = 10 ** -3
    
    
    
    
    #%%% Shadow Mask Evaluation
    
    ##### Build a dictionary of player features, used to evaluate player finding
    gCtm = CachedStream.from_SF(HPSF['H'], cap, target_color_space = 'grey', 
                                cache_size = 1, force_fill_cache = True)    
    
    ### get null shadow
    falseFrm = MaskFrame.false(medStm.stillFrm).rename('shadowMask')
    nullshadSF = ConstFeature('shadowMask', falseFrm, frameList)
    nullshadStm = StillStream(falseFrm, START_FRAME, STOP_FRAME)
    
    mulStm = MulStream([gCtm, burnStm, nullshadStm])
    plSF = mulStm.extract(ex.find_both_players_smart, **player_params)
    
    shDict = {'null': nullshadSF}
    plDict = {'null': plSF}
    for h_threshold in [None, 5, 10]:
        for s_threshold in [None, 10, 20]:
                    
            ### Shadow Mask
            mulStm = MulStream([hsvStm, medStm])
            shadow_params = {'h_threshold': h_threshold,
                      's_threshold': s_threshold,
                      'v_threshold_lower': 0.75,
                      'v_threshold_upper': 0.97,
                      'opening_kernel': (round(15 * SCALE_FACTOR), round(15 * SCALE_FACTOR)),
                      'closing_kernel': (round(15 * SCALE_FACTOR), round(15 * SCALE_FACTOR)),
                      'back_frames_are_warped': False}
            shadSF = mulStm.extract_mp(ex.get_shadow_mask, NUM_WORKERS, **shadow_params)
            shadStm = DecapStream.from_frameSF(shadSF)
            
            ### Find Players
            mulStm = MulStream([gCtm, burnStm, shadStm])
            plSF = mulStm.extract(ex.find_both_players_smart, **player_params)
    
            key = f'h{h_threshold}_s{s_threshold}'
            plDict[key] = plSF
            shDict[key] = shadSF
    
    
    keys_of_interest = None ### clearly, thresholding saturation values is completely counterproductive...
    keys_of_interest = ['hNone_sNone', 'h5_sNone', 'h10_sNone']
    
    for key in plDict:
        if not keys_of_interest is None:
            if not key in keys_of_interest:
                continue
        
        plSF = plDict[key]
        shadSF = shDict[key]
        
        ### Add Key
        plSF = plSF.merge(ConstFeature('shadFlag', key, frameList))
        
        myStm = Stream(cap, START_FRAME, STOP_FRAME, cap_color_space = 'bgr')
        
        def find_contours(shadowMask):
            shadCnts, _ = shadowMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            return shadCnts
        shadcntSF = shadSF.apply(find_contours)
        shadcntSF = shadcntSF.merge(ConstFeature('shadFlag', key, frameList))
        
        ### Evaluate player finding
        myStm = Stream(cap, START_FRAME, STOP_FRAME, cap_color_space = 'bgr', features = shadcntSF)
        myStm.show(ms_between_frames = 50)
        
        myStm = Stream(cap, START_FRAME, STOP_FRAME, cap_color_space = 'bgr', features = plSF)
        myStm.show(ms_between_frames = 50)
        
        
    
    
    
    ### frames of interest
    
    ####### Look at a single problematic frame
    frmNum = 0
    
    h_threshold = None
    s_threshold = None
    shadow_params = {'h_threshold': h_threshold,
              's_threshold': s_threshold,
              'v_threshold_lower': 0.75,
              'v_threshold_upper': 0.97,
              'opening_kernel': (round(15 * SCALE_FACTOR), round(15 * SCALE_FACTOR)),
              'closing_kernel': (round(15 * SCALE_FACTOR), round(15 * SCALE_FACTOR)),
              'back_frames_are_warped': False,
              'verbose': True}
    key = f'h{h_threshold}_s{s_threshold}'
    plSF = plDict[key]
    nullSF = plDict['null']
    
    normStm = Stream(cap, START_FRAME, STOP_FRAME, target_color_space = 'hsv')
    featStm = normStm.attach(plSF)
    featStm.show()
    
    nullStm = normStm.attach(nullSF)
    nullStm.show()
    
    hFrm = normStm[frmNum]
    featFrm = featStm[frmNum]
    featFrm.show()
    
    H = HPSF['H'][frmNum]
    median_backFrm = medStm.stillFrm
    
    locals().update(shadow_params)
    
    cStm = Stream(cap, START_FRAME, STOP_FRAME)
    cStm.show()
    
    
    
    
    
    