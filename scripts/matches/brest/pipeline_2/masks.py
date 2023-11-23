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


if __name__ == '__main__':
        
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    ### Read constant Homography and Projection Matrix
    HPSF = ConstFeatures.from_store(vid, sto, ['H', 'P'])

    ### Read Segmentation Boolean
    courtBF = BoolFeature.from_store(vid, sto, featname = 'isCourt')    
    HSFs = HPSF['H'].filter(courtBF).split()

    ### Read median and standard deviation background
    medStms = StreamList.from_store('median_backFrm', vid, sto, target_color_space = 'hsv')
    stdStms = StreamList.from_store('std_backFrm', vid, sto, target_color_space = 'hsv')
    
    ### Prepare HSV MulStreams
    hsvStms = StreamList.from_SFList(HSFs, cap, target_color_space = 'hsv')
    mulStms = StreamList.combine(hsvStms, medStms, stdStms)
    
    ### Set Up Court ROI
    courtroiSF = ConstFeature.from_store(vid, sto, 'courtroiReg', featType = Region)
    courtROI = courtroiSF.feature

    ### Burn-In Mask
    params =  {'ROI': courtROI,
               'h_threshold': 16, 
               'v_threshold': 16, 
               'gaussian_kernel': (round(21 * conf['scale_factor']), round(21 * conf['scale_factor'])), 
               'closing_kernel': (round(10 * conf['scale_factor']), round(18 * conf['scale_factor'])),
               'verbose': False}
    burnSFs = mulStms.extract_chunked_mp(ex.get_burn_in_mask_simple, nwo, nch, **params)
    burnSFs.save(vid, sto, lossless = conf['lossless'])
    
    ### Shadow Mask
    mulStms = StreamList.combine(hsvStms, medStms)
    params = {'ROI': courtROI,
              'h_threshold': 5,
              's_threshold': None,
              'v_threshold_lower': 0.75,
              'v_threshold_upper': 0.97,
              'opening_kernel': (round(15 * conf['scale_factor']), round(15 * conf['scale_factor'])),
              'closing_kernel': (round(15 * conf['scale_factor']), round(15 * conf['scale_factor'])),
              'back_frames_are_warped': False}
    shadSFs = mulStms.extract_chunked_mp(ex.get_shadow_mask, nwo, nch, **params)
    shadSFs.save(vid, sto, lossless = conf['lossless'])
    
    ### Motion Mask
    gCtms = StreamList.from_SFList(HSFs, cap, target_color_space = 'grey', 
                                   cache_size = 8, force_fill_cache = True)
    
    # a) a fine grained mask, for ball detection
    params = {'ROI': courtROI,
              'intensity_threshold': 5,
              'relevant_indices': [4, 6, 8],
              'open_kernel': None,
              'close_kernel': (3, 3)}
    motSFs = gCtms.extract_chunked_mp(ex.find_motion_mask, nwo, nch * 10, **params)
    motSFs = motSFs.rename('fine_motMask')
    motSFs.save(vid, sto, lossless = conf['lossless'])
    
    # # b) a rougher mask, for ballkid detection
    # params = {'ROI': courtROI,
    #           'intensity_threshold': 10,
    #           'relevant_indices': [4, 6, 8],
    #           'close_kernel': (17, 17),
    #           'open_kernel': (4, 4)}
    # motSFs = gCtms.extract_chunked_mp(ex.find_motion_mask, nwo, nch * 10, **params)
    # motSFs = motSFs.rename('rough_motMask')
    # motSFs.save(vid, sto, lossless = conf['lossless'])
    
    
    