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


if __name__ == '__main__':
        
    conf = ut.read_config('brest')
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    frameList = list(range(conf['start_frame'], conf['stop_frame']))
    
    ### Read constant Homography and Projection Matrix
    HPSF = ConstFeatures.from_store(vid, sto, ['H', 'P'], frameList = frameList)
    courtSF = HPSF['H'].apply(sa.find_courtlines)

    ### Read Segmentation Booleans
    # courtBF = BoolFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['isCourt']).squeeze()    
    playBF = BoolFeatures.from_store(vid, sto, featnameList = ['isPlay']).squeeze()   
    
    ### Add empty player feature
    mySF = HPSF.to_SF().add('topPlayer', None).add('botPlayer', None).drop('P')
    mySFs = mySF.filter(playBF).split()
    
    ### Get Streams, and conform them to the segmentation
    greyCtms = StreamList.from_SFList(mySFs, cap, 
                                      cache_size = 1, force_fill_cache = True,
                                      cap_color_space = 'bgr', target_color_space = 'grey')
    
    burnStms = StreamList.from_store('foregroundMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask')
    burnStms = burnStms.conform(greyCtms)
    shadStms = StreamList.from_store('shadowMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask') 
    shadStms = shadStms.conform(greyCtms)
    
    ### Set Up ROIs
    # get courtROI to focus on the court itself
    courtroiSF = ConstFeatures.from_store(vid, sto, ['courtroiReg'], 
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
              'bot_min_bounding_height': 200 * conf['scale_factor'],
              'bot_min_bounding_width': 80 * conf['scale_factor'],
              'top_min_bounding_height': 150 * conf['scale_factor'],
              'top_min_bounding_width': 60 * conf['scale_factor'],
              'initial_ROI_upper': initial_ROI_upper,
              'initial_ROI_lower': initial_ROI_lower,
              'halfcourtLine_upper': halfcourtLine_upper,
              'halfcourtLine_lower': halfcourtLine_lower}
    
    mulStms = StreamList.combine(greyCtms, burnStms, shadStms)
    plSFs = mulStms.extract_chunked_mp(ex.find_both_players_smart, nwo, nch, **params)
    
    plSFs.save(vid, sto, compression = 32)
    


