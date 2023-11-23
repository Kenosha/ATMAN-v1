# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 18:03:27 2023

@author: gianl
"""

import sys
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
        
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    FRAME_BLUR = 1
    
    ### Get Streams, and conform them to the segmentation
    burnCtms = StreamList.from_store('foregroundMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask',
                                     cache_size = FRAME_BLUR, force_fill_cache = True)
    shadCtms = StreamList.from_store('shadowMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask',
                                     cache_size = FRAME_BLUR, force_fill_cache = True)
    
    ### Set Up courtROI to focus on the court itself
    courtroiSF = ConstFeature.from_store(vid, sto, featname = 'courtroiReg', featType = Region)
    courtROI = courtroiSF.feature

    ### Find Player Candidate Contours
    params = {'min_player_size': 50, 
              'ROI': courtroiSF.feature,
              'frame_blur': FRAME_BLUR}
    mulStms = StreamList.combine(burnCtms, shadCtms)
    candSFs = mulStms.extract_chunked_mp(ex.find_player_candidate_contours_timeblur, nwo, nch, **params)
    candSFs.save(vid, sto, compression = conf['compression_players'])

