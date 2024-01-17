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

def load(vid, sto):
    ### Read Segmentation Booleans
    playBF = BoolFeature.from_store(vid, sto, featname = 'isRally')   
    
    ### Read Players
    playerSF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])
    
    ### Read Fine Motion Mask 
    motiStms = StreamList.from_store('fine_motMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask') 

    return playBF, playerSF, motiStms

def save(cntSFs, vid, sto):
    cntSFs.rename('moveCnts').save(vid, sto, compression = conf['compression_ball'])

def main(conf, filter_params, contour_params):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']

    playBF, playerSF, motiStms = load(vid, sto)
    
    playerSFs = playerSF.split()
    motiStms = motiStms.conform(playerSFs).attach(playerSFs).rename('motMask')
        
    filtSFs = motiStms.extract_chunked_mp(ex.filter_player_movement, nwo, nch, **filter_params)

    ### Get contours 
    filtStms = motiStms.from_SFList(filtSFs, None)
    
    cntSFs = filtStms.extract_chunked_mp(ex.get_contours_from_mask, nwo, nch, **contour_params)

    save(cntSFs, vid, sto)

if __name__ == '__main__':
            
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    
    ### Filter out Players from fine motion mask
    filter_params = {'top_player_dil_kernel': (12, 7),
                     'bot_player_dil_kernel': (20, 12)}
    
    contour_params = {'ROI': None, 
                      'min_area_of_contours': 0, 
                      'max_number_of_contours': None}
    
    main(conf, filter_params, contour_params)
    
    