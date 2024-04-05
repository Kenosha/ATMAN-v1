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

def load(vid, sto):
    ### Read Players
    playerSF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])
    
    ### Read Fine Motion Mask 
    motiStms = StreamList.from_store('fine_motMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask') 

    return playerSF, motiStms

def save(filtSFs, vid, sto, compression = 24):
    filtSFs.rename('fine_motMask_filt').save(vid, sto, compression = compression)

def main(conf, filter_params, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    savekwargs = {'lossless': conf['lossless']}

    playerSF, motiStms = load(vid, sto)
    
    playerSFs = playerSF.split()
    motiStms = motiStms.conform(playerSFs).attach(playerSFs)
        
    filtSFs = motiStms.extract_chunked_mp_to_disc(ex.filter_player_movement, nwo, nch, vid, sto, 
                                                  savekwargs, verbose = True, **filter_params)
    #filtStms = StreamList.from_SFList(filtSFs)

    #save(filtSFs, vid, sto, compression = conf['compression_ball'])

if __name__ == '__main__':
    
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in ut.get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[6])
    
    ### Filter out Players from fine motion mask
    filter_params = prm_mod.ball_filtered_motion_mask.filter_params

    name = ppl._name

    main(conf, filter_params, name = name)
    
    