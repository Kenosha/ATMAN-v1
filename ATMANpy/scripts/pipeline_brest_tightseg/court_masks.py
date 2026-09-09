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
import ATMANpy.funcs.multi_extract_funcs as mx
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points

def load(vid, sto):
    HSF = ConstFeature.from_store(vid, sto, featname = 'H') 
    courtroiSF = ConstFeature.from_store(vid, sto, featname = 'courtroiReg', featType = Region)
    segBF = BoolFeature.from_store(vid, sto, featname = 'isRally_sound')
    
    ### Read median and standard deviation background
    medStms = StreamList.from_store('median_backFrm', vid, sto, target_color_space = 'hsv')
    stdStms = StreamList.from_store('std_backFrm', vid, sto, target_color_space = 'hsv')
    
    return HSF, courtroiSF, segBF, medStms, stdStms

def save(burnSFs, shadSFs, motSFs, vid, sto, lossless = True):
    burnSFs.rename('foregroundMask').save(vid, sto, lossless = lossless)
    shadSFs.rename('shadowMask').save(vid, sto, lossless = lossless)
    motSFs.rename('fine_motMask').save(vid, sto, lossless = lossless)

def find_fine_motion_mask(gFrms, **kwargs):
    # Streaming saves use the returned variable name as the feature directory.
    fine_motMask = mx.find_motion_mask(gFrms, **kwargs)
    return fine_motMask

def main(conf, burn_in_params, shadow_params, motion_params, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    save_params = {'lossless': conf['lossless']}
    
    HSF, courtroiSF, segBF, medStms, stdStms = load(vid, sto)

    HSFs = HSF.filter(segBF).split()
    courtROI = courtroiSF.feature
    
    hsvStms = StreamList.from_SFList(HSFs, cap, target_color_space = 'hsv')
    mulStms = StreamList.combine(hsvStms, medStms)
    
    ### Burn In Background Mask
    mulStms.apply_chunked_mp_to_disc(mx.get_burn_in_mask_simple, nwo, nch, vid, sto, save_params, **burn_in_params, ROI = courtROI)
    
    ### Shadow Mask
    mulStms.apply_chunked_mp_to_disc(mx.get_shadow_mask_simple, nwo, nch, vid, sto, save_params, **shadow_params, ROI = courtROI)
    
    ### Motion Mask
    gStms = StreamList.from_SFList(HSFs, cap, target_color_space = 'grey')
    gStms.apply_chunked_mp_to_disc(find_fine_motion_mask, nwo, 10 * nch, vid, sto, save_params, **motion_params, ROI = courtROI)
    
    # save(burnSFs, shadSFs, motSFs, vid, sto, lossless = conf['lossless'])
    

if __name__ == '__main__':
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in ut.get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[0])
    
    burn_in_params = prm_mod.court_masks.burn_in_params
    shadow_params = prm_mod.court_masks.shadow_params
    motion_params = prm_mod.court_masks.motion_params
    
    name = ppl._name

    main(conf, burn_in_params, shadow_params, motion_params, name = name)
    