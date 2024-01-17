# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:26:01 2023

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

import ATMANpy.auxil.constants as const
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
from ATMANpy.types.Contours import Contour, Contours

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points

from ATMANpy.models.ParticleFilter import ParticleFilter

def load(vid, sto):
    HCF = ConstFeature.from_store(vid, sto, featname = 'H')
    cntSF = StreamFeature.from_store(vid, sto, featname = 'candCnts')
    hitcntSF = StreamFeature.from_store(vid, sto, featname = 'hitCnt')
    chamfSF = StreamFeature.from_store(vid, sto, featname = 'chamferCnts')
    playBF = BoolFeature.from_store(vid, sto, featname = 'isRally')
    return HCF, cntSF, hitcntSF, chamfSF, playBF
    
def save(chosePtSFs, vid, sto):
    chosePtSFs.rename('ball_choicePt').save(vid, sto)

def main(conf, filter_init_params, filter_run_params, ball_choice_params):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = get_video(vid)
    width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    
    HCF, cntSF, hitcntSF, chamfSF, playBF = load(vid, sto)
    
    ### Load Courtlines ###
    H = HCF.feature
    schemaLines = CourtLines.make().add_singles_netline()
    courtLines = sa.find_courtlines(H, schemaLines = schemaLines)
    base_length = courtLines['Bottom Baseline'].length() ### take the larger of the two (bot side) so that our maximum ratio is larger!
    side_length = courtLines['Bottom Singles Ad Line'].length() + courtLines['Top Singles Deuce Line'].length()
    
    ### Combine ball contours with hypothesized hit contours
    cntplusSF = cntSF.merge(hitcntSF, how = 'outer').apply(sa.merge_contour_into_contours)

    cntSFs = cntSF.conform(playBF.filter(playBF)).split()
    cntplusSFs = cntplusSF.conform(playBF.filter(playBF)).split()
    chamfSFs = chamfSF.conform(playBF.filter(playBF)).split()
    
    ### Simulate Particles ###
    # a) create SF with player chamfers and ball candidate contours
    mySFs = chamfSFs.merge(cntplusSFs.rename('candCnts'))
    
    # b) Initialize Particle Filter Model
    myFilter = ParticleFilter(**filter_init_params)
    
    # c) run particle filter
    filter_run_params_additional = {'frame_width': width,
                                    'frame_height': height,
                                    'pixel_per_meter_horizontal': base_length / const.COURT_WIDTH,
                                    'pixel_per_meter_vertical': side_length / const.SIDE_LENGTH}
    partSFs = mySFs.apply_whole_mp(myFilter.find_particles, nwo, **filter_run_params, **filter_run_params_additional) 
    
    ### Kitagawa Smoothing
    kitaSFs = partSFs.apply_whole_mp(sf.kitagawa_smooth, nwo)
    
    ### Choosing likely ball contour --> NOTE THAT WE ONLY USE THE CAND CONTOURS, NOT THE HIT CONTOURS
    jntSFs = kitaSFs.merge(cntSFs)
    choseSFs = jntSFs.apply(sa.choose_ball_contour, **ball_choice_params, pFilter = myFilter)
    chosePtSFs = choseSFs.apply(sa.maybe_center)

    save(chosePtSFs, vid, sto)

if __name__ == '__main__':

    conf = ut.read_config('brest', config_name = '_config_pipe2')
    
    filter_init_params = {'process_standard_deviation': 1.5,
                          'observation_standard_deviation': 2,
                          'spatial_lambda': 10 ** -4}
    
    filter_run_params = {'N': 1500,
                         'fps': 25,
                         'rectify': True}
    
    ball_choice_params = {'standard_deviation_threshold': 2}
    
    main(conf, filter_init_params, filter_run_params, ball_choice_params)
    
    
    
    
    
    
    
    