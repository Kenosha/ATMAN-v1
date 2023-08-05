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
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)


if __name__ == '__main__':

    conf = ut.read_config('brest')
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']

    frameList = list(range(conf['start_frame'], conf['stop_frame'] + 1))
    
    ### Load Courtlines ###
    H = ConstFeatures.from_store(vid, sto, featnameList = ['H'], frameList = frameList).squeeze()
    courtLines = sa.find_courtlines(H.feature)
    base_length = courtLines['Bottom Baseline'].length() ### take the larger of the two (bot side) so that our maximum ratio is larger!
    side_length = courtLines['Bottom Singles Ad Line'].length() + courtLines['Top Singles Deuce Line'].length()
    
    ### Load Candidate Contours ###
    cntSF = StreamFeatures.from_store(vid, sto, featnameList = ['candCnts']).squeeze()
    chamfSF = StreamFeatures.from_store(vid, sto, featnameList = ['chamferCnts'])
    playBF = BoolFeatures.from_store(vid, sto, featnameList = ['isPlay']).squeeze()
    
    cntSFs = cntSF.conform(playBF.filter(playBF)).split()
    chamfSFs = chamfSF.conform(playBF.filter(playBF)).split()
    
    ### Simulate Particles ###
    # a) create SF with player chamfers and ball candidate contours
    mySFs = chamfSFs.merge(cntSFs)
    
    # b) Initialize Particle Filter Model
    from ATMANpy.models.ParticleFilter import ParticleFilter
    params = {'process_standard_deviation': 3, #5 * conf['scale_factor'],
              'observation_standard_deviation': 3, #5 * conf['scale_factor'],
              'spatial_lambda': (10 ** -7) / conf['scale_factor']}
    myFilter = ParticleFilter(**params)
    
    # c) run particle filter
    width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    params = {'frame_width': width,
              'frame_height': height,
              'pixel_per_meter_horizontal': base_length / const.COURT_WIDTH,
              'pixel_per_meter_vertical': side_length / const.SIDE_LENGTH,
              'N': 1000,
              'fps': 25,
              'rectify': True}
    partSFs = mySFs.apply_whole_mp(myFilter.find_particles, nwo, **params) 
    
    ### Kitagawa Smoothing
    kitaSFs = partSFs.apply_whole_mp(sf.kitagawa_smooth, nwo)
    
    ### Choosing likely ball contour
    params = {'pFilter': myFilter,
              'standard_deviation_threshold': 2}
    jntSFs = kitaSFs.merge(mySFs.get_feature('candCnts'))
    choseSFs = jntSFs.apply(sa.choose_ball_contour, **params)
    
    ### TESTING ###
    from ATMANpy.funcs.particle_funcs_vectorized import particles_to_points
    
    seqNum = 8
    
    partSF = partSFs[seqNum]['particles']
    ptSF = partSF.apply(particles_to_points, particles_to_produce = 100)
    Stream.from_SF(ptSF, cap).show()
    
    partSF = kitaSFs[seqNum]['smooth_particles']
    ptSF = partSF.apply(particles_to_points, particles_to_produce = 100)
    Stream.from_SF(ptSF, cap).show()
    
    Stream.from_SF(choseSFs[seqNum], cap).show()
    
    