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

def load(vid, sto):
    hitSF = StreamFeature.from_store(vid, sto, featname = 'hitFlag')
    plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])
    return hitSF, plySF

def save(hitcntSF, vid, sto, **save_params):
    hitcntSF.rename('hitCnt').save(vid, sto, **save_params)

def main(conf, params):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...

    hitSF, plySF = load(vid, sto)

    ### Combine hits detected with player location to add hit candidate points
    mySF = hitSF.merge(plySF, how = 'left')

    ## TODO: maybe add more than just one candidate contour, let the particle filter decide which one is best?
    hitcntSF = mySF.apply(sa.get_inferred_hit_points, **params).dropna()
    
    save(hitcntSF, vid, sto, compression = params['resolution'])

if __name__ == '__main__':

    conf = ut.read_config('brest', config_name = '_config_pipe2')
    
    params = {'stroke_px_r_top': 10,
              'stroke_px_r_bot': 20,
              'stroke_px_u_top': -2,
              'stroke_px_u_bot': 10,
              'serve_px_u_top': 20,
              'serve_px_u_bot': 20,
              'do_bot': False,
              'do_top': True,
              'hand_top': 'right',
              'hand_bot': 'right',
              'resolution': 6}

    main(conf, params)
