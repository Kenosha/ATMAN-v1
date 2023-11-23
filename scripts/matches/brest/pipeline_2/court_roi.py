# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:45:37 2023

@author: gianl
"""

import os
import time

import cv2 as cv
import numpy as np
import pandas as pd

### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video
from ATMANpy.funcs.cv_funcs import join_video_captures, get_width_height

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
    start, stop = conf['start_frame'], conf['stop_frame']
    add_scale = conf['add_scale']

    width, height = get_width_height(cap)

    ### Make folders for feature storage
    featPath = ut.get_folder_struct('features_stored')['main']
    feat_vidPath = ut.maybe_make_folder(featPath, vid)
    storePath = ut.maybe_make_folder(feat_vidPath, sto)
    
    randStm = RandStream(cap, start, stop, number_of_frames = 1000,
                         cap_color_space = 'bgr', target_color_space = 'grey',
                         scale_factor = add_scale)
    
    ### Set up a region of interest to focus attention on court. 
    # -> region of interest will be saved as a StillFrame of Mask type
    some_frame = 300
    someFrm = randStm.get(some_frame, put = False)
    # someFrm.show()
    
    p1, p2, p3, p4 = Point2D((200, 12)), Point2D((440, 12)), Point2D((0, 295)), Point2D((width, 295))
    ROI = Region.from_points(p1, p2, p3, p4)
    someFrm.put(ROI).show()
    
    courtCF = ConstFeature('courtroiReg', ROI)
    courtCF.save(vid, sto, extract_func = ff.extract_region_values)
    
    ### Serve ROIs
    HSF = ConstFeature.from_store(vid, sto, featname = 'H')
    H = HSF.feature
    courtLines = sa.find_courtlines(H, schemaLines = schemaLines)
    
    # a) Top Serve
    loLn, hiLn = courtLines['Top Baseline'].down(20), courtLines['Top Baseline'].up(60)
    p1, p2, p3, p4 = hiLn.leftPt, hiLn.rightPt, loLn.leftPt, loLn.rightPt
    ROI = Region.from_points(p1, p2, p3, p4)
    someFrm.put(ROI).show()
    
    topsrvCF = ConstFeature('topserveReg', ROI)
    topsrvCF.save(vid, sto, extract_func = ff.extract_region_values)
    
    # b) Bot Serve
    loLn, hiLn = courtLines['Bottom Baseline'].down(30), courtLines['Bottom Baseline'].up(82)
    p1, p2, p3, p4 = hiLn.leftPt, hiLn.rightPt, loLn.leftPt, loLn.rightPt
    ROI = Region.from_points(p1, p2, p3, p4)
    someFrm.put(ROI).show()
    
    topsrvCF = ConstFeature('botserveReg', ROI)
    topsrvCF.save(vid, sto, extract_func = ff.extract_region_values)
    
    
    
    