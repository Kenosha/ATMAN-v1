# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 23:20:45 2023

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
from ATMANpy.types.Contours import Contours

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)


if __name__ == '__main__':
        
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    ### Read Player Candidate Contours
    foreSF = StreamFeature.from_store(vid, sto, featname = 'foreCnts')
    
    ### Read Serve ROIs
    topCF = ConstFeature.from_store(vid, sto, featname = 'topserveReg', featType = Region)
    botCF = ConstFeature.from_store(vid, sto, featname = 'botserveReg', featType = Region)
    topROI = topCF.feature
    botROI = botCF.feature
    
    ### Read H and make Courtlines
    HSF = ConstFeature.from_store(vid, sto, featname = 'H')
    H = HSF.feature
    courtLines = sa.find_courtlines(H)
    
    ### Define Criteria that a serving player contour has to fulfill.
    # Criterion 1) center of shape is inside of the serve ROI
    def keep_contours_in_roi(myCnts, which = 'Top'):
        assert which in ['Top', 'Bottom']
        
        if which == 'Top':
            serveReg = topROI
        elif which == 'Bottom':
            serveReg = botROI
    
        inCnts = Contours([cnt for cnt in myCnts if cnt.center().within(serveReg)])
        return inCnts
        
    # Auxiliary function
    def get_distance_to_baseline(myCnt, which = 'Top'):
        assert which in ['Top', 'Bottom']
        
        baseLine = courtLines[f'{which} Baseline']
        botPt = myCnt.enclosing_region()['Bottom'].center()
        linePt = botPt.project_vertically(baseLine)
        
        ### make it so that negative distances are not admissible -> they are in front of the line!
        ### then, the task is simply to choose the smallest positive distance.
        if which == 'Top':
            dist = linePt.y - botPt.y
        elif which == 'Bottom':
            dist = botPt.y - linePt.y
        
        return dist
        
     ### Criterion 2) bottom of shape is behind the respective baseline, keep the closest to baseline among these.
    def keep_closest_contour_behind_baseline(myCnts, which = 'Top', margin = 5):
        assert which in ['Top', 'Bottom']
       
        bestCnt, best_dist = None, None
        for myCnt in myCnts:
            mrg_dist = get_distance_to_baseline(myCnt) + margin # to not exclude players at baseline, add a little margin
            if mrg_dist > 0: ### accept only contours whose bottom is behind the respective line
                if best_dist is None:
                    bestCnt = myCnt
                    best_dist = mrg_dist
                else:
                    if mrg_dist < best_dist:
                        bestCnt = myCnt
                        best_dist = mrg_dist
        
        return bestCnt
    
    ### Apply Criteria    
    topSF_1 = foreSF.apply(keep_contours_in_roi, which = 'Top')
    topSF_2 = topSF_1.apply(keep_closest_contour_behind_baseline, which = 'Top')
    
    botSF_1 = foreSF.apply(keep_contours_in_roi, which = 'Bottom')
    botSF_2 = botSF_1.apply(keep_closest_contour_behind_baseline, which = 'Bottom')
    
    Stream(cap, 50, 200, features = topSF_2).show()
    
    
    
    
    
    
    
    
    
    
    
    
    