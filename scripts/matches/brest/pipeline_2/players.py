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

def find_player_candidate_links(candSF, frame_offset = 5):
    from ATMANpy.funcs.player_funcs import find_player_candidate_contours_with_overlap
    
    sfDict = {}
    
    player_candidates_prev = None
    for idx, frame in enumerate(candSF):
        if idx % frame_offset != 0:
            featDict = {'player_candidates': None,
                        'player_parents': None}
        else:
            candCnts = candSF[frame]
            
            if player_candidates_prev is None:
                player_candidates, player_parents = candCnts, [[]] * candCnts.length()
            else:
                player_candidates, player_parents = find_player_candidate_contours_with_overlap(candCnts, player_candidates_prev)
            
            player_candidates_prev = player_candidates
                
            featDict = {'player_candidates': player_candidates,
                        'player_parents': player_parents}
                
        sfDict[frame] = featDict
        
    return StreamFeatures(sfDict)
        
if __name__ == '__main__':
        
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    ### Read constant Homography and Projection Matrix
    HSF = ConstFeature.from_store(vid, sto, featname = 'H')
    H = HSF.feature
    
    ### Make Some Constant Features
    courtLines = sa.find_courtlines(H)
    courtCF = ConstFeature('courtLines', courtLines)
    courtroiCF = ConstFeature.from_store(vid, sto, featname = 'courtroiReg', featType = Region)
    
    ### Read Segmentation Booleans
    courtBF = BoolFeature.from_store(vid, sto, featname = 'isCourt')
    serveBF = BoolFeatures.from_store(vid, sto, featnameList = ['botPlayer_srvFlag', 'topPlayer_srvFlag'])
    no_serveBF = serveBF.any().neg()
    playBF = no_serveBF & courtBF
    
    ### Read Player Candidate Contours
    foreSF = StreamFeature.from_store(vid, sto, featname = 'foreCnts')
    foreSFs = foreSF.filter(playBF).split()
    
    FRAME_OFFSET = 5
    params = {'frame_offset': FRAME_OFFSET}
    trackSFs = foreSFs.apply_whole_mp(find_player_candidate_links, nwo, **params)    
    
    params = {'frame_offset': FRAME_OFFSET,
              'critical_number_of_tracks': 50000,
              'verbose': False,
              'super_verbose': False}
    splitDFs = trackSFs.apply_whole_mp(sf.split_player_candidate_tracks, nwo, **params)
    splitDFs = splitDFs.sfList ### have to do this so that splitDFs is not an SFList type
    
    candSFs = trackSFs.concatenate().add('courtroiReg', courtroiCF.feature).add('courtLines', courtCF.feature).split()

    params = {'granularity': 10,
              'area_outlier_iqr_multiple': 1.5,
              'overlap_optimum': 0.9 ** FRAME_OFFSET,
              'overlap_strictness': 5,
              'metric_weights': {'dist': 5, 'ovlp': 2, 'area': 1, 'move': 2}}
    plSFs = candSFs.apply_whole_mp_starmapped(sf.evaluate_player_candidates, nwo, splitDFs, **params)
    
    plSFs.save(vid, sto, compression = conf['compression_players'])
    
    StreamList.from_SFList(plSFs, cap).show(ms_between_frames = 10)
    
