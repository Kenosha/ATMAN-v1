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

def load(vid, sto):
    HSF = ConstFeature.from_store(vid, sto, featname = 'H')
    courtroiCF = ConstFeature.from_store(vid, sto, featname = 'courtroiReg', featType = Region)
    playBF = BoolFeature.from_store(vid, sto, featname = 'isServeStart')
    serveSF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayer_srvReg', 'botPlayer_srvReg'])
    foreSF = StreamFeature.from_store(vid, sto, featname = 'foreCnts')
    return HSF, courtroiCF, playBF, serveSF, foreSF

def save(topSFs, botSFs, vid, sto):
    topSFs.rename('topPlayerCnt').save(vid, sto, compression = conf['compression_players'])
    botSFs.rename('botPlayerCnt').save(vid, sto, compression = conf['compression_players'])
    
def main(conf, link_params, split_track_params, match_params, eval_params, itpl_params):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    
    HSF, courtroiCF, playBF, serveSF, foreSF = load(vid, sto)
    
    H = HSF.feature
    schemaLines = CourtLines.make().add_singles_netline()
    courtLines = sa.find_courtlines(H, schemaLines = schemaLines)
    courtCF = ConstFeature('courtLines', courtLines)
    
    # augment the serve region with a receiver region opposite
    serveSF = serveSF.dropna().apply(sa.get_locations_at_serve, courtLines = courtLines)
    serveSFs = serveSF.conform(playBF).filter(playBF).split()
    
    foreSFs = foreSF.filter(playBF).split()
    
    trackSFs = foreSFs.apply_whole_mp(sf.find_player_candidate_links, nwo, **link_params)    
    
    splitDFs = trackSFs.apply_whole_mp(sf.split_player_candidate_tracks, nwo, **split_track_params)
    splitDFs = splitDFs.sfList ### have to do this so that splitDFs is not an SFList type
    
    ### Evaluate Metrics
    # add in constants
    candSFs = trackSFs.concatenate().add('courtroiReg', courtroiCF.feature).add('courtLines', courtCF.feature).split()
    # add in serve and return points
    candSFs = candSFs.merge(serveSFs)

    ### Match Tracks that can be joined
    augmented_splitDFs = candSFs.apply_whole_mp_starmapped(sf.iteratively_match_tracks, nwo, splitDFs, **match_params)
    augmented_splitDFs = augmented_splitDFs.sfList
    
    ### Evaluate Tracks as to "Playerness"
    augmented_plSFs = candSFs.apply_whole_mp_starmapped(sf.evaluate_player_candidates, nwo, augmented_splitDFs, **eval_params)
    
    ### interpolate the player contours (between FRAME_OFFSET)
    mergeSFs = augmented_plSFs.merge(foreSFs, how = 'outer')
    topSFs = mergeSFs.apply_whole_mp(sf.interpolate_players, nwo, **itpl_params, which = 'top')
    botSFs = mergeSFs.apply_whole_mp(sf.interpolate_players, nwo, **itpl_params, which = 'bot')
    
    save(topSFs, botSFs, vid, sto)
    
    
if __name__ == '__main__':
        
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    
    frame_offset = 5
    link_params = {'frame_offset': frame_offset}
    split_track_params = {'frame_offset': frame_offset,
                          'critical_number_of_tracks': 50000,
                          'verbose': False,
                          'super_verbose': False}
    match_params = {'frame_offset': frame_offset,
                    'num_points': 6,
                    'threshold_distance': 20}
    eval_params = {'granularity': 10,
              'area_outlier_iqr_multiple': 1.5,
              'overlap_optimum': 0.9 ** frame_offset,
              'overlap_strictness': 5,
              'area_outlier_iqr_multiple': 1.5,
              'metric_weights': {'dist': 4, 'dist_time': 4, 'dist_serve': 0, 'ovlp': 2, 'area': 1, 'move': 2}}
    itpl_params = {'frame_offset': frame_offset,
                   'minimum_overlap_between_candidate_and_interpolation_position': 0.2}
    
    main(conf, link_params, split_track_params, match_params, eval_params, itpl_params)
    
    