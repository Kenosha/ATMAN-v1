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
    playBF = BoolFeature.from_store(vid, sto, featname = 'isRally_sound')
    foreSF = StreamFeature.from_store(vid, sto, featname = 'foreCnts')
    return HSF, courtroiCF, playBF, foreSF

def save(topSFs, botSFs, vid, sto, compression = 32):
    topSFs.rename('topPlayerCnt').save(vid, sto, compression = compression)
    botSFs.rename('botPlayerCnt').save(vid, sto, compression = compression)
    
def check(vid, sto, name, topSFs, botSFs, cap):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2 as cv
    
    SAVE_NAME = 'player_tracked_contours'
    
    base_graph_dir = ut.get_folder_struct('graphs')['pipeline_checks']
    store_dir = os.path.join(base_graph_dir, name, sto)
    width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    aspect_ratio = width / height
    fig_height = 24
    fig_width = fig_height * aspect_ratio
    
    ### establish how many random backgrounds we want to show
    grid_length = 6
    num_backgrounds = grid_length ** 2
    
    ### tighten the plot
    bleed = 0
    gridspec_kw = {'wspace': 0.05, 'hspace': 0.05}
    
    rng = np.random.default_rng()
    chosen_indices = rng.choice(np.arange(0, topSFs.length(), dtype = int), size = num_backgrounds)
        
    fig, axes = plt.subplots(grid_length, grid_length, figsize = (fig_width, fig_height), gridspec_kw = gridspec_kw)
    
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for plot_idx, ax in enumerate(axes.ravel()):
        sf_idx = int(chosen_indices[plot_idx])
        chosenSF = topSFs[sf_idx].merge(botSFs[sf_idx])
        sfFrm = Stream.from_SF(chosenSF, cap).sample(1)
        
        sfFrm = sfFrm.put(f'Rally #: {sf_idx}', color = (0, 0, 255))
        sfArr = np.flip(sfFrm.array, axis = -1) 
        
        ax.imshow(sfArr)
        ax.set_axis_off()
    
    store_name = SAVE_NAME + '_' + vid + '.png'
    store_path = os.path.join(store_dir, store_name)
    
    fig.savefig(store_path)
    plt.close()
    
def main(conf, link_params, split_track_params, match_params, eval_params, itpl_params, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    HSF, courtroiCF, playBF, foreSF = load(vid, sto)
    
    H = HSF.feature
    schemaLines = CourtLines.make().add_singles_netline()
    courtLines = sa.find_courtlines(H, schemaLines = schemaLines)
    courtCF = ConstFeature('courtLines', courtLines)
    
    # # augment the serve region with a receiver region opposite
    # serveSF = serveSF.dropna().apply(sa.get_locations_at_serve, courtLines = courtLines)
    # serveSFs = serveSF.conform(playBF).filter(playBF).split()
    
    foreSFs = foreSF.filter(playBF).split()
    
    trackSFs = foreSFs.apply_whole_mp(sf.find_player_candidate_links, nwo, **link_params)    
    
    splitDFs = trackSFs.apply_whole_mp(sf.split_player_candidate_tracks, nwo, **split_track_params)
    splitDFs = splitDFs.sfList ### have to do this so that splitDFs is not an SFList type
    
    ### Evaluate Metrics
    # add in constants
    candSFs = trackSFs.concatenate().add('courtroiReg', courtroiCF.feature).add('courtLines', courtCF.feature).split()
    # # add in serve and return points
    # candSFs = candSFs.merge(serveSFs)

    ### Match Tracks that can be joined
    augmented_splitDFs = candSFs.apply_whole_mp_starmapped(sf.iteratively_match_tracks, nwo, splitDFs, **match_params)
    augmented_splitDFs = augmented_splitDFs.sfList
    
    ### Evaluate Tracks as to "Playerness"
    augmented_plSFs = candSFs.apply_whole_mp_starmapped(sf.evaluate_player_candidates, nwo, augmented_splitDFs, **eval_params)
    
    ### interpolate the player contours (between FRAME_OFFSET)
    mergeSFs = augmented_plSFs.merge(foreSFs, how = 'outer')
    topSFs = mergeSFs.apply_whole_mp(sf.interpolate_players, nwo, **itpl_params, which = 'top')
    botSFs = mergeSFs.apply_whole_mp(sf.interpolate_players, nwo, **itpl_params, which = 'bot')
    
    check(vid, sto, name, topSFs, botSFs, cap)
    
    save(topSFs, botSFs, vid, sto, compression = conf['compression_players'])
    
    
if __name__ == '__main__':
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in ut.get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[2])
    
    link_params = prm_mod.player_tracking.link_params
    split_track_params = prm_mod.player_tracking.split_track_params
    match_params = prm_mod.player_tracking.match_params
    eval_params = prm_mod.player_tracking.eval_params
    itpl_params = prm_mod.player_tracking.itpl_params
    
    name = ppl._name
    
    main(conf, link_params, split_track_params, match_params, eval_params, itpl_params, name = name)
    
    