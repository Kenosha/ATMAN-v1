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
import ATMANpy.funcs.multi_extract_funcs as mx
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region

def load(vid, sto, frame_blur = 1):
    segBF = BoolFeature.from_store(vid, sto, featname = 'isRally_sound')
    courtroiSF = ConstFeature.from_store(vid, sto, featname = 'courtroiReg', featType = Region)
    
    burnCtms = StreamList.from_store('foregroundMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask',
                                     cache_size = frame_blur, force_fill_cache = True)
    shadCtms = StreamList.from_store('shadowMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask',
                                     cache_size = frame_blur, force_fill_cache = True)
    
    return segBF, courtroiSF, burnCtms, shadCtms

def save(candSFs, vid, sto, **save_params):
    candSFs.rename('foreCnts').save(vid, sto, **save_params)

def check(vid, sto, name, candSFs, cap):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2 as cv
    
    SAVE_NAME = 'player_candidate_contours'
    
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
    chosen_indices = rng.choice(np.arange(0, candSFs.length(), dtype = int), size = num_backgrounds)
        
    fig, axes = plt.subplots(grid_length, grid_length, figsize = (fig_width, fig_height), gridspec_kw = gridspec_kw)
    
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for plot_idx, ax in enumerate(axes.ravel()):
        sf_idx = int(chosen_indices[plot_idx])
        chosenSF = candSFs[sf_idx]
        sfFrm = Stream.from_SF(chosenSF, cap).sample(1)
        
        sfFrm = sfFrm.put(f'Rally #: {sf_idx}', color = (0, 0, 255))
        sfArr = np.flip(sfFrm.array, axis = -1) 
        
        ax.imshow(sfArr)
        ax.set_axis_off()
    
    store_name = SAVE_NAME + '_' + vid + '.png'
    store_path = os.path.join(store_dir, store_name)
    
    fig.savefig(store_path)
    plt.close()
    

def main(conf, params, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    segBF, courtroiSF, burnCtms, shadCtms = load(vid, sto, frame_blur = params['frame_blur'])
    
    segBFs = segBF.filter(segBF).split()
    burnCtms = burnCtms.conform(segBFs)
    shadCtms = shadCtms.conform(segBFs)
    courtROI = courtroiSF.feature

    ### Find Player Candidate Contours
    mulStms = StreamList.combine(burnCtms, shadCtms)
    
    candSFs = mulStms.apply_chunked_mp(mx.find_player_candidate_contours_timeblur, nwo, nch, **params, ROI = courtROI, verbose = True)
    
    check(vid, sto, name, candSFs, cap)
    
    save(candSFs, vid, sto, compression = conf['compression_players'])

if __name__ == '__main__':
        
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in ut.get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[0])
    
    params = prm_mod.player_candidates.params
    
    name = ppl._name
    
    main(conf, params, name = name)
