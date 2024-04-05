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

def check(vid, sto, name, plySF, hitcntSF, cap):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2 as cv
    
    SAVE_NAME = 'ball_locations_at_hit'
    
    base_graph_dir = ut.get_folder_struct('graphs')['pipeline_checks']
    store_dir = os.path.join(base_graph_dir, name, sto)
    width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    aspect_ratio = width / height
    fig_height = 36
    fig_width = fig_height * aspect_ratio
    
    ### establish how many random backgrounds we want to show
    grid_length = 10
    num_backgrounds = grid_length ** 2
    
    ### tighten the plot
    bleed = 0
    gridspec_kw = {'wspace': 0.05, 'hspace': 0.05}
    
    rng = np.random.default_rng()
    chosen_indices = rng.choice(hitcntSF.get_frames(), size = num_backgrounds)
        
    fig, axes = plt.subplots(grid_length, grid_length, figsize = (fig_width, fig_height), gridspec_kw = gridspec_kw)
    
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for plot_idx, ax in enumerate(axes.ravel()):
        chosen_frame = int(chosen_indices[plot_idx])
        chosen_point = hitcntSF[chosen_frame]
        chosen_player_top = plySF[chosen_frame]['topPlayerCnt']
        chosen_player_bot = plySF[chosen_frame]['botPlayerCnt']

        showFrm = GreyFrame.from_video(cap, chosen_frame).to_BGR()
        showFrm = showFrm.put(chosen_player_top, color = (255, 0, 0), thickness = 2)
        showFrm = showFrm.put(chosen_player_bot, color = (255, 0, 0), thickness = 2)
        showFrm = showFrm.put(chosen_point, color = (0, 255, 255), thickness = 2)

        showFrm = showFrm.put(f'Frame #: {chosen_frame}', color = (0, 0, 255))
        showFrm = np.flip(showFrm.array, axis = -1) 
        
        ax.imshow(showFrm)
        ax.set_axis_off()
    
    store_name = SAVE_NAME + '_' + vid + '.png'
    store_path = os.path.join(store_dir, store_name)
    
    fig.savefig(store_path)
    plt.close()

def main(conf, params, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    cap = conf['cap']

    hitSF, plySF = load(vid, sto)
    thin_hitSF = hitSF.thin() ### get only the central singleton in each group of hits.

    ### Combine hits detected with player location to add hit candidate points
    mySF = thin_hitSF.merge(plySF, how = 'left')
    
    ## TODO: maybe add more than just one candidate contour, let the particle filter decide which one is best?
    hitcntSF = mySF.apply(sa.get_inferred_hit_points, **params).dropna()
    
    check(vid, sto, name, plySF, hitcntSF, cap)

    save(hitcntSF, vid, sto, compression = params['resolution'])

if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[0])
    
    params = prm_mod.ball_inference_hit_candidates.params    
    
    name = ppl._name
    main(conf, params, name = name)
