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

def load(vid, sto):
    plSF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])
    return plSF

def save(chamfSFs, vid, sto, **save_params):
    chamfSFs.rename('chamferCnts').save(vid, sto, **save_params)
    
def check(vid, sto, name, chamfSFs, plSFs, cap):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2 as cv
    
    SAVE_NAME = 'chamfer_contours'
    
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
    chosen_indices = rng.choice(np.arange(0, chamfSFs.length(), dtype = int), size = num_backgrounds)
        
    fig, axes = plt.subplots(grid_length, grid_length, figsize = (fig_width, fig_height), gridspec_kw = gridspec_kw)
    
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for plot_idx, ax in enumerate(axes.ravel()):
        sf_idx = int(chosen_indices[plot_idx])
        chosenSF_chamf = chamfSFs[sf_idx]
        chosenSF_player = plSFs[sf_idx]

        frame = int(rng.choice(chosenSF_chamf.get_frames(), size = 1)[0])
        chamfCnt, plCnt = chosenSF_chamf[frame], chosenSF_player[frame]

        showFrm = GreyFrame.from_video(cap, frame).to_BGR()
        showFrm = showFrm.put(chamfCnt, color = (0, 0, 255), thickness = 2)
        showFrm = showFrm.put(plCnt['topPlayerCnt'], color = (0, 255, 255), thickness = 2)
        showFrm = showFrm.put(plCnt['botPlayerCnt'], color = (0, 255, 255), thickness = 2)

        showFrm = showFrm.put(f'Rally #: {sf_idx}', color = (0, 0, 255))
        showFrm = np.flip(showFrm.array, axis = -1) 
        
        ax.imshow(showFrm)
        ax.set_axis_off()
    
    store_name = SAVE_NAME + '_' + vid + '.png'
    store_path = os.path.join(store_dir, store_name)
    
    fig.savefig(store_path)
    plt.close()
    
def main(conf, params, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']

    plSF = load(vid, sto)
    
    plSFs = plSF.split()
    
    ### Make Chamfer Contours ###
    greyStms = StreamList.from_SFList(plSFs, cap, name = 'gFrm',
                                      cap_color_space = 'bgr', target_color_space = 'grey')
    
    chamfSFs = greyStms.extract_chunked_mp(ex.get_chamfer_contours, nwo, nch, **params)
    
    check(vid, sto, name, chamfSFs, plSFs, cap)

    save(chamfSFs, vid, sto, compression = conf['compression_players'])
    

if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[8])
    
    params = prm_mod.ball_inference_chamfer.params

    name = ppl._name

    main(conf, params, name = name)
    