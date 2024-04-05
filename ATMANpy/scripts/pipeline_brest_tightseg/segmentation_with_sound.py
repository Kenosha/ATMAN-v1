# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:14:49 2024

@author: gianl
"""


import sys
import os
import time
import dill as pickle
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from scipy.signal import find_peaks

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

from ATMANpy.funcs.segmenter_funcs import get_rally_probabilities, make_smoothing_function, make_audio_hit_series

        
def check(vid, sto, name, boolSer):
    base_graph_dir = ut.get_folder_struct('graphs')['pipeline_checks']
    store_dir = os.path.join(base_graph_dir, name, sto)
    store_name = 'audio_segmentation' + '_' + vid + '.png'
    store_path = os.path.join(store_dir, store_name)
    
    audio_true = 0.1
    line_true = 0.05
    
    fig, ax = plt.subplots(1, 1, figsize = (25, 3))
    plotSer_seg_audio = boolSer.replace(False, np.nan).replace(True, audio_true).rename('audio_segmentation')
    plotSer_seg_lines = boolSer.replace(False, line_true).replace(True, line_true).rename('line_segmentation')
    
    plotFrm = pd.concat([plotSer_seg_audio, plotSer_seg_lines], axis = 1)
    plotFrm.plot(ax = ax) ### check this, to see where to set thresholds!!
    
    ax.get_yaxis().set_visible(False)
    ax.set_ylim([0, audio_true + line_true])
    
    fig.savefig(store_path)
    plt.close()

def load(vid, sto):
    audFrm = ff.read_feature_frame('sound_hitTable', vid, sto)
    return audFrm

def save(vid, sto, segBF):
    segBF.rename('isRally_sound').save(vid, sto)

def main(conf, smooth_params, prob_params, thresh_params, shift_params, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    
    smooth_func_audio = make_smoothing_function(smooth_params['smoothing_kernel_audio'])
    
    audFrm = load(vid, sto)
    audFrms = ut.split(audFrm) ### these are split by the segmentation_with_lines segmenter!
    
    boolSers = []
    for idx, subFrm in enumerate(audFrms):
        audioSer = subFrm['hit']
        conv_audioSer = pd.Series(smooth_func_audio(audioSer), index = audioSer.index)
        conv_audio_hitSer = make_audio_hit_series(conv_audioSer)

        ### TODO: make the probability thresholding a bit more resistant to splitting the rally in the middle (if needed).
        probSer = get_rally_probabilities(conv_audio_hitSer, **prob_params) ### this is a sort of decaying envelope around the hit series
        rallySer = probSer >= thresh_params['rally_prob_threshold']
        rallySer.name = 'isRally_sound'
        boolSers.append(rallySer)
        
    frame_idxs = np.arange(conf['start_frame'], conf['stop_frame'] + 1, dtype = int)
    boolSer = pd.concat(boolSers).reindex(frame_idxs)

    ### shift so that we include the beginning of the serve, not just the impact moment
    boolSer = boolSer.shift(periods = -shift_params['shift_frames'])
    
    check(vid, sto, name, boolSer)
    
    ### fill nan with false
    boolSer = boolSer.fillna(False)
    segBF = BoolFeature.from_series(boolSer)   
    
    save(vid, sto, segBF)
    
if __name__ == '__main__':
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
    
    avail_vids = [f for f in ut.get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[0])
    
    smooth_params = prm_mod.segmentation_with_sound.smooth_params
    prob_params = prm_mod.segmentation_with_sound.prob_params
    thresh_params = prm_mod.segmentation_with_sound.thresh_params
    shift_params = prm_mod.segmentation_with_sound.shift_params
    
    name = ppl._name
    
    main(conf, smooth_params, prob_params, thresh_params, shift_params, name = name)
    








