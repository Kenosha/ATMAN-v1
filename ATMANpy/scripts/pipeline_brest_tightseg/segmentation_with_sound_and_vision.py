# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:45:55 2023

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

import torch
from torch.utils.data import DataLoader
from ATMANpy.models.TemporalCNN.nets import SmallFullConvModel
from ATMANpy.models.TemporalCNN.data import PlayerInferenceDataset
from ATMANpy.models.TemporalCNN.utils import infer_with_index

from ATMANpy.funcs.segmenter_funcs import get_rally_probabilities, make_smoothing_function, weight_cobb_douglas
from ATMANpy.funcs.segmenter_funcs import make_audio_hit_series, make_video_hit_series, make_video_hit_frame

def load(vid, sto):
    ### read in probabilities
    topFrm = ff.read_feature_frame('top_hitTable', vid, sto)
    botFrm = ff.read_feature_frame('bot_hitTable', vid, sto)
    audFrm = ff.read_feature_frame('sound_hitTable', vid, sto)
    segBF = BoolFeature.from_store(vid, sto, featname = 'isRally_sound')
    infoSF = StreamFeatures.from_store(vid, sto, featnameList = ['top_name_Flag', 'top_hand_Flag', 'bot_name_Flag', 'bot_hand_Flag'])
    infoSF_tb = StreamFeatures.from_store(vid, sto, featnameList = ['top_name_tb_Flag', 'top_hand_tb_Flag', 'bot_name_tb_Flag', 'bot_hand_tb_Flag'])
    return topFrm, botFrm, audFrm, segBF, infoSF, infoSF_tb

def save(outSF, outBF, vid, sto):
    outSF.rename('hitFlag').save(vid, sto) 
    outBF.rename('isRally').save(vid, sto)

def plot_hit_probs(conv_video_hitFrm, conv_audio_hitSer, conv_hitFrm, 
                   hitSF, rallyBF,
                   idx, save_folder_name):
    ### charts save where
    mainDir = ut.get_folder_struct('graphs')['hit_probabilities_video_and_audio_viterbi_with_overhead']
    saveDir = ut.maybe_make_folder(mainDir, save_folder_name)
    
    top_srv_viterbi_frames = [f for f, evt in hitSF.streamDict.items() if evt == 'top_srv']
    bot_srv_viterbi_frames = [f for f, evt in hitSF.streamDict.items() if evt == 'bot_srv']
    top_hit_viterbi_frames = [f for f, evt in hitSF.streamDict.items() if evt[:3] == 'top' and f not in top_srv_viterbi_frames]
    bot_hit_viterbi_frames = [f for f, evt in hitSF.streamDict.items() if evt[:3] == 'bot' and f not in bot_srv_viterbi_frames]
    
    fs, ls = rallyBF.filter(rallyBF).fs_ls()

    ### Plot 
    fig, axes = plt.subplots(2, 1, figsize = (16, 16))
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.spines[['right', 'top']].set_visible(False)
    
    # a) video, audio probabolities
    ax = axes[0]
    conv_video_hitFrm['top_hit_p'].plot(ax = ax, color = 'forestgreen', alpha = 0.7, ls = 'dashdot')
    conv_video_hitFrm['bot_hit_p'].plot(ax = ax, color = 'firebrick', alpha = 0.7, ls = 'dashdot')
    conv_audio_hitSer.plot(ax = ax, color = 'darkblue', alpha = 0.7, ls = 'dotted')
    ax.legend()
    
    # b) joint probability and viterbi events
    ax = axes[1]
    conv_hitFrm['top_hit_p'].plot(ax = ax, color = 'forestgreen', alpha = 0.7, ls = 'dashdot')
    conv_hitFrm['bot_hit_p'].plot(ax = ax, color = 'firebrick', alpha = 0.7, ls = 'dashdot')
    
    ax.legend()

    # marker points for detected hits 
    ax.scatter(bot_hit_viterbi_frames, [0.975] * len(bot_hit_viterbi_frames), marker = 'v', color = 'firebrick')
    ax.scatter(top_hit_viterbi_frames, [0.975] * len(top_hit_viterbi_frames), marker = 'v', color = 'forestgreen')
    ax.scatter(bot_srv_viterbi_frames, [0.975] * len(bot_srv_viterbi_frames), marker = 'x', color = 'firebrick')
    ax.scatter(top_srv_viterbi_frames, [0.975] * len(top_srv_viterbi_frames), marker = 'x', color = 'forestgreen')

    ax.axvline(fs, color = 'darkblue', ls = 'dashdot')
    ax.axvline(ls, color = 'darkblue', ls = 'dashdot')

    plt.tight_layout()
    
    figname = f'video_audio_{idx}.png'
    fig.savefig(os.path.join(saveDir, figname))
    plt.close()
     

def main(conf, weight_params, smooth_params, viterbi_params, widen_params, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    fs, ls = conf['start_frame'], conf['stop_frame']
    cap = conf['cap']

    smooth_func_audio = make_smoothing_function(smooth_params['smoothing_kernel_audio'], mode = 'same')
    smooth_func_video = make_smoothing_function(smooth_params['smoothing_kernel_video'], mode = 'same')
    
    topFrm, botFrm, audFrm, segBF, infoSF, infoSF_tb = load(vid, sto)
    
    topSF = StreamFeatures.from_dataframe(topFrm)
    botSF = StreamFeatures.from_dataframe(botFrm)
    audSF = StreamFeatures.from_dataframe(audFrm)

    allSF = topSF.merge(botSF, how = 'outer').merge(audSF, how = 'outer').conform(segBF.filter(segBF))
    allSFs = allSF.split()
    
    ### FIX NAMES AND HANDEDNESS IN TIEBREAKS    
    if infoSF_tb.is_empty():
        infoSF_fix = infoSF
    else:
        replSF = infoSF_tb
        for feature_name in replSF.get_features():
            replSF = replSF.rename_feature(feature_name, feature_name.replace('_tb', ''))
        infoSF_fix = infoSF.replace(replSF)

    hitSFs = []
    rallyBFs = []
    for idx, mySF in enumerate(allSFs):
        myFrm = mySF.to_dataframe()
        fillSer = pd.Series(0, index = myFrm.columns)
        fillSer['top_oth'] = 1
        fillSer['bot_oth'] = 1
        myFrm = myFrm.fillna(fillSer)

        ### get data
        videoFrm = myFrm[[col for col in myFrm if col not in ['hit', 'bounce']]]
        audioSer = myFrm['hit']
        
        ### make hit series from the frames
        video_hitFrm = make_video_hit_frame(videoFrm)
        audio_hitSer = make_audio_hit_series(audioSer)
        
        ### smooth it out
        conv_video_hitFrm = video_hitFrm.apply(smooth_func_video, axis = 0)
        conv_audio_hitSer = pd.Series(smooth_func_audio(audio_hitSer), index = audioSer.index, name = 'audio_hit_p')
        
        ### combine it, weighted.
        conv_hitFrm = weight_cobb_douglas(conv_video_hitFrm, conv_audio_hitSer, weight_params['confidence_audio_vs_video'])
        conv_hitSF = StreamFeatures.from_dataframe(conv_hitFrm)

        ### VITERBI ###
        stateSF = sf.viterbi_hit_events(conv_hitSF, **viterbi_params)
        
        # Stream.from_SF(stateSF.conform(mySF), cap).show(ms_between_frames = 40, frame_counter = True)

        if stateSF.is_empty():
            print(f'isRally_sound {idx}: No shots detected')
            continue
        
        ### Transform unspecific hit_top and hit_bot events into specific events 
        vidSF = StreamFeatures.from_dataframe(videoFrm)
        mySF = stateSF.merge(vidSF, how = 'left').merge(infoSF_fix, how = 'left')
        hitSF = sf.specify_hit_events(mySF)

        if hitSF.is_empty():
            print(f'isRally_sound {idx}: No shots detected after checking for serve at beginning')
            continue

        ### Widen, and create the rally boolean
        
        fs, ls = hitSF.fs_ls()
        fs_adjusted, ls_adjusted = fs - widen_params['add_before'], ls + widen_params['add_after']
        rallyBF = BoolFeature({frame: True for frame in range(fs_adjusted, ls_adjusted + 1)}, name = 'isRally')

        hitSFs.append(hitSF)
        rallyBFs.append(rallyBF)

        # Stream.from_SF(hitSF.conform(rallyBF.filter(rallyBF)), cap).show(ms_between_frames = 100)
        # Stream.from_SF(allSFs[0], cap).show()
        if conf.get('save_checks', True):
            SAVE_FOLDER_NAME = f'viterbi_connecting_states_lohi_hit5_moreflight_{vid}'
            plot_hit_probs(conv_video_hitFrm, conv_audio_hitSer, conv_hitFrm,
                            hitSF, rallyBF,
                            idx, SAVE_FOLDER_NAME)

    hitSFs = SFList(hitSFs)
    rallyBFs = SFList(rallyBFs)

    hitSF = hitSFs.concatenate()
    rallyBF = rallyBFs.concatenate()

    save(hitSF, rallyBF, vid, sto)
    
if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[0])
    
    weight_params = prm_mod.segmentation_with_sound_and_vision.weight_params
    smooth_params = prm_mod.segmentation_with_sound_and_vision.smooth_params
    viterbi_params = prm_mod.segmentation_with_sound_and_vision.viterbi_params
    widen_params = prm_mod.segmentation_with_sound_and_vision.widen_params
    
    name = ppl._name

    main(conf, weight_params, smooth_params, viterbi_params, widen_params, name = name)
    
        