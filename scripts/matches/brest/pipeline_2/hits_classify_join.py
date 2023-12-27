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

import torch
from torch.utils.data import DataLoader
from ATMANpy.models.TemporalCNN.nets import SmallFullConvModel
from ATMANpy.models.TemporalCNN.data import PlayerInferenceDataset
from ATMANpy.models.TemporalCNN.utils import infer_with_index

def get_rally_probabilities(hitSer, next_hit_should_come_soon = 20, next_hit_should_have_come = 50, 
                            decay_mild = 0.9, decay_strong = 0.5):

    rally_probability = 1 ### at the beginning, we are for sure in a rally
    time_since_last_potential_stroke = 0
    
    rally_probabilities = []
    for row, hit_probability in hitSer.iteritems():
        if hit_probability > rally_probability:
            rally_probability = hit_probability
            time_since_last_potential_stroke = 0
        else:
            if time_since_last_potential_stroke < next_hit_should_come_soon:
                pass  ### leave rally probability unchanged!
            elif time_since_last_potential_stroke < next_hit_should_have_come:
                rally_probability *= decay_mild
            else:
                rally_probability *= decay_strong
            
            time_since_last_potential_stroke += 1 
        
        rally_probabilities.append(rally_probability)
    
    return pd.Series(rally_probabilities, index = hitSer.index, name = 'rally_p')
    
if __name__ == '__main__':
    
    CONFIDENCE_AUDIO_VS_VIDEO = 0.7 ### we trust the audio a bit more than the video. 
    CONFIDENCE_VIDEO_VS_AUDIO = 1 - CONFIDENCE_AUDIO_VS_VIDEO
    
    prob_params = {'next_hit_should_come_soon': 20,  ### can be early, for serves. set on the basis of above boxplot
                   'next_hit_should_have_come': 120, ### can be late, for high lobs for example. set on the basis of above boxplot
                   'decay_mild': 0.99,
                   'decay_strong': 0.5}
    rally_prob_threshold = 0.2 ### lower than this probability, we believe that the rally is over 
    smoothing_kernel = np.array([0.25, 0.5, 1, 0.5, 0.25]) 
    smoothing_kernel = smoothing_kernel / smoothing_kernel.sum()
    minimum_hit_prominence = rally_prob_threshold
    
    ### TODO: Convolve (ie smooth) audio and video (esp. video) probabilities before computing hit prob -> this counteracts shifts of video hit against audio hit! 
    
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
       
    ### charts save where
    saveDir = ut.get_folder_struct('graphs')['hit_probabilities_video_and_audio']
    
    ### read in probabilities
    topFrm = ff.read_feature_frame('top_hitTable', vid, sto)
    botFrm = ff.read_feature_frame('bot_hitTable', vid, sto)
    
    topSF = StreamFeatures.from_dataframe(topFrm)
    botSF = StreamFeatures.from_dataframe(botFrm)
    
    hitSF = StreamFeature.from_store(vid, sto, featname = 'hitprobFlag')
    
    allSF = topSF.merge(botSF, how = 'outer').merge(hitSF, how = 'outer')
    allSFs = allSF.split()
    
    ### get a feeling for time elapsed between shots
    for idx, mySF in enumerate(allSFs):
        myFrm = mySF.to_dataframe()
        
        ### get video
        tophitSer = 1 - myFrm['top_oth']
        bothitSer = 1 - myFrm['bot_oth']
        tophitSer.name = 'top_hit_p'
        bothitSer.name = 'bot_hit_p'
    
        vidhitSer = pd.concat([tophitSer, bothitSer], axis = 1).max(axis = 1)
        vidhitSer.name = 'video_hit_p'
        
        ### get audio
        audiohitSer = myFrm['hitprobFlag']
        audiohitSer.name = 'audio_hit_p'
        
        ### combine video and audio
        hitSer = vidhitSer ** CONFIDENCE_VIDEO_VS_AUDIO * audiohitSer ** CONFIDENCE_AUDIO_VS_VIDEO
        hitSer.name = 'joint_hit_p'
        smooth_hitSer = pd.Series(np.convolve(hitSer, smoothing_kernel, mode = 'same'), index = hitSer.index)
        smooth_hitSer.name = 'smooth_hit_p'
        
        ### get rally probabilities
        probSer = get_rally_probabilities(hitSer, **prob_params) ### this is a sort of decaying envelope around the hit series
        out_of_rally = probSer < rally_prob_threshold
        earliest_out_of_rally = out_of_rally[out_of_rally].index.min()
        
        ### truncated values
        if np.isnan(earliest_out_of_rally):
            trunc_vidhitSer = vidhitSer
            trunc_audiohitSer = audiohitSer
            trunc_hitSer = hitSer
            trunc_smooth_hitSer = smooth_hitSer
        else:
            trunc_vidhitSer = vidhitSer[vidhitSer.index < earliest_out_of_rally]
            trunc_audiohitSer = audiohitSer[audiohitSer.index < earliest_out_of_rally]
            trunc_hitSer = hitSer[hitSer.index < earliest_out_of_rally]
            trunc_smooth_hitSer = smooth_hitSer[smooth_hitSer.index < earliest_out_of_rally]
        
        ### Plot 
        fig, axes = plt.subplots(3, 1, figsize = (21, 10))
        for ax in axes:
            ax.set_ylim(0, 1)
            ax.spines[['right', 'top']].set_visible(False)
        
        # a) video, audio and joint probability
        ax = axes[0]
        vidhitSer.plot(ax = ax, color = 'forestgreen', alpha = 0.5, ls = 'dashdot')
        audiohitSer.plot(ax = ax, color = 'lightgreen', alpha = 0.5, ls = 'dashdot')
        hitSer.plot(ax = ax, color = 'firebrick')
        ax.legend()
        
        # b) joint probability, rally probability, cutoff
        ax = axes[1]
        hitSer.plot(ax = ax, color = 'firebrick', alpha = 0.5)
        probSer.plot(ax = ax, color = 'orangered')
        ax.axvline(earliest_out_of_rally, color = 'chocolate', ls = 'dashdot')
        ax.legend()
        
        # c) truncated plot
        ax = axes[2]
        trunc_vidhitSer.plot(ax = ax, color = 'forestgreen', alpha = 0.5, ls = 'dashdot')
        trunc_audiohitSer.plot(ax = ax, color = 'lightgreen', alpha = 0.5, ls = 'dashdot')
        trunc_hitSer.plot(ax = ax, color = 'firebrick', alpha = 0.5)
        trunc_smooth_hitSer.plot(ax = ax, color = 'orangered')
        ax.legend()
        
        figname = f'video_audio_{idx}.png'
        fig.savefig(os.path.join(saveDir, figname))
        plt.close()
        
    #     # get local maxima above a certain threshold
    #     topMax = (tophitSer > tophitSer.shift(1)) & (tophitSer > tophitSer.shift(-1)) & (tophitSer > 0.5)
    #     botMax = (bothitSer > bothitSer.shift(1)) & (bothitSer > bothitSer.shift(-1)) & (bothitSer > 0.5) 
    #     # convert to list, join and sort
    #     topmaxIdxs = topMax[topMax].index.tolist()
    #     botmaxIdxs = botMax[botMax].index.tolist()
    #     maxIdxs = sorted([*topmaxIdxs, *botmaxIdxs])
    #     maxDiffs = pd.Series(maxIdxs).diff().dropna()
    #     list_of_maxima_spacings.extend(maxDiffs.values.tolist())
    
    # Stream.from_SF(allSFs[12], cap).show(frame_counter = True, ms_between_frames = 250)
    
    # spaceSer = pd.Series(list_of_maxima_spacings, name = 'frames_elapsed_between_shots')
    # spaceSer.plot.box(ylim = [0, 200])
    
    ### calculate probability of being in rally
    
    bfList = []
    for idx, mySF in enumerate(allSFs):
        myFrm = mySF.to_dataframe()
        
        tophitSer = 1 - myFrm['top_oth']
        bothitSer = 1 - myFrm['bot_oth']
        tophitSer.name = 'top_hit_p'
        bothitSer.name = 'bot_hit_p'
        
        hitFrm = pd.concat([tophitSer, bothitSer], axis = 1)
        probSer = get_rally_probabilities(hitFrm.max(axis = 1), **params)
        
        out_of_rally = probSer < rally_prob_threshold
        earliest_out_of_rally = out_of_rally[out_of_rally].index.min()
        
        boolDict = {frame: frame < earliest_out_of_rally for frame in mySF.get_frames()}
        outBF = BoolFeature(boolDict, name = 'isRally')
        bfList.append(outBF)
        
        ### Plot
        fig, ax = plt.subplots(1, 1, figsize = (10, 10))
        ax.set_ylim(0, 1)
        ax.spines[['right', 'top']].set_visible(False)
        
        tophitSer.plot(ax = ax, color = 'forestgreen')
        bothitSer.plot(ax = ax, color = 'lightgreen')
        probSer.plot(ax = ax, color = 'firebrick')
        ax.axvline(earliest_out_of_rally, color = 'chocolate', ls = 'dashdot')
        
        plt.legend()
        
        figname = f'rally_segmenter_{idx}.png'
        fig.savefig(os.path.join(saveDir, figname))
        plt.close()
        
    outBF = SFList(bfList).concatenate_and_plug().to_bool().fillna(False)
    outBF.save(vid, sto)
    
    #%% Test
    
    # idx = 130
    
    # Stream.from_SF(allSFs[idx], cap).show(frame_counter = True)
    
    # chk = allSFs[idx].to_dataframe()
        
        