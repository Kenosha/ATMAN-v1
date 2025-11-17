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
    
def make_smoothing_function(kernel):
    norm_kernel = kernel / kernel.sum()
    return partial(np.convolve, v = norm_kernel, mode = 'same')

def make_video_hit_series(videoFrm, name = 'video_hit_p'):
    ### get video
    tophitSer = 1 - videoFrm['top_oth']
    bothitSer = 1 - videoFrm['bot_oth']
    tophitSer.name = 'top_hit_p'
    bothitSer.name = 'bot_hit_p'

    vidhitSer = pd.concat([tophitSer, bothitSer], axis = 1).max(axis = 1)
    vidhitSer.name = name
    
    return vidhitSer

def make_audio_hit_series(audioSer, name = 'audio_hit_p'):
    ### get audio
    audiohitSer = audioSer
    audiohitSer.name = name
    return audiohitSer
        
def weight_cobb_douglas(leftSer, rightSer, left_lambda, name = 'joint_hit_p'):
    outSer = leftSer ** left_lambda * rightSer ** (1 - left_lambda)
    outSer.name = name
    return outSer

if __name__ == '__main__':
    
    #### PARAMETERS #######
    
    CONFIDENCE_AUDIO_VS_VIDEO = 0.7 ### we trust the audio a bit more than the video. 
    
    # for get_rally_probabilities
    prob_params = {'next_hit_should_come_soon': 20,  ### can be early, for serves. set on the basis of above boxplot
                   'next_hit_should_have_come': 120, ### can be late, for high lobs for example. set on the basis of above boxplot
                   'decay_mild': 0.99,
                   'decay_strong': 0.5}
    
    # thresholds
    rally_prob_threshold = 0.2 ### lower than this probability, we believe that the rally is over 
    hit_prob_threshold = rally_prob_threshold
    
    # smoothing 
    smoothing_kernel_audio = np.array([0.25, 0.5, 1, 0.5, 0.25]) 
    smoothing_kernel_video = np.array([0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25])
    smooth_func_audio = make_smoothing_function(smoothing_kernel_audio)
    smooth_func_video = make_smoothing_function(smoothing_kernel_video)
     
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
    
    ### read in a segmenter -> just for conforming
    segBF = BoolFeature.from_store(vid, sto, featname = 'isServeStart')
    
    frame_event_dict = {} ### this is the dictionary that will get populated with frame keys and identified hit events, to then save as SF
    frame_rally_dict = {} ### this is the dictionary that will get populated with frame keys and the isRally boolean, to then save as SF
    ### get a feeling for time elapsed between shots
    for idx, mySF in enumerate(allSFs):
        myFrm = mySF.to_dataframe()
        
        ### get data
        videoFrm = myFrm[[col for col in myFrm if col != 'hitprobFlag']]
        audioSer = myFrm['hitprobFlag']
        
        ### smooth it out
        conv_videoFrm = videoFrm.apply(smooth_func_video, axis = 0)
        conv_audioSer = pd.Series(smooth_func_audio(audioSer), index = audioSer.index)
        
        ### make hit series from the frames
        video_hitSer = make_video_hit_series(videoFrm)
        conv_video_hitSer = make_video_hit_series(conv_videoFrm)

        audio_hitSer = make_audio_hit_series(audioSer)
        conv_audio_hitSer = make_audio_hit_series(conv_audioSer)
        
        ### weight using cobb douglas form
        hitSer = weight_cobb_douglas(audio_hitSer, video_hitSer, CONFIDENCE_AUDIO_VS_VIDEO)
        conv_hitSer = weight_cobb_douglas(conv_audio_hitSer, conv_video_hitSer, CONFIDENCE_AUDIO_VS_VIDEO)
        
        ### get rally probabilities -> use here the smoothed function; the unsmoothed stuff is done only for showing why it's worse
        probSer = get_rally_probabilities(conv_hitSer, **prob_params) ### this is a sort of decaying envelope around the hit series
        out_of_rally = probSer < rally_prob_threshold
        earliest_out_of_rally = out_of_rally[out_of_rally].index.min()
        
        ### create rally segmenter
        boolDict = {frame: frame < earliest_out_of_rally for frame in mySF.get_frames()}
        frame_rally_dict.update(boolDict)
        
        ### truncated values
        if np.isnan(earliest_out_of_rally):
            trunc_vidhitSer = video_hitSer
            trunc_audiohitSer = audio_hitSer
            trunc_hitSer = hitSer
            trunc_conv_hitSer = conv_hitSer
        else:
            trunc_vidhitSer = video_hitSer[video_hitSer.index < earliest_out_of_rally]
            trunc_audiohitSer = audio_hitSer[audio_hitSer.index < earliest_out_of_rally]
            trunc_hitSer = hitSer[hitSer.index < earliest_out_of_rally]
            trunc_conv_hitSer = conv_hitSer[conv_hitSer.index < earliest_out_of_rally]
        
        ### get local maxima above certain threshold
        peak_idxs, peak_dict = find_peaks(trunc_conv_hitSer, height = hit_prob_threshold)
        peak_frames = probSer.index[peak_idxs]
        peak_heights = peak_dict['peak_heights']
        
        ### understand what is going on at these local maxima
        # for each peak frame, understand whether something is happening bot or top, then select the likeliest event
        bot_hit_peak_frames = []
        top_hit_peak_frames = []
        bot_srv_peak_frames = []
        top_srv_peak_frames = []
        for pf in peak_frames:
            pfSer = conv_videoFrm.loc[pf]
            top_oth, bot_oth = pfSer['top_oth'], pfSer['bot_oth']
            hi_prefix = 'top' if top_oth < bot_oth else 'bot'
            
            prefSer = pfSer[[c for c in pfSer.index if hi_prefix in c]]
            prefSer = prefSer.drop(f'{hi_prefix}_oth')
            maximum_event = prefSer[prefSer == prefSer.max()].index.item()
            frame_event_dict[pf] = maximum_event
            
            side, evt = maximum_event.split('_')[:2]
            
            if side == 'top':
                if evt == 'srv':
                    top_srv_peak_frames.append(pf)
                else:
                    top_hit_peak_frames.append(pf)
            elif side == 'bot':
                if evt == 'srv':
                    bot_srv_peak_frames.append(pf)
                else:
                    bot_hit_peak_frames.append(pf)
                    
        ### Plot 
        fig, axes = plt.subplots(2, 2, figsize = (16, 16))
        for ax in axes.flatten():
            ax.set_ylim(0, 1)
            ax.spines[['right', 'top']].set_visible(False)
        
        # a) video, audio and joint probability UNSMOOTHED
        ax = axes[0, 0]
        video_hitSer.plot(ax = ax, color = 'forestgreen', alpha = 0.5, ls = 'dashdot')
        audio_hitSer.plot(ax = ax, color = 'lightgreen', alpha = 0.5, ls = 'dashdot')
        hitSer.plot(ax = ax, color = 'firebrick')
        ax.legend()
        
        # b) video, audio and joint probability SMOOTHED
        ax = axes[1, 0]
        conv_video_hitSer.plot(ax = ax, color = 'forestgreen', alpha = 0.5, ls = 'dashdot')
        conv_audio_hitSer.plot(ax = ax, color = 'lightgreen', alpha = 0.5, ls = 'dashdot')
        conv_hitSer.plot(ax = ax, color = 'firebrick')
        ax.legend()
        
        # c) joint probability, rally probability, cutoff
        ax = axes[0, 1]
        conv_hitSer.plot(ax = ax, color = 'firebrick', alpha = 0.5)
        probSer.plot(ax = ax, color = 'orangered')
        ax.axvline(earliest_out_of_rally, color = 'chocolate', ls = 'dashdot')
        ax.legend()
        
        # d) truncated plot
        ax = axes[1, 1]
        trunc_vidhitSer.plot(ax = ax, color = 'forestgreen', alpha = 0.5, ls = 'dashdot')
        trunc_audiohitSer.plot(ax = ax, color = 'lightgreen', alpha = 0.5, ls = 'dashdot')
        trunc_conv_hitSer.plot(ax = ax, color = 'firebrick')
        ax.legend()

        ax.axhline(hit_prob_threshold, color = 'chocolate', ls = 'dashdot')

        # marker points for detected hits 
        ax.scatter(bot_hit_peak_frames, [0.975] * len(bot_hit_peak_frames), marker = 'v', color = 'darkblue')
        ax.scatter(top_hit_peak_frames, [0.975] * len(top_hit_peak_frames), marker = 'v', color = 'darkorange')
        ax.scatter(bot_srv_peak_frames, [0.975] * len(bot_srv_peak_frames), marker = 'x', color = 'darkblue')
        ax.scatter(top_srv_peak_frames, [0.975] * len(top_srv_peak_frames), marker = 'x', color = 'darkorange')
                
        plt.tight_layout()
        
        figname = f'video_audio_{idx}.png'
        fig.savefig(os.path.join(saveDir, figname))
        plt.close()
        
    outSF = StreamFeature(frame_event_dict, name = 'hitFlag')
    outSF.save(vid, sto)    
    
    outBF = BoolFeature(frame_rally_dict, name = 'isRally')
    outBF = outBF.conform(segBF).fillna(False)
    outBF.save(vid, sto)
        