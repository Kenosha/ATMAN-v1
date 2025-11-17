# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:47:05 2023

@author: gianl
"""


import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from functools import partial
from scipy.signal import find_peaks
import sounddevice as sd

import ATMANpy.funcs.util_funcs as ut
from ATMANpy.types.Features import BoolFeature, StreamFeature, StreamFeatures
from ATMANpy.types.Streams import Stream

def plot_waveform(ax, waveform, sample_rate, first_sample_index = 0):
    x_axis = np.arange(0, len(waveform)) + first_sample_index
    ax.plot(x_axis, waveform, linewidth = 1, alpha = 0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True)
    return ax
    
def fs_ls(myBF):
    return myBF.first()[0], myBF.last()[0]

def crop_waveform_to_sf(waveform, sf, samples_per_frame):
    fs, ls = fs_ls(sf)
    floor, ceil = np.floor(samples_per_frame).astype(int), np.ceil(samples_per_frame).astype(int), 
    fs_sample, ls_sample = fs * floor, (ls + 1) * ceil # last frame included!
    if len(waveform.shape) == 1:
        return waveform[fs_sample:ls_sample]
    else:
        return waveform[:, fs_sample:ls_sample]

def first_last_sample_index_in_frame(frame_number, first_frame_number, samples_per_frame):
    frame_index = frame_number - first_frame_number
    first_sample_in_frame = frame_index * samples_per_frame
    last_sample_in_frame = first_sample_in_frame + samples_per_frame - 1
    return first_sample_in_frame, last_sample_in_frame

def apply_func_to_window(waveform, func, 
                    window_length = 1024, hop_length = 512):
    h = window_length // 2
    sample_range = range(h, len(waveform) - h, hop_length)
    out_array = np.array([func(waveform[i-h:i+h]) for i in sample_range])
    return sample_range, out_array

def ae_applicator(window_waveform):
    return np.max(np.abs(window_waveform))

ae_func = partial(apply_func_to_window, func = ae_applicator)

### vidDict maps the video id to the first and last isCourt segment we want to label.
vidDict = {
            '2021-7341-ms001-brest-brandon-nakashima-vs-joao-sousa': (0, 2),
            '2021-7341-ms002-brest-brandon-nakashima-vs-alexandre-muller': (0, 1),
            '2021-7341-ms003-brest-joao-sousa-vs-maxime-janvier': (0, 2),
            '2021-7341-ms004-brest-alexandre-muller-vs-luca-van-assche': (0, 2),
            '2021-7341-ms005-brest-brandon-nakashima-vs-henri-laaksonen': (0, 2),
            '2021-7341-ms007-brest-joao-sousa-vs-federico-gaio': (0, 2),
            '2021-7341-ms008-brest-luca-van-assche-vs-manuel-guinard': (0, 1),
            '2021-7341-ms009-brest-alexandre-muller-vs-stefano-travaglia': (1, 1), ### has warm up as first isCourt segment!
            '2021-7341-ms010-brest-brandon-nakashima-vs-bernabe-zapata-miralles': (1, 2), ### has warm up as first isCourt segment!
            '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano': (0, 3)
           }

sto = 'segmentation_and_labelling_impactonly'
kind = 'bounce'

# for vid in vidDict:
vid = '2021-7341-ms010-brest-brandon-nakashima-vs-bernabe-zapata-miralles'
fs_crt, ls_crt = vidDict[vid]
videoDir = ut.get_folder_struct('video_files')[vid]
videoPath = os.path.join(videoDir, 'match.mp4')
fps = 25
cap = ut.get_video(vid)

# get segmenters
crtBF = BoolFeature.from_store(vid, sto, featname = 'isCourt')
segBF = BoolFeature.from_store(vid, sto, featname = 'isPlay')

# concentrate on court segments with labels
crtBFs = crtBF.filter(crtBF).split()
crtBFs = crtBFs[fs_crt:ls_crt + 1] ### only the first four court segments have consistent labels
crtBF = crtBFs.concatenate().conform(segBF).fillna(False)

# split single points
segBF = segBF.filter(crtBF).filter(segBF)

label_feat_name = f'labelled_{kind}_Flag'
if kind == 'shot':
    labSF = StreamFeatures.from_store(vid, sto, featnameList = ['labelled_top_Flag', 'labelled_bot_Flag'])
    labSF = labSF['labelled_top_Flag'].fillna(labSF['labelled_bot_Flag']).rename(label_feat_name)
else:
    labSF = StreamFeature.from_store(vid, sto, featname = label_feat_name)
# labSF_first = labSF.filter(labSF == 'bounce first') ### only interested in first bounces (others are likely not well perceptible... let's try anyway

audio_feat_name = f'labelled_{kind}_audio_sample_Flag'
try:
    smpSF = StreamFeature.from_store(vid, sto, featname = audio_feat_name)
except:
    smpSF = StreamFeature({}, name = audio_feat_name)

jntSFs = segBF.merge(labSF, how = 'left').merge(smpSF, how = 'left').drop('isPlay').split()
    
# get waveform
metadata = torchaudio.info(videoPath)
sample_rate = metadata.sample_rate
print(metadata)
samples_per_frame = sample_rate // fps

stereo_waveform, sample_rate = torchaudio.load(videoPath)
stereo_waveform = stereo_waveform.numpy()
mono_waveform = np.mean(stereo_waveform, axis = 0)
samples = len(mono_waveform)

#%%## Graph

frames_before = 5
frames_after = 10
offset = 500
ae_wl = 256
ae_hl = 16

graph_folder = ut.get_folder_struct('graphs')['audio_analysis']
save_folder = ut.maybe_make_folder(graph_folder, f'label_alignment_bounces_off{offset}_wl{ae_wl}_hl{ae_hl}_{vid}')

breakFlag = False

sampleDict = {}
for seq_idx, mySF in enumerate(jntSFs):
    mySF = mySF.dropna()
    if mySF.is_empty():
        continue
    
    shotSFs = mySF.split() ### this way, double-frame impacts are done only once.
    
    print(f'Sequence {seq_idx} out of {jntSFs.length()}...')

    for hit_idx, shotSF in enumerate(shotSFs):
        name = f'seq{seq_idx}_hit{hit_idx}'
        try:
            audio_labSF = shotSF[audio_feat_name].dropna()
        except:
            pass
        else:
            if not audio_labSF.is_empty(): ### already labelled this!
                if audio_labSF.unique()[0] == 'redo':
                    pass
                    # name = name + '_redo'
                else:
                    continue
            
        print(f'   ... hit {hit_idx} out of {shotSFs.length()}...')

        frames = shotSF.get_frames()
        chosen_peak_sample_shifted = None
        
        while chosen_peak_sample_shifted is None:
            # featDict = mySF[frame]
            
            # if featDict['labelled_top_Flag'] == 'impact':
            #     which = 'top'
            # elif featDict['labelled_bot_Flag'] == 'impact':
            #     which = 'bot'
            
            ### figure out first and last sample within the inmpact frame(s) 
            fs_frame, ls_frame = frames[0], frames[-1]
            fs_sample, _ = first_last_sample_index_in_frame(fs_frame, 0, samples_per_frame)
            _, ls_sample = first_last_sample_index_in_frame(ls_frame, 0, samples_per_frame)

            
            ### Produce the cropped waveform
            fs_sample_shift = fs_sample - samples_per_frame * frames_before
            ls_sample_shift = ls_sample + samples_per_frame * frames_after
            crop_waveform = mono_waveform[fs_sample_shift: ls_sample_shift]
            
            ### find local maxima
            wins, ae = ae_func(crop_waveform, window_length = ae_wl, hop_length = ae_hl)
            wins_samples = np.array(wins) + fs_sample_shift
            peak_idxs, peak_dict = find_peaks(ae, height = np.quantile(ae, 0.75))
            peak_heights = peak_dict['peak_heights']
            peak_samples = [wins_samples[peak_idx] for peak_idx in peak_idxs]
            # max_peak_sample = peak_samples[np.argmax(peak_heights)]
            
            ### plot, for checking
            fig, ax = plt.subplots(1, 1, figsize = (12, 10))
            ax = plot_waveform(ax, crop_waveform, sample_rate, first_sample_index = fs_sample_shift)
            ax.plot(wins_samples, ae)
            ax.axvspan(fs_sample, ls_sample, facecolor='firebrick', alpha=0.5)
            # ax.axvspan(max_peak_sample, max_peak_sample + samples_per_frame, facecolor='forestgreen', alpha=0.5)
                        
            ### additionally to above: label peaks, so we can fix the choice!
            for peak_idx, peak_sample in enumerate(peak_samples):    
                peak_height = peak_heights[peak_idx]
                ax.annotate(str(peak_idx), (peak_sample, peak_height), size = 12)    
            
            ### show graph, and make a choice of peak
            plt.show()
            sd.play(crop_waveform, samplerate = sample_rate)
            time.sleep(len(crop_waveform) / sample_rate)
            
            choiceStr = input('Choose peak index: ')
            try:
                choiceInt = int(choiceStr)
            except:
                if choiceStr.startswith('l'): ### l for listen
                    if choiceStr == 'l':
                        choiceStr = 'l_4'
                    slowdown = int(choiceStr.split('_')[1])
                    slow_sr = sample_rate // slowdown
                    sd.play(crop_waveform, samplerate = slow_sr)
                    time.sleep(len(crop_waveform) / slow_sr)
                elif choiceStr == 'v': ### v for view
                    Stream(cap, fs_frame - frames_before, ls_frame + frames_after).show(ms_between_frames = 1000)
                elif choiceStr == 'x': ### store and redo
                    chosen_peak_sample_shifted = 'redo'
                elif choiceStr == 'q':
                    breakFlag = True
                    break
                elif choiceStr.startswith('r'): ### reset frames before and frames after
                    _, bef, aft = choiceStr.split('_')
                    frames_before = int(bef)
                    frames_after = int(aft)
                elif choiceStr == 'n': ### inaudible
                    chosen_peak_sample_shifted = 'inaudible'
            else:
                chosen_peak_sample_shifted = peak_samples[choiceInt] - offset
                
                ### finish plot
                ax.axvspan(chosen_peak_sample_shifted, chosen_peak_sample_shifted + samples_per_frame, facecolor='forestgreen', alpha=0.5)
            
        if breakFlag:
            break
        
        ### confirm choice and save
        for frame in frames:
            sampleDict[frame] = chosen_peak_sample_shifted
        
        
        fig.savefig(os.path.join(save_folder, name + '.png'))
        plt.close()

    if breakFlag:
        break

sampleSF = StreamFeature(sampleDict, name = audio_feat_name)
sampleSF.save(vid, sto)

#%%% Check on some strange cases

# for seq_idx, mySF in enumerate(jntSFs):
#     mySF = mySF.dropna()
#     shotSFs = mySF.split() ### this way, double-frame impacts are done only once.
    
#     print(f'Sequence {seq_idx} out of {jntSFs.length()}...')

# frames_before = 2
# frames_after = 6
# offset = 250
# ae_wl = 256
# ae_hl = 16

# seq_idx = 57
# hit_idx = 3

# mySF = jntSFs[seq_idx].dropna()
# frame = mySF.get_frames()[hit_idx]
# featDict = mySF[frame]

# if featDict['labelled_top_Flag'] == 'impact':
#     which = 'top'
# elif featDict['labelled_bot_Flag'] == 'impact':
#     which = 'bot'

# ### Produce the cropped waveform
# fs_sample, ls_sample = first_last_sample_index_in_frame(frame, 0, samples_per_frame)
# fs_sample_shift = fs_sample - samples_per_frame * frames_before
# ls_sample_shift = ls_sample + samples_per_frame * frames_after
# crop_waveform = mono_waveform[fs_sample_shift: ls_sample_shift]

# ### find local maxima
# wins, ae = ae_func(crop_waveform, window_length = ae_wl, hop_length = ae_hl)
# wins_samples = np.array(wins) + fs_sample_shift
# peak_idxs, peak_dict = find_peaks(ae, height = np.quantile(ae, 0.75))
# peak_heights = peak_dict['peak_heights']
# peak_samples = [wins_samples[peak_idx] for peak_idx in peak_idxs]
# max_peak_sample = peak_samples[np.argmax(peak_heights)]
# # sampleDict[frame] = max_peak_sample

# ### plot, for checking
# fig, ax = plt.subplots(1, 1, figsize = (12, 10))
# ax = plot_waveform(ax, crop_waveform, sample_rate, first_sample_index = fs_sample_shift)
# ax.plot(wins_samples, ae)
# ax.axvspan(fs_sample, ls_sample, facecolor='firebrick', alpha=0.5)
# ax.axvspan(max_peak_sample, max_peak_sample + samples_per_frame, facecolor='forestgreen', alpha=0.5)

# ### additionally to above: label peaks, so we can fix the choice!
# for peak_idx, peak_sample in enumerate(peak_samples):    
#     peak_height = peak_heights[peak_idx]
#     ax.annotate(str(peak_idx), (peak_sample, peak_height))    


# ### listen and watch
# sd.play(crop_waveform)
# Stream(cap, frame - frames_before, frame + frames_after).show(ms_between_frames = 1000)


    
    











