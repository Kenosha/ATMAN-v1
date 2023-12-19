# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:29:27 2023

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

# for vid in vidDict:
vid = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
sto = 'segmentation_and_labelling_impactonly'
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

labSF = StreamFeatures.from_store(vid, sto, featnameList = ['labelled_top_Flag', 'labelled_bot_Flag'])
try:
    smpSF = StreamFeature.from_store(vid, sto, featname = 'labelled_shot_audio_sample_Flag')
except:
    smpSF = StreamFeature({}, name = 'labelled_shot_audio_sample_Flag')

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

#%% Test Window Lengths

WINDOW_LENGTH = 1024
HOP_LENGTH = 512

shot_samples = pd.Series(smpSF.filter(smpSF != 'redo').to_series().astype(int).unique())

idx = 0
beg, end = shot_samples[idx] - 5, shot_samples[idx] + WINDOW_LENGTH * 20

win_waveform = mono_waveform[beg:end]
sd.play(win_waveform, samplerate = sample_rate)

ax = mono_waveform


#%% 

def plot_spectrogram(Y, sample_rate, hop_length, y_axis = 'linear'):
    plt.figure(figsize = (25, 10))
    librosa.display.specshow(Y, sr = sample_rate, hop_length = hop_length,
                             x_axis = 'time', y_axis = y_axis)
    plt.colorbar(format = '%+2.f')
    
WINDOW_LENGTH = 1024
HOP_LENGTH = 512

HOPS = 25
QUANTILES = [0.25, 0.5, 0.75]

shot_samples = pd.Series(smpSF.filter(smpSF != 'redo').to_series().astype(int).unique())
shift_samples = shot_samples - shot_samples.shift()

begSer, endSer = shot_samples - HOP_LENGTH * HOPS, shot_samples + WINDOW_LENGTH + HOP_LENGTH * HOPS
begSer.name = 'beg'
endSer.name = 'end'
winFrm = pd.concat([begSer, endSer], axis = 1)

mfccList = []
for idx in winFrm.index:
    beg, end = winFrm.loc[idx]
    shot_waveform = mono_waveform[beg:end]
    
    mfccs = librosa.feature.mfcc(y = shot_waveform, 
                                    hop_length = HOP_LENGTH, n_fft = WINDOW_LENGTH, 
                                    n_mfcc = 13, sr = sample_rate)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order = 2)
    comp_mfccs = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])
    
    mfccList.append(comp_mfccs)

plot_spectrogram(mfccList[0], sample_rate, HOP_LENGTH)

mfccs = np.array(mfccList)
q_mfccs = np.quantile(mfccs, QUANTILES, axis = 0)

plotFrm = pd.DataFrame(q_mfccs[:, 0].T, columns = QUANTILES)
plotFrm.index = plotFrm.index - np.mean(plotFrm.index)
plotFrm.plot()












