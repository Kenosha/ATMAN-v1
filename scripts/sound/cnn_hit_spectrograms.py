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

#%% Create a bunch of spectrograms of hits and non-hits

HOP_LENGTH = 512
WINDOW_LENGTH = 1024
CNN_WING_SIZE = 10 ### this is the one-sided wing size of the clips we feed into the cnn, measured in fft hops.
CNN_SCOPE = CNN_WING_SIZE * 2 + 1  # if CNN_WING_SIZE = 10, the scope of the CNN (i.e, what it sees, in hops), is 10 * 2 + 1
SEMIHITS_PER_HIT = 4
NONHITS_PER_HIT = 50

def first_last_sample_index_in_frame(frame_number, first_frame_number, samples_per_frame):
    frame_index = frame_number - first_frame_number
    first_sample_in_frame = frame_index * samples_per_frame
    last_sample_in_frame = first_sample_in_frame + samples_per_frame - 1
    return first_sample_in_frame, last_sample_in_frame
   
def first_last_sample_index_in_window(window_index, hop_length, window_length):
    first_sample_index = window_index * hop_length
    last_sample_index = first_sample_index + window_length - 1
    return first_sample_index, last_sample_index
 
def first_window_with_sample(sample_index, hop_length, window_length):
    i = 0
    fs, ls = first_last_sample_index_in_window(i, hop_length, window_length)
    is_in_window = fs <= sample_index <= ls ### note the two weak inequalities: this is because the two first_last functions above return samples strictly inside the frame/window
    while not is_in_window:
        i += 1
        fs, ls = first_last_sample_index_in_window(i, hop_length, window_length)
        is_in_window = fs <= sample_index <= ls 
    else:
        return i

def last_window_with_sample(sample_index, hop_length, window_length):
    first_window = first_window_with_sample(sample_index, hop_length, window_length)
    
    i = first_window + 1
    fs, ls = first_last_sample_index_in_window(i, hop_length, window_length)
    is_in_window = fs <= sample_index <= ls ### note the two weak inequalities: this is because the two first_last functions above return samples strictly inside the frame/window
    while is_in_window:
        i += 1
        fs, ls = first_last_sample_index_in_window(i, hop_length, window_length)
        is_in_window = fs <= sample_index <= ls 
    else:
        return i - 1

def calculate_proximity(windows, hit_windows, maximum_allowable_distance):
    winArr, hitArr = np.array(windows)[:, np.newaxis], np.array(hit_windows)[np.newaxis, :]
    distance_to_nearest_hit_window = np.min(np.absolute(winArr - hitArr), axis = 1)
    
    overlap = np.maximum(maximum_allowable_distance - distance_to_nearest_hit_window, 0)
    overlap_share = overlap / maximum_allowable_distance
    
    return overlap_share ** 2

def plot_spectrogram(Y, sample_rate, 
                     hop_length = 512, window_length = 1024, 
                     y_axis = 'linear'):
    plt.figure(figsize = (25, 10))
    librosa.display.specshow(Y, sr = sample_rate, hop_length = hop_length,
                             n_fft = window_length,
                             x_axis = 'time', y_axis = y_axis)
    plt.colorbar(format = '%+2.f')


### plot proximity function
windows = np.arange(0, 100, 1)
hit_windows = [np.floor(np.mean(windows)), np.ceil(np.mean(windows))]
proxArr = calculate_proximity(windows, hit_windows, CNN_WING_SIZE)
plt.plot(windows, proxArr)
plt.show()
plt.close()

################################

mySF = jntSFs[0]
fs, ls = mySF.fs_ls()
audSF = mySF['labelled_shot_audio_sample_Flag'].dropna()

### get waveform
fs_sample, _ = first_last_sample_index_in_frame(fs, 0, samples_per_frame)
_, ls_sample = first_last_sample_index_in_frame(ls, 0, samples_per_frame)
sf_form = mono_waveform[fs_sample:ls_sample]

### make spectrogram
fts = librosa.stft(sf_form, n_fft = WINDOW_LENGTH, hop_length = HOP_LENGTH)
num_freqs, num_ts = fts.shape
spec = np.abs(fts) ** 2 ### squared magnitude gives spectrogram values
db_spec = librosa.power_to_db(spec)

### divide into hit regions and non-hit regions
hit_windows = []
for hit_sample in audSF.unique():    
    fs_hit_win = first_window_with_sample(hit_sample, HOP_LENGTH, WINDOW_LENGTH)
    ls_hit_win = last_window_with_sample(hit_sample, HOP_LENGTH, WINDOW_LENGTH)
    one_hit_windows = list(range(fs_hit_win, ls_hit_win + 1))
    hit_windows.append(one_hit_windows)
    
hit_windows_flat = [win for wins in hit_windows for win in wins]
hit_windows = np.array(hit_windows)
hit_windows_flat = np.array(hit_windows_flat)
    
### sample clear hits, shifted hits, and non-hits
### --> each with different sample probabilities.
rng = np.random.default_rng(seed = 42)

windows = np.arange(0, num_ts)
proxArr_strict = calculate_proximity(windows, hit_windows_flat, CNN_WING_SIZE)
proxArr_lenient = calculate_proximity(windows, hit_windows_flat, CNN_SCOPE)
plt.plot(windows, proxArr_strict, alpha = 0.9)
plt.plot(windows, proxArr_lenient, alpha = 0.9)
plt.show()
plt.close()

### Clear Hits -> take one (of the hit_windows)
assert (hit_windows_flat == windows[proxArr_strict == 1]).all()
chosen_hit_wins = np.array([rng.choice(wins, replace = False) for wins in hit_windows])

### Non-centered hits -> take 2 from the left and 2 from the right of the shot beginning, for each hit
### Note that with proxArr_strict, all semihits overlap the main spike of the shot, so that the boom is always included.
non_centered_windows = windows[(0 < proxArr_strict ) & (proxArr_strict < 1)]
adjacent_non_centered_windows = ut.split_into_contiguous(non_centered_windows) ### there will be 2 * number of shots arrays here.
chosen_semihit_wins = np.array([rng.choice(wins, size = SEMIHITS_PER_HIT // 2, replace = False) for wins in adjacent_non_centered_windows])
chosen_semihit_wins = chosen_semihit_wins.flatten()

### Non-Hits
non_hit_windows = windows[proxArr_lenient == 0]
non_hits_to_sample = NONHITS_PER_HIT * len(chosen_hit_wins)
chosen_nonhit_wins = rng.choice(non_hit_windows[CNN_WING_SIZE:-CNN_WING_SIZE], size = non_hits_to_sample)

### Collect all spectrograms, and their labels
chosen_wins = np.concatenate([chosen_hit_wins, chosen_semihit_wins, chosen_nonhit_wins])

specs = []
labels = []

for win in chosen_wins:
    left_hop, right_hop = win - CNN_WING_SIZE, win + CNN_WING_SIZE + 1
    win_spec = db_spec[:, left_hop: right_hop]
    label = proxArr_strict[win]
    
    specs.append(win_spec)
    labels.append(label)

specs = np.array(specs) ### shape: N x Freqs x Hops
labels = np.array(labels) # shape: N







