# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:22:13 2023

@author: gianl
"""


import os
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
import sounddevice as sd

import ATMANpy.funcs.util_funcs as ut
from ATMANpy.types.Features import BoolFeature, StreamFeatures

def plot_waveform(ax, waveform, sample_rate):
    ### waveform has to be mono!
    time_axis = np.arange(0, len(waveform)) / sample_rate
    ax.plot(time_axis, waveform, linewidth = 1, alpha = 0.5)
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

vid = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
sto = 'cnn_store'
videoDir = ut.get_folder_struct('video_files')[vid]
videoPath = os.path.join(videoDir, 'match.mp4')
fps = 25
cap = ut.get_video(vid)

metadata = torchaudio.info(videoPath)
sample_rate = metadata.sample_rate
print(metadata)
samples_per_frame = sample_rate // fps

stereo_waveform, sample_rate = torchaudio.load(videoPath)
stereo_waveform = stereo_waveform.numpy()
mono_waveform = np.mean(stereo_waveform, axis = 0)
samples = len(mono_waveform)

segBF = BoolFeature.from_store(vid, sto, featname = 'isRally')
segBFs = segBF.filter(segBF).split()

myBF = segBFs[0]
sf_form = crop_waveform_to_sf(mono_waveform, myBF, samples_per_frame)

fig, ax = plt.subplots(1, 1, figsize = (10, 8))
ax = plot_waveform(ax, sf_form, sample_rate)

#%%## Time-Domain Features

def apply_func_to_window(waveform, func, 
                    window_length = 1024, hop_length = 512):
    return np.array([func(waveform[i:i + window_length]) for i in range(0, len(waveform), hop_length)])

ae_func = partial(apply_func_to_window, func = np.max)

def rmse(x):
    return ((x ** 2).mean()) ** 0.5
rmse_func = partial(apply_func_to_window, func = rmse)

WINDOW_LENGTH = 1024
HOP_LENGTH = 512

ae = ae_func(sf_form, window_length = WINDOW_LENGTH, hop_length = HOP_LENGTH)
rms = rmse_func(sf_form, window_length = WINDOW_LENGTH, hop_length = HOP_LENGTH)
zcr = librosa.feature.zero_crossing_rate(sf_form, frame_length = WINDOW_LENGTH, hop_length = HOP_LENGTH)

frames = range(0, len(ae))
t = librosa.frames_to_time(frames, hop_length = HOP_LENGTH, sr = sample_rate)

def plot_feature(ax, feat, time):
    ax.plot(time, feat)
    return ax

ax = plot_feature(ax, ae, t)
ax = plot_feature(ax, rms, t)
ax = plot_feature(ax, zcr[0], t)

#%%% Try to get periods of silence

fig, ax = plt.subplots(1, 1, figsize = (10, 8))
ax = plot_waveform(ax, sf_form, sample_rate)

rms_norm = (rms - np.median(rms)) / np.std(rms)

ax = plot_feature(ax, rms_norm, t)

#%%% Frequency-Domain Features

def plot_magnitude_spectrum(waveform, title, sample_rate):
    ft = np.fft.fft(waveform)
    ft_mag = np.abs(ft)
    
    # cut off ft at halfway point so that we don't get nyquist ghosts
    cutoff = len(ft_mag) // 2 + 1
    nyquist_frequency = sample_rate // 2
    ft_crop = ft_mag[:cutoff]
    frequency = np.linspace(0, nyquist_frequency, cutoff)

    fig, ax = plt.subplots(1, 1, figsize = (18, 5))
    
    ax.plot(frequency, ft_crop)
    plt.xlabel('Frequency (Hz)')
    plt.title(title)
    plt.show()
    
plot_magnitude_spectrum(sf_form, '', sample_rate)
    
### Short-Time Fourier Transforms

def plot_spectrogram(Y, sample_rate, hop_length, y_axis = 'linear'):
    plt.figure(figsize = (25, 10))
    librosa.display.specshow(Y, sr = sample_rate, hop_length = hop_length,
                             x_axis = 'time', y_axis = y_axis)
    plt.colorbar(format = '%+2.f')

fts = librosa.stft(sf_form, n_fft = WINDOW_LENGTH, hop_length = HOP_LENGTH)
num_freqs, num_ts = fts.shape

spec = np.abs(fts) ** 2 ### squared magnitude gives spectrogram values
db_spec = librosa.power_to_db(spec)

plot_spectrogram(spec, sample_rate, HOP_LENGTH)
plot_spectrogram(db_spec, sample_rate, HOP_LENGTH)
plot_spectrogram(db_spec, sample_rate, HOP_LENGTH, y_axis = 'log')

mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = sample_rate, n_fft = WINDOW_LENGTH,
                                               hop_length = HOP_LENGTH)
mel_scale = torchaudio.transforms.MelScale(sample_rate = sample_rate, n_stft = 513)


mel_spec_array = melFunc(torch.Tensor(sf_form)).numpy()
mel_spec_array_2 = mel_scale(torch.Tensor(spec)).numpy()

plot_spectrogram(mel_spec_array, sample_rate, HOP_LENGTH)
plot_spectrogram(mel_spec_array_2, sample_rate, HOP_LENGTH)

mel_spec_array
mel_spec_array_2


#%% MFCCs

mfccs = librosa.feature.mfcc(sf_form, 
                             hop_length = HOP_LENGTH, n_fft = WINDOW_LENGTH, 
                             n_mfcc = 13, sr = sample_rate)

delta_mfccs = librosa.feature.delta(mfccs)
delta2_mfccs = librosa.feature.delta(mfccs, order = 2)

plot_spectrogram(mfccs, sample_rate, hop_length = HOP_LENGTH, y_axis = None)
plot_spectrogram(delta_mfccs, sample_rate, hop_length = HOP_LENGTH, y_axis = None)
plot_spectrogram(delta2_mfccs, sample_rate, hop_length = HOP_LENGTH, y_axis = None)

comp_mfccs = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])

plot_spectrogram(comp_mfccs, sample_rate, hop_length = HOP_LENGTH, y_axis = None)

comp_mfccs.shape

#%%% Check the audio features at and around shots - can they be identified simply?

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
    
# get segmenters
crtBF = BoolFeature.from_store(vid, sto, featname = 'isCourt')
segBF = BoolFeature.from_store(vid, sto, featname = 'isRally')

# concentrate on court segments with labels
crtBFs = crtBF.filter(crtBF).split()
crtBFs = crtBFs[:4] ### only the first four court segments have consistent labels
crtBF = crtBFs.concatenate().conform(segBF).fillna(False)

# split single points
segBF = segBF.filter(crtBF).filter(segBF)

sto_label = 'segmentation_and_labelling_impactonly'
labSF = StreamFeatures.from_store(vid, sto_label, featnameList = ['labelled_top_Flag', 'labelled_bot_Flag'])

jntSFs = segBF.merge(labSF, how = 'left').drop('isRally').split()

### initialize parameters for feature extraction
WINDOW_LENGTH = 1024
HOP_LENGTH = 512

window_duration = WINDOW_LENGTH / sample_rate

print(f'window duration: {round(window_duration * 1000, 1)} ms')

samples_per_frame = sample_rate / fps

print(f'Sound samples in a video frame: {round(samples_per_frame, 1)}')

graph_folder = ut.get_folder_struct('graphs')['audio_analysis']
save_folder = ut.maybe_make_folder(graph_folder, 'mfcc')

# check mfccs
idx = 0
mySF = jntSFs[0]
fs, ls = fs_ls(mySF)
sf_form = crop_waveform_to_sf(mono_waveform, mySF, samples_per_frame)

mfccs = librosa.feature.mfcc(sf_form, 
                             hop_length = HOP_LENGTH, n_fft = WINDOW_LENGTH, 
                             n_mfcc = 13, sr = sample_rate)
delta_mfccs = librosa.feature.delta(mfccs)
delta2_mfccs = librosa.feature.delta(mfccs, order = 2)
comp_mfccs = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])

folder_name = f'seq_{idx}'
save_where = ut.maybe_make_folder(save_folder, folder_name)

mean, std = comp_mfccs.mean(axis = 1, keepdims = True), comp_mfccs.std(axis = 1, keepdims = True)
norm_mfccs = (comp_mfccs - mean) / std
    
main_color = 'whitesmoke'
for frame in mySF:
    title_stump = f'Frame: {frame}'
    
    ### initialize impact coloring
    bot_impact = mySF[frame]['labelled_bot_Flag'] == 'impact'
    top_impact = mySF[frame]['labelled_top_Flag'] == 'impact'
    if bot_impact:
        color = 'maroon'
        title = title_stump + ' -> bot impact'
    elif top_impact:
        color = 'saddlebrown'
        title = title_stump + ' -> top impact'
    else:
        title = title_stump
        color = 'dimgrey'
        
    ### Get associated mfcc windows
    first_sample, last_sample = first_last_sample_index_in_frame(frame, fs, samples_per_frame)
    
    first_window = first_window_with_sample(first_sample, HOP_LENGTH, WINDOW_LENGTH)
    last_window = last_window_with_sample(last_sample, HOP_LENGTH, WINDOW_LENGTH)

    frame_mfccs = norm_mfccs[:, first_window:last_window+1]
    num_feats, num_windows = frame_mfccs.shape
    
    ### Plot
    fig, axes = plt.subplots(num_windows, 1, sharex = True, figsize = (16, 16))
    fig.suptitle(title, color = main_color)
    fig.set_facecolor(color) ### change color to match impact type
    
    for win in range(num_windows):
        win_mfccs = frame_mfccs[:, win]
        ax = axes[win]
        
        ax.plot(win_mfccs, color = main_color)
        
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(f'Window: {first_window + win}', size = 10)
        ax.set_facecolor(color) ### change color to match impact type
        
        ax.title.set_color(main_color)
        ax.spines[['bottom', 'left']].set_color(main_color)
        ax.tick_params(axis='x', colors = main_color)
        ax.tick_params(axis='y', colors = main_color)
        ax.yaxis.label.set_color(main_color)
        ax.xaxis.label.set_color(main_color)


    save_name = f'melspec_{frame}.png'
    fig.tight_layout()
    fig.savefig(os.path.join(save_where, save_name))
    plt.close()


    














