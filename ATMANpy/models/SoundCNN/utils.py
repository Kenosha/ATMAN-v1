# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:24:58 2023

@author: gianl
"""

import numpy as np

import librosa
import matplotlib.pyplot as plt


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


def plot_waveform(ax, waveform, sample_rate, first_sample_index = 0):
    x_axis = np.arange(0, len(waveform)) + first_sample_index
    ax.plot(x_axis, waveform, linewidth = 1, alpha = 0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True)
    return ax

def plot_spectrogram(Y, sample_rate, 
                     hop_length = 512, window_length = 1024, 
                     y_axis = 'linear'):
    plt.figure(figsize = (25, 10))
    librosa.display.specshow(Y, sr = sample_rate, hop_length = hop_length,
                             n_fft = window_length,
                             x_axis = 'time', y_axis = y_axis)
    plt.colorbar(format = '%+2.f')

