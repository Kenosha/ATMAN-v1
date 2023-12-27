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
   
from ATMANpy.models.SoundCNN.utils import first_last_sample_index_in_frame, first_last_sample_index_in_window
from ATMANpy.models.SoundCNN.utils import first_window_with_sample, last_window_with_sample
from ATMANpy.models.SoundCNN.utils import calculate_proximity, plot_waveform, plot_spectrogram

HOP_LENGTH = 512
WINDOW_LENGTH = 1024
CNN_WING_SIZE = 40 ### this is the one-sided wing size of the clips we feed into the cnn, measured in fft hops.
CNN_SCOPE = CNN_WING_SIZE * 2 + 1  # if CNN_WING_SIZE = 10, the scope of the CNN (i.e, what it sees, in hops), is 10 * 2 + 1
SEMIHITS_PER_HIT = 4
NONHITS_PER_HIT = 50
MEL_BANDS = 128
MEL_NAME = f'MEL{MEL_BANDS}' if MEL_BANDS is not None else 'SPEC' 

_sr = 44100 ### is the correct value, check sample rate below
print(f'Scope of CNN: {CNN_SCOPE} hops at {HOP_LENGTH} samples per hop')
print(f'Length of scope: {HOP_LENGTH * CNN_SCOPE / _sr} seconds')

### save where
DATA_NAME = 'train'

full_name = f'{DATA_NAME}_{MEL_NAME}_HL{HOP_LENGTH}_WL{WINDOW_LENGTH}_WS{CNN_WING_SIZE}_1_{SEMIHITS_PER_HIT}_{NONHITS_PER_HIT}'
dataPath = ut.get_folder_struct('cnn_sound')['data']
save_where = ut.maybe_make_folder(dataPath, full_name)

PLOT_PROXFUNC = False
PLOT_PROXVALS = False

SEED = 42
rng = np.random.default_rng(seed = SEED)

### Test Proximity Function

if PLOT_PROXFUNC:
    ### plot proximity function
    windows = np.arange(0, 100, 1)
    hit_windows = [np.floor(np.mean(windows)), np.ceil(np.mean(windows))]
    proxArr = calculate_proximity(windows, hit_windows, CNN_WING_SIZE)
    plt.plot(windows, proxArr)
    plt.show()
    plt.close()
 
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
            # '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano': (0, 3)
           }

specList = []
labList = []
for vid in vidDict:
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
    
    ################################
    
    all_specs = []
    all_labels = []
    for mySF in jntSFs:
        fs, ls = mySF.fs_ls()
        audSF = mySF['labelled_shot_audio_sample_Flag'].dropna()
        
        if audSF.is_empty():
            continue
        
        ### get waveform
        fs_sample, _ = first_last_sample_index_in_frame(fs, 0, samples_per_frame)
        _, ls_sample = first_last_sample_index_in_frame(ls, 0, samples_per_frame)
        sf_form = mono_waveform[fs_sample:ls_sample]
        
        ### make spectrogram
        if MEL_BANDS is None:
            fts = librosa.stft(sf_form, 
                               n_fft = WINDOW_LENGTH, 
                               hop_length = HOP_LENGTH)
            spec = np.abs(fts) ** 2 ### squared magnitude gives spectrogram values
            # db_spec = librosa.power_to_db(spec) # dont transform so that the output 
                                                  # does not depend on the reference value 
                                                  # within the spectrogram of mySF
                                                  # later, once all spectrograms are created, we transform!
            num_freqs, num_ts = spec.shape
        else:
            melFunc = T.MelSpectrogram(sample_rate = sample_rate, 
                                       n_fft = WINDOW_LENGTH, 
                                       hop_length = HOP_LENGTH, 
                                       n_mels = MEL_BANDS)
            spec = melFunc(torch.Tensor(sf_form)).numpy()
            num_freqs, num_ts = spec.shape
             
        ### divide into hit regions and non-hit regions
        hit_windows = []
        for hit_sample in audSF.unique():    
            try:
                int_hit_sample = int(hit_sample)
            except ValueError:
                if hit_sample == 'inaudible':
                    continue ### don't register as a hit sample
            else:
                hit_sample_rel = int(hit_sample) - fs_sample
                
            fs_hit_win = first_window_with_sample(hit_sample_rel, HOP_LENGTH, WINDOW_LENGTH)
            ls_hit_win = last_window_with_sample(hit_sample_rel, HOP_LENGTH, WINDOW_LENGTH)
            one_hit_windows = list(range(fs_hit_win, ls_hit_win + 1))
            hit_windows.append(one_hit_windows)
            
        hit_windows_flat = [win for wins in hit_windows for win in wins]
        hit_windows = np.array(hit_windows)
        hit_windows_flat = np.array(hit_windows_flat)
            
        ### sample clear hits, shifted hits, and non-hits
        ### --> each with different sample probabilities.        
        windows = np.arange(0, num_ts)
        proxArr_strict = calculate_proximity(windows, hit_windows_flat, CNN_WING_SIZE)
        proxArr_lenient = calculate_proximity(windows, hit_windows_flat, CNN_SCOPE)
        
        if PLOT_PROXVALS:
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
        cropped_non_hit_windows = non_hit_windows[CNN_WING_SIZE:-CNN_WING_SIZE] ### crop so that we get full size spectrogram snippets
        non_hits_to_sample = NONHITS_PER_HIT * len(chosen_hit_wins)
        chosen_nonhit_wins = rng.choice(cropped_non_hit_windows, size = non_hits_to_sample)
        
        ### Collect all spectrograms, and their labels
        chosen_wins = np.concatenate([chosen_hit_wins, chosen_semihit_wins, chosen_nonhit_wins])
        
        specs = []
        labels = []
        
        for win in chosen_wins:
            left_hop, right_hop = win - CNN_WING_SIZE, win + CNN_WING_SIZE + 1
            win_spec = spec[:, left_hop: right_hop]
            label = proxArr_strict[win]
            
            specs.append(win_spec)
            labels.append(label)
        
        # specs -> a list of N samples pertaining to mySF, each sample being an array with shape freqs x hops, representing the spectrogram. 
        # labels -> a list of N labels pertaining to mySF, each label being a number between 0 (not a shot) and 1 (shot in the middle)
        
        all_specs.extend(specs)
        all_labels.extend(labels)
    
    print('__________')
    print(vid)
    print(f'{len(all_specs)} Spectrograms')
    print(f'{len(all_labels)} Labels')
    print('___________')

    specList.extend(all_specs)
    labList.extend(all_labels)

specArr = np.array(specList)
labArr = np.array(labList)

print('____________')
print('DONE!')
print(f'Spectrograms shape: {specArr.shape}')
print(f'Spectrograms size on disc: {specArr.size * specArr.dtype.itemsize / 1000000} MB')
print(f'Labels shape: {labArr.shape}')
print(f'Spectrograms size on disc: {labArr.size * labArr.dtype.itemsize / 1000000} MB')

specFile = os.path.join(save_where, 'specs.npy')
np.save(specFile, specArr)

labFile = os.path.join(save_where, 'labels.npy')
np.save(labFile, labArr)




