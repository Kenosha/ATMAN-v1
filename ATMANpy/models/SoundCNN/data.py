# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:21:06 2023

@author: gianl
"""


import os
import math
import random
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.v2 as tf
from torchvision.io import read_video
from torch.utils.data import Dataset, IterableDataset, get_worker_info

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import ATMANpy.funcs.util_funcs as ut
from ATMANpy.types.Features import BoolFeature
from ATMANpy.models.SoundCNN.utils import first_last_sample_index_in_frame

SAMPLE_RATE = 44100
FPS = 25

class LabelledSpectroData(Dataset):
    
    def __init__(self, name = 'data', 
                 data_descriptor = None, label_descriptor = None,
                 hop_length = 512, window_length = 1024, 
                 wing_size = 10, mel_bands = None,
                 semihits_per_hit = 4, nonhits_per_hit = 50,
                 to_db = True, db_reference = None, norm_value = None):
        
        super(LabelledSpectroData).__init__()
        
        mel_name = f'MEL{mel_bands}' if mel_bands is not None else 'SPEC' 

        if data_descriptor is None:
            data_name = f'{name}_{mel_name}_HL{hop_length}_WL{window_length}_WS{wing_size}_1_{semihits_per_hit}_{nonhits_per_hit}'
        else:
            data_name = f'{name}_{data_descriptor}_{mel_name}_HL{hop_length}_WL{window_length}_WS{wing_size}_1_{semihits_per_hit}_{nonhits_per_hit}'
        data_path = os.path.join(ut.get_folder_struct('cnn_sound')['data'], data_name)
        
        specArr = np.load(os.path.join(data_path, 'specs.npy'))
        
        if label_descriptor is None:
            label_name = 'labels.npy'
        else:
            label_name = f'{label_descriptor}_labels.npy'
        labArr = np.load(os.path.join(data_path, label_name))

        ### adapt the type of the labels to the type of specArr
        labArr = labArr.astype('float32')

        ### calculate some metrics, to use for decibel reference potentially
        mean_power = specArr.mean()
        median_power = np.median(specArr)
        if db_reference == 'mean':
            db_reference = mean_power
        elif db_reference == 'median':
            db_reference = median_power
    
        ### potentially transform to decibels
        if to_db:
            specArr = librosa.power_to_db(specArr, ref = db_reference)

        ### potentially normalize values
        if norm_value is not None:
            specArr = specArr / norm_value

        assert specArr.shape[0] == labArr.shape[0]
        self.samples, self.freqs, self.hops = specArr.shape

        ### kwargs
        self.name = name
        self.hop_length = hop_length
        self.window_length = window_length
        self.wing_size = wing_size
        self.semihits_per_hit = semihits_per_hit
        self.nonhits_per_hit = nonhits_per_hit
        self.to_db = to_db
        self.db_reference = db_reference

        ### path
        self.data_name = data_name
        self.data_path = data_path
        
        ### data
        self.specArr = specArr
        self.labArr = labArr
        
        ### useful added metrics
        self.scope_size = wing_size * 2 + 1
        self.secs_in_scope = hop_length * self.scope_size / SAMPLE_RATE
        self.mean_power = mean_power
        self.median_power = median_power
        self.is_mel = mel_name[:3] == 'MEL'
        
    def __len__(self):
        return self.samples
    
    def __getitem__(self, idx):
        spec = self.specArr[idx] # shape: freqs x hops
        label = self.labArr[idx] # float between 0 and 1
        
        ### add trivial channel dimension
        spec = spec[np.newaxis, :, :]
        
        return spec, label

def waveform_generator(vid, sto, segmenter = 'isRally', yield_frame_indices = True, widen = 0):
    ### path to match file
    videoDir = ut.get_folder_struct('video_files')[vid]
    videoPath = os.path.join(videoDir, 'match.mp4')
        
    ### get waveform
    metadata = torchaudio.info(videoPath)
    sample_rate = metadata.sample_rate
    # print(metadata)
    samples_per_frame = sample_rate // FPS

    # waveform to mono
    stereo_waveform, sample_rate = torchaudio.load(videoPath)
    stereo_waveform = stereo_waveform.numpy()
    mono_waveform = np.mean(stereo_waveform, axis = 0)

    ### get segmenter 
    ### TODO: Why do we read this again, and not pass it in?
    segBF = BoolFeature.from_store(vid, sto, featname = segmenter)
    segBFs = segBF.filter(segBF).split()
    
    for myBF in segBFs:
        fs, ls = myBF.fs_ls()
        fs, ls = max(fs - widen, 0), max(0, ls + widen)

        ### crop waveform
        fs_sample, _ = first_last_sample_index_in_frame(fs, 0, samples_per_frame)
        _, ls_sample = first_last_sample_index_in_frame(ls, 0, samples_per_frame)
        sf_form = mono_waveform[fs_sample:ls_sample]

        if yield_frame_indices:
            yield sf_form, fs, ls
        else:    
            yield sf_form, fs_sample, ls_sample

def point_sound_generator(vid, sto, segmenter = 'isRally', 
                          mel_bands = 128, hop_length = 512, window_length = 1024,
                          to_db = True, norm_value = 50, db_reference = 1,
                          yield_frame_indices = True, widen = 0):
    ### path to match file
    videoDir = ut.get_folder_struct('video_files')[vid]
    videoPath = os.path.join(videoDir, 'match.mp4')
        
    ### get waveform
    metadata = torchaudio.info(videoPath)
    sample_rate = metadata.sample_rate
    # print(metadata)
    samples_per_frame = sample_rate // FPS

    # waveform to mono
    stereo_waveform, sample_rate = torchaudio.load(videoPath)
    stereo_waveform = stereo_waveform.numpy()
    mono_waveform = np.mean(stereo_waveform, axis = 0)
    # samples = len(mono_waveform)
    
    ### get segmenter 
    ### TODO: Why do we read this again, and not pass it in?
    segBF = BoolFeature.from_store(vid, sto, featname = segmenter)
    segBFs = segBF.filter(segBF).split()
    
    for myBF in segBFs:
        fs, ls = myBF.fs_ls()
        fs, ls = max(fs - widen, 0), max(0, ls + widen)

        ### crop waveform
        fs_sample, _ = first_last_sample_index_in_frame(fs, 0, samples_per_frame)
        _, ls_sample = first_last_sample_index_in_frame(ls, 0, samples_per_frame)
        sf_form = mono_waveform[fs_sample:ls_sample]

        ### make spectrogram
        if mel_bands is None:
            fts = librosa.stft(sf_form, 
                               n_fft = window_length, 
                               hop_length = hop_length)
            spec = np.abs(fts) ** 2 ### squared magnitude gives spectrogram values
            # db_spec = librosa.power_to_db(spec) # dont transform so that the output 
                                                  # does not depend on the reference value 
                                                  # within the spectrogram of mySF
                                                  # later, once all spectrograms are created, we transform!
            num_freqs, num_ts = spec.shape
        else:
            melFunc = T.MelSpectrogram(sample_rate = sample_rate, 
                                       n_fft = window_length, 
                                       hop_length = hop_length, 
                                       n_mels = mel_bands)
            spec = melFunc(torch.Tensor(sf_form)).numpy()
            num_freqs, num_ts = spec.shape
                 
        ### potentially transform to decibels
        if to_db:
            spec = librosa.power_to_db(spec, ref = db_reference)

        ### potentially normalize values
        if norm_value is not None:
            spec = spec / norm_value

        ### add trivial channel dimension
        spec = spec[np.newaxis, :, :]

        if yield_frame_indices:
            yield spec, fs, ls
        else:    
            yield spec, fs_sample, ls_sample

class PointSpectroData(IterableDataset):
    def __init__(self, vid, sto, segmenter = 'isRally', 
                 mel_bands = 128, hop_length = 512, window_length = 1024,
                 to_db = True, norm_value = 50, db_reference = 1,
                 yield_frame_indices = True):
        
        self.vid = vid
        self.sto = sto
        
        self.segmenter = segmenter
        self.mel_bands = mel_bands
        self.hop_length = hop_length
        self.window_length = window_length
        self.to_db = to_db
        self.norm_value = norm_value
        self.db_reference = db_reference
        self.yield_frame_indices = yield_frame_indices
        
    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None: ### Single Process Data Loading
            inf_gen = point_sound_generator(self.vid, self.sto, segmenter = self.segmenter,
                                            mel_bands = self.mel_bands, hop_length = self.hop_length, window_length = self.window_length,
                                            to_db = self.to_db, norm_value = self.norm_value, db_reference = self.db_reference,
                                            yield_frame_indices = self.yield_frame_indices)
            return inf_gen

    


