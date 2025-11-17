# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:29:27 2023

@author: gianl
"""

import os
import time
import pandas as pd
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
import librosa
import librosa.display

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader

from functools import partial
from scipy.signal import find_peaks
import sounddevice as sd

import ATMANpy.funcs.util_funcs as ut
from ATMANpy.types.Features import BoolFeature, StreamFeature, StreamFeatures
from ATMANpy.types.Streams import Stream
   
from ATMANpy.models.SoundCNN.nets import SoundModel_FC
from ATMANpy.models.SoundCNN.data import PointSpectroData
from ATMANpy.models.SoundCNN.utils import first_last_sample_index_in_frame, first_last_sample_index_in_window
from ATMANpy.models.SoundCNN.utils import first_window_with_sample, last_window_with_sample
from ATMANpy.models.SoundCNN.utils import calculate_proximity, plot_waveform, plot_spectrogram
from ATMANpy.models.SoundCNN.utils import run_inference_on_arbitrary_spectrum

MODEL_TO_READ = 'test_model_20231223_001241'
SAMPLE_RATE = 44100
FPS = 25
samples_per_frame = SAMPLE_RATE / FPS
central_sample_in_frame_offset = (samples_per_frame - 1) / 2

LOG_PATH = ut.get_folder_struct('cnn_sound')['runs']
RES_PATH = ut.get_folder_struct('cnn_sound')['weights']
PRM_PATH = ut.get_folder_struct('cnn_sound')['params']

prm_read_where = os.path.join(PRM_PATH, MODEL_TO_READ + '.pkl')
res_read_where = os.path.join(RES_PATH, MODEL_TO_READ)

with open(prm_read_where, 'rb') as fp:
    params = pickle.load(fp)
    print('Read parameters: ')
    print(params)
    
if __name__ == '__main__':
        
    ### CNN parameters
    device = 'cuda'
    loader_workers = 0 ### TODO: Parallelize! (see todo_cnn)
    
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']

    segmenter = 'isServeStart'

    segBF = BoolFeature.from_store(vid, sto, featname = segmenter) ### TODO: Improve rally segmentation using also the audio
    segBFs = segBF.filter(segBF).split()

    model = SoundModel_FC(params['conv_filters'], params['conv_kernels'], params['pool_kernels'], params['fc_neurons'],
                          frequencies = params['freqs'], hops = params['hops'])
    model.to(device)
    model.load_state_dict(torch.load(res_read_where))
    model.train(False)
    
    ds = PointSpectroData(vid, sto, segmenter = segmenter, mel_bands = params['mel_bands'], 
                          hop_length = params['hop_length'], window_length = params['window_length'],
                          to_db = params['to_db'], norm_value = params['norm_value'], db_reference = params['db_reference'],
                          yield_frame_indices = False)
    
    inf_loader = DataLoader(ds, batch_size = 1, num_workers = 0, pin_memory = True)
    
    specArrs, sample_bounds = run_inference_on_arbitrary_spectrum(model, inf_loader, device)
    
    specSers = []
    for idx, specArr in enumerate(specArrs):
        fs_sample, ls_sample = sample_bounds[idx]
        mySF = segBFs[idx]
        frames = mySF.get_frames()
        
        stride = np.prod(params['pool_kernels'])
        hop = params['hop_length']
        win = params['window_length']
        scope = params['wing_size'] * 2 + 1
        scope_samples = scope * hop
        hop_stride = stride * hop
        
        left_sample_indices = np.arange(fs_sample, ls_sample, hop_stride)
        right_sample_indices = left_sample_indices + scope_samples - 1
        
        left_right_sample_indices = np.stack([left_sample_indices, right_sample_indices])
        central_sample_indices = np.mean(left_right_sample_indices, axis = 0)
        
        central_sample_indices = central_sample_indices[:len(specArr)]
        
        frame_div, frame_mod = central_sample_indices // samples_per_frame, central_sample_indices % samples_per_frame
        frame_indices = frame_div + frame_mod / samples_per_frame
        
        ### interpolate values to normal frames
        specSer = pd.Series(specArr, index = frame_indices)
        new_index = sorted(frames + frame_indices.tolist())
        specSer_reidx = specSer.reindex(new_index)
        specSer_itpl = specSer_reidx.interpolate(method = 'index', limit_area='inside')
        specSer_itpl = specSer_itpl[frames].dropna()
        
        specSers.append(specSer_itpl)
        
    outSer = pd.concat(specSers)
    outSer.name = 'hitprobFlag'
    outSF = StreamFeature.from_series(outSer)
        
    outSF.save(vid, sto)
    
    
    # fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    
    # ax.plot(frame_indices, specArr)
    
    # for frame in mySF.dropna():
    #     featDict = mySF[frame]
    #     tf, bf = featDict['labelled_top_Flag'], featDict['labelled_bot_Flag']
    #     top_impact = tf == 'impact'
    #     bot_impact = bf == 'impact'
        
    #     ax.set_ylim([0, 1])
        
    #     if top_impact:
    #         ax.axvline(frame, c = 'firebrick')
    #     elif bot_impact:
    #         ax.axvline(frame, c = 'goldenrod')
        
    #     ax.spines[['top', 'right']].set_visible(False)    
    
    # fig.savefig(os.path.join(grp_save_where, f'inference_{idx}'))
        
    # plt.close()

