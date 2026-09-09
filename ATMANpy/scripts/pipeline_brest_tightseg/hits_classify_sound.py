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

import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.feature_funcs as ff
from ATMANpy.types.Features import BoolFeature, StreamFeature, StreamFeatures
from ATMANpy.types.Streams import Stream
   
from ATMANpy.models.SoundCNN.nets import SoundModel_MN
from ATMANpy.models.SoundCNN.data import PointSpectroData
from ATMANpy.models.SoundCNN.utils import first_last_sample_index_in_frame, first_last_sample_index_in_window
from ATMANpy.models.SoundCNN.utils import first_window_with_sample, last_window_with_sample
from ATMANpy.models.SoundCNN.utils import calculate_proximity, plot_waveform, plot_spectrogram
from ATMANpy.models.SoundCNN.utils import run_inference_on_arbitrary_spectrum, load_cnn

def load(vid, sto):
    segBF = BoolFeature.from_store(vid, sto, featname = 'isCourt')
    return segBF

def save(outFrm, vid, sto):
    ff.save_feature_frame(outFrm, 'sound_hitTable', vid, sto)

def main(conf, model_params, name = None):    
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    ### CNN parameters
    device = conf.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(conf.get('torch_threads', 1))
    loader_workers = 0 
    report_frequency = 100

    model_name = model_params['model_name']
    sample_rate = model_params['sample_rate']
    fps = model_params['fps']
    
    samples_per_frame = sample_rate / fps
    central_sample_in_frame_offset = (samples_per_frame - 1) / 2
    
    ### Load 
    segBF = load(vid, sto)
    segBFs = segBF.filter(segBF).split()
    if segBFs.length() == 0:
        raise ValueError('No isCourt segments are available for sound classification.')
    
    model, params = load_cnn(model_name, SoundModel_MN, device = device)
    
    stride = np.prod(params['pool_kernels'])
    scope = params['wing_size'] * 2 + 1
    scope_samples = scope * params['hop_length']
    hop_stride = stride * params['hop_length']
    
    ds = PointSpectroData(vid, sto, segmenter = 'isCourt', mel_bands = params['mel_bands'], 
                          hop_length = params['hop_length'], window_length = params['window_length'],
                          to_db = params['to_db'], norm_value = params['norm_value'], db_reference = params['db_reference'],
                          yield_frame_indices = False)
    
    inf_loader = DataLoader(ds, batch_size = 1, num_workers = 0,
                            pin_memory = torch.device(device).type == 'cuda')
    
    ### Run Inference of Spectra
    specArrs, sample_bounds = run_inference_on_arbitrary_spectrum(model, inf_loader, device, 
                                                                  as_logits = True,
                                                                  report_frequency = report_frequency)
    
    specFrms = []
    for idx, specArr in enumerate(specArrs):
        fs_sample, ls_sample = sample_bounds[idx]
        mySF = segBFs[idx]
        frames = mySF.get_frames()    
        num_labels, num_hops = specArr.shape
    
        left_sample_indices = np.arange(fs_sample, ls_sample, hop_stride)
        right_sample_indices = left_sample_indices + scope_samples - 1
        
        left_right_sample_indices = np.stack([left_sample_indices, right_sample_indices])
        central_sample_indices = np.mean(left_right_sample_indices, axis = 0)
        
        central_sample_indices = central_sample_indices[:num_hops]
        
        frame_div, frame_mod = central_sample_indices // samples_per_frame, central_sample_indices % samples_per_frame
        frame_indices = frame_div + frame_mod / samples_per_frame
        
        ### interpolate values to normal frames
        specFrm = pd.DataFrame(specArr.T, index = frame_indices, columns = ['hit', 'bounce'])
        new_index = sorted(frames + frame_indices.tolist())
        specFrm_reidx = specFrm.reindex(new_index)
        specFrm_itpl = specFrm_reidx.interpolate(method = 'index', limit_area='inside')
        specFrm_itpl = specFrm_itpl.loc[frames].dropna()
        
        specFrms.append(specFrm_itpl)
        
    outFrm = pd.concat(specFrms)
    
    save(outFrm, vid, sto)
        
    
if __name__ == '__main__':
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
    
    conf = cnf_mod.conf
    model_params = prm_mod.hits_classify_sound.model_params
    
    main(conf, model_params, name = ppl._name)    
    
