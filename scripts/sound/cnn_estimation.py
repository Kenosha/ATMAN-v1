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
from datetime import datetime
import dill as pickle

import sounddevice as sd
import librosa
import librosa.display

import torch
import torch.nn as nn
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary

import ATMANpy.funcs.util_funcs as ut
from ATMANpy.types.Features import BoolFeature, StreamFeature, StreamFeatures
from ATMANpy.types.Streams import Stream
   
from ATMANpy.models.SoundCNN.utils import train_one_epoch, run_inference
from ATMANpy.models.SoundCNN.utils import plot_waveform, plot_spectrogram, make_confusion_matrix
from ATMANpy.models.SoundCNN.utils import wave_to_spec, spec_to_wave
from ATMANpy.models.SoundCNN.data import LabelledSpectroData, SAMPLE_RATE
from ATMANpy.models.SoundCNN.nets import SoundModel_FC

# import warnings
# from cryptography.utils import CryptographyDeprecationWarning
# warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

#### PARAMETERS ########
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

LOG_PATH = ut.get_folder_struct('cnn_sound')['runs']
RES_PATH = ut.get_folder_struct('cnn_sound')['weights']
PRM_PATH = ut.get_folder_struct('cnn_sound')['params']

MODEL_RUN_IDENTIFIER = 'test_model'

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0004
REPORT_FREQUENCY = 500
TO_DB = True
NORM_VALUE = 50

DEVICE = 'cuda'
PIN_MEMORY = DEVICE == 'cuda'
NUM_WORKERS = 4
CONV_FILTERS = [32, 32, 64, 64, 128, 128]
CONV_KERNELS = [3, 3, 5, 5, 7, 7]
POOL_KERNELS = [1, 2, 1, 2, 1, 1]
FC_NEURONS = 16

### TODO: Avoid too many max pools, instead increase filter size.
### TODO: Check what happens with more initial filters -> 16 seems a bit small.

DO_CHECK_SAMPLES = False

if __name__ == '__main__':
    ### GET THE DATA ##########
    # these parameters are already determined in the cnn_hit_spectrograms script, 
    # which creates the data. we just specify them to fetch data from the right folder.
    data_creation_params = {'hop_length': 512,  
                            'window_length': 1024,
                            'wing_size': 40, 
                            'semihits_per_hit': 4, 
                            'nonhits_per_hit': 50,
                            'mel_bands': 128}
    data_transformation_params = {'to_db': TO_DB,
                                  'norm_value': NORM_VALUE}
    train_data = LabelledSpectroData(name = 'train', **data_creation_params, **data_transformation_params, db_reference = 'mean')
    test_data = LabelledSpectroData(name = 'test', **data_creation_params, **data_transformation_params, db_reference = train_data.db_reference)
    
    data_name = train_data.data_name.lstrip('train_')
    freqs, hops = train_data.freqs, train_data.hops
    
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True, 
                              pin_memory = PIN_MEMORY, num_workers = NUM_WORKERS)
    
    test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False,
                             pin_memory = PIN_MEMORY, num_workers = NUM_WORKERS)
    
    ### SET UP RESULT PATHS #######
    LOG_NAME = MODEL_RUN_IDENTIFIER + '_' + timestamp
    RES_NAME = MODEL_RUN_IDENTIFIER + '_' + timestamp
    PRM_NAME = MODEL_RUN_IDENTIFIER + '_' + timestamp + '.pkl'
    
    log_save_where = os.path.join(LOG_PATH, LOG_NAME)
    res_save_where = os.path.join(RES_PATH, RES_NAME)
    prm_save_where = os.path.join(PRM_PATH, PRM_NAME)
    
    ### SAVE RELEVANT PARAMETERS IN A DICTIONARY ######
    params = {'norm_value': NORM_VALUE,
              'to_db': TO_DB,
              'conv_filters': CONV_FILTERS,
              'conv_kernels': CONV_KERNELS,
              'pool_kernels': POOL_KERNELS,
              'fc_neurons': FC_NEURONS,
              'num_workers': NUM_WORKERS,
              'batch_size': BATCH_SIZE,
              'device': DEVICE,
              'epochs': EPOCHS,
              'learning_rate': LEARNING_RATE,
              'freqs': freqs,
              'hops': hops,
              'db_reference': train_data.db_reference,
              **data_creation_params
              }
    with open(prm_save_where, 'wb') as f:
        pickle.dump(params, f)
        print('Parameters saved to file.')
    
    ### LISTEN TO A FEW SAMPLES #########
    if DO_CHECK_SAMPLES:
        idxs = [0, 6, 100]
        
        fig_wave, axes_wave = plt.subplots(len(idxs), 1, sharex = True)
        fig_spec, axes_spec = plt.subplots(len(idxs), 1, sharex = True)
        for i, idx in enumerate(idxs):
            orig_spec, lab = train_data[idx]
            orig_spec = orig_spec[0]
            ### denormalize
            if NORM_VALUE is not None:
                orig_spec = orig_spec * NORM_VALUE
            
            ### convert
            spec = orig_spec
            if TO_DB:
                spec = librosa.db_to_power(spec, ref=train_data.db_reference)
            if data_creation_params['mel_bands'] is not None:
                spec = librosa.feature.inverse.mel_to_stft(spec, sr = SAMPLE_RATE, 
                                                           n_fft = data_creation_params['window_length'])
                
            wave = spec_to_wave(spec, 
                                hop_length = data_creation_params['hop_length'], 
                                window_length = data_creation_params['window_length'])
            sd.play(wave, samplerate = SAMPLE_RATE)
            
            plot_waveform(axes_wave[i], wave, SAMPLE_RATE)
            plot_spectrogram(axes_spec[i], orig_spec, SAMPLE_RATE, 
                             y_axis = 'mel' if train_data.is_mel else 'linear', 
                             hop_length = data_creation_params['hop_length'])
    
            print(f'label of index {idx}: {lab}')
            time.sleep(train_data.secs_in_scope + 1)
    
    
    ### PREPARE TENSORBOARD ##########
    writer = SummaryWriter(log_save_where)
    
    tb = program.TensorBoard()
    tb.configure(argv = [None, '--logdir', log_save_where])
    url = tb.launch()
    print(f'TensorBoard launched on {url}')
    
    writer.flush()
    
    ### SET UP MODEL #########
    model = SoundModel_FC(CONV_FILTERS, CONV_KERNELS, POOL_KERNELS, FC_NEURONS,
                          frequencies = freqs, hops = hops, 
                          share_shots = train_data.labArr.mean())
    model = model.to(DEVICE)
    model_summary = summary(model, (1, freqs, hops))
    
    lossFunc = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    
    # ##########
    
    # myArr, labArr = next(iter(train_loader))
    # myArr = myArr.to(DEVICE)
    # outputs = model(myArr)
    
    ### TRAINING LOOP ############
    
    best_vloss = 10 ** 6
    for epoch in range(1, EPOCHS + 1):
        print(f'EPOCH {epoch}')
        
        model.train(True)
        avg_loss = train_one_epoch(epoch, writer, 
                                   model, train_loader, lossFunc, optimizer, DEVICE,
                                   report_frequency = REPORT_FREQUENCY)
        model.train(False)
        
        avg_vloss, vprobs, vlabels = run_inference(model, test_loader, lossFunc, DEVICE)
            
        confplot = make_confusion_matrix(vprobs, vlabels)
    
        confPath = os.path.join(log_save_where, 'confmats')
        try:
            os.mkdir(confPath)
        except FileExistsError:
            pass
        finally:
            confplot.savefig(os.path.join(confPath, f'confmat_{epoch}'))
        
        writer.add_figure('Confusion Matrix', confplot, epoch)
        writer.flush()
        
        writer.add_scalars('Training vs. Validation Loss',
                            {'Training': avg_loss, 'Validation': avg_vloss},
                            epoch)
        writer.flush()
        
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), res_save_where)
