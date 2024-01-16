# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:24:58 2023

@author: gianl
"""

import os
import torch
import torch.nn as nn
import numpy as np
import dill as pickle

from ATMANpy.funcs.util_funcs import get_folder_struct

from itertools import product
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

SAMPLE_RATE = 44100
FPS = 25

def first_last_sample_index_in_frame(frame_number, first_frame_number, samples_per_frame = SAMPLE_RATE // FPS):
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

def calculate_proximity(units, hit_units, maximum_allowable_distance):
    if len(hit_units) == 0:
        return np.zeros(units.shape) 
    else:
        unitArr, hitArr = np.array(units)[:, np.newaxis], np.array(hit_units)[np.newaxis, :]
        distance_to_nearest_hit_window = np.min(np.absolute(unitArr - hitArr), axis = 1)
        
        overlap = np.maximum(maximum_allowable_distance - distance_to_nearest_hit_window, 0)
        overlap_share = overlap / maximum_allowable_distance
        
        return overlap_share ** 2

def join_hit_and_bounce_prox(hit_prox, bounce_prox):
    ### normalize the proximities so that they can not exceed 1, in sum!
    ### they can be under 1, since then we probably have neither a hit nor a bounce.
    if hit_prox + bounce_prox > 1:
        hit_num = hit_prox * (1 - bounce_prox)
        bounce_num = bounce_prox * (1 - hit_prox)
        
        den = hit_num + bounce_num
        
        hit_prob, bounce_prob = hit_num / den, bounce_num / den
    else:
        hit_prob, bounce_prob = hit_prox, bounce_prox

    return hit_prob, bounce_prob

def plot_waveform(ax, waveform, sample_rate, first_sample_index = 0):
    x_axis = np.arange(0, len(waveform)) + first_sample_index
    ax.plot(x_axis, waveform, linewidth = 1, alpha = 0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True)
    return ax

def plot_spectrogram(ax, Y, sample_rate, 
                     hop_length = None, window_length = None, 
                     y_axis = 'linear'):
    img = librosa.display.specshow(Y, sr = sample_rate, hop_length = hop_length,
                                   n_fft = window_length,
                                   x_axis = 'time', y_axis = y_axis, 
                                   ax = ax)
    plt.colorbar(img, format = '%+2.f', ax = ax)

def wave_to_spec(Y, hop_length = 512, window_length = 1024):
    fts = librosa.stft(Y, n_fft = window_length, hop_length = hop_length)
    return np.abs(fts) ** 2

def spec_to_wave(S, hop_length = 512, window_length = 1024):
    mag = S ** 0.5
    return librosa.griffinlim(mag, hop_length = hop_length, n_fft = window_length)

### CONFUSION MATRIX FUNCTIONS #########
def segment_probabilities(probs, cutoffs): 
    ### for confusion matrix probabilities
    probs, cutoffs = np.array(probs), np.array(cutoffs)
    comparisons = probs[:, np.newaxis] > cutoffs[np.newaxis, :]
    segments = comparisons.sum(axis = 1)
    return segments

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def make_confusion_matrix(vprobs, vlabels, name = 'shot'):
    CONV_MAT_CUTOFFS = [0.001, 0.4, 0.9]
    CONV_MAT_NAMES = [f'not a {name}', f'barely a {name}', f'off-center {name}', name]
    
    predicted_segment = segment_probabilities(vprobs, CONV_MAT_CUTOFFS)
    true_segment = segment_probabilities(vlabels, CONV_MAT_CUTOFFS)
    
    conf_mat = confusion_matrix(true_segment, predicted_segment)
    plot = plot_confusion_matrix(conf_mat, CONV_MAT_NAMES)
    return plot


### TRAIN LOOP FUNCTIONS ###########
def train_one_epoch(epoch_index, tb_writer,
                    model, data_loader, loss_function, optimizer, device,
                    report_frequency = 100):
    running_loss = 0
    previous_running_loss = 0
    
    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad(set_to_none = True)
        outputs = model(inputs)
        outputs = outputs.squeeze(dim = (1, 2, 3)) ### squeeze out C, H, W
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
  
        running_loss += loss.item()
        
        if i % report_frequency == report_frequency - 1:
            last_loss = (running_loss - previous_running_loss) / report_frequency
            
            print(f'   batch {i + 1} --> loss: {last_loss}')
            tb_x = (epoch_index - 1) * len(data_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_writer.flush()
            
            previous_running_loss = running_loss
            
    return running_loss / i

def run_inference(model, data_loader, loss_function, device, as_logits = False):
    with torch.inference_mode():
        running_vloss = 0
        vlabel_list = []
        vprob_list = []
        for i, vdata in enumerate(data_loader):
            vinputs, vlabels = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            
            voutputs = model(vinputs)
            voutputs = voutputs.squeeze(dim = (1, 2, 3)) ### squeeze out only T x H x W (which, with the fully convolutional models, remain)
            
            vloss = loss_function(voutputs, vlabels)
            running_vloss += vloss
            
            if as_logits:
                vprobs = nn.Sigmoid()(voutputs)
            else:
                vprobs = voutputs
            
            vprob_list.append(vprobs)
            vlabel_list.append(vlabels)
    
    avg_vloss = running_vloss / (i + 1)
    vprobs = torch.cat(vprob_list).to('cpu').numpy()
    vlabels = torch.cat(vlabel_list).to('cpu').numpy()
    
    return avg_vloss, vprobs, vlabels

def run_inference_on_arbitrary_spectrum(model, data_loader, device, as_logits = False, report_frequency = 10):
    ### this is supposed to run on the PointSpectroData -> has no labels (at the moment)
    outArrs = []
    sample_bounds = []
    with torch.inference_mode():
        for idx, batchObj in enumerate(data_loader):
            if idx % report_frequency == 0: print(f'Inferring batch {idx}')
            
            specArr, fsArr, lsArr = batchObj
            specArr = specArr.to(device)
            
            outArr = model(specArr).to('cpu').squeeze()
            
            if as_logits:
                outArr = nn.Sigmoid()(outArr)
                
            outArr = outArr.numpy()

            outArrs.append(outArr)
            
            fs = fsArr.squeeze().item() ### this works because batch size is 1
            ls = lsArr.squeeze().item()
            sample_bounds.append((fs, ls))
            
    return outArrs, sample_bounds ### we do not concatenate since the array have different sizes, for outArrs
            
def load_cnn(model_name, model_type, device = 'cuda'):
    pathDict = get_folder_struct('cnn_sound')
    resPath, prmPath = pathDict['weights'], pathDict['params']
    
    res_read_where = os.path.join(resPath, model_name)
    prm_read_where = os.path.join(prmPath, model_name + '.pkl')
    
    ### Read Parameters 
    with open(prm_read_where, 'rb') as fp:
        params = pickle.load(fp)
        print('Read parameters: ')
        print(params)

    ### Read CNN
    model = model_type(params['conv_filters'], params['conv_kernels'], params['pool_kernels'], params['fc_neurons'],
                       frequencies = params['freqs'], hops = params['hops'], initial_output_shares = [0.1, 0.1])
    
    model.to(device)
    model.load_state_dict(torch.load(res_read_where))
    model.train(False)

    return model, params







