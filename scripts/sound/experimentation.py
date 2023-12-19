# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:09:37 2023

@author: gianl
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


from ATMANpy.funcs.util_funcs import get_folder_struct, get_video, maybe_make_folder
from ATMANpy.types.Features import BoolFeature, StreamFeature, StreamFeatures
from ATMANpy.types.Streams import Stream

import sounddevice as sd

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    
def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    
# def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
#     if ax is None:
#         _, ax = plt.subplots(1, 1)
#     if title is not None:
#         ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

def play_sound(waveform, fs, ls, samples_per_frame):
    if len(waveform.shape) == 2:
        play_channel = waveform[0, fs * samples_per_frame: (ls + 1) * samples_per_frame]  # last frame included!
    else:
        play_channel = waveform[fs * samples_per_frame: (ls + 1) * samples_per_frame]
    sd.play(play_channel)

def fs_ls(myBF):
    return myBF.first()[0], myBF.last()[0]

vid = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
sto = 'cnn_store'
videoDir = get_folder_struct('video_files')[vid]
videoPath = os.path.join(videoDir, 'match.mp4')
fps = 25
cap = get_video(vid)


metadata = torchaudio.info(videoPath)
sample_rate = metadata.sample_rate
print(metadata)
samples_per_frame = sample_rate // fps

waveform, sample_rate = torchaudio.load(videoPath)
channels, samples = waveform.shape

segBF = BoolFeature.from_store(vid, sto, featname = 'isRally')
segBFs = segBF.filter(segBF).split()

myBF = segBFs[0]
fs, ls = fs_ls(myBF)
frames = myBF.length()

fs_sample, ls_sample = fs * samples_per_frame, (ls + 1) * samples_per_frame # last frame included!
crop_waveform = waveform[:, fs_sample:ls_sample]

plot_waveform(crop_waveform, sample_rate)
plot_specgram(crop_waveform, sample_rate)

left_channel, right_channel = crop_waveform.numpy()

sd.play(right_channel)

### Check the audio features at and around shots - can they be identified simply?
sto_label = 'segmentation_and_labelling_impactonly'
labSF = StreamFeatures.from_store(vid, sto_label, featnameList = ['labelled_top_Flag', 'labelled_bot_Flag'])
jntSFs = segBF.merge(labSF).filter(segBF).split()

MAKE_GRAPHS = True
if MAKE_GRAPHS:
    for PLOT_DB in [False]: ### for power in decibel: axis limits below zero, and reference db has to be by mel_bin
        unit_x = 'mel_bin'
        unit_y = 'db' if PLOT_DB else 'power'
        main_color = 'whitesmoke'
        
        graph_folder = get_folder_struct('graphs')['audio_analysis']
        save_folder = maybe_make_folder(graph_folder, unit_y)
        
        wins_per_frame = 4 ### each video frame ought to have this many spectogram windows
        win_length = samples_per_frame // wins_per_frame
        hop_length = win_length
        n_fft = 800
        
        for idx, mySF in enumerate(jntSFs[94:]):
            idx += 94
            print(f'Play Sequence {idx}...')
            folder_name = f'seq_{idx}'
            save_where = maybe_make_folder(save_folder, folder_name)
            
            ### slice out waveform
            fs, ls = mySF.first()[0], mySF.last()[0]
            fs_sample, ls_sample = fs * samples_per_frame, (ls + 1) * samples_per_frame # last frame included!
            crop_waveform = waveform[:, fs_sample:ls_sample]
            
            ### make spectrogram    
            transform = T.MelSpectrogram(sample_rate, 
                                         win_length = win_length,
                                         hop_length = hop_length,
                                         n_fft = n_fft)
            
            mel_specgram = transform(crop_waveform)
            max_power = mel_specgram.max().item()
            med_power = mel_specgram.median().item()
            
            print(f'   ... Median Power: {med_power}, Max Power: {max_power}')
            
            db_specgram = librosa.power_to_db(mel_specgram, ref = med_power)
            max_db = np.max(db_specgram)
            med_db = np.median(db_specgram)
            
            print(f'   ... Median DB: {med_db}, Max DB: {max_db}')
            
            plot_specgram = db_specgram if PLOT_DB else mel_specgram
            max_ax = max_db if PLOT_DB else max_power
            n_c, n_freqs, n_t = plot_specgram.shape
            ax_lim = (max_ax // 10 + 1) * 10
        
            for frame in mySF:
                first_sample = (frame - fs) * wins_per_frame
            
                fig, axes = plt.subplots(2, 2, figsize = (16, 10))
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
                
                fig.suptitle(title, color = main_color)
                fig.set_facecolor(color) ### change color to match impact type
                
                for i in range(wins_per_frame):
                    curr_sample = first_sample + i
                    x, y = i // 2, i % 2
                    ax = axes[x, y]
                    
                    ax.set_ylim([0, ax_lim])
                    ax.spines[['top', 'right']].set_visible(False)
                    ax.set_title(f'Sample: {curr_sample}', size = 10)
                    ax.set_facecolor(color) ### change color to match impact type
                    
                    if x == 1:
                        ax.set_xlabel(unit_x)
                    if y == 0:
                        ax.set_ylabel(unit_y)
                        
                    ax.title.set_color(main_color)
                    ax.spines[['bottom', 'left']].set_color(main_color)
                    ax.tick_params(axis='x', colors = main_color)
                    ax.tick_params(axis='y', colors = main_color)
                    ax.yaxis.label.set_color(main_color)
                    ax.xaxis.label.set_color(main_color)
                    
                    tArr = plot_specgram[0, :, curr_sample]
                    ax.plot(tArr, color = main_color)
                
                save_name = f'melspec_{frame}.png'
                fig.tight_layout()
                fig.savefig(os.path.join(save_where, save_name))
                plt.close()

# #### Recheck mel spectrograms

            
# fs, ls = fs_ls(jntSFs[1])
# play_sound(waveform, fs, ls, samples_per_frame)

# wins_per_frame = 4 ### each video frame ought to have this many spectogram windows
# win_length = samples_per_frame // wins_per_frame
# hop_length = win_length
# n_fft = 800

# mySF = jntSFs[0]
# ### slice out waveform
# fs, ls = mySF.first()[0], mySF.last()[0]
# fs_sample, ls_sample = fs * samples_per_frame, (ls + 1) * samples_per_frame # last frame included!
# crop_waveform = waveform[:, fs_sample:ls_sample]

# plot_waveform(crop_waveform, sample_rate)

# transform = T.MelSpectrogram(sample_rate, 
#                              win_length = win_length,
#                              hop_length = hop_length,
#                              n_fft = n_fft)

# mel_specgram = transform(crop_waveform)
# db_specgram = librosa.power_to_db(mel_specgram)



# mySF = jntSFs[1]
# ### slice out waveform
# fs, ls = mySF.first()[0], mySF.last()[0]
# fs_sample, ls_sample = fs * samples_per_frame, (ls + 1) * samples_per_frame # last frame included!
# crop_waveform = waveform[:, fs_sample:ls_sample]

# plot_waveform(crop_waveform, sample_rate)


# RUN_MODEL = False
# if RUN_MODEL:
#     class GreedyCTCDecoder(torch.nn.Module):
#         def __init__(self, labels, blank=0):
#             super().__init__()
#             self.labels = labels
#             self.blank = blank
    
#         def forward(self, emission: torch.Tensor) -> str:
#             """Given a sequence emission over labels, get the best path string
#             Args:
#               emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
    
#             Returns:
#               str: The resulting transcript
#             """
#             indices = torch.argmax(emission, dim=-1)  # [num_seq,]
#             indices = torch.unique_consecutive(indices, dim=-1)
#             indices = [i for i in indices if i != self.blank]
#             return "".join([self.labels[i] for i in indices])
        
#     torch.random.manual_seed(0)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_waveform = crop_waveform.to(device)
    
#     bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    
#     model = bundle.get_model().to(device)
#     labels = bundle.get_labels()
#     print(model.__class__)
#     print(labels)
    
#     with torch.inference_mode():
#         emission, _ = model(input_waveform)
    
#     decoder = GreedyCTCDecoder(labels=labels)
#     transcript = decoder(emission[0])
        
    