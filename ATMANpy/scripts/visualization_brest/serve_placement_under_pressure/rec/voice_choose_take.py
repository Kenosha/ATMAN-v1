
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import subprocess

from ATMANpy.funcs.util_funcs import get_folder_struct, make_timestamp, maybe_make_folder
from scipy.io.wavfile import write

from ATMANpy.types.Features import ConstFeature, StreamFeature
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import Stream, DecapStream

import librosa
import pandas as pd
from matplotlib import pyplot as plt

SAMPLE_RATE = 44100
PROJECT_NAME = 'serve_pressure'
VIDEOS = ['intro_20240729_025502', 'deuce_side_static_20240730_004031', 
          'transition_deuce_to_ad_20240730_012459', 'ad_side_static_20240728_214055']

IDX = 1
VIDEO = VIDEOS[IDX]

### Paths
result_path = os.path.join(get_folder_struct('results_video')[PROJECT_NAME], VIDEO)
audio_path = os.path.join(result_path, 'audio')
best_path = maybe_make_folder(audio_path, 'best')

folders = os.listdir(audio_path)
take_folders = [f for f in folders if f.startswith('take_')]
take_nums = [int(f.replace('take_', '')) for f in take_folders]
num_takes = len(take_folders)

### Get a list of voice segments from take 1 -> assumed to be complete 
take1 = take_folders[0]
take1_files = os.listdir(os.path.join(audio_path, take1))
take1_recs = [f for f in take1_files if f.startswith('seg')]

sort_func = lambda fileStr: int(fileStr.replace('seg_', '').replace('.mp3', ''))
take1_recs = sorted(take1_recs, key = sort_func)

best_waves = {}
best_graphs = {}
while set(take1_recs) != set(best_waves.keys()): 
    for rec in take1_recs:
        ### read in takes
        takeDict = {}
        for take in take_folders:
            take_file = os.path.join(audio_path, take, rec)
            take_waveform, _ = sf.read(take_file)
            takeDict[take] = take_waveform

        ### play back waveform
        for take, take_waveform in takeDict.items():
            print(take)
            sd.play(take_waveform)
            sd.wait()

        ### choose waveform
        choice = input(f'Which take is the best? press any of {take_nums} for selection, anything else for repeat.')
        choice = int(choice)
        if choice in take_nums:
            enc = rec.replace('seg', 'env').replace('.mp3', '.png')
            graph_file = os.path.join(audio_path, take, enc)
            
            best_waves[rec] = takeDict[f'take_{choice}']
            best_graphs[enc] = plt.imread(graph_file)

            print(f'Chose take {choice} for {rec}')

### SAVE FILES IN BEST FOLDER ###
for rec in best_waves:
    enc = rec.replace('seg', 'env').replace('.mp3', '.png')
    wave, graph = best_waves[rec], best_graphs[enc]

    sf.write(os.path.join(best_path, rec), wave, samplerate = SAMPLE_RATE)
    plt.imsave(os.path.join(best_path, enc), graph)


