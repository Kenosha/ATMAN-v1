
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

from shutil import copyfile

SAMPLE_RATE = 44100
PROJECT_NAME = 'return_chip_topspin'
VIDEOS = ['intro_20241010_175050', 'descriptive_20241011_173710']

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


seg_files = [f for f in os.listdir(result_path) if f.startswith('seg_')]
seg_nums = sorted([int(f.replace('.mp4', '').split('_')[1]) for f in seg_files])


def get_take_tups(take):
    take_files = os.listdir(os.path.join(audio_path, take))
    take_recs = [f for f in take_files if f.startswith('seg')]
    take_names = [f.split('.')[0] for f in take_recs]
    
    take_qual = [f.split('_')[-1][-1] for f in take_names]
    take_num = [f.split('_')[1] for f in take_names]

    take_tups = [(int(n), int(q)) for n, q in zip(take_num, take_qual)]
    return take_tups

### Get all segments from all takes, with their respective quality
take_dicts = [{num: qual for num, qual in get_take_tups(take)} for take in take_folders]

current_best_idx = {k: None for k in seg_nums}
current_best_qual = {k: 4 for k in seg_nums}

for take_idx, take_qual in enumerate(take_dicts):
    for k in take_qual:
        if take_qual[k] < current_best_qual[k]: ### if the quality in the take we are looping over is better (= lower) for the segment in question...
            current_best_idx[k] = take_idx
            current_best_qual[k] = take_qual[k]

### copy over the best video into a best_take folder
for k in current_best_idx:
    best_qual = current_best_qual[k]
    best_idx = current_best_idx[k]

    if best_idx is None:
        continue

    take_folder = take_folders[best_idx]

    seg_file_from = os.path.join(audio_path, take_folder, f'seg_{k}_q{best_qual}.mp3')
    env_file_from = os.path.join(audio_path, take_folder, f'env_{k}_q{best_qual}.png')

    seg_file_to = os.path.join(best_path, f'seg_{k}.mp3')
    env_file_to = os.path.join(best_path, f'env_{k}.png')

    copyfile(seg_file_from, seg_file_to)
    copyfile(env_file_from, env_file_to)

