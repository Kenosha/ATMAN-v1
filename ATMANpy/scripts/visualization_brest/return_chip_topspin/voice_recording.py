
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
PROJECT_NAME = 'return_chip_topspin'
VIDEOS = ['intro_20241010_175050', 'descriptive_20241011_173710']

from ATMANpy.funcs.rec_funcs import countdown, amplitude_envelope, time_needed

IDX = 1
VIDEO = VIDEOS[IDX]

### Paths
result_path = os.path.join(get_folder_struct('results_video')[PROJECT_NAME], VIDEO)
graph_path = maybe_make_folder(result_path, 'graphs')
audio_path = maybe_make_folder(result_path, 'audio')

folders = os.listdir(audio_path)
take_folders = [f for f in folders if f.startswith('take_')]
num_takes = len(take_folders)

take_path = maybe_make_folder(audio_path, f'take_{num_takes + 1}')

seg_files = [f for f in os.listdir(result_path) if f.startswith('seg_')]
seg_nums = sorted([int(f.replace('.mp4', '').split('_')[1]) for f in seg_files])

### Check for which segment we already have some quality 1 elements

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

### Read in Video Segments
ordered_dStms = []
for idx in range(len(seg_files)):
    seg = f'seg_{idx}.mp4'
    full_path = os.path.join(result_path, seg)
    ordered_dStms.append(Stream.from_file(full_path).decap())

### Read in Script
script_lines_split = []
for idx in range(len(seg_files)):
    script_path = os.path.join(result_path, f'script_{idx}.txt')
    with open(script_path) as f:
        script_lines = f.readlines()
        script_lines = [line.replace('\n', '').strip(' ') for line in script_lines]
        script_lines = [line for line in script_lines if line != '']
        
        script_line = '\n'.join(script_lines)
        script_lines_split.append(script_line)

assert len(ordered_dStms) == len(script_lines_split)
num_stms = len(ordered_dStms)

#### Where to print script lines
from ATMANpy.types.Lines import Region
from ATMANpy.types.Points import Point2D
textPt = Point2D((50, 50))
script_region = Region.from_points(textPt, textPt.down(50).right(100))

### Make Long Script Features
myDict = {}
counter = 0
for subStm, script_line in zip(ordered_dStms, script_lines_split):
    addDict = {frame: script_line for frame in range(counter, counter + subStm.length() + 1)}
    myDict.update(addDict)
    counter += subStm.length() + 1

textSF = StreamFeature(myDict, name = 'textFlag')

### Record Audio
import time
recordings = {}
countdown()
for idx in range(num_stms):
    subStm = ordered_dStms[idx]
    script_line = script_lines_split[idx]

    if script_line == '':
        continue

    qual = current_best_qual[idx]

    if qual == 1:
        continue

    time.sleep(1)
    
    showFrm = subStm.get(0).dim(0.25).put(script_line, where = textPt.coords)
    seconds = time_needed(script_line)
    sub_recording = sd.rec(int(seconds * SAMPLE_RATE), samplerate = SAMPLE_RATE, channels = 2)
    
    ### depending on how happy you are with the recording, press 1 for highest quality, 2 for medium, 3 for bad
    keypress = showFrm.show(num_seconds = seconds)
    quality_integer = int(chr(keypress)) 
    
    recordings[idx] = sub_recording, quality_integer

### Save Envelope and Recording
db_ref_in_pascal = 0.00002
db_threshold = 45
hop_size = 1000
env_size = 10000

for idx, waveform_quality_tuple in recordings.items():
    waveform, quality = waveform_quality_tuple

    mono_waveform = waveform.mean(axis = 1)
    envSer = amplitude_envelope(mono_waveform, env_size, hop_size, SAMPLE_RATE)
    envSer_db = pd.Series(librosa.amplitude_to_db(envSer, ref = db_ref_in_pascal), index = envSer.index)
    envSer_db.name = 'envelope'

    thrSer_db = (envSer_db > db_threshold) * db_threshold
    thrSer_db.name = 'threshold'
    pltFrm = pd.concat([envSer_db, thrSer_db], axis = 1)

    fig, ax = plt.subplots()
    ax = pltFrm['envelope'].plot()
    fig.savefig(os.path.join(take_path, f'env_{idx}_q{quality}.png'))
    plt.close()

    sf.write(os.path.join(take_path, f'seg_{idx}_q{quality}.mp3'), mono_waveform, samplerate = SAMPLE_RATE)

