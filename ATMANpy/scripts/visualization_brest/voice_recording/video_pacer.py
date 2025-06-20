
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

from scipy import signal
import librosa
import pandas as pd
from matplotlib import pyplot as plt
import noisereduce as nr

SAMPLE_RATE = 44100
FRAME_RATE = 25

PROJECT_NAME = 'atman_capabilities'
VIDEOS = ['short_intro_20250620_024215', 'short_raster_20250607_022044']

PROJECT_NAME = 'atman_capabilities'
VIDEOS = ['lines_20250609_173938', 'players_20250606_013734', 'balls_20250606_014742', 
          'sound_20250609_132907', 'vision_20250606_021653', 'trajectory_20250607_183929', 
          'filtering_20250607_232416', 'visual_engine_20250616_233029']

PROJECT_NAME = 'return_chip_topspin'
VIDEOS = ['intro_20250618_143225', 'descriptive_20250618_144448', 'inference_20250618_223539']

PROJECT_NAME = 'return_chip_topspin'
VIDEOS = ['short_20250615_000557']

PROJECT_NAME = 'serve_pressure_new'
VIDEOS = ['first_part_20250617_151722', 'ad_side_example_20250616_135617', 'ad_side_cases_20250616_144314',
          'transition_ad_to_deuce_20250616_145410', 'deuce_side_cases_20250616_152224']

PROJECT_NAME = 'serve_pressure_new'
VIDEOS = ['short_20250617_141905']

IDX = 0
VIDEO = VIDEOS[IDX]

### Paths
result_path = os.path.join(get_folder_struct('results_video')[PROJECT_NAME], VIDEO)
audio_path = os.path.join(result_path, 'audio', 'best')
fill_path = maybe_make_folder(result_path, 'fill_segments')
segments_joint_path = maybe_make_folder(result_path, 'segments_joint')

folders = os.listdir(audio_path)
num_takes = len([f for f in folders if f.startswith('take_')])

### Read in Video Segments
seg_files = [f for f in os.listdir(result_path) if f.startswith('seg_')]
num_segs = len(seg_files)

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

### print script lines, for reference
for idx, lineStr in enumerate(script_lines_split):
    print(idx)
    # print('   ')
    print(lineStr[:100] + '...')
    # print('   ')
    # print('   ')
    print('________')

### supply the segment you want something inserted AFTER, and the length in frames
megaDict = {'short_20250617_141905': {20: 25},
            'deuce_side_cases_20250616_152224': {2: 50},
            'filtering_20250607_232416': {7: 30},
            'intro_20250618_143225': {25: 50,
                                      28: 50,
                                      32: 25,
                                      45: 15,
                                      48: 20,
                                      49: 25,
                                      50: 50},
            'descriptive_20250618_144448': {4: 25,
                                            6: 25,
                                            8: 50,
                                            16: 40,
                                            22: 25,
                                            25: 25,
                                            29: 25,
                                            36: 50},
            'inference_20250618_223539': {5: 25,
                                          6: 35,
                                          10: 15,
                                          11: 35,
                                          14: 25,
                                          16: 20,
                                          17: 35,
                                          20: 25,
                                          24: 25,
                                          33: 50,
                                          41: 25,
                                          44: 15,
                                          46: 20,
                                          51: 50}}

try:
    paceDict = megaDict[VIDEO] 
except KeyError:
    pass
else:
    for idx, frames in paceDict.items():
        samples = frames * (SAMPLE_RATE // 25)

        ### repeat frame for duration in frames ...
        statFrm = ordered_dStms[idx].frames[-1]
        statFrms = BGRFrames.from_frame(statFrm, frames)

        ### with silence over it ...
        null_wave = np.zeros((samples))
        
        ### and save video and audio
        video_file = os.path.join(fill_path, f'video_{idx}.avi')
        statFrms.save(fill_path, f'video_{idx}')

        audio_file = os.path.join(fill_path, f'audio_{idx}.mp3')
        sf.write(audio_file, null_wave, samplerate = SAMPLE_RATE)

        ### combine
        joint_file = os.path.join(segments_joint_path, f'joint_{idx}_fill.mp4')
        subprocess.run(f'ffmpeg -y -i "{video_file}" -i "{audio_file}" -c copy "{joint_file}" -r {FRAME_RATE} -ar {SAMPLE_RATE}')


