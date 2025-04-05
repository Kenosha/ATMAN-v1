
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
import noisereduce as nr

SAMPLE_RATE = 44100
PROJECT_NAME = 'serve_pressure'
VIDEOS = ['intro_20240729_025502', 'deuce_side_static_20240730_004031', 
          'transition_deuce_to_ad_20240730_165219', 'ad_side_static_20240730_023305']

VIDEO_CROPS = {'intro_20240729_025502': {0:  (0.2, 11),
                                         1:  (0.2, 10),
                                         2:  (0.2, 6),
                                         6:  (0.2, 10),
                                         8:  (0.2, 6),
                                         9:  (0.2, 8),
                                         11: (0.2, 28),
                                         12: (0.2, 11),
                                         14: (0.2, 9),
                                         16: (0.2, 7),
                                         19: (0.5, 9),
                                         21: (0.5, 11),
                                         22: (0.2, 6),
                                         25: (0.2, 12),
                                         26: (0.2, 21),
                                         27: (0.2, 13),
                                         28: (0.2, 9),
                                         29: (0.2, 11),
                                         30: (0.2, 5)},
                'deuce_side_static_20240730_004031': {0:  (0.2, 12),
                                                      1:  (0.2, 20),
                                                      2:  (0.2, 11),
                                                      3:  (0.2, 10),
                                                      4:  (0.2, 12.5),
                                                      5:  (0.2, 11.5),
                                                      6:  (0.2, 16),
                                                      7:  (0.2, 14.5),
                                                      8:  (0.2, 15),
                                                      9:  (0.2, 12),
                                                      10: (0.2, 18),
                                                      11: (0.2, 9),
                                                      12: (0.2, 8),
                                                      13: (0.5, 7)},
                'transition_deuce_to_ad_20240730_165219': {3: (0.2, 21),
                                                           5: (0.2, 6)},
                'ad_side_static_20240730_023305': {0: (0.2, 19),
                                                   1: (0.2, 8),
                                                   2: (0.2, 13.5),
                                                   3: (0.2, 14),
                                                   4: (0.5, 8.5),
                                                   5: (0.2, 17),
                                                   6: (0.2, 11.5),
                                                   7: (0.2, 14)}}

IDX = 2
VIDEO = VIDEOS[IDX]

### Paths
result_path = os.path.join(get_folder_struct('results_video')[PROJECT_NAME], VIDEO)
audio_path = os.path.join(result_path, 'audio', 'best')

folders = os.listdir(audio_path)
num_takes = len([f for f in folders if f.startswith('take_')])

### Read in Video Segments
seg_files = [f for f in os.listdir(result_path) if f.startswith('seg')]
num_segs = len(seg_files)

ordered_dStms = {}
waveforms = {}
for idx in range(num_segs):
    seg = f'seg_{idx}.mp4'
    take = f'seg_{idx}.mp3'
    seg_path = os.path.join(result_path, seg)
    take_path = os.path.join(audio_path, take)

    ordered_dStms[idx] = Stream.from_file(seg_path).decap()
    try:
        waveforms[idx], _ = sf.read(take_path)
    except:
        waveforms[idx] = None ### some segments do not have audio

### Crop Frames and Waveforms
cropDict = VIDEO_CROPS[VIDEO]

cropped_frames = []
cropped_waveforms = []

for idx in range(num_segs):
    try:
        crop = cropDict[idx]
    except:
        crop = None
    
    waveform = waveforms[idx]
    dStm = ordered_dStms[idx]
    myFrms = dStm.frames

    if crop is None:
        ### in this case, there is no waveform -> create an empty one
        duration_in_samples = int(myFrms.length() / 25 * SAMPLE_RATE)
        waveform = np.zeros((duration_in_samples))

        cropped_frames.append(myFrms)
        cropped_waveforms.append(waveform)
    else:
        ### In this case, there is a waveform, so the frame is static -> we have to repeat it, to fit the length of narration
        fs_sec, ls_sec = crop
        fs_frm, ls_frm = int(fs_sec * 25), int(ls_sec * 25) + 1
        fs_smp, ls_smp = int(fs_sec * SAMPLE_RATE), int(ls_sec * SAMPLE_RATE) + 1

        cropped_snippet = BGRFrames.from_frame(myFrms[0], repetitions = ls_frm - fs_frm)
        cropped_waveform = waveform[fs_smp:ls_smp]

        cropped_frames.append(cropped_snippet)
        cropped_waveforms.append(cropped_waveform)

joint_cropped_frames = BGRFrames.from_frames(cropped_frames)
joint_cropped_waveforms = np.concatenate(cropped_waveforms, axis = 0)

cropped_video_path = os.path.join(result_path, 'cropped_vid.avi')
cropped_audio_path = os.path.join(result_path, 'cropped_recording.mp3')
cropped_joint_path = os.path.join(result_path, 'cropped_joint.mp4')
joint_cropped_frames.save(result_path, 'cropped_vid')
sf.write(cropped_audio_path, joint_cropped_waveforms, samplerate = SAMPLE_RATE)

### POST-PROCESSING ###
waveform_nr = nr.reduce_noise(y = joint_cropped_waveforms, sr = SAMPLE_RATE, stationary = True, prop_decrease = 0.25)

from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, Gain
board = Pedalboard([
    NoiseGate(threshold_db = -30, ratio = 1.5, release_ms = 250),
    Compressor(threshold_db = -16, ratio = 2.5),
    LowShelfFilter(cutoff_frequency_hz = 400, gain_db = 10, q = 1),
    Gain(gain_db = 13)
])

waveform_nr_pd = waveform_nr_pd = board(waveform_nr, SAMPLE_RATE)
cropped_audio_path_enhanced = os.path.join(result_path, 'cropped_recording_enhanced.mp3')
sf.write(cropped_audio_path_enhanced, waveform_nr_pd, samplerate = SAMPLE_RATE)

### WRITE
subprocess.run(f'ffmpeg -y -i "{cropped_video_path}" -i "{cropped_audio_path_enhanced}" -c copy "{cropped_joint_path}"')