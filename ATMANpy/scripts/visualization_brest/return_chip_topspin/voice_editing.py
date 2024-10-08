
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
PROJECT_NAME = 'return_chip_topspin'
VIDEOS = ['intro_20241006_135157']

# VIDEO_CROPS = {'intro_20241006_135157': {0:  (0.2, 45),
#                                          1:  (0.2, 39),
#                                          3:  (0.2, 50),
#                                          5:  (0.2, 12),
#                                          8:  (0.2, 6),
#                                          9:  (0.2, 8),
#                                          11: (0.2, 28),
#                                          12: (0.2, 11),
#                                          14: (0.2, 9),
#                                          16: (0.2, 7),
#                                          19: (0.5, 9),
#                                          21: (0.5, 11),
#                                          22: (0.2, 6),
#                                          25: (0.2, 12),
#                                          26: (0.2, 21),
#                                          27: (0.2, 13),
#                                          28: (0.2, 9),
#                                          29: (0.2, 11),
#                                          30: (0.2, 5)},
#                 }

IDX = 0
VIDEO = VIDEOS[IDX]

### Paths
result_path = os.path.join(get_folder_struct('results_video')[PROJECT_NAME], VIDEO)
audio_path = os.path.join(result_path, 'audio', 'best')
audio_enhanced_path = maybe_make_folder(result_path, 'audio_enhanced')
video_cropped_path = maybe_make_folder(result_path, 'video_cropped')
segments_joint_path = maybe_make_folder(result_path, 'segments_joint')

folders = os.listdir(audio_path)
num_takes = len([f for f in folders if f.startswith('take_')])

### Read in Video Segments
seg_files = [f for f in os.listdir(result_path) if f.startswith('seg_')]
num_segs = len(seg_files)

ordered_Stms = {}
waveforms = {}
for idx in range(num_segs):
    seg = f'seg_{idx}.mp4'
    take = f'seg_{idx}.mp3'
    seg_path = os.path.join(result_path, seg)
    take_path = os.path.join(audio_path, take)

    ordered_Stms[idx] = Stream.from_file(seg_path)
    try:
        waveforms[idx], _ = sf.read(take_path)
    except:
        waveforms[idx] = None ### some segments do not have audio

### find the crops by examining the waveform
from ATMANpy.funcs.rec_funcs import amplitude_envelope

db_ref_in_pascal = 0.00002
hop_size = 1000
env_size = 10000
db_threshold = 10

cropDict = {}
for idx, waveform in waveforms.items():
    if waveform is None:
        cropDict[idx] = None
    else:
        envSer = amplitude_envelope(waveform, env_size, hop_size, SAMPLE_RATE)
        envSer_db = pd.Series(librosa.amplitude_to_db(envSer, ref = db_ref_in_pascal), index = envSer.index)

        thrSer = envSer_db > db_threshold
        thrSer.name = 'threshold'

        rolSer = thrSer.rolling(10, center = True, min_periods = 1).mean()

        last_idx = (rolSer < 0.5).argmax()
        last_sec = rolSer.index[last_idx]

        cropDict[idx] = (0, last_sec)

for idx in range(num_segs):
    print(idx)
    try:
        crop = cropDict[idx]
    except:
        crop = None
    
    waveform = waveforms[idx]
    myStm = ordered_Stms[idx]
    myFrms = myStm.decap().frames

    if crop is None:
        ### in this case, there is no waveform -> create an empty one
        duration_in_samples = int(myFrms.length() / 25 * SAMPLE_RATE)
        
        cropped_snippet = myFrms
        cropped_waveform = np.zeros((duration_in_samples))
    else:
        ### In this case, there is a waveform, so the frame is static -> we have to repeat it, to fit the length of narration
        fs_sec, ls_sec = crop
        fs_frm, ls_frm = int(fs_sec * 25), int(ls_sec * 25) + 1
        fs_smp, ls_smp = int(fs_sec * SAMPLE_RATE), int(ls_sec * SAMPLE_RATE) + 1

        cropped_snippet = BGRFrames.from_frame(myFrms[0], repetitions = ls_frm - fs_frm)
        cropped_waveform = waveform[fs_smp:ls_smp]

    ### save cropped video
    cropped_snippet.save(video_cropped_path, f'video_{idx}')
    video_file = os.path.join(video_cropped_path, f'video_{idx}.avi')
    
    ### save enhanced and cropped audio
    waveform_nr = nr.reduce_noise(y = cropped_waveform, sr = SAMPLE_RATE, stationary = True, prop_decrease = 0.25)

    from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, Gain
    board = Pedalboard([
        NoiseGate(threshold_db = -30, ratio = 1.5, release_ms = 250),
        Compressor(threshold_db = -16, ratio = 2.5),
        LowShelfFilter(cutoff_frequency_hz = 400, gain_db = 10, q = 1),
        Gain(gain_db = 13)
    ])

    waveform_nr_pd = board(waveform_nr, SAMPLE_RATE)
    audio_file = os.path.join(audio_enhanced_path, f'audio_{idx}.mp3')
    sf.write(audio_file, waveform_nr_pd, samplerate = SAMPLE_RATE)

    ### combine
    joint_file = os.path.join(segments_joint_path, f'joint_{idx}.mp4')
    subprocess.run(f'ffmpeg -y -i "{video_file}" -i "{audio_file}" -c copy "{joint_file}"')



# cropped_frames = []
# cropped_waveforms = []

# for idx in range(num_segs):
#     try:
#         crop = cropDict[idx]
#     except:
#         crop = None
    
#     waveform = waveforms[idx]
#     dStm = ordered_dStms[idx]
#     myFrms = dStm.frames

#     if crop is None:
#         ### in this case, there is no waveform -> create an empty one
#         duration_in_samples = int(myFrms.length() / 25 * SAMPLE_RATE)
#         waveform = np.zeros((duration_in_samples))

#         cropped_frames.append(myFrms)
#         cropped_waveforms.append(waveform)
#     else:
#         ### In this case, there is a waveform, so the frame is static -> we have to repeat it, to fit the length of narration
#         fs_sec, ls_sec = crop
#         fs_frm, ls_frm = int(fs_sec * 25), int(ls_sec * 25) + 1
#         fs_smp, ls_smp = int(fs_sec * SAMPLE_RATE), int(ls_sec * SAMPLE_RATE) + 1

#         cropped_snippet = BGRFrames.from_frame(myFrms[0], repetitions = ls_frm - fs_frm)
#         cropped_waveform = waveform[fs_smp:ls_smp]

#         cropped_frames.append(cropped_snippet)
#         cropped_waveforms.append(cropped_waveform)

# ### divide cropped frames, so that we can write them without running out of memory
# max_size = 20 # gb
# curr_size = 0
# list_of_frame_lists = []
# list_of_wave_lists = []
# curr_frame_list = []
# curr_wave_list = []

# for frame, wave in zip(cropped_frames, cropped_waveforms):
#     if curr_size + frame.size() > max_size:
#         list_of_frame_lists.append(curr_frame_list)
#         list_of_wave_lists.append(curr_wave_list)

#         curr_frame_list = [frame]
#         curr_wave_list = [wave]
#         curr_size = frame.size()
#     else:
#         curr_frame_list.append(frame)
#         curr_wave_list.append(wave)
#         curr_size += frame.size()

# if len(curr_frame_list) > 0:
#     list_of_frame_lists.append(curr_frame_list)
#     list_of_wave_lists.append(curr_wave_list)

# ### write to video

# idx = 0
# while len(list_of_frame_lists) > 0:
#     ### we delete the first frame list to save on RAM
#     frame_list = list_of_frame_lists.pop(0)
#     wave_list = list_of_wave_lists.pop(0)
    
#     joint_cropped_frames = BGRFrames.from_frames(frame_list)
#     joint_cropped_waveforms = np.concatenate(wave_list)

#     cropped_video_path = os.path.join(result_path, f'cropped_vid_{idx}.avi')
#     cropped_audio_path = os.path.join(result_path, f'cropped_recording_{idx}.mp3')
#     cropped_joint_path = os.path.join(result_path, f'cropped_joint_{idx}.mp4')
#     joint_cropped_frames.save(result_path, f'cropped_vid_{idx}')
#     sf.write(cropped_audio_path, joint_cropped_waveforms, samplerate = SAMPLE_RATE)

#     ### POST-PROCESSING ###
#     waveform_nr = nr.reduce_noise(y = joint_cropped_waveforms, sr = SAMPLE_RATE, stationary = True, prop_decrease = 0.25)

#     from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, Gain
#     board = Pedalboard([
#         NoiseGate(threshold_db = -30, ratio = 1.5, release_ms = 250),
#         Compressor(threshold_db = -16, ratio = 2.5),
#         LowShelfFilter(cutoff_frequency_hz = 400, gain_db = 10, q = 1),
#         Gain(gain_db = 13)
#     ])

#     waveform_nr_pd = waveform_nr_pd = board(waveform_nr, SAMPLE_RATE)
#     cropped_audio_path_enhanced = os.path.join(result_path, f'cropped_recording_enhanced_{idx}.mp3')
#     sf.write(cropped_audio_path_enhanced, waveform_nr_pd, samplerate = SAMPLE_RATE)

#     ### WRITE
#     subprocess.run(f'ffmpeg -y -i "{cropped_video_path}" -i "{cropped_audio_path_enhanced}" -c copy "{cropped_joint_path}"')

#     idx += 1