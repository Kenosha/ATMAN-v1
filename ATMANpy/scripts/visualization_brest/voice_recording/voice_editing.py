
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
audio_enhanced_path = maybe_make_folder(result_path, 'audio_enhanced')
video_cropped_path = maybe_make_folder(result_path, 'video_cropped')
segments_joint_path = maybe_make_folder(result_path, 'segments_joint')

folders = os.listdir(audio_path)
num_takes = len([f for f in folders if f.startswith('take_')])

### Read in Video Segments
seg_files = [f for f in os.listdir(result_path) if f.startswith('seg_')]
num_segs = len(seg_files)

ordered_Stms = {}
rec_waves = {}
for idx in range(num_segs):
    seg = f'seg_{idx}.mp4'
    take = f'seg_{idx}.mp3'
    seg_path = os.path.join(result_path, seg)
    take_path = os.path.join(audio_path, take)

    ordered_Stms[idx] = Stream.from_file(seg_path)
    try:
        rec_waves[idx], _ = sf.read(take_path)
    except:
        rec_waves[idx] = None ### some segments do not have audio

### Read In Soundwaves
sound_waves = {}
for idx in range(len(seg_files)):
    try:
        script_path = os.path.join(result_path, f'wave_{idx}.mp3')
        sound_wave, _ = sf.read(script_path)
    except:
        sound_wave = None
    
    sound_waves[idx] = sound_wave

### find the crops by examining the waveform
from ATMANpy.funcs.rec_funcs import amplitude_envelope

db_ref_in_pascal = 0.00002
hop_size = 1000
env_size = 10000
db_threshold_before = 50
db_threshold_after = 10

cropDict = {}
for idx, waveform in rec_waves.items():
    if waveform is None:
        cropDict[idx] = None
    else:
        envSer = amplitude_envelope(waveform, env_size, hop_size, SAMPLE_RATE)
        envSer_db = pd.Series(librosa.amplitude_to_db(envSer, ref = db_ref_in_pascal), index = envSer.index)

        thrSer_aft = envSer_db > db_threshold_after
        rolSer_aft = thrSer_aft.rolling(10, center = True, min_periods = 1).mean()

        thrSer_bef = envSer_db > db_threshold_before
        rolSer_bef = thrSer_bef.rolling(10, center = True, min_periods = 1).mean()

        first_idx = (rolSer_bef > 0.5).argmax()
        first_sec = rolSer_bef.index[first_idx]

        last_idx = (rolSer_aft < 0.5).argmax()
        last_sec = rolSer_aft.index[last_idx]

        ### ensure that the entire duration of the waveform fits within an integer number of frames.
        crop_frames = (last_sec - first_sec) * FRAME_RATE
        ceil_frames = np.ceil(crop_frames)
        redist_frames = (ceil_frames - crop_frames) / 2
        redist_secs = redist_frames / FRAME_RATE

        first_sec -= redist_secs
        last_sec += redist_secs

        cropDict[idx] = (first_sec, last_sec)

for idx in range(num_segs):
    print(idx)
    try:
        crop = cropDict[idx]
    except:
        crop = None
    
    rec_waveform = rec_waves[idx]
    sound_waveform = sound_waves[idx]
    myStm = ordered_Stms[idx]
    myFrms = myStm.decap().frames

    if crop is None:
        ### in this case, there is no waveform -> create an empty one
        duration_in_samples = int(myFrms.length() / FRAME_RATE * SAMPLE_RATE)
        
        cropped_snippet = myFrms
        cropped_waveform = np.zeros((duration_in_samples))

        if sound_waveform is not None:
            ### check the length divergence
                # if the frames are not slowed down, these should match
                # if the frames are slowed down, then some get lost (WHY, again?)
                    # ensure that the divergence is not too great, and fit the wave to the length of the frames.

            duration_sound = len(sound_waveform) 
            div = (duration_sound / duration_in_samples) - 1
            print(f'Divergence between Length of Background Sound and Frames: {div:.3%}')

            sound_waveform_fit = signal.resample(sound_waveform, duration_in_samples)
            cropped_waveform += sound_waveform_fit


    else:
        ### In this case, there is a waveform.
        ### 1) if the frame is longer than the waveform, we pad with silence. 
        ### 2) if it is shorter, then we repeat the last frame, on still.
        fs_sec, ls_sec = crop
        fs_frm, ls_frm = int(fs_sec * FRAME_RATE), int(ls_sec * FRAME_RATE) 
        fs_smp, ls_smp = int(fs_sec * SAMPLE_RATE), int(ls_sec * SAMPLE_RATE)

        snippet_len = ls_frm - fs_frm
        frames_len = myFrms.length()
        frames_samples = int(frames_len / FRAME_RATE * SAMPLE_RATE)
        avail_samples = ls_smp - fs_smp 
        if avail_samples < frames_samples:  ### 1) frame longer than waveform
                                            ### -> pad with empty waveform
            empty_waveform = np.zeros(frames_samples - avail_samples)

            cropped_snippet = myFrms
            cropped_waveform = np.concatenate([rec_waveform[fs_smp:ls_smp], empty_waveform])

            if sound_waveform is not None:
                sound_waveform_fit = signal.resample(sound_waveform, frames_samples)
                cropped_waveform += sound_waveform_fit

        else:                               ### 2) frame shorter than waveform
                                            ### -> pad with repeated frames
            excessFrms = BGRFrames.from_frame(myFrms[-1], repetitions = snippet_len - myFrms.length())
            
            cropped_snippet = myFrms.join(excessFrms)
            cropped_waveform = rec_waveform[fs_smp:ls_smp]

            if sound_waveform is not None:
                ### the waveform comes from the video, so it can't extend into the excessFrms
                sound_waveform_fit = signal.resample(sound_waveform, frames_samples)
                empty_waveform = np.zeros(avail_samples - frames_samples)
                sound_waveform_fit = np.concatenate([sound_waveform_fit, empty_waveform])

                cropped_waveform += sound_waveform_fit


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
    subprocess.run(f'ffmpeg -y -i "{video_file}" -i "{audio_file}" -c copy "{joint_file}" -r {FRAME_RATE} -ar {SAMPLE_RATE}')

