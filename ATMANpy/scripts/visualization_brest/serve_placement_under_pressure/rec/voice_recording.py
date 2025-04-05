
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
          'transition_deuce_to_ad_20240730_012459', 'ad_side_static_20240730_023305']

def countdown():
    ### add in a countdown, so we open the opencv window
    from ATMANpy.types.Frames import BGRFrame
    emptyFrm = BGRFrame(np.zeros((1000, 1000, 3), dtype = 'uint8'))
    
    f3 = emptyFrm.put('3', size = 10, where = (450, 450))
    f2 = emptyFrm.put('2', size = 10, where = (450, 450))
    f1 = emptyFrm.put('1', size = 10, where = (450, 450))
    
    countdownFrms = BGRFrames.from_frames([ BGRFrames.from_frame(f3, 25),
                                            BGRFrames.from_frame(f2, 25),
                                            BGRFrames.from_frame(f1, 25)])
    DecapStream(countdownFrms, 0, 75).show()

def plot_waveform(waveform, sample_rate):
    import matplotlib.pyplot as plt
    num_frames, num_channels = waveform.shape
    time_axis = np.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[:, c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    figure.savefig(os.path.join(graph_path, 'wave.png'))

def amplitude_envelope(signal, frame_size, hop_length, sr) -> pd.Series:
    """Calculate the amplitude envelope of a signal

    Returns the envelope values and timestamps
    """
    amplitude_envelope = []
    
    # calculate amplitude envelope for each frame
    for i in range(0, len(signal), hop_length): 
        amplitude_envelope_current_frame = max(signal[i:i+frame_size]) 
        amplitude_envelope.append(amplitude_envelope_current_frame)
    
    values = np.array(amplitude_envelope)
    times = librosa.frames_to_time(np.arange(1, len(values)+1), hop_length=hop_length, sr=sr)
    series = pd.Series(values, index=times, name='envelope')
    return series

### calculate how much time the text ought to be on stream
def time_needed(script_line, words_per_minute = 100, pause_after_period = 1):
    sentences = script_line.split('\n')
    pause_time = len(sentences) * pause_after_period

    words = script_line.split(' ')
    speak_time = len(words) / words_per_minute * 60
    return speak_time + pause_time

IDX = 3
VIDEO = VIDEOS[IDX]

### Paths
result_path = os.path.join(get_folder_struct('results_video')[PROJECT_NAME], VIDEO)
graph_path = maybe_make_folder(result_path, 'graphs')
audio_path = maybe_make_folder(result_path, 'audio')
script_path = os.path.join(result_path, 'script.txt')

folders = os.listdir(audio_path)
num_takes = len(folders)

take_path = maybe_make_folder(audio_path, f'take_{num_takes + 1}')

### Read in Video Segments
seg_files = [f for f in os.listdir(result_path) if f.startswith('seg')]

ordered_dStms = []
for idx in range(len(seg_files)):
    seg = f'seg_{idx}.mp4'
    full_path = os.path.join(result_path, seg)
    ordered_dStms.append(Stream.from_file(full_path).decap())

### Read in Script
with open(script_path) as f:
    script_lines = [line.replace('\n', '') for line in f.readlines()]
script_lines_split = [ln.replace('.', '\n') for ln in script_lines]

assert len(ordered_dStms) == len(script_lines_split)
num_stms = len(ordered_dStms)

#### Where to print script lines
from ATMANpy.types.Lines import Region
from ATMANpy.types.Points import Point2D
textPt = Point2D((50, 920))
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

    time.sleep(1)
    
    showFrm = subStm.get(0).put(script_line, where = textPt.coords)
    seconds = time_needed(script_line)
    sub_recording = sd.rec(int(seconds * SAMPLE_RATE), samplerate = SAMPLE_RATE, channels = 2)
    showFrm.show(num_seconds = seconds)
    recordings[idx] = sub_recording

### Save Envelope and Recording
db_ref_in_pascal = 0.00002
db_threshold = 45
hop_size = 1000
env_size = 10000

for idx, waveform in recordings.items():
    mono_waveform = waveform.mean(axis = 1)
    envSer = amplitude_envelope(mono_waveform, env_size, hop_size, SAMPLE_RATE)
    envSer_db = pd.Series(librosa.amplitude_to_db(envSer, ref = db_ref_in_pascal), index = envSer.index)
    envSer_db.name = 'envelope'

    thrSer_db = (envSer_db > db_threshold) * db_threshold
    thrSer_db.name = 'threshold'
    pltFrm = pd.concat([envSer_db, thrSer_db], axis = 1)

    fig, ax = plt.subplots()
    ax = pltFrm['envelope'].plot()
    fig.savefig(os.path.join(take_path, f'env_{idx}.png'))
    plt.close()

    sf.write(os.path.join(take_path, f'seg_{idx}.mp3'), mono_waveform, samplerate = SAMPLE_RATE)

