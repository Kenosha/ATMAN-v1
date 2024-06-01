
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import subprocess

from ATMANpy.funcs.util_funcs import get_folder_struct, make_timestamp
from scipy.io.wavfile import write

from ATMANpy.types.Features import ConstFeature
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import Stream, DecapStream

SAMPLE_RATE = 44100
VIDEO_NAME = 'First Serves Deuce_20240601_163302'
PROJECT_NAME = 'first_scripted_video'

### Paths
result_path = os.path.join(get_folder_struct('results_video')['scripted'], VIDEO_NAME)
script_path = os.path.join(result_path, 'script.txt')
video_path = os.path.join(result_path, 'full.mp4')
audio_path = os.path.join(result_path, 'recording.mp3')
joint_path = os.path.join(result_path, 'joint.mp4')

### Read in Video Segments
seg_files = [f for f in os.listdir(result_path) if f.startswith('seg')]

ordered_dStms = []
for idx in range(len(seg_files)):
    seg = f'seg_{idx}.mp4'
    full_path = os.path.join(result_path, seg)
    ordered_dStms.append(Stream.from_file(full_path).decap())

### Read in Full Video
fullStm = Stream.from_file(video_path)

### Read in Script
with open(script_path) as f:
    script_lines = [line.replace('\n', '') for line in f.readlines()]

### Record Audio
recordings = []
for subStm, script_line in zip(ordered_dStms, script_lines):
    txtSF = ConstFeature('text', script_line)
    
    seconds = subStm.length() / 25
    sub_recording = sd.rec(int(seconds * SAMPLE_RATE), samplerate = SAMPLE_RATE, channels = 2)

    subStm.attach(txtSF).show()

    recordings.append(sub_recording)

joint_recording = np.concatenate(recordings, axis = 0)

### write audio file
sf.write(audio_path, joint_recording, samplerate = SAMPLE_RATE)

### write combined video and audio file
subprocess.run(f'ffmpeg -y -i "{video_path}" -i "{audio_path}" -c copy "{joint_path}"')

