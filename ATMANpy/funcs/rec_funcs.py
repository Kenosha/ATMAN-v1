import os
import librosa
import pandas as pd
import numpy as np

from ATMANpy.types.Features import ConstFeature, StreamFeature
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import Stream, DecapStream

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

def plot_waveform(waveform, sample_rate, graph_path = None):
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

    if graph_path is not None:
        figure.savefig(os.path.join(graph_path, 'wave.png'))

    return figure

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
def time_needed(script_line, words_per_minute = 100, pause_after_period = 1, min_seconds = 10):
    sentences = script_line.split('\n')
    pause_time = len(sentences) * pause_after_period

    words = script_line.split(' ')
    speak_time = len(words) / words_per_minute * 60
    return max(min_seconds, speak_time + pause_time)