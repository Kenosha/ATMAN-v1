# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:27:15 2022

@author: gianl
"""

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.funcs.util_funcs as atu
from ATMANpy.types.Points import Point, SafePoint
from ATMANpy.types.Lines import Line, Lines, CourtLines, Region
from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame
from ATMANpy.types.Contours import Contour

import cv2 as cv
import pandas as pd
import numpy as np

import os

############# STREAM FUNCTIONS ########

import ATMANpy.funcs.high_level_funcs as hlf

def init_featDict():
    # H = None
    # config = None
    # topplayerRegion, botplayerRegion = None, None
    # particles, idxHistories, closeHistories, modelHistories = None, None, None, None    
    
    featDict = {'H': None, 'config': None,
                'courtLines': None, 'netLines': None, 'trackFlag': None}
    return featDict

def init_frameDict(frmCache):
    
    cFrm = frmCache[0]
    gFrm = cFrm.to_grey()
    
    frameDict = {'cFrm': cFrm, 'gFrm': gFrm}
    
    return frameDict
    
def get_court_lines(frmDict, featDict):
    
    gFrm = frmDict['gFrm']
    H = featDict['H']
    config = featDict['config']
    
    try:
        courtLines, netLines, H, config, trackFlag = hlf.track_courtlines(gFrm, H, config)
    except Exception as e:
        print('Exception in track_courtlines:')
        print(e)
        courtLines, netLines, H, config = None
        trackFlag = 'Not Found'
    
    featDict['courtLines'] = courtLines
    featDict['netLines'] = netLines
    featDict['H'] = H
    featDict['config'] = config
    featDict['trackFlag'] = trackFlag
    
    return featDict

#%%### DRAW FUNCTIONS #####  

def draw_lines(frmDict, featDict):
    
    cFrm = frmDict['cFrm']
    
    if featDict['trackFlag'] == 'Not Found':
        sendFrm = cFrm.put('No Lines!', thickness = 2, color = (0, 0, 255))
    else:
        courtLines = featDict['courtLines']
        
        sendFrm = cFrm.put(courtLines, thickness = 2, color = (0, 255, 0))

    return sendFrm
    
    


#%% Collect all Sequences of Play

### IDEA:
# Look for courtlines (if not found, also try general homography func, not just 225)
# save courtlines and homography
# do this parallelized, for the whole video
# evaluate consequent chunks of video with a stable homography/similar courtlines.


### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
# cap = atu.get_video(VIDEO_ID)
################

from functools import partial
from ATMANpy.funcs.cv_funcs import get_stream_features, replay
from ATMANpy.funcs.high_level_funcs import postprocess_ball_flight
import time

my_stream_func = partial(get_stream_features, VIDEO_ID, 
                        init_feat_func = init_featDict, 
                        init_frame_func = init_frameDict,
                        court_line_func = get_court_lines,
                        draw_func = draw_lines,
                        verbose = False)

####### MULTIPROCESS ########
# WORKERS = os.cpu_count()
WORKERS = 24

import multiprocessing as mp
def mulproc(mp_func, mp_iterator, num_workers = WORKERS):
    with mp.Pool(num_workers) as p:
        outList = p.starmap(mp_func, mp_iterator)
    return outList

def subdivide_frames(start_frame, stop_frame, num_workers = WORKERS):
    start_frames = np.linspace(start_frame, stop_frame, num = num_workers, 
                               endpoint = False, dtype = 'int32')
    end_frames = [*start_frames[1:] - 1, stop_frame]
    return list(zip(start_frames, end_frames))
    

#### Experiment -> how considerable is the speed-up?

"""
This does not work well! 
Multiprocessing with multiple video capture instances is either 
much slower (in the low run_length cases), or
does not terminate (in all cases). I suspect this is because of blocking behaviour
caused by multiple processes creating a video capture using the same file.

"""

start_frame = 11000

### a)
run_length = 50
print(f'Run Length: {run_length} frames!')
stop_frame = start_frame + run_length

time_before = time.time()
stream_features = my_stream_func(start_frame, stop_frame)
time_after = time.time()
timeStr = round(time_after - time_before, 2)
print(f'   Serial Run Done! Time Elapsed: {timeStr} seconds')    


time_before = time.time()
mp_iterator = subdivide_frames(start_frame, stop_frame)
if __name__ == '__main__':
    out = mulproc(my_stream_func, mp_iterator, num_workers = WORKERS)
time_after = time.time()
timeStr = round(time_after - time_before, 2)
print(f'   Multiprocessed Run Done! Time Elapsed: {timeStr} seconds')    
  
### b)
run_length = 500
print(f'Run Length: {run_length} frames!')
stop_frame = start_frame + run_length

time_before = time.time()
stream_features = my_stream_func(start_frame, stop_frame)
time_after = time.time()
timeStr = round(time_after - time_before, 2)
print(f'   Serial Run Done! Time Elapsed: {timeStr} seconds')    


time_before = time.time()
mp_iterator = subdivide_frames(start_frame, stop_frame)
if __name__ == '__main__':
    out = mulproc(my_stream_func, mp_iterator, num_workers = WORKERS)
time_after = time.time()
timeStr = round(time_after - time_before, 2)
print(f'   Multiprocessed Run Done! Time Elapsed: {timeStr} seconds')    
  
# ### c)
# run_length = 5000
# print(f'Run Length: {run_length} frames!')
# stop_frame = start_frame + run_length

# time_before = time.time()
# stream_features = my_stream_func(start_frame, stop_frame)
# time_after = time.time()
# timeStr = round(time_after - time_before, 2)
# print(f'   Serial Run Done! Time Elapsed: {timeStr} seconds')    


# time_before = time.time()
# mp_iterator = subdivide_frames(start_frame, stop_frame)
# if __name__ == '__main__':
#     out = mulproc(my_stream_func, mp_iterator, num_workers = WORKERS)
# time_after = time.time()
# timeStr = round(time_after - time_before, 2)
# print(f'   Multiprocessed Run Done! Time Elapsed: {timeStr} seconds')    
  

# ### d)
# run_length = 50000
# print(f'Run Length: {run_length} frames!')
# stop_frame = start_frame + run_length

# time_before = time.time()
# stream_features = my_stream_func(start_frame, stop_frame)
# time_after = time.time()
# timeStr = round(time_after - time_before, 2)
# print(f'   Serial Run Done! Time Elapsed: {timeStr} seconds')    


# time_before = time.time()
# mp_iterator = subdivide_frames(start_frame, stop_frame)
# if __name__ == '__main__':
#     out = mulproc(my_stream_func, mp_iterator, num_workers = WORKERS)
# time_after = time.time()
# timeStr = round(time_after - time_before, 2)
# print(f'   Multiprocessed Run Done! Time Elapsed: {timeStr} seconds')    
  


# # #### Simulation 
# # video_segments = [(11830, 11930)]

# # for start_frame, stop_frame in video_segments:
# #     stream_features = my_stream_func(cap, start_frame, stop_frame)
# #     stream_features_smooth = postprocess_ball_flight(stream_features)
# #     replay(cap, stream_features_smooth, 
# #            cache_frame_offset = CURR_FRAME_IDX, drawFunc = draw_replay)

# # #############





