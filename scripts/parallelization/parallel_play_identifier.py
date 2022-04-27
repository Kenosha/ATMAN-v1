#!/usr/bin/env python

'''
BASED ON: open cv video_threded sample script

Multithreaded video processing sample.
Usage:
   video_threaded.py {<video device number>|<video file name>}
   Shows how python threading capabilities can be used
   to organize parallel captured frame processing pipeline
   for smoother playback.
Keyboard shortcuts:
   ESC - exit
   space - switch between multi and single threaded processing
'''

import numpy as np
import cv2 as cv

from multiprocessing.pool import ThreadPool
from collections import deque

import ATMANpy.funcs.util_funcs as atu

#%% Functions ###

import time
from copy import deepcopy
from ATMANpy.types.Frames import BGRFrame, GreyFrame

from ATMANpy.funcs.high_level_funcs import track_courtlines
from functools import partial


VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
start_frame = 10280
stop_frame = 20000

def get_features_parallel(cap, start_frame, stop_frame, verbose = True):

    ### do the fast function -> 225 only
    my_courtline_func = partial(track_courtlines, get_net = False, only_225 = True)    

    time_start = time.time()
    time_now = time_start
    
    stream_features = {}
    H, config = None, None
    
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    
    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    pending = deque()

    while True:
        
        if len(pending) < threadn:
            frmNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            _ret, frame = cap.read()
            
            cFrm = BGRFrame(frame)
            gFrm = cFrm.to_grey()
            
            task = pool.apply_async(my_courtline_func, (gFrm, H, config))

            pending.append((frmNum, gFrm, task))
            
        while len(pending) > 0 and pending[0][2].ready():
            num, frm, task = pending.popleft()
            courtLines, H, config, trackFlag = task.get()
            line_feature_dict = {'courtLines': courtLines, 
                                 'H': H, 
                                 'trackFlag': trackFlag}
            stream_features[num] = deepcopy(line_feature_dict)
            
            if num % 1000 == 0:
                time_lst = time_now
                time_now = time.time()
                time_elapsed = round(time_now - time_lst, 2)
                print(f'  Done with {num}, took {time_elapsed} seconds...')
            
            if verbose:
                showArr = frm.to_BGR().put(courtLines).array
                cv.imshow('threaded video', showArr)
            
        ch = cv.waitKey(1)
        if ch == 27:
            break
        
        if frmNum > stop_frame:
            break

    totFrames = stop_frame - start_frame
    time_elapsed = time.time() - time_start
    minutes_elapsed = round(time_elapsed / 60, 1)
    print(f'Done with {totFrames} frames. Took {minutes_elapsed} minutes')

    cv.destroyAllWindows()
    return stream_features

my_features = get_features_parallel(cap, start_frame, stop_frame, verbose = False)

###################
### Second Pass ###
###################

# here, we:
# -> re-try "stutter frames" (where tracking was not found for a few frames)

def get_redo_frames(my_features, 
                    redo_threshold = 24): 
                    ### below this many consequent frames that were not found,
                    ### we will retry all of them.
    
    def list_to_ranges(inList):
        outList = []
        left_elem = None
        last_elem = None
        for i in inList:
            if last_elem is None:
                last_elem = i
                left_elem = i
                continue
                
            if left_elem is None:
                left_elem = i
                continue
            
            if i == last_elem + 1:
                last_elem = i
                continue
            elif i > last_elem + 1:
                outList.append((left_elem, last_elem))
                left_elem = i
                last_elem = i
                continue
            
        return outList
            
    def flatten(list_of_lists):
        if list_of_lists == []:
            return []
        else:
            return [*list_of_lists[0], *flatten(list_of_lists[1:])]
    
    not_found = [frm for frm, frmDict in my_features.items() if frmDict['trackFlag'] == 'Not Found']
    contiguous = list_to_ranges(not_found)      
    to_retry = [list(range(start, end + 1)) for start, end in contiguous if end - start < redo_threshold]
    to_retry_flat = flatten(to_retry)
    
    return to_retry_flat

def redo_features_parallel(cap, frameList, verbose = True):
    
    ### do the thorough function -> general homography finding.
    my_courtline_func = partial(track_courtlines, get_net = False, only_225 = False)    

    time_start = time.time()
    
    stream_features = {}
    H, config = None, None
    
    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    pending = deque()

    while frameList != []:
        
        if len(pending) < threadn:
            frmNum, frameList = frameList[0], frameList[1:]
            cap.set(cv.CAP_PROP_POS_FRAMES, frmNum)
            _ret, frame = cap.read()
            
            cFrm = BGRFrame(frame)
            gFrm = cFrm.to_grey()
            
            task = pool.apply_async(my_courtline_func, (gFrm, None, None))

            pending.append((frmNum, gFrm, task))
            
        while len(pending) > 0 and pending[0][2].ready():
            num, frm, task = pending.popleft()
            courtLines, H, config, trackFlag = task.get()
            line_feature_dict = {'courtLines': courtLines, 
                                 'H': H, 
                                 'trackFlag': trackFlag}
            stream_features[num] = deepcopy(line_feature_dict)
            
            if verbose:
                showArr = frm.to_BGR().put(courtLines).array
                cv.imshow('threaded video', showArr)
            
        ch = cv.waitKey(1)
        if ch == 27:
            break
        
        if frmNum > stop_frame:
            break

    totFrames = len(frameList)
    time_elapsed = time.time() - time_start
    minutes_elapsed = round(time_elapsed / 60, 1)
    print(f'Done with {totFrames} frames. Took {minutes_elapsed} minutes')

    cv.destroyAllWindows()
    return stream_features


redos = get_redo_frames(my_features)
redone_features = redo_features_parallel(cap, redos)

len(redos)


for frm, featDict in redone_features.items():
    gFrm = featDict['gFrm']
    cLines = featDict['courtLines']
    gFrm.to_BGR().put(cLines).show()


frmNum = 17296

storedFrm = redone_features[frmNum]['gFrm']
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
gFrm = GreyFrame.from_video(cap, frmNum)  

(storedFrm.array == gFrm.array).all()

# gFrm.show()
# 


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

