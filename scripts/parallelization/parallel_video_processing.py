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

from ATMANpy.types.Frames import BGRFrame

from ATMANpy.funcs.high_level_funcs import track_courtlines
from functools import partial
my_courtline_func = partial(track_courtlines, get_net = False)

# courtLines, H, config, trackFlag = hlf.track_courtlines(gFrm, H, config,
#                                                         get_net = False)


class DummyTask:
    def __init__(self, data):
        self.data = data
    def ready(self):
        return True
    def get(self):
        return self.data

def main():
    
    start_frame = 11000
    stop_frame = 12000
    H, config = None, None
    
    VIDEO_ID = 'pqxAmUWIoKc'
    cap = atu.get_video(VIDEO_ID)
    
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    frmNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    

    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    pending = deque()

    threaded_mode = True

    while cap.isOpened() and (start_frame <= frmNum < stop_frame):
        
        if len(pending) < threadn:
            _ret, frame = cap.read()
            cFrm = BGRFrame(frame)
            gFrm = cFrm.to_grey()
            
            if threaded_mode:
                task = pool.apply_async(my_courtline_func, (gFrm, H, config))
            else:
                task = DummyTask(my_courtline_func(gFrm, H, config))
            pending.append(task)
            
        while len(pending) > 0 and pending[0].ready():
            courtLines, H, config, trackFlag = pending.popleft().get()
            showArr = cFrm.put(courtLines).array
            cv.imshow('threaded video', showArr)
            
        ch = cv.waitKey(1)
        if ch == ord(' '):
            threaded_mode = not threaded_mode
        if ch == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()