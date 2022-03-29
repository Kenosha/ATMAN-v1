# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:17:12 2021

Utility Functions

@author: gianl
"""

import cv2 as cv

def show(frame, name = 'frame'):
    cv.imshow(name, frame)
    cv.waitKey(0)
    #closing all open windows  
    cv.destroyAllWindows()  

def get_frame(cap, frame_num):
    
    total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)  ### total number of frames
    
    if frame_num < 0 or frame_num > total_frames:
        print(f'No such frame, choose number between 0 and {total_frames}')
        return None
    
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    return frame

def show_video(cap, start_frame = None, stop_frame = None, ms_between_frames = None,
               frame_counter = False):
       
    if ms_between_frames is None:
        ms_between_frames = 25
    if start_frame is None:
        start_frame = 0
    if stop_frame is None:
        stop_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
    
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    frame_num = start_frame
    
    while cap.isOpened() and (start_frame <= frame_num <= stop_frame):
        ret, frame = cap.read()
        
        if frame_counter:
            cv.putText(frame, str(frame_num), org = (40, 40), 
                       fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 2,
                       color = (0, 255, 0), thickness = 3)
        
        if ret:
            cv.imshow('frame', frame)
            frame_num += 1
            if cv.waitKey(ms_between_frames) & 0xFF == ord('q'):
                break
        else:
            break
        
    cap.release()
    cv.destroyAllWindows()
    
    
def video_stream(cap, start_frame = None, stop_frame = None, ms_between_frames = None):
    if ms_between_frames is None:
        ms_between_frames = 25
    if start_frame is None:
        start_frame = 0
    if stop_frame is None:
        stop_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
    
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    frmNum = cap.get(cv.CAP_PROP_POS_FRAMES)
    
    while cap.isOpened() and (start_frame <= frmNum <= stop_frame):
        
        frmNum = cap.get(cv.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        
        frame = yield frmNum, frame # take in frame, possibly altered, to show
        # print(frame)
        
        cv.imshow('frame', frame)
        if cv.waitKey(ms_between_frames) & 0xFF == ord('q'):
            cv.destroyAllWindows()  
            yield None
                
        
    
        
        
        
        
        
        
        
        
        
        