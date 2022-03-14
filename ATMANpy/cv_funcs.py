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
    
    total_frames = cap.get(7)  ### total number of frames
    
    if frame_num < 0 or frame_num > total_frames:
        print(f'No such frame, choose number between 0 and {total_frames}')
        return None
    
    cap.set(1, frame_num)
    ret, frame = cap.read()
    
    return frame
