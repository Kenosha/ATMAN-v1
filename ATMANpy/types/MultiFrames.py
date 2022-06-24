# -*- coding: utf-8 -*-
"""
Created on Sat May 21 22:18:33 2022

@author: gianl
"""


import os
import cv2 as cv
import numpy as np
from dataclasses import dataclass

import ATMANpy.funcs.util_funcs as atu
from ATMANpy.funcs.cv_funcs import get_frames   
from ATMANpy.types.Frames import HSVFrame, BGRFrame, GreyFrame


# ### VIDEO ######
# VIDEO_ID = 'pqxAmUWIoKc'
# cap = atu.get_video(VIDEO_ID)
# ################

        

@dataclass(frozen = True)
class Frames:
    video_id: str
    start_frame: int
    stop_frame: int
    color_space: str
    
    def __post_init__(self):
        
        ### get the actual frame values ###
        # start_cache = start_frame - cache_offset
        # stop_cache = stop_frame + cache_offset
        cap = atu.get_video(self.video_id)
        frames = get_frames(cap, self.start_frame, self.stop_frame, color_space = self.color_space)
        # cap.release()
        
        #### IMMUTABILITY BREAKS ####
        object.__setattr__(self, 'frames', frames)
        object.__setattr__(self, 'unitMap', {'B': 1, 'KB': 1000, 'MB': 1000 ** 2, 'GB': 1000 ** 3})
        object.__setattr__(self, 'colorMap', {'hsv': HSVFrame, 'grey': GreyFrame, 'bgr': BGRFrame})
        
        ####
    
    def __getitem__(self, frmItem):
        ### Note! We can only address items by their frame number!
        
        if type(frmItem) is slice:
            arrSlice = slice(frmItem.start - self.start_frame, 
                             frmItem.stop - self.start_frame, 
                             frmItem.step - self.start_frame)
            return self.frames[arrSlice] ### might potentially also return another Frames object?
        else:
            myType = self.colorMap[self.color_space]
            arrItem = frmItem - self.start_frame
            return myType(self.frames[arrItem])
     
    def __iter__(self):
        return self.frames.__iter__()
    
    @staticmethod
    def from_sf(video_id, sf, color_space = 'bgr'):
        ks = sorted(sf.keys())
        start, stop = ks[0], ks[-1]
        return Frames(video_id, start, stop, color_space)
        
    
    def to_generator(self):
        frame_counter = self.start_frame
        for frame in self:
            yield frame_counter, frame
            frame_counter += 1
            
    def size(self, unit = 'GB'):
        divisor = self.unitMap[unit]
        return self.frames.size / divisor
        
    
    
    
# start_frame = 10200
# stop_frame = 10500
    
# # myFrms = Frames.from_cap(cap, start_frame, stop_frame)


# myFrms = Frames(VIDEO_ID, start_frame, stop_frame, 'hsv')

# myFrms[0]

# # for f in myFrms:
# #     print(f[0, 0])


# myFrms[:2]

# myGen = myFrms.to_generator()
# for g in myGen:
#     print(g)


# myGen = myFrms.to_generator()
# frmIdx, frm = next(myGen)

# myGen.send('bizbaz') ##3 works!



