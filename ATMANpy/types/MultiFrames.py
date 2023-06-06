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
from ATMANpy.funcs.cv_funcs import get_frames, get_framelist
from ATMANpy.types.Frames import HSVFrame, BGRFrame, GreyFrame


class FramesIterator:
    def __init__(self, frameType, array, name):
        self._array = array
        self._frameType = frameType
        self._index = 0
        self._name = name

    def __next__(self):
        try:
            nextArr = self._array[self._index]
        except IndexError:
            raise StopIteration
        else:
            self._index += 1
            return self._frameType(nextArr, name = self._name)
        
@dataclass(frozen = True)
class Frames:
    array: np.array
    name: str = 'frames'
    
    
    @classmethod
    def from_video(cls, cap, start_frame, stop_frame, name = None):
        frames = get_frames(cap, start_frame, stop_frame, color_space = cls.get_color())
        if name is None:
            return cls(frames)
        else:
            return cls(frames, name = name)
    
    @classmethod
    def from_list(cls, cap, frameList, name = None):
        frames = get_framelist(cap, frameList, color_space = cls.get_color())
        if name is None:
            return cls(frames)
        else:
            return cls(frames, name = name)
    
    def __getitem__(self, item):
        framesType = self.__class__
        frameType = framesType.get_constituent()
        if isinstance(item, int):
            return frameType(self.array[item], name = self.name)
        else:
            return framesType(self.array[item], name = self.name)
    
    def __repr__(self):
        framesType = self.__class__
        return f'{framesType}(length = {self.length()}, size = {self.size(unit = "MB")} MB)'
    
    def __iter__(self):
        framesType = self.__class__
        frameType = framesType.get_constituent()
        return FramesIterator(frameType, self.array, self.name)
    
    def size(self, unit = 'GB'):
        unitMap = {'B': 1, 'KB': 1000, 'MB': 1000 ** 2, 'GB': 1000 ** 3}
        return round(self.array.size / unitMap[unit], 1)
    
    def shape(self):
        return self.array.shape
    
    def length(self):
        return self.shape[0]
    
@dataclass(frozen = True)
class GreyFrames(Frames):
    name: str = 'grey_frames'
    
    @classmethod
    def get_constituent(cls):
        return GreyFrame
    
    @classmethod
    def get_color(cls):
        return 'grey'
    
@dataclass(frozen = True)
class BGRFrames(Frames):
    name: str = 'bgr_frames'
    
    @classmethod
    def get_constituent(cls):
        return BGRFrame
    
    @classmethod
    def get_color(cls):
        return 'bgr'
    
    
@dataclass(frozen = True)
class HSVFrames(Frames):
    name: str = 'hsv_frames'
    
    @classmethod
    def get_constituent(cls):
        return HSVFrame
    
    @classmethod
    def get_color(cls):
        return 'hsv'
    
if __name__ == '__main__':
        
    ### VIDEO ######
    VIDEO_ID = 'pqxAmUWIoKc'
    cap = atu.get_video(VIDEO_ID)
    ################

    start, stop = 3000, 3100
    gFrms = GreyFrames.from_video(cap, start, stop)
    cFrms = BGRFrames.from_video(cap, start, stop)
    hFrms = HSVFrames.from_video(cap, start, stop)

    print(gFrms.shape())
    print(cFrms.shape())
    print(hFrms.shape())

