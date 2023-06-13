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
from ATMANpy.types.Frames import HSVFrame, BGRFrame, GreyFrame, BoolFrame


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

    def __repr__(self):
        framesType = self.__class__.__name__
        if self.cols() is None and self.rows() is None:
            dimStr = 'None'
        else:
            dimStr = f'{self.cols()} x {self.rows()}'
        return f'{framesType}(name = {self.name}, length = {self.length()}, size = {self.size(unit = "MB")} MB, dims = {dimStr})'
    
    @classmethod
    def from_video(cls, cap, start_frame, stop_frame, 
                   cap_color_space = 'bgr', 
                   scale_factor = 1,
                   name = None):
        typeMap = {'bgr': BGRFrames,
                   'grey': GreyFrames,
                   'hsv': HSVFrames,
                   'bool': BoolFrames}
        capType = typeMap[cap_color_space]
        
        frames = get_frames(cap, start_frame, stop_frame)
        
        if name is None:
            capFrms = capType(frames)
        else:
            capFrms = capType(frames, name = name)
        
        if scale_factor != 1:
            capFrms = capFrms.rescale(scale_factor)
        
        conversionFuncStr = cls.get_converter()
        
        if cap_color_space == cls.get_color():
            return capFrms
        else:
            return getattr(capFrms, conversionFuncStr)()
        
    @classmethod
    def from_list(cls, cap, frameList, 
                  cap_color_space = 'bgr', 
                  scale_factor = 1,
                  name = None):
        typeMap = {'bgr': BGRFrames,
                   'grey': GreyFrames,
                   'hsv': HSVFrames,
                   'bool': BoolFrames}
        capType = typeMap[cap_color_space]
        
        frames = get_framelist(cap, frameList)
        
        if name is None:
            capFrms = capType(frames)
        else:
            capFrms = capType(frames, name = name)
        
        conversionFuncStr = cls.get_converter()
        
        if cap_color_space == cls.get_color():
            convFrms = capFrms
        else:
            convFrms = getattr(capFrms, conversionFuncStr)()
        
        if convFrms.is_empty():
            return convFrms
        
        if scale_factor != 1:
            convFrms = convFrms.rescale(scale_factor)
        
        return convFrms
    
    def __getitem__(self, item):
        framesType = self.__class__
        frameType = framesType.get_constituent()
        if isinstance(item, int):
            return frameType(self.array[item], name = self.name)
        else:
            return framesType(self.array[item], name = self.name)
    
    def __iter__(self):
        framesType = self.__class__
        frameType = framesType.get_constituent()
        return FramesIterator(frameType, self.array, self.name)
    
    def size(self, unit = 'GB'):
        unitMap = {'B': 1, 'KB': 1000, 'MB': 1000 ** 2, 'GB': 1000 ** 3}
        return round(self.array.size / unitMap[unit], 2)
    
    def shape(self):
        return self.array.shape
    
    def length(self):
        return self.shape()[0]
    
    def rows(self):
        try:
            return self.shape()[1]
        except IndexError: ### happens if the Frames Object is empty
            return None
    
    def cols(self):
        try:
            return self.shape()[2]
        except IndexError: ### happens if the Frames Object is empty
            return None
    
    def is_empty(self):
        return self.length() == 0
    
    def rescale(self, scale_factor):
        framesType = self.__class__
        
        width, height = self.cols(), self.rows()
        dim = round(width * scale_factor), round(height * scale_factor)
        
        resArr = np.array([cv.resize(imgArr, dim, interpolation = cv.INTER_AREA) for imgArr in self.array])
        return framesType(resArr, name = self.name)
        
@dataclass(frozen = True)
class BGRFrames(Frames):
    name: str = 'bgr_frames'
    
    def __repr__(self):
        return Frames.__repr__(self)
    
    @classmethod
    def get_constituent(cls):
        return BGRFrame
    
    @classmethod
    def get_color(cls):
        return 'bgr'
    
    @classmethod
    def get_converter(cls):
        return 'to_BGR'
    
    def to_grey(self):
        convArr = np.array([cv.cvtColor(imgArr, cv.COLOR_BGR2GRAY) for imgArr in self.array])
        return GreyFrames(convArr, name = self.name)
    
    def to_HSV(self):
        convArr = np.array([cv.cvtColor(imgArr, cv.COLOR_BGR2HSV) for imgArr in self.array])
        return HSVFrames(convArr, name = self.name)
    
@dataclass(frozen = True)
class HSVFrames(Frames):
    name: str = 'hsv_frames'
    
    def __repr__(self):
        return Frames.__repr__(self)
    
    @classmethod
    def get_constituent(cls):
        return HSVFrame
    
    @classmethod
    def get_color(cls):
        return 'hsv'
    
    @classmethod
    def get_converter(cls):
        return 'to_HSV'
    
    def to_grey(self):
        convArr = np.array([cv.cvtColor(imgArr, cv.COLOR_HSV2GRAY) for imgArr in self.array])
        return GreyFrames(convArr, name = self.name)
    
    def to_BGR(self):
        convArr = np.array([cv.cvtColor(imgArr, cv.COLOR_HSV2BGR) for imgArr in self.array])
        return BGRFrames(convArr, name = self.name)
    
@dataclass(frozen = True)
class GreyFrames(Frames):
    name: str = 'grey_frames'
    
    def __post_init__(self):
        if len(self.array.shape) == 4: ### need to convert to gray...
            convArr = np.array([cv.cvtColor(imgArr, cv.COLOR_BGR2GRAY) for imgArr in self.array])
            object.__setattr__(self, 'array', convArr)
    
    def __repr__(self):
        return Frames.__repr__(self)
    
    @classmethod
    def get_constituent(cls):
        return GreyFrame
    
    @classmethod
    def get_color(cls):
        return 'grey'
    
    @classmethod
    def get_converter(cls):
        return 'to_grey'
    
    def to_BGR(self):
        convArr = np.array([cv.cvtColor(imgArr, cv.COLOR_GRAY2BGR) for imgArr in self.array])
        return BGRFrames(convArr, name = self.name)
    
    def to_HSV(self):
        convArr = np.array([cv.cvtColor(imgArr, cv.COLOR_GRAY2HSV) for imgArr in self.array])
        return HSVFrames(convArr, name = self.name)
        
    # def to_bool(self): 
    #     return BoolFrames(self.array, name = self.name) ### NO!! Does not guarantee a true 0/255 boolean frame
        

@dataclass(frozen = True)
class BoolFrames(Frames):
    name: str = 'bool_frames'
    
    def __post_init__(self):
        if len(self.array.shape) == 4: ### need to convert to gray...
            convArr = np.array([cv.cvtColor(imgArr, cv.COLOR_BGR2GRAY) for imgArr in self.array])
            object.__setattr__(self, 'array', convArr)
            
    def __repr__(self):
        return Frames.__repr__(self)
    
    @classmethod
    def get_constituent(cls):
        return BoolFrame

    @classmethod
    def get_color(cls):
        return 'bool'
    
    @classmethod
    def get_converter(cls):
        return 'to_bool'
    
    def to_grey(self):
        return GreyFrames(self.array, name = self.name)
        
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

    cFrms[0].show()
    hFrms[0].show()
    gFrms[0].show()

    cFrms.to_HSV()[0].show()
