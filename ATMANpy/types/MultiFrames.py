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
from ATMANpy.types.Frames import HSVFrame, BGRFrame, GreyFrame, BoolFrame, MaskFrame

from skvideo.io import FFmpegWriter

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
        
        if isinstance(item, slice):
            return framesType(self.array[item], name = self.name)
        else:
            try:
                len(item)
            except:
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
    
    def join(self, otherFrms):
        framesType = self.__class__
        otherType = otherFrms.__class__
        assert framesType == otherType, f'Attempting to join MultiFrame of {framesType} with MultiFrame of {otherType}'
    
        joinArr = np.concatenate((self.array, otherFrms.array), axis = 0)
        return framesType(joinArr, self.name)
    
    def sum(self, axes):
        return self.array.sum(axes)
    
    def median(self):
        framesType = self.__class__
        frameType = framesType.get_constituent()
        myArr = np.median(self.array, axis = 0).astype('uint8')
        return frameType(myArr)
    
    def mean(self):
        framesType = self.__class__
        frameType = framesType.get_constituent()
        myArr = np.mean(self.array, axis = 0).astype('uint8')
        return frameType(myArr)
    
    def std(self):
        framesType = self.__class__
        frameType = framesType.get_constituent()
        myArr = np.std(self.array, axis = 0).astype('uint8')
        return frameType(myArr)
    
    def mask(self, maskFrms):
        assert isinstance(maskFrms, MaskFrames)
        assert self.length() == maskFrms.length()
        framesType = self.__class__
        
        outArr = np.array([cv.bitwise_and(imgArr, imgArr, mask = maskArr) for imgArr, maskArr in zip(self.array, maskFrms.array)])
        return framesType(outArr, name = self.name)
        
    
    def rescale(self, scale_factor):
        framesType = self.__class__
        
        width, height = self.cols(), self.rows()
        dim = round(width * scale_factor), round(height * scale_factor)
        
        resArr = np.array([cv.resize(imgArr, dim, interpolation = cv.INTER_AREA) for imgArr in self.array])
        return framesType(resArr, name = self.name)
        
    def pad(self, to_height = None, to_width = None):
        framesType = self.__class__
        
        if to_height is None:
            to_height = self.rows()
        if to_width is None:
            to_width = self.cols()
            
        hdiff = to_height - self.rows()
        wdiff = to_width - self.cols()
    
        if hdiff > 0:
            hdiv, hmod = hdiff // 2, hdiff % 2
            top_add = hdiv
            bot_add = hdiv + hmod
        else:
            top_add = 0
            bot_add = 0
        
        if wdiff > 0:
            wdiv, wmod = wdiff // 2, wdiff % 2
            left_add = wdiv
            right_add = wdiv + wmod
        else:
            left_add = 0
            right_add = 0
        
        outArr = np.array([cv.copyMakeBorder(imgArr, 
                                             top_add, bot_add, left_add, right_add, 
                                             cv.BORDER_CONSTANT, value = 0) for imgArr in self.array])
        
        return framesType(outArr, name = self.name)
        
    def warp(self, HArr, cols = None, rows = None):
        assert len(HArr) == self.length(), f'Trying to warp MultiFrame of length {self.length()} with {len(HArr)} Homographies.'
        
        framesType = self.__class__
        
        if cols is None:
            cols = self.cols()
        if rows is None:
            rows = self.rows()
        
        outArr = np.array([cv.warpPerspective(imgArr, H, (cols, rows)) for imgArr, H in zip(self.array, HArr)])
        return framesType(outArr, name = self.name)
    
    def crop(self, cropRegion):
        framesType = self.__class__
        top_left, bot_right = cropRegion.p1, cropRegion.p4
        
        outArr = np.array([imgArr[top_left.y:bot_right.y, top_left.x:bot_right.x] for imgArr in self.array])
        return framesType(outArr, name = self.name)
    
    def zoom(self, zoom_factor, interpolation = cv.INTER_LINEAR):
        framesType = self.__class__
        outArr = np.array([cv.resize(imgArr, None, fx = zoom_factor, fy = zoom_factor, interpolation = interpolation) for imgArr in self.array])
        if framesType.__name__ == 'MaskFrames':
            return GreyFrames(outArr) > 127
        elif framesType.__name__ == 'BoolFrames':
            return BGRFrames(outArr) > 127
        else:
            return framesType(outArr, name = self.name)
        
    def save(self, savePath, saveName, lossless = True):
        if not saveName.endswith('.avi'):
            saveName = saveName + '.avi'
            
        fullPath = os.path.join(savePath, saveName)
        
        if lossless:
            outputdict = {
              '-vcodec': 'libx264',  #use the h.264 codec
              '-crf': '0',           #set the constant rate factor to 0, which is lossless
              '-preset':'veryslow'   #the slower the better compression, in princple, try 
                                     #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
             }
        else:
            outputdict = {}
            
        writer = FFmpegWriter(fullPath, outputdict = outputdict)
        
        rgbArr = np.flip(self.array, 3) ### self.array.shape = T, H, W, C --> need to flip the channels to get RGB.
        for myArr in rgbArr:
            writer.writeFrame(myArr)
        
        writer.close()
    
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
        
@dataclass(frozen = True)
class BoolFrames(Frames):
    name: str = 'bool_frames'
    
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
    
    def to_mask(self):
        return MaskFrames(self.array, name = self.name)
           

@dataclass(frozen = True)
class MaskFrames(Frames):
    name: str = 'mask_frames'
    
    def __post_init__(self):
        if len(self.array.shape) == 4: ### need to convert to gray...
            convArr = np.array([cv.cvtColor(imgArr, cv.COLOR_BGR2GRAY) for imgArr in self.array])
            object.__setattr__(self, 'array', convArr)
            
    def __repr__(self):
        return Frames.__repr__(self)
    
    def __or__(self, other):
        return MaskFrames(cv.bitwise_or(self.array, other.array), name = f'{self.name}_or_{other.name}')

    def __and__(self, other):
        return MaskFrames(cv.bitwise_and(self.array, other.array), name = f'{self.name}_and_{other.name}')
    
    def __invert__(self):
        return MaskFrames(cv.bitwise_not(self.array), name = f'{self.name}_not')
    
    @classmethod
    def get_constituent(cls):
        return MaskFrame

    @classmethod
    def get_color(cls):
        return 'mask'
    
    @classmethod
    def get_converter(cls):
        return 'to_mask'
    
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
