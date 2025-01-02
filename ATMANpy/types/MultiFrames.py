# -*- coding: utf-8 -*-
"""
Created on Sat May 21 22:18:33 2022

@author: gianl
"""


import os
import cv2 as cv
import numpy as np
from dataclasses import dataclass

from ATMANpy.funcs.util_funcs import shift_array
from ATMANpy.funcs.cv_funcs import get_frames, get_framelist
from ATMANpy.types.Contours import Contour, Contours
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
    def from_frames(cls, list_of_multiframes, name = None):
        if name is None:
            name = list_of_multiframes[0].name
        jntArr = np.concatenate([frm.array for frm in list_of_multiframes], axis = 0)
        return cls(jntArr, name = name)

    @classmethod
    def from_single_frames(cls, list_of_frames, name = None):
        if name is None:
            name = list_of_frames[0].name
        jntArr = np.concatenate([frm.array[np.newaxis] for frm in list_of_frames], axis = 0)
        return cls(jntArr, name = name)

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
    
    @classmethod
    def from_frame(cls, myFrm, repetitions):
        conversionFuncStr = cls.get_converter()
        try:
            convFrm = getattr(myFrm, conversionFuncStr)()
        except AttributeError:
            convFrm = myFrm
        convArr = np.repeat(convFrm.array[np.newaxis], repetitions, axis = 0)
        return cls(convArr)

    # @staticmethod       
    # def dispatch_arithmetic(x, y, func, *args, **kwargs):
        # try:
        #     outArr = func(x, y, *args, **kwargs)
        # except TypeError:
        #     outArrs = [func(chImg, chVal, *args, **kwargs) for chImg, chVal in zip(cv.split(x), y)] ### if y is a list: we subtract channels one-by-one
        #     outArr = cv.merge(outArrs)
        # return outArr
        # outArr = func(x, y, *args, **kwargs)
        # return outArr
    
    @staticmethod      
    def get_operant(op):
        try: 
            operant = op.array
        except:
            operant = op
        finally:
            return operant    
        
    def __add__(self, y):
        InstClass = self.__class__
        try: ### first, try with the fast, but capricious cv functions (they can not do broadcasting, for example!)
            outArr = cv.add(self.array, Frames.get_operant(y))
        except:
            fall_back_function = np.add
            outArr = Frames.dispatch_arithmetic(self.array, Frames.get_operant(y), fall_back_function)
        return InstClass(outArr.astype('uint8'))
    
    def __sub__(self, y):
        InstClass = self.__class__
        try:
            outArr = cv.absdiff(self.array, Frames.get_operant(y))
        except:
            def fall_back_function(x, y):
                a = x-y
                b = y-x
                c = x>y
                return a*c + b*(~c)
            outArr = fall_back_function(self.array, Frames.get_operant(y))
        return InstClass(outArr.astype('uint8'))
    
    def __truediv__(self, y):
        InstClass = self.__class__
        try:
            outArr = cv.divide(self.array, Frames.get_operant(y))
        except:
            fall_back_function = np.divide
            outArr = fall_back_function(self.array, Frames.get_operant(y))
        return InstClass(outArr.astype('uint8'))
    
    def __mod__(self, y):
        InstClass = self.__class__
        outArr = np.mod(self.array, Frames.get_operant(y))
        return InstClass(outArr.astype('uint8'))
    
    def __mul__(self, y):
        InstClass = self.__class__
        outArr = self.array.astype(float) * Frames.get_operant(y)
        return InstClass(outArr.astype('uint8'))
    
    def __pow__(self, y):
        InstClass = self.__class__
        outArr = self.array.astype(float) ** y
        return InstClass(outArr.astype('uint8'))

    def __lt__(self, y):
        try:
            outArr = cv.compare(self.array, Frames.get_operant(y), cv.CMP_LT)
        except:
            outArr = self.array < Frames.get_operant(y)
            
        if not self.has_color():
            return MaskFrames(outArr)
        else:
            return BoolFrames(outArr)
        
    def __le__(self, y):
        try:
            outArr = cv.compare(self.array, Frames.get_operant(y), cv.CMP_LE)
        except:
            outArr = self.array <= Frames.get_operant(y)
            
        if not self.has_color():
            return MaskFrames(outArr)
        else:
            return BoolFrames(outArr)
        
    def __gt__(self, y):
        try:
            outArr = cv.compare(self.array, Frames.get_operant(y), cv.CMP_GT)
        except:
            outArr = self.array > Frames.get_operant(y)
            
        if not self.has_color():
            return MaskFrames(outArr)
        else:
            return BoolFrames(outArr)
        
    def __ge__(self, y):
        try:
            outArr = cv.compare(self.array, Frames.get_operant(y), cv.CMP_GE)
        except:
            outArr = self.array >= Frames.get_operant(y)
            
        if not self.has_color():
            return MaskFrames(outArr)
        else:
            return BoolFrames(outArr)
        
    def __eq__(self, y):
        try:
            outArr = cv.compare(self.array, Frames.get_operant(y), cv.CMP_EQ)
        except:
            outArr = self.array == Frames.get_operant(y)
            
        if not self.has_color():
            return MaskFrames(outArr)
        else:
            return BoolFrames(outArr)
       
    
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
    
    def has_color(self):
        return self.shape() == 4
    
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
    
    def right_of(self, aLn):
        outArray = np.zeros((self.rows(), self.cols()), dtype = 'uint8')
        for y in range(self.rows()):
            threshold_x = aLn.weak_x(y)
            first_beyond_threshold = max(int(threshold_x) + 1, 0)
            outArray[y, first_beyond_threshold:] = 255
        outArrays = np.tile(outArray, (self.length(), 1, 1))
        return MaskFrames(outArrays)
        
    def left_of(self, aLn):
        outArray = np.zeros((self.rows(), self.cols()), dtype = 'uint8')
        for y in range(self.rows()):
            threshold_x = aLn.weak_x(y)
            first_beyond_threshold = max(int(threshold_x) + 1, 0)
            outArray[y, :first_beyond_threshold] = 255
        outArrays = np.tile(outArray, (self.length(), 1, 1))
        return MaskFrames(outArrays)
    
    def below(self, aLn):
        outArray = np.zeros((self.rows(), self.cols()), dtype = 'uint8')
        for x in range(self.cols()):
            threshold_y = aLn.weak_y(x)
            first_beyond_threshold = max(int(threshold_y) + 1, 0)
            outArray[first_beyond_threshold:, x] = 255
        outArrays = np.tile(outArray, (self.length(), 1, 1))
        return MaskFrames(outArrays)
            
    def above(self, aLn):
        outArray = np.zeros((self.rows(), self.cols()), dtype = 'uint8')
        for x in range(self.cols()):
            threshold_y = aLn.weak_y(x)
            first_beyond_threshold = max(int(threshold_y) + 1, 0)
            outArray[:first_beyond_threshold, x] = 255
        outArrays = np.tile(outArray, (self.length(), 1, 1))
        return MaskFrames(outArrays)
          
    def within_region(self, aReg):
        l, r, t, b = aReg['Left'], aReg['Right'], aReg['Top'], aReg['Bottom']
        return self.right_of(l) & self.left_of(r) & self.below(t) & self.above(b)
    
    def mask(self, maskFrms):
        assert isinstance(maskFrms, MaskFrames)
        assert self.length() == maskFrms.length()
        framesType = self.__class__
        
        outArr = np.array([cv.bitwise_and(imgArr, imgArr, mask = maskArr) for imgArr, maskArr in zip(self.array, maskFrms.array)])
        return framesType(outArr, name = self.name)
        
    def gaussian_blur(self, kernelSize):
        framesType = self.__class__
        
        resArr = np.array([cv.GaussianBlur(imgArr, kernelSize, 0) for imgArr in self.array])
        return framesType(resArr)
    
    def rescale(self, scale_factor):
        framesType = self.__class__
        
        width, height = self.cols(), self.rows()
        dim = round(width * scale_factor), round(height * scale_factor)
        
        resArr = np.array([cv.resize(imgArr, dim, interpolation = cv.INTER_AREA) for imgArr in self.array])
        return framesType(resArr, name = self.name)
        
    def rename(self, new_name):
        framesType = self.__class__
        return framesType(self.array, name = new_name)
    
    def slow(self, slow_factor: int = 2):
        ### TODO: does not work with slow_factor != 2
        framesType = self.__class__
        framesArr = self.array
        t, other_dims = self.shape()[0], self.shape()[1:]
        
        ### create target array, and fill it in every slow_factor position
        targetArr = np.full((slow_factor * t, *other_dims), fill_value = np.nan)
        targetArr[::slow_factor] = framesArr

        ### TODO: --> all that follows will have to be adapted if we want to make this work with slow_factor != 2
        ### find mean values to fill with
        fore_array = np.roll(framesArr, 1, axis = 0)[1:] ### kill first because of rolling over.
        back_array = np.roll(framesArr, -1, axis = 0)[:-1] ## kill last because of rolling over
        fore_array, back_array = fore_array.astype('float'), back_array.astype('float')
        mean_array = (fore_array + back_array) / 2
        
        ### fill nans with means
        targetArr = targetArr[:-1] ### can not fill last value: we can not extrapolate
        targetArr[1::slow_factor] = mean_array
        targetArr = targetArr.astype('uint8')

        return framesType(targetArr, name = self.name)

    def pad(self, to_height = None, to_width = None, height_how = 'center', width_how = 'center'):
        ### height_how: center, up, down; width_how: center, left, right.
        assert height_how in ['center', 'up', 'down']
        assert width_how in ['center', 'left', 'right']

        framesType = self.__class__
        
        if to_height is None:
            to_height = self.rows()
        if to_width is None:
            to_width = self.cols()
            
        hdiff = to_height - self.rows()
        wdiff = to_width - self.cols()
    
        if hdiff > 0:
            if height_how == 'center':
                hdiv, hmod = hdiff // 2, hdiff % 2
                top_add = hdiv
                bot_add = hdiv + hmod
            elif height_how == 'up':
                top_add = 0
                bot_add = hdiff
            elif height_how == 'down':
                top_add = hdiff
                bot_add = 0
        else:
            top_add = 0
            bot_add = 0
        
        if wdiff > 0:
            if width_how == 'center':
                wdiv, wmod = wdiff // 2, wdiff % 2
                left_add = wdiv
                right_add = wdiv + wmod
            elif width_how == 'left':
                left_add = 0
                right_add = wdiff
            elif width_how == 'right':
                left_add = wdiff
                right_add = 0
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
        
    def shift(self, shift_by, fill_value = np.nan):
        framesType = self.__class__
        
        ##### we always fill by the first or the last frame, depending on the direction of the shift
        ##### this is to preserve uint8 type (avoid nans in array).
        if shift_by < 0:
            fillArr = self[-1].array
        elif shift_by > 0:
            fillArr = self[0].array
        else:
            return self
        
        shiftedArr = shift_array(self.array, shift_by, axis = 0, fill_value = fillArr)
        return framesType(shiftedArr)
        
    def reverse(self):
        framesType = self.__class__
        flippedArr = np.flip(self.array, axis = 0)
        return framesType(flippedArr)

    def open(self, kernelSize, kernelType = cv.MORPH_ELLIPSE):
        framesType = self.__class__
        self_copy = self.array.copy()
        
        kernel = cv.getStructuringElement(kernelType, kernelSize)
        arrs = [cv.morphologyEx(arr, cv.MORPH_OPEN, kernel) for arr in self_copy]
        arr = np.stack(arrs, axis = 0)
        return framesType(arr)
    
    def close(self, kernelSize, kernelType = cv.MORPH_ELLIPSE):
        framesType = self.__class__
        self_copy = self.array.copy()
        
        kernel = cv.getStructuringElement(kernelType, kernelSize)
        arrs = [cv.morphologyEx(arr, cv.MORPH_CLOSE, kernel) for arr in self_copy]
        arr = np.stack(arrs, axis = 0)
        return framesType(arr)
        
    def find_contours(self, contour_retrieval_mode, contour_approx_mode):
        ### retrieval: cv.RETR_TREE
        ### approx: cv.CHAIN_APPROX_SIMPLE
        resList = [cv.findContours(arr, contour_retrieval_mode, contour_approx_mode) for arr in self.array]
        list_of_contour_arrays, list_of_hierarchies = [res[0] for res in resList], [res[1] for res in resList]
        list_of_typed_cts = [Contours([Contour(cnt) for cnt in cnts]) for cnts in list_of_contour_arrays]
        return list_of_typed_cts, list_of_hierarchies
    
    def save(self, savePath, saveName, lossless = True, save_as = 'avi'):
        if not saveName.endswith(f'.{save_as}'):
            saveName = saveName + f'.{save_as}'
            
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
        
        if self.__class__.get_color() == 'bgr':
            saveArr = np.flip(self.array, 3) ### self.array.shape = T, H, W, C --> need to flip the channels to get RGB.
        else:
            saveArr = self.array
            
        for myArr in saveArr:
            writer.writeFrame(myArr)
        
        writer.close()

    def put(self, anyObj, **kwargs):
        framesType = self.__class__
        return framesType.from_single_frames([subFrm.put(anyObj, **kwargs) for subFrm in self])

    
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
    
    def B(self):
        self_copy = self.array.copy()
        return GreyFrames(self_copy[:, :, :, 0], name = 'b_channels')
    
    def G(self):
        self_copy = self.array.copy()
        return GreyFrames(self_copy[:, :, :, 1], name = 'g_channels')
    
    def R(self):
        self_copy = self.array.copy()
        return GreyFrames(self_copy[:, :, :, 2], name = 'r_channels')
    
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
     
    def H(self):
        self_copy = self.array.copy()
        return GreyFrames(self_copy[:, :, :, 0], name = 'h_channels')
    
    def S(self):
        self_copy = self.array.copy()
        return GreyFrames(self_copy[:, :, :, 1], name = 's_channels')
    
    def V(self):
        self_copy = self.array.copy()
        return GreyFrames(self_copy[:, :, :, 2], name = 'v_channels')
     
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
    
    @staticmethod
    def false(like):
        if isinstance(like, tuple):
            shape = like
        else:
            shape = like.shape()
            if len(shape) == 3:
                shape = tuple([*shape, 3])
        return BoolFrames(np.zeros(shape, dtype = 'uint8'), name = like.name)
    
    @staticmethod
    def true(like):
        return ~BoolFrames.false(like)
    
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

    @staticmethod
    def false(like):
        if isinstance(like, tuple):
            shape = like
        else:   ### assumed to be a frame
            shape = like.shape()[:3]
        return MaskFrames(np.zeros(shape, dtype = 'uint8'), name = like.name)
        
    @staticmethod
    def true(like):
        return ~MaskFrames.false(like)
    
    def to_grey(self):
        return GreyFrames(self.array, name = self.name)
    

if __name__ == '__main__':
        
    from ATMANpy.funcs.util_funcs import get_video
    
    ### VIDEO ######
    VIDEO_ID = 'pqxAmUWIoKc'
    cap = get_video(VIDEO_ID)
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
