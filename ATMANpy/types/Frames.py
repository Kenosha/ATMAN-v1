# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:30:35 2022

@author: gianl
"""

import os
import cv2 as cv
import numpy as np
from dataclasses import dataclass

from ATMANpy.funcs.type_funcs import typecheck, arraycheck
from ATMANpy.funcs.cv_funcs import show, get_frame    
from ATMANpy.types.Points import Point
from ATMANpy.types.Lines import Line, Lines, Region
from ATMANpy.types.Contours import Contour

from ATMANpy.types.PutDispatcher import put


class FrameIterator:
    def __init__(self, frame):
        self._frame = frame
        self._rowIdx = 0
        self._colIdx = 0
        
    def __next__(self):
        rows, cols = self._frame.rows, self._frame.cols
        if self._colIdx >= cols:
            self._colIdx = 0
            self._rowIdx += 1
        
        if self._rowIdx < rows:
            value = self._frame.array[self._rowIdx, self._colIdx]
            out = Point((self._colIdx, self._rowIdx), value = value)
            self._colIdx += 1
            return out
        else:
            raise StopIteration
        
    
@dataclass(frozen = True)
class Frame:
    array: np.ndarray
    name: str = 'frame'
    
    def __post_init__(self):
        #### TYPE CHECKS ####
        typecheck(self)
        #####################
        
        #### IMMUTABILITY BREAKS ####
        object.__setattr__(self, 'rows', self.array.shape[0])
        object.__setattr__(self, 'cols', self.array.shape[1])
        object.__setattr__(self, 'shape', self.array.shape)
        object.__setattr__(self, 'size', self.rows * self.cols)
        
        if len(self.array.shape) == 1:
            object.__setattr__(self, 'array', np.array([self.array]))
        #############################  
    
    @staticmethod       
    def dispatch_arithmetic(x, y, func, *args, **kwargs):
        try:
            outArr = func(x, y, *args, **kwargs)
        except TypeError:
            outArrs = [func(chImg, chVal, *args, **kwargs) for chImg, chVal in zip(cv.split(x), y)] ### if y is a list: we subtract channels one-by-one
            outArr = cv.merge(outArrs)
        return outArr
    
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
        outArr = Frame.dispatch_arithmetic(self.array, Frame.get_operant(y), cv.add)
        return InstClass(outArr)
    def __sub__(self, y):
        InstClass = self.__class__
        outArr = Frame.dispatch_arithmetic(self.array, Frame.get_operant(y), cv.absdiff)
        return InstClass(outArr)
    def __truediv__(self, y):
        InstClass = self.__class__
        outArr = Frame.dispatch_arithmetic(self.array, Frame.get_operant(y), cv.divide)
        return InstClass(outArr)
    
    def __mod__(self, y):
        InstClass = self.__class__
        outArr = np.mod(self.array, Frame.get_operant(y))
        return InstClass(outArr)
    
    def __mul__(self, y):
        return Frame(self.array.astype(float) * Frame.get_operant(y))
    def __pow__(self, y):
        return Frame(self.array.astype(float) ** y)

    def __lt__(self, y):
        outArr = Frame.dispatch_arithmetic(self.array, Frame.get_operant(y), cv.compare, cv.CMP_LT)
        return BoolFrame(outArr)
    def __le__(self, y):
        outArr = Frame.dispatch_arithmetic(self.array, Frame.get_operant(y), cv.compare, cv.CMP_LE)
        return BoolFrame(outArr)
    def __gt__(self, y):
        outArr = Frame.dispatch_arithmetic(self.array, Frame.get_operant(y), cv.compare, cv.CMP_GT)
        return BoolFrame(outArr)
    def __ge__(self, y):
        outArr = Frame.dispatch_arithmetic(self.array, Frame.get_operant(y), cv.compare, cv.CMP_GE)
        return BoolFrame(outArr)
    def __eq__(self, y):
        outArr = Frame.dispatch_arithmetic(self.array, Frame.get_operant(y), cv.compare, cv.CMP_EQ)
        return BoolFrame(outArr)
    
    def __iter__(self):
        return FrameIterator(self)
    
    ### SHOW METHODS ###
    def show(self):
        show(self.array, name = self.name)

    def unique(self):
        return np.unique(self.array)

    def eval_at(self, point: Point):
        point = point.conform_to(self)
        return self.array[point.y, point.x]
    
    def fast_eval_at(self, point: Point):
        return self.array[point.y, point.x]
    
    def maybe_eval_at(self, point: Point):
        try:
            return self.array[point.y, point.x]
        except IndexError:
            return None
    
    def mean_std(self, mask = None):
        if mask is not None:
            mask = mask.array
        mean, std = cv.meanStdDev(self.array, mask = mask)
        return mean.squeeze().tolist(), std.squeeze().tolist()
    
    def sum(self, axes):
        return self.array.sum(axes)
    
    def clip_lo(self, y):
        InstClass = self.__class__
        return InstClass(cv.max(self.array, Frame.get_operant(y)))
    def clip_hi(self, y):
        InstClass = self.__class__
        return InstClass(cv.min(self.array, Frame.get_operant(y))) 
    
    def mask(self, boolFrm):
        maskArr = boolFrm.array.astype('uint8') # cv mask does not accept bools
        return Frame(cv.bitwise_and(self.array, self.array, mask = maskArr))
    ###############
        
    def right_of(self, aLn: Line):
        outArray = np.zeros((self.rows, self.cols), dtype = 'uint8')
        for y in range(self.rows):
            threshold_x = aLn.weak_x(y)
            first_beyond_threshold = max(int(threshold_x) + 1, 0)
            outArray[y, first_beyond_threshold:] = 255
        return BoolFrame(outArray)
        
    def left_of(self, aLn: Line):
        outArray = np.zeros((self.rows, self.cols), dtype = 'uint8')
        for y in range(self.rows):
            threshold_x = aLn.weak_x(y)
            first_beyond_threshold = max(int(threshold_x) + 1, 0)
            outArray[y, :first_beyond_threshold] = 255
        return BoolFrame(outArray)
    
    def below(self, aLn: Line):
        outArray = np.zeros((self.rows, self.cols), dtype = 'uint8')
        for x in range(self.cols):
            threshold_y = aLn.weak_y(x)
            first_beyond_threshold = max(int(threshold_y) + 1, 0)
            outArray[first_beyond_threshold:, x] = 255
        return BoolFrame(outArray)
            
    def above(self, aLn: Line):
        outArray = np.zeros((self.rows, self.cols), dtype = 'uint8')
        for x in range(self.cols):
            threshold_y = aLn.weak_y(x)
            first_beyond_threshold = max(int(threshold_y) + 1, 0)
            outArray[:first_beyond_threshold, x] = 255
        return BoolFrame(outArray)
          
    def within_region(self, region: Region):
        l, r, t, b = region['Left'], region['Right'], region['Top'], region['Bottom']
        return self.right_of(l) & self.left_of(r) & self.below(t) & self.above(b)
        
    def within_contour(self, contour: Contour):
        zeroArr = np.zeros((self.rows, self.cols), dtype = 'uint8')
        cntMask = cv.fillPoly(zeroArr, [contour.points], 255)
        return BoolFrame(cntMask)
    
    def roll(self, how = 'down', shift = 1, nan_sentinel = np.nan):
        if shift < 0:
            raise Exception('Roll with positive shifts only! Give direction via the how keyword!')
        
        axis = 1 if how in ['left', 'right'] else 0
        shift = - shift if how in ['left', 'up'] else shift
            
        if nan_sentinel == np.nan:
            rArr = self.array.astype(float)
        else:
            rArr = self.array
        rArr = np.roll(rArr, axis = axis, shift = shift) 
        
        if how == 'down':
            rArr[:shift] = nan_sentinel # first {shift} rows have to be set to nan ... 
        elif how == 'up':
            rArr[shift:] = nan_sentinel # last {- shift} rows have to be set to nan (note that shift is negative!)
        elif how == 'right':
            rArr[:, :shift] = nan_sentinel # first {shift} columns have to be set to nan 
        elif how == 'left':
            rArr[:, shift:] = nan_sentinel # last {- shift} columns have to be set to nan (note that shift is negative!)
        else:
            raise Exception('Give down, up, right or left as keywords for how in roll method of frame!')
        
        return Frame(rArr)
               
    def crop(self, cropRegion: Region):
        InstClass = self.__class__
        self_copy = self.array.copy()
        top_left, bot_right = cropRegion.p1, cropRegion.p4
        cropArr = self_copy[top_left.y:bot_right.y, top_left.x:bot_right.x]
        return InstClass(cropArr)
    
    def zoom(self, zoom_factor, interpolation = cv.INTER_LINEAR):
        InstClass = self.__class__
        self_copy = self.array.copy()
        zoomArr = cv.resize(self_copy, None, 
                            fx = zoom_factor, fy = zoom_factor, interpolation = interpolation)
        return InstClass(zoomArr)

    def gaussian_blur(self, kernelSize):
        self_copy = self.array.copy()
        InstClass = self.__class__
        
        blurArr = cv.GaussianBlur(self_copy, kernelSize, 0)
        return InstClass(blurArr)
    
    def find_contours(self, contour_retrieval_mode, contour_approx_mode):
        ### retrieval: cv.RETR_TREE
        ### approx: cv.CHAIN_APPROX_SIMPLE
        self_copy = self.array.copy()
        cts, hierarchy = cv.findContours(self_copy, 
                                         contour_retrieval_mode, contour_approx_mode)
        return cts, hierarchy
        
    def warp(self, H, cols = None, rows = None):
        InstClass = self.__class__
        self_copy = self.array.copy()
        
        if cols is None:
            cols = self.cols
        if rows is None:
            rows = self.rows
        wrpArr = cv.warpPerspective(self_copy, H, (cols, rows))
        return InstClass(wrpArr)
    ###############

@dataclass(frozen = True)
class BoolFrame(Frame):
    ### amended 14.04.22 to contain only 0 and 255, not bools or 0 and 1.
    ### this makes it predictable, and fit for fast cv arithmetic and logical funcs
    
    array: np.ndarray
    name: str = 'frame'
    
    def __post_init__(self):
        ### call post init of super class
        super(BoolFrame, self).__post_init__()

        #### TYPE CHECKS ####
        # arraycheck(self, member_type = 'bool', 
        #            channel_number = 2, color_channel_length = None)
        #####################

    def _check_values(self):
        return set([0, 255]).issubset(self.unique())

    def __or__(self, other):
        return BoolFrame(cv.bitwise_or(self.array, other.array))

    def __and__(self, other):
        return BoolFrame(cv.bitwise_and(self.array, other.array))
    
    def __invert__(self):
        return BoolFrame(cv.bitwise_not(self.array)) 

    # def any(self, axes):
    #     return BoolFrame(np.any(self.array, axis = axes))
    # def all(self, axes):
    #     return BoolFrame(np.all(self.array, axis = axes))
    
    def put(self, anyObj, 
            color = 255, radius = 1, thickness = 1, where = (50, 50), size = 1):
        return put(self.to_grey(), anyObj, 
                   color = color, radius = radius, thickness = thickness, where = where, size = size)
    
    def any_channel(self):
        c1, c2, c3 = cv.split(self.array)
        return BoolFrame(cv.bitwise_or(cv.bitwise_or(c1, c2), c3))
    def all_channels(self):
        c1, c2, c3 = cv.split(self.array)
        return BoolFrame(cv.bitwise_and(cv.bitwise_and(c1, c2), c3))

    # def to_grey(self):
    #     if 255 in self.unique(): ### this is a problem... unify the values of bool frames!
    #         return GreyFrame(self.array.astype('uint8'))
    #     else:
    #         return GreyFrame(self.array.astype('uint8') * 255)

    def to_grey(self):
        return GreyFrame(self.array)

    def zero(self):
        return BoolFrame(np.zeros(self.shape, dtype = 'uint8'))

    def to_false(self):
        return BoolFrame(self.zero().array, name = self.name)
    def to_true(self):
        return ~self.to_false()
    def to_csv(self, savePath, name = None):
        if name == None:
            name = self.name
        np.savetxt(os.path.join(savePath, f'{name}_bool.csv'), self.array, delimiter = ',')
        
    def open(self, kernel):
        self_copy = self.array.copy()
        arr = cv.morphologyEx(self_copy, cv.MORPH_OPEN, kernel)
        return BoolFrame(arr)
        
    def close(self, kernel):
        self_copy = self.array.copy()
        arr = cv.morphologyEx(self_copy, cv.MORPH_CLOSE, kernel)
        return BoolFrame(arr)
    
    def dilate(self, kernel):
        self_copy = self.array.copy()
        arr = cv.dilate(self_copy, kernel)
        return BoolFrame(arr)
    
    def erode(self, kernel):
        self_copy = self.array.copy()
        arr = cv.erode(self_copy, kernel)
        return BoolFrame(arr)
        
        
    def prob_hough(self, 
                   rho = 1,  ### the resolution of parameter r in pixels
                   theta = np.pi/180, ### the resolution of parameter theta in radians
                   min_votes = 100, ### the minimum number of intersections to detect a line,
                   minLineLength = 50,  ### the minimum length of a candidate line 
                   maxLineGap = 10): ### the maximum gap between two points to be considered the same line
        
        hough_lines = cv.HoughLinesP(self.array, rho, theta, min_votes, 
                                     minLineLength = minLineLength, maxLineGap = maxLineGap)
        hough_lines = Lines.from_coordList(hough_lines)
        return hough_lines

    ## override show
    def show(self):
        show(self.array, name = self.name)
        

@dataclass(frozen = True)
class GreyFrame(Frame):
    array: np.ndarray
    name: str = 'frame'
    
    def __post_init__(self):
        ### call post init of super class
        super(GreyFrame, self).__post_init__()
        
        #### TYPE CHECKS ####
        # we can actually deal with a three dim array, by turning it back to Grey
        arraycheck(self, channel_number = 3, color_channel_length = None)
        ########################  
        
        #### IMMUTABILITY BREAKS ####        
        ### convert BGR to Grey
        if len(self.array.shape) == 3:
            object.__setattr__(self, 'array', cv.cvtColor(self.array,  cv.COLOR_BGR2GRAY))
        #############################  
    
    # ### INEQUALITIES ###
    # #for a grey frame, these are defined with respect to integers, and return a boolean frame.
    # def __lt__(self, other: int):
    #     return BoolFrame(self.array < other)
    # def __le__(self, other: int):
    #     return BoolFrame(self.array <= other)
    # def __gt__(self, other: int):
    #     return BoolFrame(self.array > other)
    # def __ge__(self, other: int):
    #     return BoolFrame(self.array >= other)
    #####################
    
    ### CONSTRUCTORS ###
    @staticmethod
    def from_video(cap, frame_num, name = 'frame'):
        return GreyFrame(get_frame(cap, frame_num))
    ####################
    
    ### TRANSFORMS ###
    def to_BGR(self):
        self_copy = self.array.copy() # to preverve immutability of frame
        return BGRFrame(cv.cvtColor(self_copy, cv.COLOR_GRAY2BGR), name = self.name)
    
    def to_HSV(self):
        self_copy = self.array.copy() # to preverve immutability of frame
        return HSVFrame(cv.cvtColor(self_copy, cv.COLOR_GRAY2HSV), name = self.name)
    
    def to_csv(self, savePath, name = None):
        if name is None:
            name = self.name
        np.savetxt(os.path.join(savePath, f'{name}_grey.csv'), self.array, delimiter = ',')
    ##################     
    
    def put(self, anyObj, 
            color = 255, radius = 1, thickness = 1, where = (50, 50), size = 1):
        return put(self, anyObj, 
                   color = color, radius = radius, thickness = thickness, where = where, size = size)
        
    def zero(self):
        return GreyFrame(np.zeros(self.shape, dtype = 'uint8'))
    
    def mask(self, boolFrm):
        maskArr = boolFrm.array.astype('uint8') # cv mask does not accept bools
        return GreyFrame(cv.bitwise_and(self.array, self.array, mask = maskArr))
    
    def sobel(self):
        self_copy = self.array.copy()
        grad_x = cv.Sobel(self_copy, cv.CV_64F, 1, 0)
        grad_y = cv.Sobel(self_copy, cv.CV_64F, 0, 1)
        grad = np.sqrt(grad_x**2 + grad_y**2)
        grad_norm = (grad * 255 / grad.max()).astype(np.uint8)
        return GreyFrame(grad_norm)
    
    def sobel_dirs(self, toward_high_intensity = True):
        ### how: atan or atan2
        # - atan: calculate arctan of quotient, losing info if x is negative
        # - atan2: two argument arctan.
        self_copy = self.array.copy()
        grad_x = cv.Sobel(self_copy, cv.CV_64F, 1, 0)
        grad_y = cv.Sobel(self_copy, cv.CV_64F, 0, 1)
        theta = np.arctan2(grad_y, grad_x)
        
        ### by default, sobel gradients point toward high intensity regions
        # we can also turn this around if we want
        if not toward_high_intensity:
            theta = theta + np.pi
        
        return theta
    
    def open(self, kernel):
        self_copy = self.array.copy()
        arr = cv.morphologyEx(self_copy, cv.MORPH_OPEN, kernel)
        return GreyFrame(arr)
        
    def close(self, kernel):
        self_copy = self.array.copy()
        arr = cv.morphologyEx(self_copy, cv.MORPH_CLOSE, kernel)
        return GreyFrame(arr)
    
    def clopen(self, close_kernel, open_kernel):
        self_copy = self.array.copy()
        arr = cv.morphologyEx(self_copy, cv.MORPH_CLOSE, close_kernel)
        arr = cv.morphologyEx(self_copy, cv.MORPH_OPEN, open_kernel)
        return GreyFrame(arr)

    def opse(self, open_kernel, close_kernel):
        self_copy = self.array.copy()
        arr = cv.morphologyEx(self_copy, cv.MORPH_OPEN, open_kernel)
        arr = cv.morphologyEx(self_copy, cv.MORPH_CLOSE, close_kernel)
        return GreyFrame(arr)
    
    def connected_regions(self, size_threshold = 0):
        SCREEN_SIZE_CONSTANT = 0.95 ### sometime, the entire screen is chosen as region...
                                    ## if a region is larger than SCREEN_SIZE_CONSTANT of the screen, 
                                    ## we remove it.
        
        num, frm, boxes, centrs = cv.connectedComponentsWithStats(self.array)
        
        regionList = []
        for box, centr in zip(boxes, centrs):
            t_x, t_y, w, h, a = box
            
            if a > SCREEN_SIZE_CONSTANT * self.size: 
                continue
            
            top_left = Point((t_x, t_y))
            bot_right = top_left.right(w - 1).down(h - 1) #note: if w is 1 pixel, then top_left = bot_right!
            myRegion = Region.from_points(top_left, bot_right, 
                                          area = a, centroid = Point(centr))
            if a > size_threshold:
                regionList.append(myRegion)
                
        return regionList
    ###############

@dataclass(frozen = True)
class BGRFrame(Frame):
    array: np.ndarray
    name: str = 'frame'
    
    def __post_init__(self):
        ### call post init of super class
        super(BGRFrame, self).__post_init__()
        
        #### TYPE CHECKS ####
        arraycheck(self, channel_number = 3, color_channel_length = 3)
        #####################   
        
        #### IMMUTABILITY BREAKS ####
        # convert grayscale to a grey BGR
        if len(self.array.shape) == 2:
            object.__setattr__(self, 'array', cv.cvtColor(self.array,  cv.COLOR_GRAY2BGR))
        #############################  
    
    ### CONSTRUCTORS ###
    @staticmethod
    def from_video(cap, frame_num, name = 'frame'):
        return BGRFrame(get_frame(cap, frame_num))
    
    @staticmethod
    def from_csv(readpath, name = 'frame'):
        colMap = {0: 'blue', 1: 'green', 2: 'red'}
        arrList = []
        for idx, col in colMap.items():
            arr = np.genfromtxt(os.path.join(readpath, f'{name}_{col}.csv'), delimiter = ',', dtype = 'uint8')
            arrList.append(arr)
        arr = np.stack(arrList, axis = -1)
        return BGRFrame(arr, name = name)
    ####################
    
    ### TRANSFORMS ###
    def to_grey(self):
        self_copy = self.array.copy() # to preverve immutability of frame
        return GreyFrame(cv.cvtColor(self_copy, cv.COLOR_BGR2GRAY), name = self.name)
    
    def to_HSV(self):
        self_copy = self.array.copy() # to preverve immutability of frame
        return HSVFrame(cv.cvtColor(self_copy, cv.COLOR_BGR2HSV), name = self.name)
    
    def to_csv(self, savePath, name = None):
        if name is None:
            name = self.name
            
        colMap = {0: 'blue', 1: 'green', 2: 'red'}
        for idx, col in colMap.items():
            np.savetxt(os.path.join(savePath, f'{name}_{col}.csv'), self.array, delimiter = ',')
    ##################     
    
    def put(self, anyObj, 
            color = (0, 255, 0), radius = 1, thickness = 1, where = (50, 50), size = 1):
        return put(self, anyObj, 
                   color = color, radius = radius, thickness = thickness, where = where, size = size)
        
    def zero(self):
        return BGRFrame(np.zeros(self.shape[:2], dtype = 'uint8'))
    
    def mask(self, boolFrm):
        maskArr = boolFrm.array.astype('uint8') # cv mask does not accept bools
        return BGRFrame(cv.bitwise_and(self.array, self.array, mask = maskArr))
    
    ###############
    

@dataclass(frozen = True)
class HSVFrame(Frame):
    array: np.ndarray
    name: str = 'frame'
    
    def __post_init__(self):
        ### call post init of super class
        super(HSVFrame, self).__post_init__()
        
        #### TYPE CHECKS ####
        arraycheck(self, channel_number = 3, color_channel_length = 3)
        #####################   
        
        #### IMMUTABILITY BREAKS ####
        # convert grayscale to a grey HSV
        if len(self.array.shape) == 2:
            object.__setattr__(self, 'array', cv.cvtColor(self.array,  cv.COLOR_GRAY2HSV))
        #############################  
    
    ### CONSTRUCTORS ###
    @staticmethod
    def from_video(cap, frame_num, name = 'frame'):
        return BGRFrame(get_frame(cap, frame_num)).to_HSV()
    
    @staticmethod
    def from_csv(readpath, name = 'frame'):
        colMap = {0: 'hue', 1: 'saturation', 2: 'value'}
        arrList = []
        for idx, col in colMap.items():
            arr = np.genfromtxt(os.path.join(readpath, f'{name}_{col}.csv'), delimiter = ',', dtype = 'uint8')
            arrList.append(arr)
        arr = np.stack(arrList, axis = -1)
        return HSVFrame(arr, name = name)
    ####################
    
    ### TRANSFORMS ###
    def H(self):
        self_copy = self.array.copy() # to preverve immutability of frame
        return GreyFrame(self_copy[:, :, 0], name = self.name)
    
    def S(self):
        self_copy = self.array.copy() # to preverve immutability of frame
        return GreyFrame(self_copy[:, :, 1], name = self.name)
    
    def V(self):
        self_copy = self.array.copy() # to preverve immutability of frame
        return GreyFrame(self_copy[:, :, 2], name = self.name)
    
    def to_grey(self):
        return self.V()
    
    def to_BGR(self):
        self_copy = self.array.copy() # to preverve immutability of frame
        return BGRFrame(cv.cvtColor(self_copy, cv.COLOR_HSV2BGR), name = self.name)
    
    def to_csv(self, savePath, name = None):
        if name is None:
            name = self.name
            
        colMap = {0: 'hue', 1: 'saturation', 2: 'val'}
        for idx, col in colMap.items():
            np.savetxt(os.path.join(savePath, f'{name}_{col}.csv'), self.array, delimiter = ',')
    ##################     
    
    def put(self, anyObj, 
            color = (0, 255, 0), radius = 1, thickness = 1, where = (50, 50), size = 1):
        return put(self, anyObj, 
                   color = color, radius = radius, thickness = thickness, where = where, size = size)
        
    def zero(self):
        return HSVFrame(np.zeros(self.shape[:2], dtype = 'uint8'))
    
    def mask(self, boolFrm):
        maskArr = boolFrm.array.astype('uint8') # cv mask does not accept bools
        return HSVFrame(cv.bitwise_and(self.array, self.array, mask = maskArr))
        
    ###############
    










