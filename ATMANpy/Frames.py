# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:30:35 2022

@author: gianl
"""

import cv2 as cv
import numpy as np
from dataclasses import dataclass

from ATMANpy.type_funcs import typecheck, arraycheck
from ATMANpy.cv_funcs import show, get_frame    
from ATMANpy.Points import Point
from ATMANpy.Lines import Line, Lines, Region

from multipledispatch import dispatch

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
    def get_operant(op):
        try: 
            operant = op.array
        except:
            operant = op
        finally:
            return operant
            
    def __add__(self, y):
        return Frame(self.array + Frame.get_operant(y))
    def __sub__(self, y):
        return Frame(self.array - Frame.get_operant(y))
    def __mul__(self, y):
        return Frame(self.array * Frame.get_operant(y))
    def __truediv__(self, y):
        return Frame(self.array / Frame.get_operant(y))
    
    def __pow__(self, y):
        return Frame(self.array ** y)
    def __abs__(self):
        return Frame(np.abs(self.array))

    def __lt__(self, y):
        return BoolFrame(self.array < Frame.get_operant(y))
    def __le__(self, y):
        return BoolFrame(self.array <= Frame.get_operant(y))
    def __gt__(self, y):
        return BoolFrame(self.array > Frame.get_operant(y))
    def __ge__(self, y):
        return BoolFrame(self.array >= Frame.get_operant(y))

    def __iter__(self):
        return FrameIterator(self)
    
    ### METHODS ###
    def show(self):
        show(self.array, name = self.name)

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
        
    def sum(self, axes):
        return self.array.sum(axes)
    
    def mask(self, boolFrm):
        maskArr = boolFrm.array.astype('uint8') # cv mask does not accept bools
        return Frame(cv.bitwise_and(self.array, self.array, mask = maskArr))
    ###############
        
    def right_of(self, aLn: Line):
        outArray = np.zeros((self.rows, self.cols), dtype = 'bool')
        for y in range(self.rows):
            threshold_x = aLn.weak_x(y)
            first_beyond_threshold = max(int(threshold_x) + 1, 0)
            outArray[y, first_beyond_threshold:] = True
        return BoolFrame(outArray)
        
    def left_of(self, aLn: Line):
        outArray = np.zeros((self.rows, self.cols), dtype = 'bool')
        for y in range(self.rows):
            threshold_x = aLn.weak_x(y)
            first_beyond_threshold = max(int(threshold_x) + 1, 0)
            outArray[y, :first_beyond_threshold] = True
        return BoolFrame(outArray)
    
    def below(self, aLn: Line):
        outArray = np.zeros((self.rows, self.cols), dtype = 'bool')
        for x in range(self.cols):
            threshold_y = aLn.weak_y(x)
            first_beyond_threshold = max(int(threshold_y) + 1, 0)
            outArray[first_beyond_threshold:, x] = True
        return BoolFrame(outArray)
            
    def above(self, aLn: Line):
        outArray = np.zeros((self.rows, self.cols), dtype = 'bool')
        for x in range(self.cols):
            threshold_y = aLn.weak_y(x)
            first_beyond_threshold = max(int(threshold_y) + 1, 0)
            outArray[:first_beyond_threshold, x] = True
        return BoolFrame(outArray)
          
    def within(self, region: Region):
        l, r, t, b = region['Left'], region['Right'], region['Top'], region['Bottom']
        return self.right_of(l) & self.left_of(r) & self.below(t) & self.above(b)
        
    def roll(self, how = 'down', shift = 1):
        if shift < 0:
            raise Exception('Roll with positive shifts only! Give direction via the how keyword!')
        
        axis = 1 if how in ['left', 'right'] else 0
        shift = - shift if how in ['left', 'up'] else shift
            
        floatArr = self.array.astype(float)
        rArr = np.roll(floatArr, axis = axis, shift = shift) 
        
        if how == 'down':
            rArr[:shift] = np.nan # first {shift} rows have to be set to nan 
        elif how == 'up':
            rArr[shift:] = np.nan # last {- shift} rows have to be set to nan (note that shift is negative!)
        elif how == 'right':
            rArr[:, :shift] = np.nan # first {shift} columns have to be set to nan 
        elif how == 'left':
            rArr[:, shift:] = np.nan # last {- shift} columns have to be set to nan (note that shift is negative!)
        else:
            raise Exception('Give down, up, right or left as keywords for how in roll method of frame!')
        
        return Frame(rArr)
            
            
    ###############

@dataclass(frozen = True)
class BoolFrame(Frame):
    array: np.ndarray
    name: str = 'frame'
    
    def __post_init__(self):
        ### call post init of super class
        super(BoolFrame, self).__post_init__()

        #### TYPE CHECKS ####
        # arraycheck(self, member_type = 'bool', 
        #            channel_number = 2, color_channel_length = None)
        #####################

    def __or__(self, other):
        return BoolFrame(np.bitwise_or(self.array, other.array))

    def __and__(self, other):
        return BoolFrame(np.bitwise_and(self.array, other.array))
    
    def __invert__(self):
        return BoolFrame(np.logical_not(self.array)) 


    def any(self, axes):
        return BoolFrame(np.any(self.array, axis = axes))
    def all(self, axes):
        return BoolFrame(np.any(self.array, axis = axes))

    def to_grey(self):
        return GreyFrame(self.array.astype('uint8') * 255)
    def to_false(self):
        return BoolFrame(self.to_grey().zero().array.astype('bool'), name = self.name)
    def to_true(self):
        return ~self.to_false()
        
    def prob_hough(self, 
                   rho = 1,  ### the resolution of parameter r in pixels
                   theta = np.pi/180, ### the resolution of parameter theta in radians
                   min_votes = 100, ### the minimum number of intersections to detect a line,
                   minLineLength = 50,  ### the minimum length of a candidate line 
                   maxLineGap = 10): ### the maximum gap between two points to be considered the same line
        
        hough_lines = cv.HoughLinesP(self.to_grey().array, rho, theta, min_votes, 
                                     minLineLength = minLineLength, maxLineGap = maxLineGap)
        hough_lines = Lines.from_coordList(hough_lines)
        return hough_lines

    ## override show
    def show(self):
        show(self.to_grey().array, name = self.name)

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
    
    ### INEQUALITIES ###
    #for a grey frame, these are defined with respect to integers, and return a boolean frame.
    def __lt__(self, other: int):
        return BoolFrame(self.array < other)
    def __le__(self, other: int):
        return BoolFrame(self.array <= other)
    def __gt__(self, other: int):
        return BoolFrame(self.array > other)
    def __ge__(self, other: int):
        return BoolFrame(self.array >= other)
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
    ##################     
    
    ### METHODS ###
    @dispatch(Point)
    def put(self, point: Point, color = 255, radius = 0, thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        # conf_point = point.conform_to(self)
        return GreyFrame(cv.circle(self_copy, point.round_coords(), radius, color, thickness))
    
    @dispatch(Line)
    def put(self, line: Line, color = 255, thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        # conf_point = point.conform_to(self)
        c1, c2 = line.leftPt.round_coords(), line.rightPt.round_coords()
        return GreyFrame(cv.line(self_copy, c1, c2, color, thickness))
    
    @dispatch(Lines)
    def put(self, lines: Lines, color = 255, thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        for line in lines:
            c1, c2 = line.leftPt.round_coords(), line.rightPt.round_coords()
            self_copy = cv.line(self_copy, c1, c2, color, thickness)
        return GreyFrame(self_copy)
        
    @dispatch(Region)
    ### maybe at some point fill the region in or something....
    def put(self, region: Region, color = 255, thickness = 1):
        return self.put(Lines(region.lineList), color = color, thickness = thickness)
    
    @dispatch(str)
    def put(self, text: str, where = (50, 50), size = 1, color = 255, thickness = 1):
        self_copy = self.array.copy()
        font = cv.FONT_HERSHEY_SIMPLEX
        return GreyFrame(cv.putText(self_copy, text, where, font, size, color, thickness))
        
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
    
    def open(self, kernel):
        self_copy = self.array.copy()
        arr = cv.morphologyEx(self_copy, cv.MORPH_OPEN, kernel)
        return GreyFrame(arr)
        
    def close(self, kernel):
        self_copy = self.array.copy()
        arr = cv.morphologyEx(self_copy, cv.MORPH_CLOSE, kernel)
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
    ####################
    
    ### TRANSFORMS ###
    def to_grey(self):
        self_copy = self.array.copy() # to preverve immutability of frame
        return GreyFrame(cv.cvtColor(self_copy, cv.COLOR_BGR2GRAY), name = self.name)
    
    def to_HSV(self):
        self_copy = self.array.copy() # to preverve immutability of frame
        return HSVFrame(cv.cvtColor(self_copy, cv.COLOR_BGR2HSV), name = self.name)
    ##################     
    
    ### METHODS ###
    @dispatch(Point)
    def put(self, point: Point, color = (0, 255, 0), radius = 0, thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        # conf_point = point.conform_to(self)
        return BGRFrame(cv.circle(self_copy, point.round_coords(), radius, color, thickness))
    
    @dispatch(Line)
    def put(self, line: Line, color = (0, 255, 0), thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        # conf_point = point.conform_to(self)
        c1, c2 = line.leftPt.round_coords(), line.rightPt.round_coords()
        return BGRFrame(cv.line(self_copy, c1, c2, color, thickness))
    
    @dispatch(Lines)
    def put(self, lines: Lines, color = (0, 255, 0), thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        for line in lines:
            c1, c2 = line.leftPt.round_coords(), line.rightPt.round_coords()
            self_copy = cv.line(self_copy, c1, c2, color, thickness)
        return BGRFrame(self_copy)
    
    @dispatch(Region)
    ### maybe at some point fill the region in or something....
    def put(self, region: Region, color = (0, 255, 0), thickness = 1):
        return self.put(Lines(region.lineList), color = color, thickness = thickness)
    
    @dispatch(str)
    def put(self, text: str, where = (50, 50), size = 1, color = (0, 255, 0), thickness = 1):
        self_copy = self.array.copy()
        font = cv.FONT_HERSHEY_SIMPLEX
        return BGRFrame(cv.putText(self_copy, text, where, font, size, color, thickness))
        
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
        return HSVFrame(get_frame(cap, frame_num))
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
    
    def to_RGB(self):
        self_copy = self.array.copy() # to preverve immutability of frame
        return BGRFrame(cv.cvtColor(self_copy, cv.COLOR_HSV2BGR), name = self.name)
    ##################     
    
    ### METHODS ###
    @dispatch(Point)
    def put(self, point: Point, color = (120, 100, 100), radius = 0, thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        # conf_point = point.conform_to(self)
        return HSVFrame(cv.circle(self_copy, point.round_coords(), radius, color, thickness))
    
    @dispatch(Line)
    def put(self, line: Line, color = (120, 100, 100), thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        # conf_point = point.conform_to(self)
        c1, c2 = line.leftPt.round_coords(), line.rightPt.round_coords()
        return HSVFrame(cv.line(self_copy, c1, c2, color, thickness))
    
    @dispatch(Lines)
    def put(self, lines: Lines, color = (120, 100, 100), thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        for line in lines:
            c1, c2 = line.leftPt.round_coords(), line.rightPt.round_coords()
            self_copy = cv.line(self_copy, c1, c2, color, thickness)
        return HSVFrame(self_copy)
    
    @dispatch(Region)
    ### maybe at some point fill the region in or something....
    def put(self, region: Region, color = (120, 100, 100), thickness = 1):
        return self.put(Lines(region.lineList), color = color, thickness = thickness)
    
    @dispatch(str)
    def put(self, text: str, where = (50, 50), size = 1, color = (120, 100, 100), thickness = 1):
        self_copy = self.array.copy()
        font = cv.FONT_HERSHEY_SIMPLEX
        return HSVFrame(cv.putText(self_copy, text, where, font, size, color, thickness))
        
    def zero(self):
        return HSVFrame(np.zeros(self.shape[:2], dtype = 'uint8'))
    
    def mask(self, boolFrm):
        maskArr = boolFrm.array.astype('uint8') # cv mask does not accept bools
        return HSVFrame(cv.bitwise_and(self.array, self.array, mask = maskArr))
    ###############
    










