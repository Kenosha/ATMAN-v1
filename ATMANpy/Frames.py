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
from ATMANpy.Lines import Line, Lines

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
        
        if len(self.array.shape) == 1:
            object.__setattr__(self, 'array', np.array([self.array]))
        #############################  
    
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
        
    def mask(self, boolFrm):
        maskArr = boolFrm.array.astype('uint8') # cv mask does not accept bools
        return GreyFrame(cv.bitwise_and(self.array, self.array, mask = maskArr))
        
    ###############

@dataclass(frozen = True)
class BoolFrame(Frame):
    array: np.ndarray
    name: str = 'frame'
    
    def __post_init__(self):
        ### call post init of super class
        super(BoolFrame, self).__post_init__()

        #### TYPE CHECKS ####
        arraycheck(self, member_type = 'bool', 
                   channel_number = 2, color_channel_length = None)
        #####################

    def __or__(self, other):
        return BoolFrame(np.bitwise_or(self.array, other.array), 
                         name = f"{self.name} or {other.name}")

    def __and__(self, other):
        return BoolFrame(np.bitwise_and(self.array, other.array), 
                         name = f"{self.name} and {other.name}")
    
    def __invert__(self):
        return BoolFrame(np.logical_not(self.array), 
                         name = f"not {self.name}") 

    def to_grey(self):
        return GreyFrame(self.array.astype('uint8') * 255)

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
        
    @dispatch(str)
    def put(self, text: str, where = (50, 50), size = 1, color = 255, thickness = 1):
        self_copy = self.array.copy()
        font = cv.FONT_HERSHEY_SIMPLEX
        return GreyFrame(cv.putText(self_copy, text, where, font, size, color, thickness))
        
    def zero(self):
        return GreyFrame(np.zeros(self.shape, dtype = 'uint8'))
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
    
    @dispatch(str)
    def put(self, text: str, where = (50, 50), size = 1, color = (0, 255, 0), thickness = 1):
        self_copy = self.array.copy()
        font = cv.FONT_HERSHEY_SIMPLEX
        return BGRFrame(cv.putText(self_copy, text, where, font, size, color, thickness))
        
    
    def zero(self):
        return BGRFrame(np.zeros(self.shape[:2], dtype = 'uint8'))
    ###############
    

    