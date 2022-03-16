# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:30:35 2022

@author: gianl
"""

import cv2 as cv
import numpy as np
from dataclasses import dataclass

from ATMANpy.type_funcs import typecheck, arraycheck, pointcheck
from ATMANpy.cv_funcs import show, get_frame    

from typing import Optional, List, Union
from multipledispatch import dispatch

@dataclass(frozen = True)
class Point:
    """
    A point: tuple of integers, signifying x and y coordinates.
    """   
    coords: tuple
    value: Optional[None] = None
    
    def __post_init__(self):
        #### IMMUTABILITY BREAKS ####
        object.__setattr__(self, 'x', self.coords[0])
        object.__setattr__(self, 'y', self.coords[1])
        #############################  
        
    ### METHODS ###
    def up(self, points):
        return Point((self.x, max(self.y - points, 0)))

    def down(self, points):
        return Point((self.x, self.y + points))

    def left(self, points):
        return Point((max(self.x - points, 0), self.y))

    def right(self, points):
        return Point((self.x + points, self.y))
    
    def conform_to(self, image):
        # force the point to be in bounds
        corr_x = min(max(0, self.x), image.cols - 1)
        corr_y = min(max(0, self.y), image.rows - 1)
        return Point((corr_x, corr_y))
    
    def is_left(self, inObj):
        if isinstance(inObj, Point):
            comp = inObj.x
        else:
            comp = inObj
        return self.x < comp
    
    def is_right(self, inObj):
        if isinstance(inObj, Point):
            comp = inObj.x
        else:
            comp = inObj
        return self.x > comp
    
    def is_above(self, inObj):
        if isinstance(inObj, Point):
            comp = inObj.y
        else:
            comp = inObj
        return self.y < comp
    
    def is_below(self, inObj):
        if isinstance(inObj, Point):
            comp = inObj.y
        else:
            comp = inObj
        return self.y > comp
    

    
    
    
@dataclass(frozen = True)
class SafePoint(Point):
    """
    A typechecked point
    """
    def __post_init__(self):
        #### TYPE CHECKS ####
        typecheck(self)
        pointcheck(self)
        
        super(SafePoint, self).__post_init__()
        #####################  

@dataclass(frozen = True)
class Line:
    """
    A line between two points
    """
    leftPt: Point
    rightPt: Point
    
    def __post_init__(self):
        
        ### make it so that lines always go from left to right. 
        # If there is a tie, they go from up to down.
        _reinst = False
        l, r = self.leftPt, self.rightPt
        if not l.is_left(r):
            if not l.is_right(r):
                ### same x coord!
                if l.is_below(r):
                    _reinst = True
            else:
                _reinst = True
        if _reinst:
            object.__setattr__(self, 'leftPt', r)
            object.__setattr__(self, 'rightPt', l)
            
    
    ### CONSTRUCTORS ###
    @staticmethod
    def from_coords(four_coords):
        x1, y1, x2, y2 = four_coords
        return Line(Point((x1, y1)), Point((x2, y2)))
    
    ### METHODS ### 
    def append(self, other):
        if isinstance(other, Line):
            return Lines([self, other])
        elif isinstance(other, Lines):
            return Lines([self, *other.lineList])
        
    def left_point(self):
        if self.leftPt.is_left(self.rightPt):
            return self.leftPt
        else:
            return self.rightPt
    
    def right_point(self):
        if self.leftPt.is_right(self.rightPt):
            return self.leftPt
        else:
            return self.rightPt
        
    def low_point(self):
        if self.leftPt.is_below(self.rightPt):
            return self.leftPt
        else:
            return self.rightPt
    
    def high_point(self):
        if self.leftPt.is_above(self.rightPt):
            return self.leftPt
        else:
            return self.rightPt
            
    def is_horizontal(self):
        return abs(self.rightPt.x - self.leftPt.x) > abs(self.rightPt.y - self.leftPt.y)
    
    def is_weakly_left(self, x_of_int: int):
        return self.left_point().is_left(x_of_int)
    def is_weakly_right(self, x_of_int: int):
        return self.right_point().is_right(x_of_int)
    def is_weakly_above(self, y_of_int: int):
        return self.high_point().is_above(y_of_int)
    def is_weakly_below(self, y_of_int: int):
        return self.low_point().is_below(y_of_int)
    
    def is_strictly_left(self, x_of_int: int):
        return self.right_point().is_left(x_of_int)
    def is_strictly_right(self, x_of_int: int):
        return self.left_point().is_right(x_of_int)
    def is_strictly_above(self, y_of_int: int):
        return self.low_point().is_above(y_of_int)
    def is_strictly_below(self, y_of_int: int):
        return self.high_point().is_below(y_of_int)
    
    # note that as opposed to the points (with the exception of points on the line), 
    # the complement of lines to the left of x is not all lines to the right of x, due to the OR! 
    # hence we need meaningful separate methods for is_right and is_below

    def slope(self):
        try:
            slp = (self.rightPt.y - self.leftPt.y) / (self.rightPt.x - self.leftPt.x)
        except ZeroDivisionError:
            slp = None
        finally:
            return slp
            
    def intercept(self):
        slp = self.slope()
        if slp is None:
            return None
        else:
            return self.leftPt.y - slp * self.leftPt.x
    
    def weak_intersect(self, bLn, to_int = True):
        """
        Intersect lines self and bLn, allowing for infinite extension of lines.
        The only time this will NOT yield a result is for parallel lines.       
        We do not distinguish between parallel identical and parallel non-identical lines
        If you need it, implement it.
        """
        slp1, int1 = self.slope(), self.intercept()
        slp2, int2 = bLn.slope(), bLn.intercept()
        
        if slp1 is None and slp2 is None: ### vertical, parallel lines
            return None 
        elif slp1 is None:
            x_int = self.x
            y_int = slp2 * self.x + int2
        elif slp2 is None:
            x_int = bLn.x
            y_int = slp1 * bLn.x + int1
        else:
            if slp1 == slp2:
                return None
            else:
                x_int = (int2 - int1) / (slp1 - slp2)
                y_int = slp1 * x_int + int1
                
        if to_int:
            x_int, y_int = round(x_int), round(y_int)
        return Point((x_int, y_int))
            

@dataclass(frozen = True)
class Lines:
    """
    A list of lines, sortable and displayable
    """
    lineList: List[Line]
    name: str = 'Lines'
    
    ### CONSTRUCTORS ###
    @staticmethod
    def from_coordList(coordList):
        coordList = np.squeeze(coordList) ## hough lines sometimes have an extra dim...
        return Lines([Line.from_coords(l) for l in coordList])

    def __iter__(self):
        return iter(self.lineList)
    
    def __getitem__(self, item):
        elements = self.lineList[item]
        if isinstance(elements, List):
            return Lines(elements)
        else:
            return elements ### a single line

    def __len__(self):
        return len(self.lineList)

    #### METHODS ###
    def append(self, other):
        if isinstance(other, Line):
            return Lines([*self.lineList, other])
        elif isinstance(other, Lines):
            return Lines([*self.lineList, *other.lineList])
    
    def sort(self, which_point, primary_direction, secondary_direction):
        ### do we sort by left point, or right point or high point or low point?
        ### t2b, b2t, l2r or r2l?
        
        ### supply a key which maps element to sorting order
        def key_func(aLine):
            relevantPt = getattr(aLine, which_point)()
            
            if primary_direction in ['l2r', 'r2l']:
                primary_key = relevantPt.x
            else:
                primary_key = relevantPt.y
                
            if primary_direction in ['r2l', 'b2t']:
                primary_key = - primary_key # in this case, a higher coordinate is better, so we switch sign
            
            if secondary_direction in ['l2r', 'r2l']:
                secondary_key = relevantPt.x
            else:
                secondary_key = relevantPt.y
                
            if secondary_direction in ['r2l', 'b2t']:
                secondary_key = - secondary_key # in this case, a higher coordinate is better, so we switch sign
            
            return (primary_key, secondary_key)
        
        return Lines(sorted(self.lineList, key = key_func))
        
    def merge(self, mergeFunc, flushedLines = None):
        """
        This method recursively applies a mergeFunc to merge Lines according to 
        the binary comparison done in mergeFunc.
        
        The mergeFunc takes two lines and returns a Lines object of length 1 or 2, 
        depending on whether the lines where assessed to have to be merged or not.        
        """
        if flushedLines is None:
            print(f"{self.name}: {len(self)} before merge...")
            flushedLines = Lines([]) ### these are the lines that have 'survived' the merge
            
        if len(self) <= 1:
            outLines = flushedLines.append(self)
            print(f"... and {len(outLines)} after merge!")
            return outLines
        else:
            l1, l2, ls = self[0], self[1], self[2:]
            mergedLines = mergeFunc(l1, l2)
            if len(mergedLines) == 1:
                newLines = mergedLines.append(ls)
                return newLines.merge(mergeFunc, flushedLines = flushedLines)
            else:
                newLines = l2.append(ls)
                return newLines.merge(mergeFunc, flushedLines = flushedLines.append(l1))
    
    def filt(self, filtFunc):
        print(f"{self.name}: {len(self)} before filter...")
        filtList = list(filter(filtFunc, self.lineList))
        print(f"... and {len(filtList)} after filter!")
        try:
            filtName = filtFunc.__str__().split(' ')[1]
        except:
            filtName = 'Unknown Function'
        return Lines(filtList, name = f'{self.name}, filtered by {filtName}')
    
    def highest(self):
        extremum = self[0].high_point()
        for line in self:
            extremum_candidate = line.high_point()
            if extremum_candidate.is_above(extremum):
                extremum = extremum_candidate
        return extremum
    
    def lowest(self):
        extremum = self[0].low_point()
        for line in self:
            extremum_candidate = line.low_point()
            if extremum_candidate.is_below(extremum):
                extremum = extremum_candidate
        return extremum

    def leftest(self):
        extremum = self[0].left_point()
        for line in self:
            extremum_candidate = line.left_point()
            if extremum_candidate.is_left(extremum):
                extremum = extremum_candidate
        return extremum
    
    def rightest(self):
        extremum = self[0].right_point()
        for line in self:
            extremum_candidate = line.right_point()
            if extremum_candidate.is_right(extremum):
                extremum = extremum_candidate
        return extremum
    


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
        return BoolFrame(np.bitwise_or(self.array, other.array), 
                         name = f"{self.name} and {other.name}")

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
        return GreyFrame(cv.circle(self_copy, point.coords, radius, color, thickness))
    
    @dispatch(Line)
    def put(self, line: Line, color = 255, thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        # conf_point = point.conform_to(self)
        c1, c2 = line.leftPt.coords, line.rightPt.coords
        return GreyFrame(cv.line(self_copy, c1, c2, color, thickness))
    
    @dispatch(Lines)
    def put(self, lines: Lines, color = 255, thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        for line in lines:
            c1, c2 = line.leftPt.coords, line.rightPt.coords
            self_copy = cv.line(self_copy, c1, c2, color, thickness)
        return GreyFrame(self_copy)
        
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
        return BGRFrame(cv.circle(self_copy, point.coords, radius, color, thickness))
    
    @dispatch(Line)
    def put(self, line: Line, color = (0, 255, 0), thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        # conf_point = point.conform_to(self)
        c1, c2 = line.leftPt.coords, line.rightPt.coords
        return BGRFrame(cv.line(self_copy, c1, c2, color, thickness))
    
    @dispatch(Lines)
    def put(self, lines: Lines, color = (0, 255, 0), thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        for line in lines:
            c1, c2 = line.leftPt.coords, line.rightPt.coords
            self_copy = cv.line(self_copy, c1, c2, color, thickness)
        return BGRFrame(self_copy)
    
    def zero(self):
        return BGRFrame(np.zeros(self.shape[:2], dtype = 'uint8'))
    ###############
    

if __name__ == '__main__':
    
    ### Point and ArbPoint: Does typechecking work?
    coords = (-1, 2)
    coords = (1.5, 2)
    Point(coords)
    # SafePoint(coords)
    
    coords = (0, 0)
    a = np.zeros((100, 100), dtype = 'uint8')
    
    aFrm = BGRFrame(a)
    aPt = Point(coords)
    
    oFrm = aFrm.put(aPt, radius = 5)
    oFrm.show()
    
    gFrm = GreyFrame(a)
    oFrm2 = gFrm.put(aPt, radius = 5)
    oFrm2.show()
    
    cFrm = gFrm.to_BGR()
    cFrm = cFrm.put(aPt, radius = 5)
    cFrm.show()


    Point((4, 4)).up(10)

    ### typecheck works? ###
    # aFrm = BGRFrame(1)

    ### immutability preserved?
    gFrm = GreyFrame(a)
    print(gFrm.shape)
    cv.cvtColor(gFrm.array, cv.COLOR_GRAY2BGR)
    print(gFrm.shape)
    # gFrm.array = cv.cvtColor(gFrm.array, cv.COLOR_GRAY2BGR)
    # print(gFrm.shape)


    ### Check Line
    aPt = Point((0, 0))
    bPt = Point((100, 100))
    aLn = Line(aPt, bPt)

    bLn = Line(aPt.down(10), bPt.down(5))

    ### Check Lines
    aLns = Lines([aLn, bLn])

    ##### Check Iteration
    for cPt in cFrm:
        print(cPt)
        
    for cLn in aLns:
        print(cLn)

    ### Check multiple dispatch
    a = np.zeros((200, 200), dtype = 'uint8')
    gFrm = GreyFrame(a)
    gFrm.put(aPt, radius = 4).show()
    gFrm.put(aLn).show()


    ##### Test Intersection of Lines
    a = np.zeros((500, 500), dtype = 'uint8')
    gFrm = GreyFrame(a)
    
    aPt = Point((0, 0))
    bPt = Point((300, 100))
    
    ### normal case
    aLn = Line(aPt, bPt)
    bLn = Line(aPt.down(10), bPt.up(40))
    
    lnFrm = gFrm.put(aLn).put(bLn)
    intPt = aLn.weak_intersect(bLn)
    lnFrm.to_BGR().put(intPt, radius = 5).show()
    
    ### parallel case
    bLn = Line(aPt.down(100), bPt.down(100))
    
    lnFrm = gFrm.put(aLn).put(bLn)
    intPt = aLn.weak_intersect(bLn)
    
    if intPt is None:
        lnFrm.to_BGR().show()
    else:
        lnFrm.to_BGR().put(intPt, radius = 5).show()
    
    
    ### parallel vertical case
    aPt = Point((100, 0))
    bPt = Point((100, 200))
    
    aLn = Line(aPt, bPt)
    bLn = Line(aPt.right(100), bPt.right(100))
    
    lnFrm = gFrm.put(aLn).put(bLn)
    intPt = aLn.weak_intersect(bLn)
    
    if intPt is None:
        lnFrm.to_BGR().show()
    else:
        lnFrm.to_BGR().put(intPt, radius = 5).show()
    
    
    
    
    
    
    
    
    
    
    
    
    