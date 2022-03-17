# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:30:35 2022

@author: gianl
"""

import numpy as np
from dataclasses import dataclass

from ATMANpy.Points import Point

from typing import List


@dataclass(frozen = True)
class Line:
    """
    A line between two points
    """
    leftPt: Point
    rightPt: Point
    
    name: str = 'Line'
    
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
    def from_coords(four_coords, name = 'Line'):
        x1, y1, x2, y2 = four_coords
        return Line(Point((x1, y1)), Point((x2, y2)), name = name)
    
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

    ### METHODS ###
    def up(self, points, name = 'Line'):
        return Line(self.leftPt.up(points), self.rightPt.up(points), name = name)

    def down(self, points, name = 'Line'):
        return Line(self.leftPt.down(points), self.rightPt.down(points), name = name)

    def left(self, points, name = 'Line'):
        return Line(self.leftPt.left(points), self.rightPt.left(points), name = name)

    def right(self, points, name = 'Line'):
        return Line(self.leftPt.right(points), self.rightPt.right(points), name = name)

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
            
    def weak_y(self, x):
        """
        get y value at x, on the line or beyond.    
        """
        slp, itc = self.slope(), self.intercept()
        if slp is None:
            return None
        else:
            return slp * x + itc

    def weak_x(self, y):
        """
        get x value at y, on the line or beyond.    
        """
        slp, itc = self.slope(), self.intercept()
        if slp is None:
            return self.leftPt.x
        else:
            return (y - itc) / slp
        
    def flip_vertically(self, aLn):
        return Line(self.leftPt.flip_vertically(aLn), self.rightPt.flip_vertically(aLn))
        
    def flip_horizontally(self, aLn):
        return Line(self.leftPt.flip_horizontally(aLn), self.rightPt.flip_horizontally(aLn))
        

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
        if isinstance(item, str):
            ### get line by name
            matchLines = Lines([l for l in self.lineList if l.name == item], name = item)
            if len(matchLines) == 1:
                return matchLines[0]
            elif len(matchLines) > 1:
                return matchLines
        else:
            ### get line(s) by position in list.
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
    
