# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:30:35 2022

@author: gianl
"""

import numpy as np
from dataclasses import dataclass

from ATMANpy.types.Points import Point, SafePoint

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

    def length(self):
        return self.leftPt.distance_to(self.rightPt)

    def slope(self):
        try:
            slp = (self.rightPt.y - self.leftPt.y) / (self.rightPt.x - self.leftPt.x)
        except:
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
            x_int = self.leftPt.x
            y_int = slp2 * x_int + int2
        elif slp2 is None:
            x_int = bLn.leftPt.x
            y_int = slp1 * x_int + int1
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
        
    def flip_vertically(self, aLn, name = 'Line'):
        if name is None:
            name = f'Flipped {self.name}'
        return Line(self.leftPt.flip_vertically(aLn), self.rightPt.flip_vertically(aLn),
                    name = name)
        
    def flip_horizontally(self, aLn, name = None):
        if name is None:
            name = f'Flipped {self.name}'
        return Line(self.leftPt.flip_horizontally(aLn), self.rightPt.flip_horizontally(aLn), 
                    name = name)
    
    def apply_homography(self, H, name = None):
        if name is None:
            name = self.name
        return Line(self.leftPt.apply_homography(H), self.rightPt.apply_homography(H), name = name)
    
    def round_points(self, name = None):
        if name is None:
            name = self.name
        return Line(self.leftPt.round_point(), self.rightPt.round_point(), name = name)

    def subdivide(self, elements = 100):
        x_space = np.linspace(self.left_point().x, self.right_point().x, num = elements, dtype = 'int32')
        y_space = np.linspace(self.high_point().y, self.low_point().y, num = elements, dtype = 'int32')
        if self.left_point() != self.high_point():
            y_space = np.flip(y_space)
        x_y_space = zip(x_space, y_space)
        return [Point(x_y) for x_y in x_y_space]
    
    def join(self, bLn):
        if self.rightPt == bLn.leftPt:
            return Line(self.leftPt, bLn.rightPt)
        elif self.leftPt == bLn.rightPt:
            return Line(bLn.leftPt, self.rightPt)

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
        try:
            return Lines([Line.from_coords(l) for l in coordList])
        except:
            return Lines([])

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
    
    def sort(self, which, primary_direction, secondary_direction):
        ### do we sort by left point, or right point or high point or low point?
        ### t2b, b2t, l2r or r2l?
        
        def point_func(aLine):
            try:
                relevantPt = getattr(aLine, which)()
            except:
                relevantPt = aLine.weak_intersect(which)
            return relevantPt
        
        ### supply a key which maps element to sorting order
        def key_func(aLine):
            
            relevantPt = point_func(aLine)
            
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
            # print(f"{self.name}: {len(self)} before merge...")
            flushedLines = Lines([]) ### these are the lines that have 'survived' the merge
            
        if len(self) <= 1:
            outLines = flushedLines.append(self)
            # print(f"... and {len(outLines)} after merge!")
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
    
    def transform(self, lineFunc, *args, **kwargs):
        if isinstance(lineFunc, str):
            return Lines([getattr(l, lineFunc)(*args, name = l.name, **kwargs) for l in self.lineList])
        else:
            return Lines([lineFunc(l, *args, **kwargs) for l in self.lineList])
            
    def filt(self, filtFunc):
        # print(f"{self.name}: {len(self)} before filter...")
        filtList = list(filter(filtFunc, self.lineList))
        # print(f"... and {len(filtList)} after filter!")
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
    
    

@dataclass(frozen = True)
class CourtLines(Lines):
    lineList: List[Line]
    name: str = 'Lines'
    size: int = None
    margin: int = None
    force_dimensions: bool = False
    
    def _set_dimensions(self, size = 4, force_dimensions = False):
        ### This method is just for convenience, so we can initialize the dimensions
        # created in 'get_dimensions' 
        dimDict = CourtLines.get_dimensions(size = size, force_dimensions = force_dimensions)
        
        for attr, value in dimDict.items():
            object.__setattr__(self, attr, value)
    
    def __post_init__(self):
        self._set_dimensions(size = self.size, force_dimensions = self.force_dimensions)
    
    @staticmethod
    def get_dimensions(size = 4, force_dimensions = False):
        ## court dimensions in feet
        court_width = 36
        side_length = 39
        doubles_alley_width = 4.5
        singles_width = court_width - 2 * doubles_alley_width
        service_box_length = 21
        service_box_width = singles_width / 2
        net_beyond_line = 3
    
        scale_minimum = 2 ### minimum scale, so we have integers
    
        scale = size * scale_minimum
        
        outDict = {'court_width': scale * court_width,
                   'side_length': scale * side_length,
                   'doubles_alley_width': scale * doubles_alley_width,
                   'singles_width': scale * singles_width,
                   'service_box_length': scale * service_box_length,
                   'service_box_width': scale * service_box_width,
                   'net_beyond_line': scale * net_beyond_line}
        
        if force_dimensions:
            outDict = {k: round(v) for k, v in outDict.items()}
            
        return outDict
        
    
    ### always use this method to construct the court lines!
    @staticmethod
    def make(size = 4, margin = 10, force_dimensions = False):
        
        dim = CourtLines.get_dimensions(size = size, force_dimensions = force_dimensions)
        if force_dimensions:
            margin = round(margin)

        aspect_ratio = dim['side_length'] * 2 / dim['court_width']
        _origin = SafePoint((margin, round(margin * aspect_ratio)))
        
        ### top half of court
        baseLine = Line(_origin, _origin.right(dim['court_width']), name = 'Top Baseline')
        netLine = baseLine.down(dim['side_length'])
        doubles_deuceLine = Line(_origin, _origin.down(dim['side_length']), name = 'Top Doubles Deuce Line')
        singles_deuceLine = doubles_deuceLine.right(dim['doubles_alley_width'], name = 'Top Singles Deuce Line')
        singles_adLine = singles_deuceLine.right(dim['singles_width'], name = 'Top Singles Ad Line')
        doubles_adLine = singles_adLine.right(dim['doubles_alley_width'], name = 'Top Doubles Ad Line')
        serviceLine = Line(baseLine.weak_intersect(singles_deuceLine), baseLine.weak_intersect(singles_adLine)).down(dim['side_length'] - dim['service_box_length'], name = 'Top Service Line').round_points()
        centerLine = Line(singles_deuceLine.weak_intersect(serviceLine), singles_deuceLine.weak_intersect(netLine)).right(dim['service_box_width'], name = 'Top Center Line').round_points()
        
        topLines = Lines([baseLine, serviceLine, centerLine,
                          doubles_deuceLine, singles_deuceLine, singles_adLine, doubles_adLine])
        
        def switch_name(nameStr):
            to_switch = [('Top ', 'Bottom '), ('Deuce ', 'Ad ')]
            
            for x, y in to_switch:
                if x in nameStr:
                    nameStr = nameStr.replace(x, y)
                elif y in nameStr:
                    nameStr = nameStr.replace(y, x)
            return nameStr
        
        botLines = Lines([l.flip_vertically(netLine, name = switch_name(l.name)).round_points() for l in topLines])
        
        courtLines = topLines.append(botLines)
            
        courtLines = CourtLines(courtLines.lineList, 
                                name = 'Court Lines', 
                                size = size, 
                                margin = margin,
                                force_dimensions = force_dimensions)
        
        return courtLines
    
    def add_singles_netline(self):
        netLine = self['Top Baseline'].down(self.side_length)
        netLine = Line(netLine.leftPt.right(self.doubles_alley_width).left(self.net_beyond_line),
                                netLine.rightPt.left(self.doubles_alley_width).right(self.net_beyond_line),
                                name = 'Net Line')
        return CourtLines(self.append(netLine), 
                          name = self.name, size = self.size, margin = self.margin, force_dimensions = self.force_dimensions)
        
    def add_doubles_netline(self):
        netLine = self['Top Baseline'].down(self.side_length)
        netLine = Line(netLine.leftPt.left(self.net_beyond_line),
                                netLine.rightPt.right(self.net_beyond_line),
                                name = 'Net Line')
        return CourtLines(self.append(netLine), 
                          name = self.name, size = self.size, margin = self.margin, force_dimensions = self.force_dimensions)
        
    def get_half_region(self, side = 'Top', width = 'Doubles'):
        p1 = self[f'{side} {width} Ad Line'].weak_intersect(self['Net Line'])
        p2 = self[f'{side} {width} Deuce Line'].weak_intersect(self['Net Line'])
        p3 = self[f'{side} {width} Ad Line'].weak_intersect(self[f'{side} Baseline'])
        p4 = self[f'{side} {width} Deuce Line'].weak_intersect(self[f'{side} Baseline'])
        return Region.from_points(p1, p2, p3, p4)
    
    def get_court_region(self, width = 'Doubles'):
        p1 = self[f'Top {width} Ad Line'].weak_intersect(self['Top Baseline'])
        p2 = self[f'Top {width} Deuce Line'].weak_intersect(self['Top Baseline'])
        p3 = self[f'Bottom {width} Ad Line'].weak_intersect(self['Bottom Baseline'])
        p4 = self[f'Bottom {width} Deuce Line'].weak_intersect(self['Bottom Baseline'])
        return Region.from_points(p1, p2, p3, p4)
    
    def transform(self, lineFunc, *args, **kwargs):
        if isinstance(lineFunc, str):
            return CourtLines([getattr(l, lineFunc)(*args, name = l.name, **kwargs) for l in self.lineList], 
                              name = self.name, size = self.size, margin = self.margin, force_dimensions = self.force_dimensions)
        else:
            return CourtLines([lineFunc(l, *args, name = l.name, **kwargs) for l in self.lineList],
                              name = self.name, size = self.size, margin = self.margin, force_dimensions = self.force_dimensions)
    
    def pixel_in_meter(self, horiLine = 'Top Baseline', vertLine = 'Top Center Line'):
        ### how many pixel are in a meter?
        hori_scale = self[horiLine].length() / 10.97 ### 36 foot in metres
        vert_scale = self[vertLine].length() / 6.4
        return hori_scale, vert_scale


    
@dataclass(frozen = True)
class Region(Lines):
    lineList: List[Line]
    name: str = 'Region'
    area: int = None
    centroid: Point = None
    
    def __post_init__(self):
        object.__setattr__(self, 'p1', self['Top'].leftPt)
        object.__setattr__(self, 'p2', self['Top'].rightPt)
        object.__setattr__(self, 'p3', self['Bottom'].leftPt)
        object.__setattr__(self, 'p4', self['Bottom'].rightPt)
        object.__setattr__(self, 'points', (self.p1, self.p2, self.p3, self.p4))
    
    @staticmethod
    def from_points(p1: Point, p2: Point, p3: Point = None, p4: Point = None,
                    area = None, centroid = None):
        is_rectangle = p3 is None and p4 is None
        if is_rectangle:
            p3 = Point((p1.x, p2.y))
            p4 = Point((p2.x, p1.y))
            
        ### to connect arbitrary four points without crossing lines,
        # check the two leftmost points, and the two upper points.
        l1, l2, r1, r2 = sorted([p1, p2, p3, p4], key = lambda pt: pt.x)   
        lt, lb = sorted([l1, l2], key = lambda pt: pt.y)   
        rt, rb = sorted([r1, r2], key = lambda pt: pt.y)   
        
        lineList = [Line(lt, lb, name = 'Left'), 
                    Line(lt, rt, name = 'Top'), 
                    Line(rt, rb, name = 'Right'), 
                    Line(lb, rb, name = 'Bottom')]
        
        if is_rectangle:
            h, w = lineList[0].length(), lineList[1].length()
            if area is None:
                area = (h + 1) * (w + 1) # note: a 'region' where all points are the same has h, w = zero, but contains one pixel -> therefore, area has to be one.
            if centroid is None:
                centroid = lt.down(h/2).right(w/2)
        
        return Region(lineList, area = area, centroid = centroid)
        
    @staticmethod
    def from_lines(lines: Lines):
        unique_points = [l.leftPt for l in lines] + [l.rightPt for l in lines]
        pointSet = set(unique_points)
        try:
            p1, p2, p3, p4 = pointSet
        except ValueError:
            raise Exception("Construction of Region with Lines: Needs lines with four unique points!")
        else:
            return Region.from_points(p1, p2, p3, p4)
            
    def to_points(self):
        t, b = self['Top'], self['Bottom']
        return t.leftPt, t.rightPt, b.leftPt, b.rightPt
        
    def join_corners(self):
        ### issue: given 4 lines, we might have eight different points.
        ### this function ensures that the corner points are identical
        tl1, tl2 = self['Left'].high_point(), self['Top'].left_point()
        bl1, bl2 = self['Left'].low_point(), self['Bottom'].left_point()
        tr1, tr2 = self['Right'].high_point(), self['Top'].right_point()
        br1, br2 = self['Right'].low_point(), self['Bottom'].right_point()
        
        avg = lambda p1, p2: Point((0.5 * (p1.x + p2.x), 0.5 * (p1.y + p2.y))).round_point()
        
        tl, bl, tr, br = avg(tl1, tl2), avg(bl1, bl2), avg(tr1, tr2), avg(br1, br2)
        
        return Region.from_points(tl, bl, tr, br)
        
    def scale(self, factor):
        diag = self.p4 - self.p1 ### width and the height of the region
        diag_scaled = diag * (factor - 1) ### the width and heigth the region has to be increased or reduced by.
        shift_x, shift_y = (diag_scaled // 2).coords ### the amount that each point has to move towards the middle.
        
        p1_new = self.p1.left(shift_x).up(shift_y)
        p2_new = self.p2.right(shift_x).up(shift_y)
        p3_new = self.p3.left(shift_x).down(shift_y)
        p4_new = self.p4.right(shift_x).down(shift_y)
    
        return Region.from_points(p1_new, p2_new, p3_new, p4_new)
    
    def scale_equal(self, factor):
        ### as above, but it increases height and weight equally.
        diag = self.p4 - self.p1 ### width and the height of the region
        mean_diag = (diag.coords[0] + diag.coords[1]) / 2
        scale_amount = mean_diag * (factor - 1) ### the amount that each point has to move towards the middle.
        
        p1_new = self.p1.left(scale_amount).up(scale_amount)
        p2_new = self.p2.right(scale_amount).up(scale_amount)
        p3_new = self.p3.left(scale_amount).down(scale_amount)
        p4_new = self.p4.right(scale_amount).down(scale_amount)
    
        return Region.from_points(p1_new, p2_new, p3_new, p4_new)
        
    def enlarge(self, to_min_height = None, to_min_width = None):
        diag = self.p4 - self.p1
        width, height = diag.coords
        
        p1, p2, p3, p4 = self.p1, self.p2, self.p3, self.p4
        
        if to_min_height is not None:
            add_on_both_sides = (to_min_height - height) // 2 ### floor div so we keep an integer
            
            if add_on_both_sides > 0:
                p1 = p1.up(add_on_both_sides)
                p2 = p2.up(add_on_both_sides)
                p3 = p3.down(add_on_both_sides)
                p4 = p4.down(add_on_both_sides)
        
        if to_min_width is not None:
            add_on_both_sides = (to_min_width - width) // 2 ### floor div so we keep an integer
            
            if add_on_both_sides > 0:
                p1 = p1.left(add_on_both_sides)
                p2 = p2.right(add_on_both_sides)
                p3 = p3.left(add_on_both_sides)
                p4 = p4.right(add_on_both_sides)
                
        return Region.from_points(p1, p2, p3, p4) 
    
    
    def get_homography(self, othRegion):
        from ATMANpy.funcs.line_funcs import calculate_homography
        return calculate_homography(self.points, othRegion.points)
        
        
        
    
    