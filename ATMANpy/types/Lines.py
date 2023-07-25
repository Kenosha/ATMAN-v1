# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:30:35 2022

@author: gianl
"""

import numpy as np
from dataclasses import dataclass

from ATMANpy.types.Points import Point2D, SafePoint2D

from typing import List

import ATMANpy.auxil.constants as const


@dataclass(frozen = True)
class Line:
    """
    A line between two points
    """
    leftPt: Point2D
    rightPt: Point2D
    
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
        return Line(Point2D((x1, y1)), Point2D((x2, y2)), name = name)
    
    @staticmethod
    def from_series(lnSer):
        x1 = lnSer.loc['x1']
        y1 = lnSer.loc['y1']
        x2 = lnSer.loc['x2']
        y2 = lnSer.loc['y2']
        leftPt = Point2D((x1, y1))
        rightPt = Point2D((x2, y2))
        return Line(leftPt, rightPt, name = lnSer.name)
    
    ### METHODS ### 
    def append(self, other):
        try:
            return Lines([self, *other.lineList])
        except AttributeError:
            return Lines([self, other])
    
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
    
    def is_vertical(self):
        return not self.is_horizontal()
    
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
        denom = (self.rightPt.x - self.leftPt.x)
        if denom == 0:
            return None
        else:
            try:
                slp = (self.rightPt.y - self.leftPt.y) / denom
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
        return Point2D((x_int, y_int))
            
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
        return [Point2D(x_y) for x_y in x_y_space]
    
    def join(self, bLn):
        if self.rightPt == bLn.leftPt:
            return Line(self.leftPt, bLn.rightPt)
        elif self.leftPt == bLn.rightPt:
            return Line(bLn.leftPt, self.rightPt)
        
    def center(self):
        return (self.leftPt + self.rightPt) // 2
    
    def distance_to(self, aPt):
        ### find perpendicular distance of a line to a point
        lineVec = self.rightPt - self.leftPt
        orthLine = Line(aPt, aPt + lineVec.find_perpendicular())
        nearestPt = self.weak_intersect(orthLine, to_int = False)
        return aPt.distance_to(nearestPt)
    
    def rescale(self, scale_factor):
        return Line(self.leftPt.rescale(scale_factor), self.rightPt.rescale(scale_factor), name = self.name)
    
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

    @staticmethod
    def from_frame(lnFrm):
        outList = []
        for idx, rowSer in lnFrm.iterrows():
            outList.append(Line.from_series(rowSer))
        return Lines(outList)

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
    def names(self):
        return [aLn.name for aLn in self.lineList]
    
    def append(self, other):
        try:
            return Lines([*self.lineList, *other.lineList])
        except:
            return Lines([*self.lineList, other])
    
    def drop(self, line_to_drop):
        return Lines([aLn for aLn in self.lineList if aLn.name != line_to_drop], name = self.name)
    
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
    
    def rescale(self, scale_factor):
        return self.__class__([ln.rescale(scale_factor) for ln in self.lineList], name = self.name)

@dataclass(frozen = True)
class NetLines(Lines):
    lineList: List[Line]
    name: str = 'NetLines'
    
    @staticmethod
    def from_points(bot_leftPt, top_leftPt, top_midPt, top_rightPt, bot_rightPt):
        
        leftpostLine = Line(bot_leftPt, top_leftPt, name = 'Left Post').round_points()
        rightpostLine = Line(bot_rightPt, top_rightPt, name = 'Right Post').round_points()
        leftnetLine = Line(top_leftPt, top_midPt, name = 'Left Net').round_points()
        rightnetLine = Line(top_midPt, top_rightPt, name = 'Right Net').round_points()
    
        return NetLines([leftpostLine, rightpostLine, leftnetLine, rightnetLine])

@dataclass(frozen = True)
class CourtLines(Lines):
    lineList: List[Line]
    name: str = 'Lines'
    scale: int = None
    margin: int = None
    force_dimensions: bool = False
    
    def _set_dimensions(self, scale = const.SCALE, force_dimensions = False):
        ### This method is just for convenience, so we can initialize the dimensions
        # created in 'get_dimensions' 
        dimDict = CourtLines.get_dimensions(scale = scale, force_dimensions = force_dimensions)
        
        for attr, value in dimDict.items():
            object.__setattr__(self, attr, value)
    
    def __post_init__(self):
        self._set_dimensions(scale = self.scale, force_dimensions = self.force_dimensions)
    
    @staticmethod
    def get_dimensions(scale = const.SCALE, force_dimensions = False):
        
        ### dimensions of court, in feet
        feetDict = {'scale': 1,
                   'court_width': const.COURT_WIDTH,
                   'side_length': const.SIDE_LENGTH,
                   'doubles_alley_width': const.DOUBLES_ALLEY_WIDTH,
                   'singles_width': const.SINGLES_WIDTH,
                   'service_box_length': const.SERVICE_BOX_LENGTH,
                   'service_box_width': const.SERVICE_BOX_WIDTH,
                   'net_beyond_line': const.NET_BEYOND_LINE,
                   'net_height_post': const.NET_BEYOND_POST,
                   'net_height_middle': const.NET_HEIGHT_MIDDLE,
                   'baseline_width': const.BASELINE_WIDTH,
                   'otherline_width': const.OTHERLINE_WIDTH}
        ### dimensions of court, scaled up
        outDict = {k: const.SCALE * v for k, v in feetDict.items()}
        
        if force_dimensions:
            outDict = {k: round(v) for k, v in outDict.items()}
            
        return outDict
        
    
    ### always use this method to construct the court lines!
    @staticmethod
    def make(scale = const.SCALE, margin = const.MARGIN, force_dimensions = False):
        
        dim = CourtLines.get_dimensions(scale = scale, force_dimensions = force_dimensions)
        if force_dimensions:
            margin = round(margin)

        aspect_ratio = dim['side_length'] * 2 / dim['court_width']
        _origin = SafePoint2D((margin, round(margin * aspect_ratio)))
        
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
                                scale = scale, 
                                margin = margin,
                                force_dimensions = force_dimensions)
        
        return courtLines
    
    def see_dimensions(self):
        return CourtLines.get_dimensions(scale = self.scale, 
                                         force_dimensions = self.force_dimensions)
    
    def add_singles_netline(self):
        netLine = self['Top Baseline'].down(self.side_length)
        netLine = Line(netLine.leftPt.right(self.doubles_alley_width).left(self.net_beyond_line),
                                netLine.rightPt.left(self.doubles_alley_width).right(self.net_beyond_line),
                                name = 'Net Line')
        return CourtLines(self.append(netLine), 
                          name = self.name, scale = self.scale, margin = self.margin, force_dimensions = self.force_dimensions)
        
    def add_doubles_netline(self):
        netLine = self['Top Baseline'].down(self.side_length)
        netLine = Line(netLine.leftPt.left(self.net_beyond_line),
                                netLine.rightPt.right(self.net_beyond_line),
                                name = 'Net Line')
        return CourtLines(self.append(netLine), 
                          name = self.name, scale = self.scale, margin = self.margin, force_dimensions = self.force_dimensions)
        
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
    
    def get_service_box(self, side = 'Top', which = 'Deuce'):
        p1 = self[f'{side} Singles {which} Line'].weak_intersect(self['Net Line'])
        p2 = self[f'{side} Center Line'].weak_intersect(self['Net Line'])
        p3 = self[f'{side} Singles {which} Line'].weak_intersect(self[f'{side} Service Line'])
        p4 = self[f'{side} Center Line'].weak_intersect(self[f'{side} Service Line'])
        return Region.from_points(p1, p2, p3, p4)
    
    def get_service_boxes(self, side = 'Top'):
        p1 = self[f'{side} Singles Deuce Line'].weak_intersect(self['Net Line'])
        p2 = self[f'{side} Singles Ad Line'].weak_intersect(self['Net Line'])
        p3 = self[f'{side} Singles Deuce Line'].weak_intersect(self[f'{side} Service Line'])
        p4 = self[f'{side} Singles Ad Line'].weak_intersect(self[f'{side} Service Line'])
        return Region.from_points(p1, p2, p3, p4)
    
    def transform(self, lineFunc, *args, **kwargs):
        if isinstance(lineFunc, str):
            return CourtLines([getattr(l, lineFunc)(*args, name = l.name, **kwargs) for l in self.lineList], 
                              name = self.name, scale = self.scale, margin = self.margin, force_dimensions = self.force_dimensions)
        else:
            return CourtLines([lineFunc(l, *args, name = l.name, **kwargs) for l in self.lineList],
                              name = self.name, scale = self.scale, margin = self.margin, force_dimensions = self.force_dimensions)
    
    def pixel_in_meter(self, horiLine = 'Top Baseline', vertLine = 'Top Center Line'):
        ### how many pixel are in a meter?
        hori_scale = self[horiLine].length() / 10.97 ### 36 foot in metres
        vert_scale = self[vertLine].length() / 6.4
        return hori_scale, vert_scale
    
    def pixel_in_foot(self):
        return self['Bottom Baseline'].length() / 36

    def get_3D_corner_points(self, width = 'Doubles'):
        courtReg = self.get_court_region(width = width)
        points = courtReg.to_points()
        coords_3D = [[*pt.coords, 0] for pt in points]
        return coords_3D
    
    def get_3D_net_points(self):
        FOOT_HEIGHT_AT_POST = 3.5
        FOOT_HEIGHT_AT_CENTER = 3
        
        scale_factor = self.pixel_in_foot()
        net = self['Net Line']
        l, c, r = net.leftPt, net.weak_intersect(self['Bottom Center Line']), net.rightPt 
        coords_3D = [[*l.coords, FOOT_HEIGHT_AT_POST * scale_factor],
                     [*c.coords, FOOT_HEIGHT_AT_CENTER * scale_factor],
                     [*r.coords, FOOT_HEIGHT_AT_POST * scale_factor]]
        return coords_3D

    
@dataclass(frozen = True)
class Region(Lines):
    lineList: List[Line]
    name: str = 'Region'
    # area: int = None
    # centroid: Point2D = None
    
    def __post_init__(self):
        object.__setattr__(self, 'p1', self['Top'].leftPt)
        object.__setattr__(self, 'p2', self['Top'].rightPt)
        object.__setattr__(self, 'p3', self['Bottom'].leftPt)
        object.__setattr__(self, 'p4', self['Bottom'].rightPt)
        object.__setattr__(self, 'points', (self.p1, self.p2, self.p3, self.p4))
    
    @staticmethod
    def from_array(regArr):
        assert len(regArr) == 8, f'Array used to build region needs to be of length 8, has length {len(regArr)}'
        p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y = regArr
        p1, p2, p3, p4 = Point2D((p1x, p1y)), Point2D((p2x, p2y)), Point2D((p3x, p3y)), Point2D((p4x, p4y))
        return Region.from_points(p1, p2, p3, p4)
    
    @staticmethod
    def from_points(p1: Point2D, p2: Point2D, p3: Point2D = None, p4: Point2D = None,
                    area = None, centroid = None):
        is_rectangle = p3 is None and p4 is None
        if is_rectangle:
            p3 = Point2D((p1.x, p2.y))
            p4 = Point2D((p2.x, p1.y))
            
        ### to connect arbitrary four points without crossing lines,
        # check the two leftmost points, and the two upper points.
        l1, l2, r1, r2 = sorted([p1, p2, p3, p4], key = lambda pt: pt.x)   
        lt, lb = sorted([l1, l2], key = lambda pt: pt.y)   
        rt, rb = sorted([r1, r2], key = lambda pt: pt.y)   
        
        lineList = [Line(lt, lb, name = 'Left'), 
                    Line(lt, rt, name = 'Top'), 
                    Line(rt, rb, name = 'Right'), 
                    Line(lb, rb, name = 'Bottom')]
        
        # if is_rectangle:
        #     h, w = lineList[0].length(), lineList[1].length()
        #     if area is None:
        #         area = (h + 1) * (w + 1) # note: a 'region' where all points are the same has h, w = zero, but contains one pixel -> therefore, area has to be one.
        #     if centroid is None:
        #         centroid = lt.down(h/2).right(w/2)
        
        return Region(lineList)
        
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
       
    @staticmethod
    def from_center(centerPt: Point2D, radius = 10):
        p1 = centerPt.up(radius).left(radius)
        p2 = centerPt.down(radius).left(radius)
        p3 = centerPt.up(radius).right(radius)
        p4 = centerPt.down(radius).right(radius)
        
        return Region.from_points(p1, p2, p3, p4)
        
    def is_rectangle(self):
        #### -> top left and lower right point (p1 and p4) fully identify!
        p2_is_redundant = self.p2 == Point2D((self.p4.x, self.p1.y))
        p3_is_redundant = self.p3 == Point2D((self.p1.x, self.p4.y))
        return p2_is_redundant & p3_is_redundant
    
    def height(self):
        assert self.is_rectangle()
        return self.lineList[0].length()
    
    def width(self):
        assert self.is_rectangle()
        return self.lineList[1].length() 
    
    def area(self):
        # note: a 'region' where all points are the same has h, w = zero, but contains one pixel -> therefore, area has to be one.
        assert self.is_rectangle()
        return (self.height() + 1) * (self.width() + 1)
    
    def pseudo_trapezius_area(self):
        tLn, bLn = self['Top'], self['Bottom']
        width = (tLn.length() + bLn.length()) / 2
        height = tLn.distance_to(bLn.center())
        return round((width + 1) * (height + 1)) ### plus one -> see above!
        
    def centroid(self):
        assert self.is_rectangle()
        return self.p1.down(self.height() / 2).right(self.width() / 2)
    
    def rename(self, new_name):
        return Region(self.lineList, name = new_name)
    
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
        
        avg = lambda p1, p2: Point2D((0.5 * (p1.x + p2.x), 0.5 * (p1.y + p2.y))).round_point()
        
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
        
    def intersection(self, othRegion):
        assert self.is_rectangle() and othRegion.is_rectangle(), 'Intersection can only be calculated for rectangular regions! Implement if needed more generally.'
        p1x = max(self.p1.x, othRegion.p1.x)
        p1y = max(self.p1.y, othRegion.p1.y)
        p4x = min(self.p4.x, othRegion.p4.x)
        p4y = min(self.p4.y, othRegion.p4.y)
        
        p1, p4 = Point2D((p1x, p1y)), Point2D((p4x, p4y))
        
        if p1.is_above(p4) and p1.is_left(p4):
            return Region.from_points(p1, p4)
        else:
            return None
        
    
@dataclass(frozen = True)
class Regions:
    regList: List[Region]    
    name: str = 'Regions'
    
    @staticmethod
    def from_array(regArr):
        return Regions([Region.from_array(subArr) for subArr in regArr])
        
    def __iter__(self):
        return iter(self.regList)
    
    def __getitem__(self, item):
        if isinstance(item, str):
            regList = [myReg for myReg in self if myReg.name == item]
            if len(regList) == 1:
                return regList[0]
            else:
                return Regions(regList)
        else:
            return self.regList[item]
    
    def length(self):
        return len(self.regList)
    
    def heights(self):
        return [myReg.height() for myReg in self]
    
    def widths(self):
        return [myReg.width() for myReg in self]
    
    def areas(self):
        return [myReg.area() for myReg in self]
    
    def centroids(self):
        return [myReg.centroid() for myReg in self]
    
    