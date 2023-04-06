

import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from ATMANpy.types.Points import Point
from ATMANpy.types.Lines import Line, Lines, CourtLines

from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN

from ATMANpy.types.Shapes3D import Line3D, Region3D, Lines3D, Regions3D


# @dataclass(frozen = True)
# class Court(Lines3D):
#     lineList: List[Line3D]
#     name: str = 'CourtLines3D'
    
#     @staticmethod
#     def make():
#         return Lines3D.from_lines(CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN), z = 0, name = 'CourtLines3D')


@dataclass(frozen = True)
class Court(Regions3D):
    """
    A list of regions, i.e. a set of planes that make some composite shape.
    """
    _LINE_COLOR = (255, 255, 255)
    
    regList: List[Region3D]
    
    name: str = 'CourtRegions3D'
    color: Optional[tuple] = _LINE_COLOR

    @staticmethod
    def make(color = _LINE_COLOR):
        courtLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN)
        courtDims = courtLines.see_dimensions()
        
        courtLines3D = Lines3D.from_lines(courtLines, z = 0, name = 'CourtLines3D')
        
        courtRegions3D = []
        for cLn in courtLines3D:
            ### determine the dimension in which to thicken the lines
            if cLn.is_left_to_right():
                dim = 'y'
            elif cLn.is_front_to_back():
                dim = 'x'
            else:
                raise Exception('Court Lines can not have a non-zero z dimension!')
            
            ### determine the width of the line
            if 'Baseline' in cLn.name:
                width = courtDims['baseline_width']
            else:
                width = courtDims['otherline_width']
            
            courtRegions3D.append(Region3D.from_line(cLn, width = width, dim = dim, name = cLn.name))
            
        return Court(courtRegions3D, name = 'CourtLines3D', color = color)    
    
@dataclass(frozen = True)
class Net(Regions3D):
    _NET_COLOR = (212, 127, 255) ### a nice aquamarine
    
    regList: List[Region3D]
    
    name: str = 'CourtRegions3D'
    color: Optional[tuple] = _NET_COLOR
    
    @staticmethod
    def make_singles(color = _NET_COLOR):
        courtLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN)

        netLn = Line3D.from_line(courtLines.add_singles_netline()['Net Line'], z = 0)
        lPt, cPt, rPt = netLn.subdivide(3)
        
        lPt_up, cPt_up, rPt_up = [lPt.up(courtLines.net_height_post), 
                                  cPt.up(courtLines.net_height_middle), 
                                  rPt.up(courtLines.net_height_post)]
        leftReg = Region3D.from_points(lPt_up, cPt_up, lPt, cPt)
        rightReg = Region3D.from_points(cPt_up, rPt_up, cPt, rPt)
        
        return Net([leftReg, rightReg], name = 'Net3D', color = color)
        
        