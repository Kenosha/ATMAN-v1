

import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from ATMANpy.types.Points import Point
from ATMANpy.types.Lines import Line, Lines, CourtLines

from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
SCALE = 2 * SCHEMA_SIZE  ## times two, because there is a minimum scale factor... 

from ATMANpy.types.Shapes3D import Point3D, Ball3D, Line3D, Region3D, Lines3D, Regions3D


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
            
        return Court(courtRegions3D, name = 'Courtlines', color = color)    
    
@dataclass(frozen = True)
class Net(Regions3D):
    _NET_COLOR = (212, 127, 255) ### a nice aquamarine
    
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
        
        return Net([leftReg, rightReg], name = 'Net', color = color)
        
@dataclass(frozen = True)
class Ball(Ball3D):
    _RADIUS_IN_FEET = 0.108  ### what is the correct default radius? 
                              ## should be the tennis ball in the usual coord system of court lines.
                              ## ball radius in feet (2.6 inch diameter -> 1.3 inch radius = 0.108 ft.)
    _BALL_COLOR = (79, 255, 223) ### color of tennis ball in BGR
    
    @staticmethod
    def make(coords):
        return Ball(coords, Ball._RADIUS_IN_FEET * SCALE, 
                    color = Ball._BALL_COLOR, name = 'Tennis Ball')
    
@dataclass(frozen = True)
class Player(Region3D):
    _DEFAULT_HEIGHT = 6 ### in feet
    _DEFAULT_WIDTH = 1
    _DEFAULT_COLOR = (170, 255, 127) ### complement of net
    _DEFAULT_NAME = 'Player'
    
    # coords: Point3D() ### 3D coordinates
    # height: Optional[float] = _DEFAULT_HEIGHT * SCALE
    
    @staticmethod
    def make(coords, height = _DEFAULT_HEIGHT * SCALE, width = _DEFAULT_WIDTH * SCALE,
             color = _DEFAULT_COLOR, name = _DEFAULT_NAME):
        if len(coords) == 2:
            x, y = coords
            z = 0
        elif len(coords) == 3:
            x, y, z = coords
            
        loPt = Point3D((x, y, z))
        hiPt = loPt.up(height)
        
        centerLn = Line3D(loPt, hiPt)
        return Player.from_line(centerLn, width, dim = 'x',
                                     left_right = '+x', top_bottom = '-z',
                                     color = color, name = name)
    
    
@dataclass(frozen = True)
class Racquet(Region3D):
    _DEFAULT_GRIP_HEIGHT = 4 * SCALE
    # _DEFAULT_HEAD_HEIGHT = 5 * SCALE
    _DEFAULT_LENGTH = 2.25 * SCALE
    _DEFAULT_GRIP_WIDTH = 0.125 * SCALE
    _DEFAULT_HEAD_WIDTH = 0.85 * SCALE
    _DEFAULT_COLOR = (255, 212, 127)
    _GRIP_TO_SWEET = 3/4 ### distance from grip end to sweet spot, as proportion of entire racquet length. 
    
    @staticmethod
    def make(gripPt, spotPt, inclination = 0, color = _DEFAULT_COLOR):
        ### make from two points: bottom of the grip, and sweet spot
        racquetVector = spotPt - gripPt

        ### lengthen the line beyond the sweet spot, to include also the top
        topPt = gripPt + racquetVector / Racquet._GRIP_TO_SWEET
        
        ### find the points on the frame, to the left and right of the sweet spot
        incl_rad = np.pi * inclination
        # find two perpendicular vectors to racquet
        orthVector_1 = racquetVector.find_perpendicular()
        orthVector_2 = orthVector_1.cross(racquetVector)
        # normalize to unit length
        orthVector_1, orthVector_2 = orthVector_1 / orthVector_1.length(), orthVector_2 / orthVector_2.length()
        # find unit_circle orthogonal to racket
        unitPt = orthVector_1 * np.cos(incl_rad) + orthVector_2 * np.sin(incl_rad)
        # scale and shift circle
        scaledPt = unitPt * Racquet._DEFAULT_HEAD_WIDTH / 2
        framePt_1, framePt_2, = spotPt + scaledPt, spotPt - scaledPt
        
        ### finalize racquet
        # myRacquet = Racquet.from_points(t1, t2, b1, b2, color = color) ## -> sadly, this returns a region!
        racquetLines = Region3D.from_points(gripPt, framePt_1, topPt, framePt_2)
        myRacquet = Racquet(racquetLines, color = color, name = 'Racquet')
        
        ### immutability breaks...
        # a) Angles
        object.__setattr__(myRacquet, 'inclination', inclination)
        dir_hori, dir_vert = myRacquet.find_orientation_hori_vert()
        object.__setattr__(myRacquet, 'dir_hori', dir_hori)
        object.__setattr__(myRacquet, 'dir_vert', dir_vert)
        
        # b) Lines and Vectors
        object.__setattr__(myRacquet, 'racquetLine', Line3D(gripPt, spotPt)) 
        object.__setattr__(myRacquet, 'planeVec1', orthVector_1)
        object.__setattr__(myRacquet, 'planeVec2', orthVector_2)
        
        return myRacquet
    
    
    @staticmethod
    def make_at_player(aPlayer, distance_from_player = 0, height = _DEFAULT_GRIP_HEIGHT,
                       dir_hori = 0, dir_vert = 0, inclination = 0, color = _DEFAULT_COLOR):
        ### distance from player: the distance at which the player holds the racket from his trunk
        ### height: the height at which the player holds the racket
        ### dir_hori, dir_vert: horizontal and vertical angles, expressed in multiples of pi
        centerPt = aPlayer['Bottom'].center_point().up(height)
        gamma = np.pi * dir_hori
        theta = np.pi * dir_vert
        
        ### make grip point
        radius_inner = distance_from_player
        unit_coords = np.sin(theta) * np.cos(gamma), np.sin(theta) * np.sin(gamma), np.cos(theta)
        gripPt = Point3D(unit_coords) * radius_inner + centerPt
       
        ### assumption here is that the angles are still direction_horizontal and _vertical.
        radius_outer = radius_inner + Racquet._DEFAULT_LENGTH * Racquet._GRIP_TO_SWEET
        spotPt = Point3D(unit_coords) * radius_outer + centerPt
        
        return Racquet.make(gripPt, spotPt, inclination = inclination, color = color)
    
    @staticmethod
    def make_at_contact(aPlayer, aBall, 
                        aBall_before = None, aBall_after = None, ### provide these to make inclination, using law of reflection.
                        height = _DEFAULT_GRIP_HEIGHT, color = _DEFAULT_COLOR):
        centerPt = aPlayer['Bottom'].center_point().up(height)
        ballPt = aBall.get_point()
        
        lmbd = Racquet._DEFAULT_LENGTH / centerPt.distance_to(ballPt)
        gripPt = ballPt + lmbd * (centerPt - ballPt)
        
        return Racquet.make(gripPt, ballPt)
        
    def find_orientation_hori_vert(self):
        return self.racquetLine.find_orientation_hori_vert()
        
        