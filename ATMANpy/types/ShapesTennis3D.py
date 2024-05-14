

import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from ATMANpy.types.Lines import Line, Lines, CourtLines

from ATMANpy.types.Shapes3D import Point3D, Ball3D, Line3D, Region3D, Lines3D, Regions3D

import ATMANpy.auxil.constants as const

# @dataclass(frozen = True)
# class Court(Lines3D):
#     lineList: List[Line3D]
#     name: str = 'CourtLines3D'
    
#     @staticmethod
#     def make():
#         return Lines3D.from_lines(CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN), z = 0, name = 'CourtLines3D')


@dataclass(frozen = True, repr = False)
class Court(Regions3D):
    """
    A list of regions, i.e. a set of planes that make some composite shape.
    """
    _LINE_COLOR = const.BGR['bright_grey']
    _LINE_ALPHA = 1
    
    @staticmethod
    def make(courtLines = None, color = _LINE_COLOR, alpha = _LINE_ALPHA):
        if courtLines is None:
            courtLines = CourtLines.make()
        courtDims = courtLines.see_dimensions()
        
        courtLines3D = Lines3D.from_lines2D(courtLines, z = 0, name = 'CourtLines3D')
        
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
            
            courtRegions3D.append(Region3D.from_line3D(cLn, width = width, dim = dim, name = cLn.name))
            
        return Court(courtRegions3D, name = 'Courtlines', color = color, alpha = alpha)    
    
    
@dataclass(frozen = True, repr = False)
class Net(Regions3D):
    _NET_COLOR = const.BGR['bright_blue'] ### a nice aquamarine
    _NET_ALPHA = 1
    
    @staticmethod
    def make_singles(color = _NET_COLOR, alpha = _NET_ALPHA):
        courtLines = CourtLines.make()

        netLn = Line3D.from_line2D(courtLines.add_singles_netline()['Net Line'], z = 0)
        lPt, cPt, rPt = netLn.subdivide(3)
        
        lPt_up, cPt_up, rPt_up = [lPt.up(courtLines.net_height_post), 
                                  cPt.up(courtLines.net_height_middle), 
                                  rPt.up(courtLines.net_height_post)]
        leftReg = Region3D.from_points3D(lPt_up, cPt_up, lPt, cPt, name = 'Left Net')
        rightReg = Region3D.from_points3D(cPt_up, rPt_up, cPt, rPt, name = 'Right Net')
        
        return Net([leftReg, rightReg], name = 'Net', color = color, alpha = alpha)
        
    def get_rectangular_net_region(self):
        ### this is a region approximating the full net -> rectangular, without the dip in the middle.
        leftLn, rightLn = self['Left Net']['Left'], self['Right Net']['Right']
        p1, p2, p3, p4 = leftLn.high_point(), rightLn.high_point(), leftLn.low_point(), rightLn.low_point()
        return Region3D.from_points3D(p1, p2, p3, p4)
    
    
@dataclass(frozen = True, repr = False)
class Ball(Ball3D):
    _RADIUS_IN_FEET = 0.108  ### what is the correct default radius? 
                              ## should be the tennis ball in the usual coord system of court lines.
                              ## ball radius in feet (2.6 inch diameter -> 1.3 inch radius = 0.108 ft.)
    _BALL_COLOR = const.BGR['ball_yellow'] ### color of tennis ball in BGR
    _BALL_ALPHA = 1
    
    @staticmethod
    def make(coords, color = _BALL_COLOR, alpha = _BALL_ALPHA):
        return Ball(coords, Ball._RADIUS_IN_FEET * const.SCALE, 
                    color = color, alpha = alpha, 
                    name = 'Tennis Ball')
    
    def distance_to(self, otherBall):
        return self.get_point().distance_to(otherBall.get_point())
    
@dataclass(frozen = True, repr = False)
class Mark(Lines3D):
    _MARK_COLOR = const.BGR['grass_green'] ### green is default
    _MARK_ALPHA = 1
    _MARK_NAME = 'Mark'
    _SIZE = 0.25 * const.SCALE 
    
    def make(cPt, flat_along_dimension = 'z', size = _SIZE, 
             color = _MARK_COLOR, alpha = _MARK_ALPHA,
             name = _MARK_NAME):
        r = size / 2
        if flat_along_dimension == 'x':
            x1, x2 =  cPt.back(r).up(r), cPt.front(r).down(r)
            y1, y2 =  cPt.back(r).down(r), cPt.front(r).up(r)
        if flat_along_dimension == 'y':
            x1, x2 =  cPt.left(r).up(r), cPt.right(r).down(r)
            y1, y2 =  cPt.left(r).down(r), cPt.right(r).up(r)
        if flat_along_dimension == 'z':
            x1, x2 =  cPt.left(r).back(r), cPt.right(r).front(r)
            y1, y2 =  cPt.left(r).front(r), cPt.right(r).back(r)
        aLn, bLn = Line3D(x1, x2), Line3D(y1, y2)
        
        return Mark([aLn, bLn], color = color, alpha = alpha, name = name)
     
    
@dataclass(frozen = True, repr = False)
class Player(Region3D):
    _DEFAULT_HEIGHT = 6 ### in feet
    _DEFAULT_WIDTH = 1
    _DEFAULT_COLOR = const.BGR['rose_orange'] ### complement of net
    _DEFAULT_ALPHA = 1
    _DEFAULT_NAME = 'Player'
    
    # coords: Point3D() ### 3D coordinates
    # height: Optional[float] = _DEFAULT_HEIGHT * SCALE
    
    @staticmethod
    def make(coords, height = _DEFAULT_HEIGHT * const.SCALE, width = _DEFAULT_WIDTH * const.SCALE,
             color = _DEFAULT_COLOR, alpha = _DEFAULT_ALPHA, name = _DEFAULT_NAME):
        if len(coords) == 2:
            x, y = coords
            z = 0
        elif len(coords) == 3:
            x, y, z = coords
            
        loPt = Point3D((x, y, z))
        hiPt = loPt.up(height)
        
        centerLn = Line3D(loPt, hiPt)
        return Player.from_line3D(centerLn, width, dim = 'x',
                                left_right = '+x', top_bottom = '-z',
                                color = color, alpha = alpha, 
                                name = name)
    
    def get_foot_point(self):
        return self.lineList['Bottom'].center_point()
    
    def get_shoulder_point(self, which = 'right'):
        assert which in ['left', 'right']
        _DEFAULT_SHOULDER_HEIGHT = 4 * const.SCALE
        
        bottomLn = self.lineList['Bottom']
        bottomPt = bottomLn.right_point() if which == 'right' else bottomLn.left_point()
        
        return bottomPt.up(_DEFAULT_SHOULDER_HEIGHT)
    
@dataclass(frozen = True, repr = False)
class Racquet(Region3D):
    _DEFAULT_GRIP_HEIGHT = 4 * const.SCALE
    # _DEFAULT_HEAD_HEIGHT = 5 * SCALE
    _DEFAULT_LENGTH = 2.25 * const.SCALE
    _DEFAULT_GRIP_WIDTH = 0.125 * const.SCALE
    _DEFAULT_HEAD_WIDTH = 0.85 * const.SCALE
    _DEFAULT_COLOR = const.BGR['strong_pink']
    _DEFAULT_ALPHA = 1
    _GRIP_TO_SWEET = 3/4 ### distance from grip end to sweet spot, as proportion of entire racquet length. 
    
    @staticmethod
    def make(gripPt, spotPt, inclination = 0, color = _DEFAULT_COLOR, alpha = _DEFAULT_ALPHA):
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
        racquetLines = Region3D.from_points3D(gripPt, framePt_1, topPt, framePt_2)
        myRacquet = Racquet(racquetLines, color = color, alpha = alpha, name = 'Racquet')
        
        ### immutability breaks...
        # a) Lines and Vectors
        object.__setattr__(myRacquet, 'racquetLine', Line3D(gripPt, spotPt)) 
        object.__setattr__(myRacquet, 'planeVec1', orthVector_1)
        object.__setattr__(myRacquet, 'planeVec2', orthVector_2)
        
        # b) Angles
        object.__setattr__(myRacquet, 'inclination', inclination)
        dir_hori, dir_vert = myRacquet.find_orientation_hori_vert()
        object.__setattr__(myRacquet, 'dir_hori', dir_hori)
        object.__setattr__(myRacquet, 'dir_vert', dir_vert)
        
        return myRacquet
    
    
    @staticmethod
    def make_at_player(aPlayer, distance_from_player = 0, height = _DEFAULT_GRIP_HEIGHT,
                       dir_hori = 0, dir_vert = 0, inclination = 0, color = _DEFAULT_COLOR, alpha = _DEFAULT_ALPHA):
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
        
        return Racquet.make(gripPt, spotPt, inclination = inclination, color = color, alpha = alpha)
    
    @staticmethod
    def make_at_contact(aPlayer, aBall, 
                        aBall_before = None, aBall_after = None, ### provide these to make inclination, using law of reflection.
                        height = _DEFAULT_GRIP_HEIGHT, color = _DEFAULT_COLOR, alpha = _DEFAULT_ALPHA):
                
        ###### Prepare uninclined racquet ######
        centerPt = aPlayer['Bottom'].center_point().up(height)
        ballPt = aBall.get_point()
        
        lmbd = Racquet._DEFAULT_LENGTH / centerPt.distance_to(ballPt)
        gripPt = ballPt + (centerPt - ballPt) * lmbd
        
        myRacquet_uninclined = Racquet.make(gripPt, ballPt)
        
        ###### Incline Racquet #####
        ### get plane parameters
        a, b = myRacquet_uninclined.planeVec1, myRacquet_uninclined.planeVec2
        
        ### get vector of incoming and outgoing ball, both oriented away from the racquet face.
        if aBall_before is None and aBall_after is None:
            return Racquet.make(gripPt, ballPt, inclination = 0)
        elif aBall_before is None:
            afterVector = aBall_after.get_point() - aBall.get_point() 
            normalVector = afterVector.normalize()
        elif aBall_after is None:
            beforeVector = aBall_before.get_point() - aBall.get_point()
            normalVector = beforeVector.normalize()
        else:
            beforeVector = aBall_before.get_point() - aBall.get_point()
            afterVector = aBall_after.get_point() - aBall.get_point() 
            
            ### get normal vector relative to racquet plane
            normalVector = ((beforeVector.normalize() + afterVector.normalize()) / 2).normalize()
        
        if normalVector is None: ### can happen if beforeVec = -afterVec
            return Racquet.make(gripPt, ballPt, inclination = 0)
        
        ### get vector describing racquet line
        racquetVector = myRacquet_uninclined.racquetLine.to_vector().normalize()
        
        M = normalVector.cross(racquetVector).x
        sign = np.sign(a.x) if a.x != 0 else +1
        c = sign * (a.x ** 2 + b.x ** 2) ** 0.5
        g = np.arctan(-b.x/a.x)
        
        incl_rad = np.arccos(M / c) - g
        inclination = incl_rad / np.pi
        
        return Racquet.make(gripPt, ballPt, inclination = inclination, color = color, alpha = alpha)
        
    @staticmethod
    def make_between(befRacquet, aftRacquet, lambda_bef_aft = 0.5):
        ### interpolate a racquet between two positions
        befLine, aftLine = befRacquet.racquetLine, aftRacquet.racquetLine
        # leftPt -> grip, rightPt -> spotPt

        itpl_gripPt = befLine.leftPt * (1 - lambda_bef_aft) + aftLine.leftPt * lambda_bef_aft
        itpl_spotPt = befLine.rightPt * (1 - lambda_bef_aft) + aftLine.rightPt * lambda_bef_aft

        itpl_racLine = Line3D(itpl_gripPt, itpl_spotPt) 
        ### this line is potentially too long or short! 
        ### --> shorten it at the grip, so the racquet head stays on interpolated line.
        
        unit_racVec = itpl_racLine.to_vector().normalize()
        corr_gripPt = itpl_spotPt - unit_racVec * Racquet._DEFAULT_LENGTH
        
        ### interpolate inclination
        current_inclination = befRacquet.inclination * (1 - lambda_bef_aft) + aftRacquet.inclination * lambda_bef_aft
        
        return Racquet.make(corr_gripPt, itpl_spotPt, inclination = current_inclination)
    
    
    def find_orientation_hori_vert(self):
        return self.racquetLine.find_orientation_hori_vert()
        

        