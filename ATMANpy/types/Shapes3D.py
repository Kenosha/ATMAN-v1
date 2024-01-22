# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:25:12 2023

@author: gianl
"""


import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line, Lines, CourtLines

import ATMANpy.auxil.constants as const

@dataclass(frozen = True)
class Point3D:
    """
    A point in 3D: tuple of integers, signifying x, y and z coordinates
    """
    coords: tuple ### a tuple of numbers (this is a single point) or vectors (a set of points)
    
    _DEFAULT_COLOR = const.BGR['bright_grey']
    color: Optional[tuple] = _DEFAULT_COLOR
    
    def __post_init__(self):
        #### IMMUTABILITY BREAKS ####
        object.__setattr__(self, 'x', self.coords[0])
        object.__setattr__(self, 'y', self.coords[1])
        object.__setattr__(self, 'z', self.coords[2])
        #############################

    @staticmethod
    def from_point(aPt, x = None, y = None, z = None):
        if x is not None:
            return Point3D((x, aPt.x, aPt.y))
        elif y is not None:
            return Point3D((aPt.x, y, aPt.y))
        elif z is not None:
            return Point3D((aPt.x, aPt.y, z))
        else:
            raise Exception("Specify a coordinate to add when transforming a point to a Point3D.")
    
    ### ARITHMETIC ###
    @staticmethod      
    def get_operant(op):
        try: 
            op_x, op_y, op_z = op.coords
        except:
            op_x, op_y, op_z = op, op, op
        finally:
            return op_x, op_y, op_z    
    
    def __add__(self, other):
        op_x, op_y, op_z = Point3D.get_operant(other)
        outCoords = self.x + op_x, self.y + op_y, self.z + op_z
        return Point3D(outCoords)
    def __sub__(self, other):
        op_x, op_y, op_z = Point3D.get_operant(other)
        outCoords = self.x - op_x, self.y - op_y, self.z - op_z
        return Point3D(outCoords)
    def __mul__(self, other):
        op_x, op_y, op_z = Point3D.get_operant(other)
        outCoords = self.x * op_x, self.y * op_y, self.z * op_z
        return Point3D(outCoords)
    def __truediv__(self, other):
        op_x, op_y, op_z = Point3D.get_operant(other)
        outCoords = self.x / op_x, self.y / op_y, self.z / op_z
        return Point3D(outCoords)
    def __floordiv__(self, other):
        op_x, op_y, op_z = Point3D.get_operant(other)
        outCoords = self.x // op_x, self.y // op_y, self.z // op_z
        return Point3D(outCoords)
    def __abs__(self):
        outCoords = abs(self.x), abs(self.y), abs(self.z)
        return Point3D(outCoords)
    def __round__(self, ndigits = None):
        outCoords = round(self.x, ndigits = ndigits), round(self.y, ndigits = ndigits), round(self.z, ndigits = ndigits)
        return Point3D(outCoords)
    
    ### METHODS ###
    def left(self, units):
        return Point3D((max(self.x - units, 0), self.y, self.z))

    def right(self, units):
        return Point3D((self.x + units, self.y, self.z))
    
    def front(self, units):
        return Point3D((self.x, max(self.y - units, 0), self.z))

    def back(self, units):
        return Point3D((self.x, self.y + units, self.z))

    def down(self, units):
        return Point3D((self.x, self.y, max(self.z - units, 0)))
    
    def up(self, units):
        return Point3D((self.x, self.y, self.z + units))

    def distance_to(self, aPt):
        x_dist = self.x - aPt.x
        y_dist = self.y - aPt.y
        z_dist = self.z - aPt.z
        res = (x_dist ** 2 + y_dist ** 2 + z_dist ** 2) ** 0.5
        return res
    
    def is_left(self, inObj):
        if isinstance(inObj, Point3D):
            comp = inObj.x
        else:
            comp = inObj
        return self.x < comp
    
    def is_right(self, inObj):
        if isinstance(inObj, Point3D):
            comp = inObj.x
        else:
            comp = inObj
        return self.x > comp
    
    def is_shallower(self, inObj):
        if isinstance(inObj, Point3D):
            comp = inObj.y
        else:
            comp = inObj
        return self.y < comp
    
    def is_deeper(self, inObj):
        if isinstance(inObj, Point3D):
            comp = inObj.y
        else:
            comp = inObj
        return self.y > comp
    
    def is_lower(self, inObj):
        if isinstance(inObj, Point3D):
            comp = inObj.z
        else:
            comp = inObj
        return self.z < comp
    
    def is_higher(self, inObj):
        if isinstance(inObj, Point3D):
            comp = inObj.z
        else:
            comp = inObj
        return self.z > comp

    ### VECTOR FUNCTIONS
    def length(self):
        return self.distance_to(Point3D((0, 0, 0)))
    
    def normalize(self):
        if self.length() == 0:
            return None
        else:
            return self / self.length()
    
    def find_perpendicular(self):
        ### finds one (of infinitely many) perpendicular vectors
        v1, v2, v3 = self.coords

        orthVec = Point3D((v3, v3, -v1-v2))
        orthLen = orthVec.length()
        if orthLen == 0:
            orthVec = Point3D(v3, v3, -v1-v2)
        return orthVec
    
    def find_orientation_hori_vert(self):
        x, y, z = self.coords
        r = self.length()
        theta = np.arccos(z / r)
        gamma = np.arcsin(y / (r * np.sin(theta)))
       
        dir_hori = gamma / np.pi
        dir_vert = theta / np.pi
        return dir_hori, dir_vert
    
    def cross(self, otherPt):
        return Point3D(tuple(np.cross(self.coords, otherPt.coords)))
    
    def is_vectorized(self):
        try:
            len(self.x)
        except:
            return False
        else:
            return True
        
    def get_ground_point(self):
        return Point2D((self.x, self.y))
        
    def rasterize(self, resolution = None):
        if not self.is_vectorized():
            ptArr = np.array(self.coords)[np.newaxis, :] 
        else:
            ptArr = np.vstack(self.coords).T
        return ptArr
        
    def project(self, P):
        ### make a ptArr with shape N x 3
        ptArr = self.rasterize()
        num_points, dims = ptArr.shape
        
        ### Turn coordinates homogenous
        homArr = np.concatenate([ptArr, np.ones((num_points, 1))], axis = 1)
        
        ### 
        homArr_2D = np.einsum('ij, nj -> ni', P, homArr)
        pixArr = homArr_2D[:, :2] / homArr_2D[:, 2][:, np.newaxis]
        
        return pixArr
    
    def recolor(self, new_bgr):
        return self.__class__(self.coords, color = new_bgr)    
    

@dataclass(frozen = True)
class Ball3D(Point3D):
    """
    A sphere: tuple of integers, signifying x, y and z coordinates of the center, plus a radius
    """   
    _DEFAULT_COLOR = const.BGR['bright_grey']
    radius: float = 1
    
    color: Optional[tuple] = _DEFAULT_COLOR
    limit_phi: Optional[tuple] = None
    limit_theta: Optional[tuple] = None
    
    name: str = 'Ball3D'
    
    def __post_init__(self):
        #### IMMUTABILITY BREAKS ####
        object.__setattr__(self, 'x', self.coords[0])
        object.__setattr__(self, 'y', self.coords[1])
        object.__setattr__(self, 'z', self.coords[2])
        #############################  

    ### REINSTANTIATION ###
    def reposition(self, new_coords):
        return self.__class__(new_coords, self.radius, name = self.name, color = self.color)    

    def resize(self, new_radius):
        return self.__class__(self.coords, new_radius, name = self.name, color = self.color)    

    def rename(self, new_name):
        return self.__class__(self.coords, self.radius, name = new_name, color = self.color)    

    def recolor(self, new_bgr):
        return self.__class__(self.coords, self.radius, name = self.name, color = new_bgr)    

    ### METHODS ###
    def get_point(self):
        return Point3D(self.coords)
    
    def rasterize(self, resolution, offset = 0.5):
        ### resolution -> points in a area of 1 sq. unit size 
        ## area, not volume, since we're only rasterizing surfaces! -> everything pictured is hollow!
        limit_phi = self.limit_phi
        limit_theta = self.limit_theta
        
        surface = 4 * np.pi * self.radius ** 2 
        num_points = int(surface * resolution) + 1 ### ceiling function
        
        indices = np.arange(0, num_points, dtype = float) + offset
        
        ### sunflower spacing (i.e. fibonacci spacing), from:
            # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
        phi = np.arccos(1 - 2 * indices / num_points)
        theta = np.pi * (1 + 5 ** 0.5) * indices 
        theta = theta % (2 * np.pi)
        
        ### filter out points that are outside the angle limits specified
        ## -> this is how to create half balls, ball sections, etc.
        if limit_phi is not None:
            lo, hi = limit_phi
            phi_filter = (lo < phi) & (phi < hi)
        else:
            phi_filter = np.ones(num_points, dtype = 'bool')
        
        if limit_theta is not None:
            lo, hi = limit_theta
            theta_filter = (lo < theta) & (theta < hi)
        else:
            theta_filter = np.ones(num_points, dtype = 'bool')
        
        angle_filter = phi_filter & theta_filter
        phi_filtered = phi[angle_filter]
        theta_filtered = theta[angle_filter]
        
        unit_circle = np.array([np.cos(theta_filtered) * np.sin(phi_filtered), np.sin(theta_filtered) * np.sin(phi_filtered), np.cos(phi_filtered)]).T
        
        return unit_circle * self.radius + np.array(self.coords)[np.newaxis, :]

@dataclass(frozen = True)
class Line3D:
    """
    A line in three dimensions, between two Points3D.
    """
    _DEFAULT_COLOR = const.BGR['bright_grey']
    
    leftPt: Point3D
    rightPt: Point3D
    
    name: str = 'Line3D'
    color: Optional[tuple] = _DEFAULT_COLOR
    thickness: Optional[int] = 0

    @staticmethod
    def from_line(aLn, x = None, y = None, z = None):
        leftPt = Point3D.from_point(aLn.leftPt, x = x, y = y, z = z)
        rightPt = Point3D.from_point(aLn.rightPt, x = x, y = y, z = z)
        return Line3D(leftPt, rightPt, name = aLn.name)
    
    ### REINSTANTIATION ###
    def rename(self, new_name):
        return self.__class__(self.leftPt, self.rightPt, name = new_name, color = self.name, thickness = self.thickness)
        
    def recolor(self, new_bgr):
        return self.__class__(self.leftPt, self.rightPt, name = self.name, color = new_bgr, thickness = self.thickness)
    
    def resize(self, new_thickness):
        return self.__class__(self.leftPt, self.rightPt, name = self.name, color = self.color, thickness = new_thickness)    
    
    ### METHODS ###
    def left(self, units):
        return Line3D(self.leftPt.left(units), self.rightPt.left(units), name = self.name)

    def right(self, units):
        return Line3D(self.leftPt.right(units), self.rightPt.right(units), name = self.name)
    
    def back(self, units):
        return Line3D(self.leftPt.back(units), self.rightPt.back(units), name = self.name)

    def front(self, units):
        return Line3D(self.leftPt.front(units), self.rightPt.front(units), name = self.name)

    def up(self, units):
        return Line3D(self.leftPt.up(units), self.rightPt.up(units), name = self.name)

    def down(self, units):
        return Line3D(self.leftPt.down(units), self.rightPt.down(units), name = self.name)
    
    def is_left_to_right(self):
        x_dist = abs(self.rightPt.x - self.leftPt.x)
        y_dist = abs(self.rightPt.y - self.leftPt.y)
        z_dist = abs(self.rightPt.z - self.leftPt.z) 
        
        return x_dist > y_dist and x_dist > z_dist 
    
    def is_front_to_back(self):
        x_dist = abs(self.rightPt.x - self.leftPt.x)
        y_dist = abs(self.rightPt.y - self.leftPt.y)
        z_dist = abs(self.rightPt.z - self.leftPt.z) 
        
        return y_dist > x_dist and y_dist > z_dist 
    
    def is_down_to_up(self):
        x_dist = abs(self.rightPt.x - self.leftPt.x)
        y_dist = abs(self.rightPt.y - self.leftPt.y)
        z_dist = abs(self.rightPt.z - self.leftPt.z) 
        
        return z_dist > x_dist and z_dist > y_dist 
    
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
        
    def shallow_point(self):
        if self.leftPt.is_shallower(self.rightPt):
            return self.leftPt
        else:
            return self.rightPt
    
    def deep_point(self):
        if self.leftPt.is_deeper(self.rightPt):
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
        
    def center_point(self):
        return (self.leftPt + self.rightPt) / 2
            
    def subdivide(self, elements = 100, return_coords = False):
        x_space = np.linspace(self.leftPt.x, self.rightPt.x, num = elements)
        y_space = np.linspace(self.leftPt.y, self.rightPt.y, num = elements)
        z_space = np.linspace(self.leftPt.z, self.rightPt.z, num = elements)

        x_y_z_space = zip(x_space, y_space, z_space)
        
        if return_coords:
            return np.array(list(x_y_z_space))
        else:
            return [Point3D(x_y_z) for x_y_z in x_y_z_space]
    
    def length(self):
        return self.leftPt.distance_to(self.rightPt)        
    
    def to_vector(self):
        return self.rightPt - self.leftPt
    
    def find_perpendicular(self):
        return Line3D(Point3D((0, 0, 0)), self.to_vector().find_perpendicular())
    
    def find_orientation_hori_vert(self):
        return self.to_vector().find_orientation_hori_vert()
        
    def rasterize(self, resolution):
        lineArr = self.subdivide(elements = int(resolution * self.length()) + 1, return_coords = True)
        if self.thickness == 0:
            return lineArr
        else:
            radius = self.thickness / 2
            ### 6 points around a circle, scaled up with resolution and the thickness
            # -> the thicker the line, the farther out the points, so we need more.
            num_circle_points = int(radius * resolution * 6)
            radList = np.linspace(-np.pi, np.pi, num = num_circle_points, endpoint = False) 
            ### turn into 2D -> n x dim
            radArr = radList[:, np.newaxis] 
            ### make orthogonal vectors
            orthVector_1 = self.find_perpendicular().to_vector()
            orthVector_2 = orthVector_1.cross(self.to_vector())
            # normalize to unit length
            orthVector_1, orthVector_2 = orthVector_1 / orthVector_1.length(), orthVector_2 / orthVector_2.length()
            # turn into 2D -> n x dim
            orthVector_1, orthVector_2 = np.array(orthVector_1.coords)[np.newaxis, :] , np.array(orthVector_2.coords)[np.newaxis, :] 
            # find unit_circle points
            unitArr = orthVector_1 * np.cos(radArr) + orthVector_2 * np.sin(radArr)
            # scale circle
            scaledArr = unitArr * radius
            # add center points
            outArr = np.tile(scaledArr, (len(lineArr), 1)) + np.repeat(lineArr, num_circle_points, axis = 0)
            return outArr
    
    def parametrise(self):
        """
        Point y on line can be described by:
            
            y = line_orig + n * line_vec
        """

        line_orig = self.leftPt
        line_vec = self.to_vector()
        return line_orig, line_vec
    
    def intersect(self, myReg, rectify_plane = False):
        """
        find intersection of this line with a region (a plane)
        """
        
        ### point on plane can now be described by:
            # x = reg_orig + l * reg_vec1 + m * reg_vec2
        reg_orig, reg_vec1, reg_vec2 = myReg.parametrise()
        
        if rectify_plane: ## in this case, we check if the plane is orthogonal to the ground.
                            # if it is, reg_vec1 is made to run on ground, and reg_vec2 looks up!
            if reg_vec1.z == 0 and reg_vec2.x == 0 and reg_vec2.y == 0:
                reg_orig = Point3D((reg_orig.x, reg_orig.y, 0))
                reg_vec2 = Point3D((0, 0, -reg_vec2.z))
        
        ### point on line can now be described by:
            # y = line_origin + n * line_vec
        line_orig, line_vec = self.parametrise()
            
        ### -> solve for x = y is equivalent to solving
        # A * (lambda, mu, nu) = line_origin - region_origin
        # where A = ...
        A = np.vstack([np.array(reg_vec1.coords), 
                       np.array(reg_vec2.coords), 
                       - np.array(line_vec.coords)]).T
        
        orig_diff = line_orig - reg_orig
        
        l, m, n = np.linalg.inv(A) @ np.array(orig_diff.coords)
        
        outPt = line_orig + line_vec * n
        
        return outPt, (l, m, n)
    
@dataclass(frozen = True)
class Lines3D:
    """
    A list of Lines3D
    """ 
    _DEFAULT_COLOR = const.BGR['bright_grey']
    
    lineList: List[Line3D]
    name: str = 'Lines3D'
    
    color: Optional[tuple] = _DEFAULT_COLOR
    
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
                return Lines3D(elements)
            else:
                return elements ### a single line

    def __len__(self):
        return len(self.lineList)
    
    @staticmethod
    def from_lines(aLns, x = None, y = None, z = None, name = 'Lines3D'):
        return Lines3D([Line3D.from_line(aLn, x = x, y = y, z = z) for aLn in aLns], 
                      name = name)
    
    ### REINSTANTIATION ###
    def rename(self, new_name):
        return self.__class__(self.lineList, name = new_name, color = self.color)   
        
    def recolor(self, new_bgr):
        return self.__class__(self.lineList, name = self.name, color = new_bgr)    
    
    def rasterize(self, resolution):
        return np.concatenate([aLn.rasterize(resolution) for aLn in self.lineList], axis = 0)
        
        
    
@dataclass(frozen = True)
class Region3D():
    """
    A region, i.e. a plane, in three dimensions
    """
    _DEFAULT_COLOR = const.BGR['bright_grey']
    
    lineList: List[Line3D]
    
    color: Optional[tuple] = _DEFAULT_COLOR
    name: str = 'Region3D'
    

    def __post_init__(self):
        object.__setattr__(self, 'p1', self['Top'].leftPt)
        object.__setattr__(self, 'p2', self['Top'].rightPt)
        object.__setattr__(self, 'p3', self['Bottom'].leftPt)
        object.__setattr__(self, 'p4', self['Bottom'].rightPt)
        object.__setattr__(self, 'points', (self.p1, self.p2, self.p3, self.p4))
    
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
                return Region3D(elements)
            else:
                return elements ### a single line

    def __len__(self):
        return len(self.lineList)
    
    @classmethod
    def from_points(cls, p1: Point3D, p2: Point3D, p3: Point3D, p4: Point3D, 
                    left_right = '+x', top_bottom = '+y',
                    name = 'Region3D', color = _DEFAULT_COLOR):
        # is_rectangle = p3 is None and p4 is None
        # if is_rectangle:
        #     p3 = Point2D((p1.x, p2.y))
        #     p4 = Point2D((p2.x, p1.y))
            
        ### to connect arbitrary four points without crossing lines,
        # check the two leftmost points, and the two upper points.
        def make_key_func(sign_and_coord):
            sign, coord = sign_and_coord
            if sign == '-':
                return lambda pt: - getattr(pt, coord)
            else:
                return lambda pt: getattr(pt, coord)
                
        
        l1, l2, r1, r2 = sorted([p1, p2, p3, p4], key = lambda pt: make_key_func(left_right)(pt))   
        lt, lb = sorted([l1, l2], key = make_key_func(top_bottom))
        rt, rb = sorted([r1, r2], key = make_key_func(top_bottom)) 
        
        lineList = [Line3D(lt, lb, name = 'Left'), 
                    Line3D(lt, rt, name = 'Top'), 
                    Line3D(rt, rb, name = 'Right'), 
                    Line3D(lb, rb, name = 'Bottom')]
        regLines = Lines3D(lineList)
        
        # if is_rectangle:
        #     h, w = lineList[0].length(), lineList[1].length()
        #     if area is None:
        #         area = (h + 1) * (w + 1) # note: a 'region' where all points are the same has h, w = zero, but contains one pixel -> therefore, area has to be one.
        #     if centroid is None:
        #         centroid = lt.down(h/2).right(w/2)
        
        return cls(regLines, name = name, color = color)
    
    @classmethod
    def from_line(cls, aLn, width, dim = 'x', left_right = '+x', top_bottom = '+y',
                  name = 'Region3D', color = _DEFAULT_COLOR):
        ### basically thicken a line to yield a region!
        ### the width dimension 'dim' is the dimension along which the thickness is introduced!
        if dim == 'x':
            return cls.from_points(aLn.leftPt.left(width/2),
                                   aLn.leftPt.right(width/2),
                                   aLn.rightPt.left(width/2),
                                   aLn.rightPt.right(width/2),
                                   left_right = left_right, top_bottom = top_bottom,
                                   name = name, color = color)
        if dim == 'y':
            return cls.from_points(aLn.leftPt.front(width/2),
                                   aLn.leftPt.back(width/2),
                                   aLn.rightPt.front(width/2),
                                   aLn.rightPt.back(width/2),
                                   left_right = left_right, top_bottom = top_bottom,
                                   name = name, color = color)
        if dim == 'z':
            return cls.from_points(aLn.leftPt.up(width/2),
                                   aLn.leftPt.down(width/2),
                                   aLn.rightPt.up(width/2),
                                   aLn.rightPt.down(width/2),
                                   left_right = left_right, top_bottom = top_bottom,
                                   name = name, color = color)
    
    ### REINSTANTIATION ###
    def rename(self, new_name):
        return self.__class__(self.lineList, name = new_name, color = self.color)    
    
    def recolor(self, new_bgr):
        return self.__class__(self.lineList, name = self.name, color = new_bgr)    
    
    ### METHODS ###
    def rasterize(self, resolution):
        ### resolution -> points in a area of 1 sq. unit size 
        ## area, not volume, since we're only rasterizing surfaces! -> everything pictured is hollow!
        leftLn = self['Left']
        rightLn = self['Right']
        
        linear_density = resolution ** 0.5
        
        average_length_along = (leftLn.length() + rightLn.length()) / 2
        num_points_along = int(linear_density * average_length_along) + 1
            
        leftPts = leftLn.subdivide(elements = num_points_along)
        rightPts = rightLn.subdivide(elements = num_points_along)
        lineList = [Line3D(leftPt, rightPt) for leftPt, rightPt in zip(leftPts, rightPts)]
        
        average_length_across = np.average([aLn.length() for aLn in lineList])
        num_points_across = int(linear_density * average_length_across) + 1 
        
        outArr = np.array([aLn.subdivide(elements = num_points_across, return_coords = True) for aLn in lineList])
        outArr = outArr.reshape(-1, outArr.shape[-1]) ### flatten the first two dims
        
        return outArr
    
    def parametrise(self):
        """
        Point x on plane can now be described by:
            
            x = reg_orig + l * reg_vec1 + m * reg_vec2
        """
        
        topLn, leftLn = self['Top'], self['Left']
        assert topLn.leftPt == leftLn.leftPt
        
        reg_orig = topLn.leftPt
        reg_vec1, reg_vec2 = topLn.to_vector(), leftLn.to_vector()
        return reg_orig, reg_vec1, reg_vec2

    def intersect(self, aLn):
        return aLn.intersect(self)


@dataclass(frozen = True)
class Regions3D():
    """
    A list of regions, i.e. a set of planes that make some composite shape.
    """
    _DEFAULT_COLOR = const.BGR['bright_grey']
    
    regList: List[Region3D]
    
    name: str = 'Regions3D'
    color: Optional[tuple] = _DEFAULT_COLOR

    def __iter__(self):
        return iter(self.regList)
    
    def __getitem__(self, item):
        if isinstance(item, str):
            ### get line by name
            matchLines = Lines([l for l in self.regList if l.name == item], name = item)
            if len(matchLines) == 1:
                return matchLines[0]
            elif len(matchLines) > 1:
                return matchLines
        else:
            ### get line(s) by position in list.
            elements = self.regList[item]
            if isinstance(elements, List):
                return Regions3D(elements)
            else:
                return elements ### a single line

    def __len__(self):
        return len(self.regList)

    @staticmethod
    def from_lines(aLns, width, dim = 'x', name = 'Regions3D', color = _DEFAULT_COLOR):
        return Regions3D([Region3D.from_line(aLn, width, dim = dim) for aLn in aLns], 
                         name = name, color = color)
        
    ### REINSTANTIATION ###
    def rename(self, new_name):
        return self.__class__(self.regList, name = new_name, color = self.color)   
    
    def recolor(self, new_bgr):
        return self.__class__(self.regList, name = self.name, color = new_bgr)    
    
    ### METHODS ###
    def rasterize(self, resolution):
        return np.concatenate([aReg.rasterize(resolution) for aReg in self.regList], axis = 0)
    
    
    
if __name__ == '__main__':
    
    from ATMANpy.types.Visual3D import Scene, Camera, View
    
    ### Testing angular limits on the ball
    courtScn = Scene.make_court()
    myCam = Camera.calibrated_AO_behind()
    resolution = 1
    
    # offset = 0.5
    # height = 5 * const.SCALE ### in feet
    # radius = 3 * const.SCALE ### in feet
    
    # myBall = Ball3D((200, 200, height), radius = radius)
    # myScn = courtScn.put(myBall)
    # View.make(myCam, myScn, resolution = resolution).snap().show()
    
    # limit_phi = (0, 0.5 * np.pi)
    # limit_theta = None
    # myBall = Ball3D((200, 200, height), radius = radius, limit_phi = limit_phi, limit_theta = limit_theta)
    # myScn = courtScn.put(myBall)
    # View.make(myCam, myScn, resolution = resolution).snap().show()
    
    # limit_phi = None
    # limit_theta = (0, 0.5 * np.pi)
    # myBall = Ball3D((200, 200, height), radius = radius, limit_phi = limit_phi, limit_theta = limit_theta)
    # myScn = courtScn.put(myBall)
    # View.make(myCam, myScn, resolution = resolution).snap().show()
    
    ### Testing intersection of Line and Ball
    
    regLn = Line3D(Point3D((270, 340, 0)), Point3D((340, 990, 0)))
    myReg = Region3D.from_line(regLn, width = 150, dim = 'z', color = const.BGR['leafy_green'])
    
    myLn = Line3D(Point3D((200, 1000, 50)), Point3D((400, 200, 50)), color = const.BGR['sign_orange'])
    
    myScn = courtScn.put(myReg).put(myLn)
    View.make(myCam, myScn, resolution = resolution).snap().show()
    
    intPt, params = myLn.intersect(myReg)
    intBall = Ball3D(intPt.coords, radius = 4, color = const.BGR['strong_pink'])
    
    myScn = myScn.put(intBall)
    View.make(myCam, myScn, resolution = resolution).snap().show()
    
    
    
    
    
    