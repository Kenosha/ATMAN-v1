# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:25:12 2023

@author: gianl
"""

import cv2 as cv
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line, Lines, CourtLines

import ATMANpy.auxil.constants as const

from ATMANpy.funcs.util_funcs import get_updated_attribute_dictionary, interpolate_tuples

def strict_equality(someShape, otherShape):
    if not isinstance(otherShape, someShape.__class__):
        return False

    self_attrs = get_updated_attribute_dictionary(someShape, {})
    othr_attrs = get_updated_attribute_dictionary(otherShape, {})

    if self_attrs.keys() != othr_attrs.keys():
        return False
    else:
        ### compare entry by entry, and not the whole dictionary, so we catch differences by value, and not by reference
        for k in self_attrs:
            eqBool = self_attrs[k] != othr_attrs[k]
            if isinstance(eqBool, np.array): ### deal with arrays
                eqBool == eqBool.all()
            if not eqBool:
                return False
        else:
            return True

@dataclass(frozen = True)
class Point3D:
    """
    A point in 3D: tuple of integers, signifying x, y and z coordinates
    """
    coords: tuple ### a tuple of numbers (this is a single point) or vectors (a set of points) ### NEW: a np.array of vectors
    
    _DEFAULT_COLOR = const.BGR['bright_grey']
    _DEFAULT_ALPHA = 1
    
    color: Optional[tuple] = _DEFAULT_COLOR
    alpha: Optional[float] = _DEFAULT_ALPHA

    name: str = 'Point3D'
    
    def __post_init__(self):
        #### IMMUTABILITY BREAKS ####
        object.__setattr__(self, 'x', self.coords[0])
        object.__setattr__(self, 'y', self.coords[1])
        object.__setattr__(self, 'z', self.coords[2])
        #############################

    def __repr__(self):
        return f'{self.__class__.__name__}({self.coords})'

    @staticmethod
    def from_point2D(aPt, x = None, y = None, z = None):
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
    
    @classmethod
    def get_container(cls):
        return Points3D
    
    def __add__(self, other):
        op_x, op_y, op_z = Point3D.get_operant(other)
        outCoords = self.x + op_x, self.y + op_y, self.z + op_z
        return Point3D(outCoords).to_array_formatted_vectors()
    def __sub__(self, other):
        op_x, op_y, op_z = Point3D.get_operant(other)
        outCoords = self.x - op_x, self.y - op_y, self.z - op_z
        return Point3D(outCoords).to_array_formatted_vectors()
    def __mul__(self, other):
        op_x, op_y, op_z = Point3D.get_operant(other)
        outCoords = self.x * op_x, self.y * op_y, self.z * op_z
        return Point3D(outCoords).to_array_formatted_vectors()
    def __truediv__(self, other):
        op_x, op_y, op_z = Point3D.get_operant(other)
        outCoords = self.x / op_x, self.y / op_y, self.z / op_z
        return Point3D(outCoords).to_array_formatted_vectors()
    def __floordiv__(self, other):
        op_x, op_y, op_z = Point3D.get_operant(other)
        outCoords = self.x // op_x, self.y // op_y, self.z // op_z
        return Point3D(outCoords).to_array_formatted_vectors()
    def __abs__(self):
        outCoords = abs(self.x), abs(self.y), abs(self.z)
        return Point3D(outCoords).to_array_formatted_vectors()
    def __round__(self, ndigits = None):
        outCoords = round(self.x, ndigits = ndigits), round(self.y, ndigits = ndigits), round(self.z, ndigits = ndigits)
        return Point3D(outCoords).to_array_formatted_vectors()
    def __neg__(self):
        outCoords = -self.x, -self.y, -self.z
        return Point3D(outCoords).to_array_formatted_vectors()
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        
        slfArr, othArr = np.array(self.coords), np.array(other.coords) ### make into np array so it works for single points and vectors
        try:
            return (slfArr == othArr).all() ### equality ONLY by location! (relevant for scene filtering)
        except:
            return False

    ### REINSTANTIATION ###
    
    def to_array_formatted_vectors(self):
        ### old vectorized point format: tuple of vectors
        ### new vectorized point format: np.array of vectors
        x, y, z = self.coords
        if not self.is_vectorized():
            new_coords = (x, y, z)
        else:
            new_coords = np.vstack([x, y, z])

        updDict = {'coords': new_coords}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def reposition(self, new_coords):
        updDict = {'coords': new_coords}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def rename(self, new_name):
        updDict = {'name': new_name}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def recolor(self, new_color):
        updDict = {'color': new_color}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def realpha(self, new_alpha):
        updDict = {'alpha': new_alpha}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def resize(self, new_radius): ### point does not have radius, but subclasses may!
        updDict = {'radius': new_radius}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def rephi(self, new_limit_phi):
        updDict = {'limit_phi': new_limit_phi}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def retheta(self, new_limit_theta):
        updDict = {'limit_theta': new_limit_theta}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def reoffset(self, new_offset):
        updDict = {'offset': new_offset}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def rerot_x(self, new_rot_x):
        updDict = {'rot_x': new_rot_x}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def rerot_y(self, new_rot_y):
        updDict = {'rot_y': new_rot_y}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def rerot_z(self, new_rot_z):
        updDict = {'rot_z': new_rot_z}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def rotate_x(self, rot_x):
        try:
            self.rot_x
        except AttributeError:
            return self
        else:
            return self.rerot_x(self.rot_x + rot_x)

    def rotate_y(self, rot_y):
        try:
            self.rot_y
        except AttributeError:
            return self
        else:
            return self.rerot_y(self.rot_y + rot_y)
    
    def rotate_z(self, rot_z):
        try:
            self.rot_z
        except AttributeError:
            return self
        else:
            return self.rerot_z(self.rot_z + rot_z)
    
    def interpolate(self, otherPt, itpl_lambda = 0.5):
        if not self.is_vectorized() and not otherPt.is_vectorized():
            avg_coords = interpolate_tuples(self.coords, otherPt.coords, itpl_lambda = itpl_lambda)
        elif self.is_vectorized() and otherPt.is_vectorized():
            avg_coords = self.coords * (1 - itpl_lambda) + otherPt.coords * itpl_lambda
        else:
            print(f'Cannot interpolate vectorized and unvectorized Point.')

        avg_color = interpolate_tuples(self.color, otherPt.color, itpl_lambda = itpl_lambda)
        avg_alpha = self.alpha * (1 - itpl_lambda) + otherPt.alpha * itpl_lambda
        
        updDict = {'coords': avg_coords,
                   'color': avg_color,
                   'alpha': avg_alpha}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def dim(self, dim_factor):
        dimmed_color = tuple([max(0, min(int(c * dim_factor), 255)) for c in self.color]) ### at dim_factor = 1, nothing changes
        is_extreme = [(dim_c == 0 < old_c) or (old_c < dim_c == 255) for old_c, dim_c in zip(self.color, dimmed_color)]
        # if any(is_extreme):
        #     print(f'Dimming {self.color} to {dimmed_color} hit 0 or 255, verify if color satisfactory!')
        updDict = {'color': dimmed_color} 
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def transparify(self, transparence_factor):
        transparent_alpha = np.clip(self.alpha * transparence_factor, 0, 1) ### at transparence_factor = 1, nothing changes
        updDict = {'alpha': transparent_alpha}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    ### METHODS ###
    def left(self, units):
        x, y, z = self.x - units, self.y, self.z
        
        if not self.is_vectorized():
            new_coords = (x, y, z)
        else:
            new_coords = np.vstack([x, y, z])

        return self.reposition(new_coords)

    def right(self, units):
        x, y, z = self.x + units, self.y, self.z
        
        if not self.is_vectorized():
            new_coords = (x, y, z)
        else:
            new_coords = np.vstack([x, y, z])

        return self.reposition(new_coords)
    
    def front(self, units):
        x, y, z = self.x, self.y - units, self.z
        
        if not self.is_vectorized():
            new_coords = (x, y, z)
        else:
            new_coords = np.vstack([x, y, z])

        return self.reposition(new_coords)

    def back(self, units):
        x, y, z = self.x, self.y + units, self.z
        
        if not self.is_vectorized():
            new_coords = (x, y, z)
        else:
            new_coords = np.vstack([x, y, z])

        return self.reposition(new_coords)

    def down(self, units):
        x, y, z = self.x, self.y, self.z - units
        
        if not self.is_vectorized():
            new_coords = (x, y, z)
        else:
            new_coords = np.vstack([x, y, z])

        return self.reposition(new_coords)
    
    def up(self, units):
        x, y, z = self.x, self.y, self.z + units
        
        if not self.is_vectorized():
            new_coords = (x, y, z)
        else:
            new_coords = np.vstack([x, y, z])

        return self.reposition(new_coords)

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

    def center_point(self):
        return self
            
    def center_coords(self):
        return self.center_point().coords

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
            orthVec = Point3D((-v2-v3, v1, v1))
        return orthVec
    
    def find_orientation_hori_vert(self):
        #### https://math.stackexchange.com/questions/2247039/angles-of-a-known-3d-vector, but not really...
        ### but i am fairly sure it's the same coordinate system.
        x, y, z = self.coords
        r = self.length()
        theta = np.arccos(z / r)
        # gamma = np.arcsin(y / (r * np.sin(theta)))
        gamma = np.arctan2(y, x)
       
        dir_hori = gamma / np.pi
        dir_vert = theta / np.pi
        return dir_hori, dir_vert
    
    def cross(self, otherPt):
        return self.__class__(tuple(np.cross(self.coords, otherPt.coords)))
    
    def is_vectorized(self):
        try:
            len(self.x)
        except:
            return False
        else:
            return True
    
    def devectorize(self):
        if not self.is_vectorized():
            return self
        else:
            tuples = [(x, y, z) for x, y, z in zip(self.x, self.y, self.z)]
            attr_dicts = [get_updated_attribute_dictionary(self, {'coords': tup}) for tup in tuples]
            return Points3D([self.__class__(**attr_dict) for attr_dict in attr_dicts])

    def num_elements(self):
        if self.is_vectorized():
            return len(self.x)
        else:
            return 1

    def get_ground_point(self):
        return Point2D((self.x, self.y))
        
    def rasterize(self, resolution = None):
        if not self.is_vectorized():
            ptArr = np.array(self.coords)[np.newaxis, :] 
        else:
            ptArr = np.vstack(self.coords).T
        return ptArr
        
    def project(self, P, resolution = 5):
        ### make a ptArr with shape N x 3
        ptArr = self.rasterize(resolution = resolution)
        num_points, dims = ptArr.shape
        
        ### Turn coordinates homogenous
        homArr = np.concatenate([ptArr, np.ones((num_points, 1))], axis = 1)
        
        ### 
        homArr_2D = np.einsum('ij, nj -> ni', P, homArr)
        pixArr = homArr_2D[:, :2] / homArr_2D[:, 2][:, np.newaxis]
        
        return pixArr
    
    def annotate(self, text):
        attrDict = get_updated_attribute_dictionary(self, {})
        return AnnotatedPoint3D(text = text, **attrDict)

@dataclass(frozen = True, eq = False)
class AnnotatedPoint3D(Point3D):
    ### Used to store a block of text that appears somewhere in a 3D scene, until after the scene has been rasterized and snapped to 2D
    text: str = None
    size: int = 1
    thickness: int = 1
    color: tuple = (0, 255, 0)
    font: int = cv.FONT_HERSHEY_PLAIN
    lineType: int = cv.LINE_AA
    
    def __repr__(self):
        if len(self.text) > 20:
            textStr = self.text[:10] + ' ... ' + self.text[-10:]
        else:
            textStr = self.text
        return f'{self.__class__.__name__}({self.coords}, text = {textStr})'
    
    def get_parameters(self):
        return {'text': self.text,
                'size': self.size,
                'thickness': self.thickness,
                'color': self.color,
                'font': self.font,
                'lineType': self.lineType}
    
    def retext(self, new_text): ### point does not have radius, but subclasses may!
        updDict = {'text': new_text}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def resize(self, new_size): ### point does not have radius, but subclasses may!
        updDict = {'size': new_size}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def rethicken(self, new_thickness): ### point does not have radius, but subclasses may!
        updDict = {'thickness': new_thickness}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def refont(self, new_font): ### point does not have radius, but subclasses may!
        updDict = {'font': new_font}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def reline(self, new_lineType):
        updDict = {'lineType': new_lineType}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    

@dataclass(frozen = True, eq = False)
class Circle3D(Point3D):
    """
    A circle: tuple of integers, signifying x, y and z coordinates of the center, plus a radius
    """
    _DEFAULT_COLOR = const.BGR['bright_grey']
    _DEFAULT_ALPHA = 1
    
    radius: float = 1
    color: tuple = _DEFAULT_COLOR
    alpha: float = _DEFAULT_ALPHA
    limit_theta: tuple = None
    start_theta: float = 0 ### this is to start the circle somewhere else than (x, y) = (-1, 0)
    fixed_dimension: str = 'z' ### this is the dimension along which the circle is flat

    rot_x: float = 0
    rot_y: float = 0
    rot_z: float = 0
    
    name: str = 'Circle3D'
    
    def __post_init__(self):
        #### IMMUTABILITY BREAKS ####
        object.__setattr__(self, 'x', self.coords[0])
        object.__setattr__(self, 'y', self.coords[1])
        object.__setattr__(self, 'z', self.coords[2])
        #############################  
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.coords}, radius = {self.radius})'

    def __eq__(self, otherCrc):
        if not Point3D.__eq__(self, otherCrc): ### check for coordinate equality
            return False
        else:
            rad_eq = self.radius == otherCrc.radius
            lim_eq = self.limit_theta == otherCrc.limit_theta
            fix_eq = self.fixed_dimension == otherCrc.fixed_dimension
            return rad_eq and lim_eq and fix_eq

    @staticmethod
    def from_point3D(myPnt, radius = 1, limit_theta = None, start_theta = 0,
                     fixed_dimension = 'z', name = 'Circle3D', 
                     rot_x = 0, rot_y = 0, rot_z = 0):
        if limit_theta is None:
            limit_theta = (0, 2 * np.pi)
        return Circle3D(myPnt.coords, color = myPnt.color, alpha = myPnt.alpha,
                        radius = radius, limit_theta = limit_theta, start_theta = start_theta,
                        fixed_dimension = fixed_dimension,
                        rot_x = rot_x, rot_y = rot_y, rot_z = rot_z,
                        name = name)

    @classmethod
    def get_container(cls):
        return Circles3D

    ### METHODS ###
    def get_point(self):
        return Point3D(self.coords)
    
    def interpolate(self, otherCrc, itpl_lambda = 0.5):
        assert self.fixed_dimension == otherCrc.fixed_dimension

        selfPt, otherPt = self.get_point(), otherCrc.get_point()
        itplPt = selfPt.interpolate(otherPt, itpl_lambda = 0.5)
        itpl_coords = itplPt.coords

        itpl_color = interpolate_tuples(self.color, otherCrc.color, itpl_lambda = itpl_lambda)
        itpl_limit_theta = interpolate_tuples(self.limit_theta, otherCrc.limit_theta, itpl_lambda = itpl_lambda)
        itpl_start_theta = (1 - itpl_lambda) * self.start_theta + itpl_lambda * otherCrc.start_theta
        itpl_alpha = (1 - itpl_lambda) * self.alpha + itpl_lambda * otherCrc.alpha
        itpl_radius = (1 - itpl_lambda) * self.radius + itpl_lambda * otherCrc.radius

        return Circle3D(coords = itpl_coords, color = itpl_color, alpha = itpl_alpha,
                        radius = itpl_radius, limit_theta = itpl_limit_theta, start_theta = itpl_start_theta,
                        fixed_dimension = self.fixed_dimension)

    def rasterize(self, resolution):
        ### resolution: number of points on one arc length
        if self.limit_theta is None:
            limit_theta = (0, 2 * np.pi)
        else:
            limit_theta = self.limit_theta
        
        circumference = 2 * np.pi * self.radius
        num_points = int(circumference * resolution) + 1 ### ceiling function
        
        zeros = np.zeros(num_points)
        theta = np.linspace(0, 2 * np.pi, endpoint = False, num = num_points)
        
        ### filter out points that are outside the angle limits specified
        ## -> this is how to create semicircles, circle sections, etc.    
        lo, hi = limit_theta
        theta_filter = (lo < theta) & (theta < hi)
        
        zeros_filtered = zeros[theta_filter]
        theta_filtered = theta[theta_filter] + self.start_theta
        
        if self.fixed_dimension == 'x':
            unit_circle = np.array([zeros_filtered, np.cos(theta_filtered), np.sin(theta_filtered)]).T
        if self.fixed_dimension == 'y':
            unit_circle = np.array([np.cos(theta_filtered), zeros_filtered, np.sin(theta_filtered)]).T
        if self.fixed_dimension == 'z':
            unit_circle = np.array([np.cos(theta_filtered), np.sin(theta_filtered), zeros_filtered]).T

        ### rotate around three axes, to customize angle if needed
        # 1) x
        g = self.rot_x
        cg, sg = np.cos(g), np.sin(g)
        R = np.array([[1    , 0     , 0     ],
                      [0    , cg    , -sg   ],
                      [0    , sg    , cg    ]])
        unit_circle = np.einsum('ij, ni -> nj', R, unit_circle)

        # 2) y
        g = self.rot_y
        cg, sg = np.cos(g), np.sin(g)
        R = np.array([[cg   , 0     , sg    ],
                      [0    , 1     , 0     ],
                      [-sg  , 0     , cg    ]])
        unit_circle = np.einsum('ij, ni -> nj', R, unit_circle)

        # 3) z        
        g = self.rot_z
        cg, sg = np.cos(g), np.sin(g)
        R = np.array([[cg   , -sg   , 0     ],
                      [sg   , cg    , 0     ],
                      [0    , 0     , 1     ]])
        unit_circle = np.einsum('ij, ni -> nj', R, unit_circle)

        ### translate the unit circle coordinates by the center, and stretch by radius
        coordArr = np.array(self.coords)

        if len(coordArr.shape) == 1: ### in this case, we have a simple point -> self.coords is a tuple
            outArr = unit_circle * self.radius + coordArr[np.newaxis, :]
        elif len(coordArr.shape) == 2: ### in this case, self.coords is already an array, holding a vectorized set of circle centers
            outArr = unit_circle[:, :, np.newaxis] * self.radius + coordArr[np.newaxis, :, :]
            outArr = outArr.swapaxes(1, 2) ### now, the array is: resolution_points x vector_points x dimensions (=3) 
            n_res, n_vec, n_dims = outArr.shape
            outArr = outArr.reshape((n_res * n_vec, n_dims))
        return outArr


@dataclass(frozen = True, eq = False)
class Ellipse3D(Point3D):
    """
    A circle: tuple of integers, signifying x, y and z coordinates of the center, plus a radius
    ### TODO: for the moment, the circle is flat on the ground -> generalize!
    """
    _DEFAULT_COLOR = const.BGR['bright_grey']
    _DEFAULT_ALPHA = 1
    
    radius_long: float = 1
    radius_short: float = 1
    theta: float = 0
    color: Optional[tuple] = _DEFAULT_COLOR
    alpha: Optional[float] = _DEFAULT_ALPHA
    limit_angle: Optional[tuple] = None
    fixed_dimension: str = 'z' ### this is the dimension along which the circle is flat
    rot_x: float = 0
    rot_y: float = 0
    rot_z: float = 0

    name: str = 'Ellipse3D'
    
    def __post_init__(self):
        #### IMMUTABILITY BREAKS ####
        object.__setattr__(self, 'x', self.coords[0])
        object.__setattr__(self, 'y', self.coords[1])
        object.__setattr__(self, 'z', self.coords[2])
        #############################  
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.coords}, long radius = {self.radius_long}, short radius = {self.radius_short})'

    def __eq__(self, otherEll):
        if not Point3D.__eq__(self, otherEll): ### check for coordinate equality
            return False
        else:
            rad_long_eq = self.radius_long == otherEll.radius_long
            rad_short_eq = self.radius_short == otherEll.radius_short
            tht_eq = self.theta == otherEll.theta
            lim_eq = self.limit_angle == otherEll.limit_angle
            fix_eq = self.fixed_dimension == otherEll.fixed_dimension
            return rad_long_eq and rad_short_eq and tht_eq and lim_eq and fix_eq
        
    @staticmethod
    def from_point3D(myPnt, radius_long = 1, radius_short = 1, theta = 0, limit_angle = None, fixed_dimension = 'z', 
                     rot_x = 0, rot_y = 0, rot_z = 0,
                     name = 'Ellipse3D'):
        return Ellipse3D(myPnt.coords, color = myPnt.color, alpha = myPnt.alpha,
                            radius_long = radius_long, radius_short = radius_short, theta = theta,
                            limit_angle = limit_angle, fixed_dimension = fixed_dimension,
                            rot_x = rot_x, rot_y = rot_y, rot_z = rot_z,
                            name = name)

    @classmethod
    def get_container(cls):
        # return Circles3D
        return Ellipses3D

    ### METHODS ###
    def get_point(self):
        return Point3D(self.coords)
    
    def rasterize(self, resolution):
        ### resolution: number of points on one arc length
        limit_angle = self.limit_angle
        r1, r2 = self.radius_long, self.radius_short
        theta = self.theta

        h = ((r1 - r2) ** 2) / ((r1 + r2) ** 2)
        circumference = np.pi * (r1 + r2) * (1 + (3 * h) / (10 + (4 - 3 * h) ** 2)) ### ramanujan's approximation
        num_points = int(circumference * resolution) + 1 ### ceiling function
        
        zeros = np.zeros(num_points)
        angles = np.linspace(0, 2 * np.pi, endpoint = False, num = num_points)
        
        ### filter out points that are outside the angle limits specified
        ## -> this is how to create semicircles, circle sections, etc.
        if limit_angle is not None:
            lo, hi = limit_angle
            angle_filter = (lo < angles) & (angles < hi)
        else:
            angle_filter = np.ones(num_points, dtype = 'bool')
        
        zeros_filtered = zeros[angle_filter]
        angles_filtered = angles[angle_filter]
        
        d1 = r1 * np.cos(theta) * np.cos(angles_filtered) - r2 * np.sin(theta) * np.sin(angles_filtered)
        d2 = r1 * np.sin(theta) * np.cos(angles_filtered) + r2 * np.cos(theta) * np.sin(angles_filtered) 

        if self.fixed_dimension == 'x':
            base_ellipse = np.array([zeros_filtered, d1, d2]).T
        if self.fixed_dimension == 'y':
            base_ellipse = np.array([d1, zeros_filtered, d2]).T
        if self.fixed_dimension == 'z':
            base_ellipse = np.array([d1, d2, zeros_filtered]).T

        ### rotate around three axes, to customize angle if needed
        # 1) x
        g = self.rot_x
        cg, sg = np.cos(g), np.sin(g)
        R = np.array([[1    , 0     , 0     ],
                      [0    , cg    , -sg   ],
                      [0    , sg    , cg    ]])
        base_ellipse = np.einsum('ij, ni -> nj', R, base_ellipse)

        # 2) y
        g = self.rot_y
        cg, sg = np.cos(g), np.sin(g)
        R = np.array([[cg   , 0     , sg    ],
                      [0    , 1     , 0     ],
                      [-sg  , 0     , cg    ]])
        base_ellipse = np.einsum('ij, ni -> nj', R, base_ellipse)

        # 3) z        
        g = self.rot_z
        cg, sg = np.cos(g), np.sin(g)
        R = np.array([[cg   , -sg   , 0     ],
                      [sg   , cg    , 0     ],
                      [0    , 0     , 1     ]])
        base_ellipse = np.einsum('ij, ni -> nj', R, base_ellipse)
    
        ### translate the ellipse coordinates by the center
        coordArr = np.array(self.coords)

        if len(coordArr.shape) == 1: ### in this case, we have a simple point -> self.coords is a tuple
            outArr = base_ellipse + coordArr[np.newaxis, :]
        elif len(coordArr.shape) == 2: ### in this case, self.coords is already an array, holding a vectorized set of circle centers
            outArr = base_ellipse[:, :, np.newaxis] + coordArr[np.newaxis, :, :]
            outArr = outArr.swapaxes(1, 2) ### now, the array is: resolution_points x vector_points x dimensions (=3) 
            n_res, n_vec, n_dims = outArr.shape
            outArr = outArr.reshape((n_res * n_vec, n_dims))
        return outArr

@dataclass(frozen = True, eq = False)
class Ball3D(Point3D):
    """
    A sphere: tuple of integers, signifying x, y and z coordinates of the center, plus a radius
    """   
    _DEFAULT_COLOR = const.BGR['bright_grey']
    radius: float = 1
    
    color: Optional[tuple] = _DEFAULT_COLOR
    alpha: Optional[float] = 1
    
    limit_phi: Optional[tuple] = None
    limit_theta: Optional[tuple] = None
    offset: Optional[float] = 0.5 ### this is for pseudo-rotations in the rasterization pattern

    rot_x: float = 0
    rot_y: float = 0
    rot_z: float = 0
    
    name: str = 'Ball3D'
    
    def __post_init__(self):
        #### IMMUTABILITY BREAKS ####
        object.__setattr__(self, 'x', self.coords[0])
        object.__setattr__(self, 'y', self.coords[1])
        object.__setattr__(self, 'z', self.coords[2])
        #############################  

    def __repr__(self):
        return f'{self.__class__.__name__}({self.coords}, radius = {self.radius})'

    def __eq__(self, otherBall):
        if not Point3D.__eq__(self, otherBall): ### check for coordinate equality
            return False
        else:
            rad_eq = self.radius == otherBall.radius
            phi_eq = self.limit_phi == otherBall.limit_phi
            tht_eq = self.limit_theta == otherBall.limit_theta
            return rad_eq and phi_eq and tht_eq
        
    @staticmethod
    def from_point3D(myPnt, radius = 1, limit_phi = None, limit_theta = None, offset = 0.5, 
                     rot_x = 0, rot_y = 0, rot_z = 0,
                     name = 'Ball3D'):
        return Ball3D(myPnt.coords, color = myPnt.color, alpha = myPnt.alpha, 
                      radius = radius, limit_phi = limit_phi, limit_theta = limit_theta, offset = offset,
                      rot_x = rot_x, rot_y = rot_y, rot_z = rot_z,
                      name = name)

    @classmethod
    def get_container(cls):
        return Balls3D

    ### METHODS ###
    def get_point(self):
        return Point3D(self.coords)
    
    def rasterize(self, resolution):
        ### resolution -> points in a area of 1 sq. unit size 
        ## area, not volume, since we're only rasterizing surfaces! -> everything pictured is hollow!
        limit_phi = self.limit_phi
        limit_theta = self.limit_theta
        
        surface = 4 * np.pi * self.radius ** 2 
        num_points = int(surface * resolution) + 1 ### ceiling function
        
        indices = np.arange(0, num_points, dtype = float) + self.offset
        
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

        ### rotate around three axes, to customize angle if needed
        # 1) x
        g = self.rot_x
        cg, sg = np.cos(g), np.sin(g)
        R = np.array([[1    , 0     , 0     ],
                      [0    , cg    , -sg   ],
                      [0    , sg    , cg    ]])
        unit_circle = np.einsum('ij, ni -> nj', R, unit_circle)

        # 2) y
        g = self.rot_y
        cg, sg = np.cos(g), np.sin(g)
        R = np.array([[cg   , 0     , sg    ],
                      [0    , 1     , 0     ],
                      [-sg  , 0     , cg    ]])
        unit_circle = np.einsum('ij, ni -> nj', R, unit_circle)

        # 3) z        
        g = self.rot_z
        cg, sg = np.cos(g), np.sin(g)
        R = np.array([[cg   , -sg   , 0     ],
                      [sg   , cg    , 0     ],
                      [0    , 0     , 1     ]])
        unit_circle = np.einsum('ij, ni -> nj', R, unit_circle)

        ### translate the unit ball coordinates by the center, and stretch by radius
        coordArr = np.array(self.coords)

        if len(coordArr.shape) == 1: ### in this case, we have a simple point -> self.coords is a tuple
            outArr = unit_circle * self.radius + coordArr[np.newaxis, :]
        elif len(coordArr.shape) == 2: ### in this case, self.coords is already an array, holding a vectorized set of circle centers
            outArr = unit_circle[:, :, np.newaxis] * self.radius + coordArr[np.newaxis, :, :]
            outArr = outArr.swapaxes(1, 2) ### now, the array is: resolution_points x vector_points x dimensions (=3) 
            n_res, n_vec, n_dims = outArr.shape
            outArr = outArr.reshape((n_res * n_vec, n_dims))
        return outArr
        
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
    alpha: Optional[float] = 1
    contains_endpoint: Optional[bool] = True ### if False, the Line is half-open:
                                               # it contains every point arbitrarily close, but not equal, to the endpoint.
                                               # this is used in subdivision (and, subsequently, rasterization.)
    def __repr__(self):
        return f'{self.__class__.__name__}(left = {self.leftPt}, right = {self.rightPt})'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.leftPt == other.leftPt) and (self.rightPt == other.rightPt)

    @staticmethod
    def from_line2D(aLn, x = None, y = None, z = None):
        leftPt = Point3D.from_point2D(aLn.leftPt, x = x, y = y, z = z)
        rightPt = Point3D.from_point2D(aLn.rightPt, x = x, y = y, z = z)
        return Line3D(leftPt, rightPt, name = aLn.name)
    
    @classmethod
    def get_container(cls):
        return Lines3D
    
    def __add__(self, other):
        if isinstance(other, Line3D):
            new_leftPt = self.leftPt + other.leftPt
            new_rightPt = self.rightPt + other.rightPt
        elif isinstance(other, Point3D):
            new_leftPt = self.leftPt + other
            new_rightPt = self.rightPt + other
        else:
            new_leftPt = self.leftPt + other
            new_rightPt = self.rightPt + other
        updDict = {'leftPt': new_leftPt, 'rightPt': new_rightPt}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
        
    def __sub__(self, other):
        if isinstance(other, Line3D):
            new_leftPt = self.leftPt - other.leftPt
            new_rightPt = self.rightPt - other.rightPt
        elif isinstance(other, Point3D):
            new_leftPt = self.leftPt - other
            new_rightPt = self.rightPt - other
        else:
            new_leftPt = self.leftPt - other
            new_rightPt = self.rightPt - other
        updDict = {'leftPt': new_leftPt, 'rightPt': new_rightPt}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def __mul__(self, other):
        if isinstance(other, Line3D):
            raise TypeError('Multiplying Line3D with Line3D not supported!')
        elif isinstance(other, Point3D):
            new_leftPt = self.leftPt * other
            new_rightPt = self.rightPt * other
        else:
            new_leftPt = self.leftPt * other
            new_rightPt = self.rightPt * other
        updDict = {'leftPt': new_leftPt, 'rightPt': new_rightPt}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def __div__(self, other):
        if isinstance(other, Line3D):
            raise TypeError('Dividing Line3D with Line3D not supported!')
        elif isinstance(other, Point3D):
            new_leftPt = self.leftPt / other
            new_rightPt = self.rightPt / other
        else:
            new_leftPt = self.leftPt / other
            new_rightPt = self.rightPt / other
        updDict = {'leftPt': new_leftPt, 'rightPt': new_rightPt}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def __truediv__(self, other):
        if isinstance(other, Line3D):
            raise TypeError('Floor Dividing Line3D with Line3D not supported!')
        elif isinstance(other, Point3D):
            new_leftPt = self.leftPt // other
            new_rightPt = self.rightPt // other
        else:
            new_leftPt = self.leftPt // other
            new_rightPt = self.rightPt // other
        updDict = {'leftPt': new_leftPt, 'rightPt': new_rightPt}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def __abs__(self):
        new_leftPt, new_rightPt = abs(self.leftPt), abs(self.rightPt)
        updDict = {'leftPt': new_leftPt, 'rightPt': new_rightPt}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def __round__(self):
        new_leftPt, new_rightPt = round(self.leftPt), round(self.rightPt)
        updDict = {'leftPt': new_leftPt, 'rightPt': new_rightPt}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def __neg__(self):
        new_leftPt, new_rightPt = -self.leftPt, -self.rightPt
        updDict = {'leftPt': new_leftPt, 'rightPt': new_rightPt}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    ### REINSTANTIATION ###
    def rename(self, new_name):
        updDict = {'name': new_name}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def recolor(self, new_color):
        updDict = {'color': new_color}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def realpha(self, new_alpha):
        updDict = {'alpha': new_alpha}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def resize(self, new_thickness):
        updDict = {'thickness': new_thickness}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def reverse(self):
        updDict = {'leftPt': self.rightPt, 'rightPt': self.leftPt}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def dim(self, dim_factor):
        return Point3D.dim(self, dim_factor) ### Point3D is class agnostic for dimming. TODO: repeat this pattern for other cases!
        
    def transparify(self, transparence_factor):
        transparent_alpha = np.clip(self.alpha * transparence_factor, 0, 1) ### at transparence_factor = 1, nothing changes
        updDict = {'alpha': transparent_alpha}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def interpolate(self, otherLn, itpl_lambda = 0.5):
        avg_leftPt = self.leftPt.interpolate(otherLn.leftPt, itpl_lambda = itpl_lambda)
        avg_rightPt = self.rightPt.interpolate(otherLn.rightPt, itpl_lambda = itpl_lambda)
        avg_color = interpolate_tuples(self.color, otherLn.color, itpl_lambda = itpl_lambda)
        avg_alpha = self.alpha * (1 - itpl_lambda) + otherLn.alpha * itpl_lambda
        avg_thickness = self.thickness * (1 - itpl_lambda) + otherLn.thickness * itpl_lambda

        updDict = {'leftPt': avg_leftPt,
                   'rightPt': avg_rightPt,
                   'color': avg_color,
                   'alpha': avg_alpha,
                   'thickness': avg_thickness}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    ### METHODS ###
    def left(self, units):
        updDict = {'leftPt': self.leftPt.left(units), 'rightPt': self.rightPt.left(units)}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def right(self, units):
        updDict = {'leftPt': self.leftPt.right(units), 'rightPt': self.rightPt.right(units)}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def back(self, units):
        updDict = {'leftPt': self.leftPt.back(units), 'rightPt': self.rightPt.back(units)}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def front(self, units):
        updDict = {'leftPt': self.leftPt.front(units), 'rightPt': self.rightPt.front(units)}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def up(self, units):
        updDict = {'leftPt': self.leftPt.up(units), 'rightPt': self.rightPt.up(units)}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def down(self, units):
        updDict = {'leftPt': self.leftPt.down(units), 'rightPt': self.rightPt.down(units)}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def flip(self):
        updDict = {'leftPt': self.rightPt, 'rightPt': self.leftPt}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def is_left_to_right(self):
        x_dist = abs(self.rightPt.x - self.leftPt.x)
        y_dist = abs(self.rightPt.y - self.leftPt.y)
        z_dist = abs(self.rightPt.z - self.leftPt.z) 
        
        return x_dist >= y_dist and x_dist >= z_dist 
    
    def is_front_to_back(self):
        x_dist = abs(self.rightPt.x - self.leftPt.x)
        y_dist = abs(self.rightPt.y - self.leftPt.y)
        z_dist = abs(self.rightPt.z - self.leftPt.z) 
        
        return y_dist >= x_dist and y_dist >= z_dist 
    
    def is_down_to_up(self):
        x_dist = abs(self.rightPt.x - self.leftPt.x)
        y_dist = abs(self.rightPt.y - self.leftPt.y)
        z_dist = abs(self.rightPt.z - self.leftPt.z) 
        
        return z_dist >= x_dist and z_dist >= y_dist 
    
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
        if self.leftPt.is_lower(self.rightPt):
            return self.leftPt
        else:
            return self.rightPt
    
    def high_point(self):
        if self.leftPt.is_higher(self.rightPt):
            return self.leftPt
        else:
            return self.rightPt
        
    def center_point(self):
        return (self.leftPt + self.rightPt) / 2
            
    def center_coords(self):
        return self.center_point().coords

    def subdivide(self, elements = 100, return_coords = False):
        x_space = np.linspace(self.leftPt.x, self.rightPt.x, num = elements, endpoint = self.contains_endpoint)
        y_space = np.linspace(self.leftPt.y, self.rightPt.y, num = elements, endpoint = self.contains_endpoint)
        z_space = np.linspace(self.leftPt.z, self.rightPt.z, num = elements, endpoint = self.contains_endpoint)

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
        
    def is_vectorized(self):
        return self.leftPt.is_vectorized()

    def devectorize(self):
        if not self.is_vectorized():
            return self
        else:
            left_tuples = [tuple(c) for c in self.leftPt.coords.T]
            right_tuples = [tuple(c) for c in self.rightPt.coords.T]

            attr_dicts = [get_updated_attribute_dictionary(self, {'leftPt': Point3D(l_tup), 'rightPt': Point3D(r_tup)}) for l_tup, r_tup in zip(left_tuples, right_tuples)]
            return Lines3D([self.__class__(**attr_dict) for attr_dict in attr_dicts])

    def rasterize(self, resolution):
        ### idea: if necessary, de-vectorize the Line3D to Lines3D, and rasterize that
        if self.is_vectorized():
            return self.devectorize().rasterize(resolution)
        else:
            line_len = self.length()
            if not isinstance(line_len, int): ### in this case, our Line3D object is a vectorized representation of multiple lines
                line_len = np.max(line_len)

            lineArr = self.subdivide(elements = int(resolution * line_len) + 1, return_coords = True)
            if self.thickness == 0:
                return lineArr
            else:
                radius = self.thickness / 2
                ### 6 points around a circle, scaled up with resolution and the thickness
                # -> the thicker the line, the farther out the points, so we need more.
                num_circle_points = max(1, int(radius * resolution * 6))
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
    
    def dash(self, num_dashes = 10):
        subPnts = self.subdivide(num_dashes * 2)
        subLns = Lines3D([Line3D(subPnts[idx], subPnts[idx + 1]) for idx in range(len(subPnts) - 1)])
        return subLns[::2]
    
    def shorten_by(self, factor, N = 100, which = 'left'):
        assert which in ['both', 'left', 'right']
        assert 0 < factor < 1

        subPnts = self.subdivide(N)

        factor_idx = max(1, round(factor * N)) - 1
        factor_idx_left = factor_idx if which in ['both', 'left'] else 0
        factor_idx_right = -factor_idx if which in ['both', 'right'] else 0

        cutPt_left = subPnts[factor_idx_left]
        cutPt_right = subPnts[factor_idx_right]

        updDict = {'leftPt': cutPt_left, 'rightPt': cutPt_right}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return Line3D(**newDict)
    
    def lengthen_by(self, factor, which = 'left'):
        assert which in ['both', 'left', 'right']

        p1, p2 = self.leftPt, self.rightPt

        if which == 'left':
            add_left = factor - 1
            add_right = 0
        elif which == 'right':
            add_left = 0
            add_right = factor - 1
        else:
            add_left = add_right = (factor - 1)

        p1_new = (p1 - p2) * add_left + p1
        p2_new = (p2 - p1) * add_right + p2

        updDict = {'leftPt': p1_new, 'rightPt': p2_new}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return Line3D(**newDict)


@dataclass(frozen = True)
class Region3D():
    """
    A region, i.e. a plane, in three dimensions
    """
    _DEFAULT_COLOR = const.BGR['bright_grey']
    _DEFAULT_ALPHA = 1
    
    lineList: List[Line3D]
    
    color: Optional[tuple] = _DEFAULT_COLOR
    alpha: Optional[float] = _DEFAULT_ALPHA
    
    name: str = 'Region3D'
    
    def __post_init__(self):
        object.__setattr__(self, 'p1', self['Top'].leftPt)
        object.__setattr__(self, 'p2', self['Top'].rightPt)
        object.__setattr__(self, 'p3', self['Bottom'].leftPt)
        object.__setattr__(self, 'p4', self['Bottom'].rightPt)
        object.__setattr__(self, 'points', (self.p1, self.p2, self.p3, self.p4))
    
    def __repr__(self):
        return f'{self.__class__.__name__}(p1 = {self.p1}, p2 = {self.p2}, p3 = {self.p3}, p4 = {self.p4})'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return all([myLn == othLn for myLn, othLn in zip(self, other)])

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
    def from_region2D(cls, myReg, x = None, y = None, z = None):
        return Region3D([Line3D.from_line2D(myLn, x = x, y = y, z = z) for myLn in myReg.lineList])

    @classmethod
    def from_points3D(cls, p1: Point3D, p2: Point3D, p3: Point3D, p4: Point3D, 
                    left_right = '+x', top_bottom = '+y',
                    name = 'Region3D', color = _DEFAULT_COLOR, alpha = _DEFAULT_ALPHA):
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
    def from_line3D(cls, aLn, width, dim = 'x', left_right = '+x', top_bottom = '+y',
                  name = 'Region3D', color = _DEFAULT_COLOR, alpha = _DEFAULT_ALPHA):
        ### basically thicken a line to yield a region!
        ### the width dimension 'dim' is the dimension along which the thickness is introduced!
        if dim == 'x':
            return cls.from_points3D(aLn.leftPt.left(width/2),
                                   aLn.leftPt.right(width/2),
                                   aLn.rightPt.left(width/2),
                                   aLn.rightPt.right(width/2),
                                   left_right = left_right, top_bottom = top_bottom,
                                   name = name, color = color, alpha = alpha)
        if dim == 'y':
            return cls.from_points3D(aLn.leftPt.front(width/2),
                                   aLn.leftPt.back(width/2),
                                   aLn.rightPt.front(width/2),
                                   aLn.rightPt.back(width/2),
                                   left_right = left_right, top_bottom = top_bottom,
                                   name = name, color = color, alpha = alpha)
        if dim == 'z':
            return cls.from_points3D(aLn.leftPt.up(width/2),
                                   aLn.leftPt.down(width/2),
                                   aLn.rightPt.up(width/2),
                                   aLn.rightPt.down(width/2),
                                   left_right = left_right, top_bottom = top_bottom,
                                   name = name, color = color, alpha = alpha)
    
    @classmethod
    def get_container(cls):
        return Regions3D
    
    ### REINSTANTIATION ###
    def rename(self, new_name):
        updDict = {'name': new_name}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def recolor(self, new_color):
        updDict = {'color': new_color}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def realpha(self, new_alpha):
        updDict = {'alpha': new_alpha}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def rescale(self, scale_factor):
        c = self.center_point()
        scale_func = lambda p: p * scale_factor + c * (1 - scale_factor)
        q1, q2, q3, q4 = scale_func(self.p1), scale_func(self.p2), scale_func(self.p3), scale_func(self.p4)
        return Region3D.from_points3D(q1, q2, q3, q4, color = self.color, alpha = self.alpha, name = self.name)

    def transparify(self, transparence_factor):
        transparent_alpha = np.clip(self.alpha * transparence_factor, 0, 1) ### at transparence_factor = 1, nothing changes
        updDict = {'alpha': transparent_alpha}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def dim(self, dim_factor):
        return Point3D.dim(self, dim_factor) ### Point3D is class agnostic for dimming. TODO: repeat this pattern for other cases!

    def interpolate(self, otherReg, itpl_lambda = 0.5):
        avg_lineList = [x.interpolate(y, itpl_lambda = itpl_lambda) for x, y in zip(self.lineList, otherReg.lineList)]
        avg_color = interpolate_tuples(self.color, otherReg.color, itpl_lambda = itpl_lambda)
        avg_alpha = self.alpha * (1 - itpl_lambda) + otherReg.alpha * itpl_lambda
        
        updDict = {'lineList': avg_lineList,
                   'color': avg_color,
                   'alpha': avg_alpha}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def left(self, units):
        moved_lineList = [ln.left(units) for ln in self.lineList]
        updDict = {'lineList': moved_lineList}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def right(self, units):
        moved_lineList = [ln.right(units) for ln in self.lineList]
        updDict = {'lineList': moved_lineList}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def back(self, units):
        moved_lineList = [ln.back(units) for ln in self.lineList]
        updDict = {'lineList': moved_lineList}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def front(self, units):
        moved_lineList = [ln.front(units) for ln in self.lineList]
        updDict = {'lineList': moved_lineList}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def up(self, units):
        moved_lineList = [ln.up(units) for ln in self.lineList]
        updDict = {'lineList': moved_lineList}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def down(self, units):
        moved_lineList = [ln.down(units) for ln in self.lineList]
        updDict = {'lineList': moved_lineList}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def center_point(self):
        return Line3D(self.p1, self.p4).center_point()
            
    def center_coords(self):
        return self.center_point().coords

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

    def split(self, numerator = 1, denominator = 2, how = 'horizontal'):
        ### split a region into two parts, which might be of unequal size.
        ### keyword arguments numerator and denominator determine how the region is split in two
        ### how is in [horizontal, vertical] -> which of the lines to split along!

        ### collect the appropriate lines
        if how == 'horizontal':
            l1, l2 = self['Left'], self['Right']
        elif how == 'vertical':
            l1, l2 = self['Top'], self['Bottom']

        ### find the right point at which to split
        ps1, ps2 = l1.subdivide(elements = denominator + 1), l2.subdivide(elements = denominator + 1) ### plus one since endpoint is included
        p1, p2 = ps1[numerator], ps2[numerator]

        if how == 'horizontal':
            r1 = Region3D.from_points3D(self.p1, self.p2, p1, p2) ### the upper region
            r2 = Region3D.from_points3D(p1, p2, self.p3, self.p4) ### the lower region
        else:
            r1 = Region3D.from_points3D(self.p1, p1, self.p3, p2) ### the left region
            r2 = Region3D.from_points3D(p1, self.p2, p2, self.p4) ### the right region

        return Regions3D([r1, r2])
    
    def subdivide(self, elements = 10, how = 'horizontal'):
        ### subdivide a region into elements regions of equal size
        
        ### collect the appropriate lines
        if how == 'horizontal':
            l1, l2 = self['Left'], self['Right']
        elif how == 'vertical':
            l1, l2 = self['Top'], self['Bottom']

        l1s, l2s = l1.subdivide(elements = elements + 1), l2.subdivide(elements = elements + 1) ## plus one since endpoint is included

        regList = []
        prev_p1, prev_p2 = None, None
        for curr_p1, curr_p2 in zip(l1s, l2s):
            if prev_p1 is not None and prev_p2 is not None:
                if how == 'horizontal':
                    subReg = Region3D.from_points3D(prev_p1, prev_p2, curr_p1, curr_p2)
                elif how == 'vertical':
                    subReg = Region3D.from_points3D(prev_p1, curr_p1, prev_p2, curr_p2)
                regList.append(subReg)

            prev_p1, prev_p2 = curr_p1, curr_p2

        return Regions3D(regList)

    def intersect(self, aLn):
        return aLn.intersect(self)


@dataclass(frozen = True)
class Container3D:
    """
    A list of some Shape3D
    """ 
    _DEFAULT_COLOR = const.BGR['bright_grey']
    _DEFAULT_ALPHA = 1
    
    shapeList: List
    name: str = 'Shapes3D'
    
    color: Optional[tuple] = _DEFAULT_COLOR
    alpha: Optional[float] = _DEFAULT_ALPHA
    
    def __repr__(self):
        fsStr = f'{self.__class__.__name__}('
        sStrs = ['  ' + repr(shape) + ',' for shape in self.shapeList]
        lsStr = ')'
        return '\n'.join([fsStr, *sStrs, lsStr])

    def __iter__(self):
        return iter(self.shapeList)
    
    def __getitem__(self, item):
        if isinstance(item, str):
            ### get line by name
            matchShapes = self.__class__([l for l in self.shapeList if l.name == item], name = item)
            if len(matchShapes) == 1:
                return matchShapes[0]
            elif len(matchShapes) > 1:
                return matchShapes
        else:
            ### get line(s) by position in list.
            elements = self.shapeList[item]
            if isinstance(elements, List):
                updDict = {'shapeList': elements}
                newDict = get_updated_attribute_dictionary(self, updDict)
                return self.__class__(**newDict)
            else:
                return elements ### a single line

    def __len__(self):
        return len(self.shapeList)
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        
        if len(self) != len(other):
            return False
        else:
            return all([myShape == otherShape for myShape, otherShape in zip(self, other)])

    ### REINSTANTIATION ###
    def rename(self, new_name):
        updDict = {'name': new_name}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def recolor(self, new_color):
        updDict = {'color': new_color}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def realpha(self, new_alpha):
        updDict = {'alpha': new_alpha}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def interpolate(self, otherContainer, itpl_lambda = 0.5):
        assert len(self.shapeList) == len(otherContainer.shapeList)
        avg_shapeList = [x.interpolate(y, itpl_lambda = itpl_lambda) for x, y in zip(self, otherContainer)]
        avg_color = interpolate_tuples(self.color, otherContainer.color, itpl_lambda = itpl_lambda)
        avg_alpha = self.alpha * (1 - itpl_lambda) + otherContainer.alpha * itpl_lambda
        
        updDict = {'shapeList': avg_shapeList,
                   'color': avg_color,
                   'alpha': avg_alpha}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def dim(self, dim_factor):
        return Point3D.dim(self, dim_factor)
        
    def transparify(self, transparence_factor):
        transparent_alpha = np.clip(self.alpha * transparence_factor, 0, 1) ### at transparence_factor = 1, nothing changes
        updDict = {'alpha': transparent_alpha}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def center_point(self):
        center_coords = np.array([shape.center_coords() for shape in self.shapeList])
        mean_center_coord = center_coords.mean(axis = 0)
        return Point3D(tuple(mean_center_coord))
            
    def center_coords(self):
        return self.center_point().coords

    ### METHODS ###
    def rasterize(self, resolution):
        return np.concatenate([el.rasterize(resolution) for el in self.shapeList], axis = 0)

    def join(self, otherContainer):
        assert self.__class__ == otherContainer.__class__
        return self.__class__([*self.shapeList, *otherContainer.shapeList], 
                              color = self.color, alpha = self.alpha, name = self.name)


@dataclass(frozen = True, eq = False)
class Points3D(Container3D):
    """
    A list of Point3D
    """

    name: str = 'Points3D'

    @staticmethod
    def from_points2D(aPnts, x = None, y = None, z = None, name = 'Points3D'):
        return Points3D([Point3D.from_point2D(aPnt, x = x, y = y, z = z) for aPnt in aPnts], 
                      name = name)

    @classmethod
    def get_constituent(cls):
        return Point3D
    
    def vectorize(self):
        ### go from Points3D to a Point3D object with a vector of coordinates within
        coordArr = np.array([myPt.coords for myPt in self])
        return Point3D(coordArr.T, color = self.color, alpha = self.alpha)


@dataclass(frozen = True, eq = False)
class Circles3D(Container3D):
    """
    A list of Circle3D
    """

    name: str = 'Circles3D'

    @staticmethod
    def from_points3D(aPnts, radius = 1, limit_theta = None, name = 'Circles3D'):
        return Circles3D([Circle3D.from_point3D(aPnt, radius = radius, limit_theta = limit_theta) for aPnt in aPnts], 
                      name = name)
    
    @classmethod
    def get_constituent(cls):
        return Circle3D
    
    def vectorize(self):
        ### since the vectorised object needs a single parameter for each argument fed to .rasterize() ...
        canonical_radius, canonical_limit_theta = self[0].radius, self[0].limit_theta

        coordArr = np.array([myCrc.coords for myCrc in self])
        return Circle3D(coordArr.T, color = self.color, alpha = self.alpha, 
                        radius = canonical_radius, limit_theta = canonical_limit_theta)
    
@dataclass(frozen = True, eq = False)
class Ellipses3D(Container3D):
    """
    A list of Ellipses3D
    """

    name: str = 'Ellipseses3D'
    
    @classmethod
    def get_constituent(cls):
        return Ellipse3D
    

@dataclass(frozen = True, eq = False)
class Balls3D(Container3D):
    """
    A list of Ball3D
    """

    name: str = 'Balls3D'

    @staticmethod
    def from_points3D(aPnts, radius = 1, limit_theta = None, limit_phi = None, name = 'Balls3D'):
        return Balls3D([Ball3D.from_point3D(aPnt, radius = radius, limit_theta = limit_theta, limit_phi = limit_phi) for aPnt in aPnts], 
                      name = name)

    @classmethod
    def get_constituent(cls):
        return Ball3D
        
    def vectorize(self):
        ### since the vectorised object needs a single parameter for each argument fed to .rasterize() ...
        canonical_radius, canonical_limit_theta, canonical_limit_phi = self[0].radius, self[0].limit_theta, self[0].limit_phi

        coordArr = np.array([myBall.coords for myBall in self])
        return Ball3D(coordArr.T, color = self.color, alpha = self.alpha, 
                      radius = canonical_radius, limit_theta = canonical_limit_theta, limit_phi = canonical_limit_phi)

@dataclass(frozen = True, eq = False)
class Lines3D(Container3D):
    """
    A list of Line3D
    """

    name: str = 'Lines3D'

    @staticmethod
    def from_lines2D(aLns, x = None, y = None, z = None, name = 'Lines3D'):
        return Lines3D([Line3D.from_line2D(aLn, x = x, y = y, z = z) for aLn in aLns], 
                      name = name)
    
    @classmethod
    def get_constituent(cls):
        return Line3D

    def resize(self, new_thickness):
        shapeList = [ln.resize(new_thickness) for ln in self]
        paramDict = get_updated_attribute_dictionary(self, {'shapeList': shapeList})
        outLns = self.__class__(**paramDict)
        return outLns

    def vectorize(self):
        ### since the vectorised object needs a single parameter for each argument fed to .rasterize() ...
        canonical_thickness = self[0].thickness

        left_coordArr = np.array([myLn.leftPt.coords for myLn in self])
        right_coordArr = np.array([myLn.rightPt.coords for myLn in self])
        leftPt, rightPt = Point3D(left_coordArr.T), Point3D(right_coordArr.T)
        return Line3D(leftPt, rightPt, name = self.name, color = self.color, 
                      thickness = canonical_thickness)

@dataclass(frozen = True, eq = False)
class Regions3D(Container3D):
    """
    A list of Region3D
    """

    name: str = 'Regions3D'

    @staticmethod
    def from_lines3D(aLns, width, dim = 'x', name = 'Regions3D'):
        return Regions3D([Region3D.from_line3D(aLn, width, dim = dim) for aLn in aLns], 
                         name = name)
    
    @classmethod
    def get_constituent(cls):
        return Region3D
    
    def parametrise(self):
        return [myReg.parametrise() for myReg in self]
        
    ### TODO: NOT IMPLEMENTED!
    # def vectorize(self):
    #     l1s = Lines3D([myReg.lineList[0] for myReg in self])
    #     l2s = Lines3D([myReg.lineList[1] for myReg in self])
    #     l3s = Lines3D([myReg.lineList[2] for myReg in self])
    #     l4s = Lines3D([myReg.lineList[3] for myReg in self])

    #     l1, l2, l3, l4 = l1s.vectorize(), l2s.vectorize(), l3s.vectorize(), l4s.vectorize()
    #     return Region3D([l1, l2, l3, l4], color = self.color, alpha = self.alpha)


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
    
    ### Testing recoloring, renameing, realpha, dimming, transparifying

    ### Testing intersection of Line and Ball
    regLn = Line3D(Point3D((270, 340, 0)), Point3D((340, 990, 0)))
    myReg = Region3D.from_line3D(regLn, width = 150, dim = 'z', color = const.BGR['leafy_green'])
    
    myLn = Line3D(Point3D((200, 1000, 50)), Point3D((400, 200, 50)), color = const.BGR['sign_orange'])
    
    myScn = courtScn.put(myReg).put(myLn)
    View.make(myCam, myScn, resolution = resolution).snap().show()
    
    intPt, params = myLn.intersect(myReg)
    intBall = Ball3D(intPt.coords, radius = 4, color = const.BGR['strong_pink'])
    
    myScn = myScn.put(intBall)
    View.make(myCam, myScn, resolution = resolution).snap().show()
    
    ### Testing Circles
    coords = (300, 500, 0)
    r = 50
    baseCrc = Circle3D(coords, radius = r)

    myCrc = baseCrc
    myScn = courtScn.put(myCrc)
    View.make(myCam, myScn, resolution = 1).snap().show()

    myCrc = baseCrc.retheta((0, 1 * np.pi))
    myScn = courtScn.put(myCrc)
    View.make(myCam, myScn, resolution = 1).snap().show()

    myCrc = baseCrc.rerot_x(0.25 * np.pi)
    myScn = courtScn.put(myCrc)
    View.make(myCam, myScn, resolution = 1).snap().show()
    
    myCrc = baseCrc.rerot_y(0.25 * np.pi)
    myScn = courtScn.put(myCrc)
    View.make(myCam, myScn, resolution = 1).snap().show()

    myCrc = baseCrc.rerot_z(0.25 * np.pi)
    myScn = courtScn.put(myCrc)
    View.make(myCam, myScn, resolution = 1).snap().show()


    ### Testing Containers
    regLn = Line3D(Point3D((270, 340, 0)), Point3D((340, 990, 0)))
    myLn = Line3D(Point3D((200, 1000, 50)), Point3D((400, 200, 50)), color = const.BGR['sign_orange'])
    someLns = Lines3D([regLn, myLn]) 

    myScn = courtScn.put(someLns) ### note that any one shape has the same color, so that color = 'sign_orange' above is overwritten!
    View.make(myCam, myScn, resolution = resolution).snap().show()

    myScn = courtScn.put(regLn).put(myLn) ### see note above
    View.make(myCam, myScn, resolution = resolution).snap().show()

    ## 1) Points
    print('Testing Points as Container3D object and vectorized')
    a, b, c, d = Point3D((250, 340, 0)), Point3D((100, 600, 0)), Point3D((600, 20, 100)), Point3D((40, 140, 20))
    myPnts = Points3D([a, b, c, d], color = const.BGR['sign_orange'])
    
    # 1a) Points as Container
    myScn = courtScn.put(myPnts)
    View.make(myCam, myScn, resolution = resolution).snap().show()

    # 1b) Points vectorized
    myScn = courtScn.put(myPnts.vectorize())
    View.make(myCam, myScn, resolution = resolution).snap().show()

    ## 2) Circles 
    print('Testing Circles as Container3D object and vectorized')
    myCrcs = Circles3D.from_points3D(myPnts, radius = 5)
    
    # 2a) Circles as Container
    myScn = courtScn.put(myCrcs)
    View.make(myCam, myScn, resolution = resolution).snap().show()

    # 2b) Circles vectorized
    myScn = courtScn.put(myCrcs.vectorize())
    View.make(myCam, myScn, resolution = resolution).snap().show()

    ## 3) Balls
    print('Testing Balls as Container3D object and vectorized')
    myBlls = Balls3D.from_points3D(myPnts, radius = 5)
    
    # 3a) Balls as Container
    myScn = courtScn.put(myBlls)
    View.make(myCam, myScn, resolution = resolution).snap().show()

    # 3b) Balls vectorized
    myScn = courtScn.put(myBlls.vectorize())
    View.make(myCam, myScn, resolution = resolution).snap().show()

    ## 4) Lines
    print('Testing Lines as Container3D object and vectorized')
    l1, l2 = Line3D(a, b), Line3D(c, d)
    myLns = Lines3D([l1, l2])

    # 4a) Lines as Container
    myScn = courtScn.put(myLns)
    View.make(myCam, myScn, resolution = resolution).snap().show()

    # 4b) Lines vectorized
    myScn = courtScn.put(myLns.vectorize())
    View.make(myCam, myScn, resolution = resolution).snap().show()

    # 5) Ellipses
    print('Testing Ellipses as Container3D object and vectorized')
    myEll = Ellipse3D.from_point3D(myPnts[0], radius_long = 50, radius_short = 25)

    myScn = courtScn.put(myEll)
    View.make(myCam, myScn, resolution = resolution).snap().show()

    # 5) Ellipses
    myEll = Ellipse3D.from_point3D(myPnts[0], radius_long = 50, radius_short = 25, theta = np.pi / 2)

    myScn = courtScn.put(myEll)
    View.make(myCam, myScn, resolution = resolution).snap().show()
    
    # 5) Ellipses
    myEll = Ellipse3D.from_point3D(myPnts[0], radius_long = 50, radius_short = 25, theta = -np.pi / 4)
    
    myScn = courtScn.put(myEll)
    View.make(myCam, myScn, resolution = resolution).snap().show()













    ### BELOW TODO: Not implemented yet if thickness > 0
    # # 4c) Lines with thickness as Container
    # l1, l2 = Line3D(a, b, thickness = 2), Line3D(c, d, thickness = 2)
    # myLns = Lines3D([l1, l2])
    # print('Testing thick Lines as Container3D object and vectorized')
    # myScn = courtScn.put(myLns)
    # View.make(myCam, myScn, resolution = resolution).snap().show()

    # # 4d) Lines with thickness vectorized
    # myScn = courtScn.put(myLns.vectorize())
    # View.make(myCam, myScn, resolution = resolution).snap().show()

    ### BELOW TODO: Not implemented yet if Regions
    # ## 5) Regions
    # print('Testing Regions as Container3D object and vectorized')
    # r1, r2 = Region3D.from_line3D(l1, width = 10), Region3D.from_line3D(l2, width = 10)
    # myRegs = Regions3D([r1, r2])
    
    # # 5a) Regions as Container
    # myScn = courtScn.put(myRegs)
    # View.make(myCam, myScn, resolution = resolution).snap().show()

    # # 5b) Regions vectorized
    # myScn = courtScn.put(myRegs.vectorize())
    # View.make(myCam, myScn, resolution = resolution).snap().show()
