# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:26:23 2023

@author: gianl
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from ATMANpy.types.Frames import BGRFrame
from ATMANpy.types.Shapes3D import Point3D, Region3D
from ATMANpy.types.ShapesTennis3D import Court, Net
from ATMANpy.types.Lines import CourtLines

import ATMANpy.auxil.constants as const

@dataclass(frozen = True)
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            scalar = self.w * other.w - self.vector() @ other.vector()
            vector = self.w * other.vector() + other.w * self.vector() + np.cross(self.vector(), other.vector())
            x, y, z = vector
            w = scalar
        else:
            w, x, y, z = other * self.w, other * self.x, other * self.y, other * self.z

        return Quaternion(w, x, y, z)
    
    def __truediv__(self, other):
        return self * (1 / other)

    def scalar(self):
        return self.w
    
    def vector(self):
        return np.array([self.x, self.y, self.z])
    
    def tuple(self):
        return self.w, self.x, self.y, self.z

    def norm(self):
        return (self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def unit(self):
        return self / self.norm()
    
    def vector_norm(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
    
    def unit_vector(self):
        return self.vector() / self.vector_norm()
    
    def gamma(self):
        return np.arccos(self.w / self.norm())
    
    def inv(self):
        norm_squared = self.norm() ** 2
        scalar = self.scalar() / norm_squared
        vector = - self.vector() / norm_squared
        
        w = scalar
        x, y, z = vector
        return Quaternion(w, x, y, z)

    def rotate(self, vector):
        rot_q = self.inv() * Quaternion.from_vector(vector) * self
        return rot_q.vector()

    def to_rotation_matrix(self):
        w, x, y, z = self.unit().tuple()    

        R = np.array([[1 - 2*(y**2 + z**2), 2*(x*y + z*w), 2*(x*z - y*w)],
                      [2*(x*y - z*w), 1 - 2*(x**2 + z**2), 2*(y*z + x*w)],
                      [2*(x*z + y*w), 2*(y*z - x*w), 1 - 2*(x**2 + y**2)]])

        return R
    
    @staticmethod
    def from_rotation_matrix(R):
        ### using Bar-Itzhack (2015): New Method for Extracting the Quaternion from a Rotation Matrix
        ### version 3 in the paper.
        ### WEIRDNESS: invert_w: w needs its sign switched. This is weird, but produces correct quaternions
        ### i have no idea why.
        d11, d12, d13 = R[0]
        d21, d22, d23 = R[1]
        d31, d32, d33 = R[2]

        K_3 = 1/3 * np.array([[d11 - d22 - d33, d21 + d12, d31 + d13, d23 - d32],
                              [d21 + d12, d22 - d11 - d33, d32 + d23, d31 - d13],
                              [d31 + d13, d32 + d23, d33 - d11 - d22, d12 - d21],
                              [d23 - d32, d31 - d13, d12 - d21, d11 + d22 + d33]])

        eig_vals, eig_vecs = np.linalg.eig(K_3)
        max_idx = eig_vals.argmax()
        x, y, z, w = eig_vecs[:, max_idx]
        
        return Quaternion(w, x, y, z) 
    
    @staticmethod
    def from_vector(vec):
        return Quaternion(0, *vec)

@dataclass(frozen = True)
class Scene:
    shapes: List ### a collection of various 3D shapes
        
    def __post_init__(self):
        object.__setattr__(self, 'shapes', [s for s in self.shapes if s is not None]) ### prune nones

    def __repr__(self):
        fsStr = f'{self.__class__.__name__}(['
        sStrs = ['  ' + repr(shape) + ',' for shape in self.shapes]
        lsStr = '])'
        if len(sStrs) == 0:
            return 'Scene()'
        else:
            return '\n'.join([fsStr, *sStrs, lsStr])

    def __getitem__(self, item):
        return self.shapes[item]
    
    def __len__(self):
        return len(self.shapes)

    @staticmethod
    def make_empty():
        return Scene([])
    
    @staticmethod
    def make_court(courtLines = None, make_net = True, singles_net = True, net_mesh_density = 1, 
                   line_color = const.BGR['bright_grey'], line_alpha = 1, 
                   net_color = const.BGR['smokey_grey'], net_alpha = 1):
        
        courtRegions3D = Court.make(courtLines = courtLines, color = line_color, alpha = line_alpha)
        myScn = Scene.make_empty().put(courtRegions3D)

        if make_net:
            if singles_net:
                net3D = Net.make_singles(courtLines = courtLines, color = net_color, alpha = net_alpha, net_mesh_density = net_mesh_density)
                myScn = myScn.put(net3D)
            else:
                net3D = Net.make_doubles(courtLines = courtLines, color = net_color, alpha = net_alpha, net_mesh_density = net_mesh_density)
                myScn = myScn.put(net3D)

        return myScn
    
    def is_empty(self):
        return len(self.shapes) == 0
    
    def put(self, someShape):
        if someShape is None:
            return self
        elif isinstance(someShape, list):
            return Scene([*self.shapes, *someShape])
        else:
            return Scene([*self.shapes, someShape])
        
    def drop(self, index):
        if index < -len(self):
            raise IndexError(f'Index {index} not in Scene of length {len(self)}')
        elif index < 0:
            index = len(self) + index
        elif index < len(self):
            pass
        else:
            raise IndexError(f'Index {index} not in Scene of length {len(self)}')

        new_shapes = [shape for idx, shape in enumerate(self) if idx != index]
        return Scene(new_shapes)

    ### REINSTANTIATION ###
    def recolor(self, new_color):
        return Scene([s.recolor(new_color) for s in self.shapes])
    
    def realpha(self, new_alpha):
        return Scene([s.realpha(new_alpha) for s in self.shapes])
    
    def dim(self, dim_factor):
        return Scene([s.dim(dim_factor) for s in self.shapes])
    
    def transparify(self, transparence_factor):
        return Scene([s.transparify(transparence_factor) for s in self.shapes])

    def prune_transparents(self, alpha_threshold = 0.05):
        return Scene([s for s in self.shapes if s.alpha > alpha_threshold])

    def combine(self, otherScn, keep_which = 'both'):
        ### keep_which = 'self', 'other', 'both' -> what to keep in case of duplicates
        if keep_which == 'both':
            return Scene([*self.shapes, *otherScn.shapes])
        elif keep_which == 'self':
            unmatched_shapes = [shape for shape in otherScn.shapes if shape not in self.shapes]
            return Scene([*self.shapes, *unmatched_shapes])
        elif keep_which == 'other':
            unmatched_shapes = [shape for shape in self.shapes if shape not in otherScn.shapes]
            return Scene([*unmatched_shapes, *otherScn.shapes])

    ### METHODS ###
    def rasterize(self, resolution, alpha_threshold = 0.05, 
                  camera_point = None, resnorm_point = None, adaptive_resolution_exponent = 0, ### these factors are relevant if we want adaptive resolution
                  concatenate = True): 
        RES_BOUND_MULTIPLICATIVE = 10
        ### yields a point-cloud representation of the objects in the scene, 
        # with points stored in rastArr_all and their colors stored in colArr_all
        pruned_self = self.prune_transparents(alpha_threshold = alpha_threshold)

        if pruned_self.is_empty():
            return None, None, None
        else:
            rastArr_all = []
            colArr_all = []
            alphaArr_all = []
            for shape3D in pruned_self.shapes:
                ### If needed: Adaptive Resolution!
                ### Problem: if an object is further away, it will appear denser at constant resolution, and brighter therefore!
                ### Solution: reduce resolution of objects further away, and increase it for closer objects
                if adaptive_resolution_exponent != 0: 
                    norm_dist = camera_point.distance_to(resnorm_point)
                    shape_dist = camera_point.distance_to(shape3D.center_point())
                    if isinstance(shape_dist, np.ndarray):
                        shape_dist = shape_dist.mean()
                    lo, hi = resolution / RES_BOUND_MULTIPLICATIVE, resolution * RES_BOUND_MULTIPLICATIVE
                    resolution_modifier = (norm_dist / shape_dist) ** adaptive_resolution_exponent
                    adaptive_resolution = max(lo, min(resolution * resolution_modifier, hi))
                    rastArr = shape3D.rasterize(adaptive_resolution)
                else:
                    rastArr = shape3D.rasterize(resolution)
                
                colTup = shape3D.color
                alpha = shape3D.alpha
                colArr = np.tile(np.array(colTup), (len(rastArr), 1))
                alphaArr = np.full((len(rastArr), 1), fill_value = alpha)
                
                rastArr_all.append(rastArr)
                colArr_all.append(colArr)
                alphaArr_all.append(alphaArr)
            
            if concatenate:
                rastArr_all = np.concatenate(rastArr_all)
                colArr_all = np.concatenate(colArr_all)
                alphaArr_all = np.concatenate(alphaArr_all)
                
            return rastArr_all, colArr_all, alphaArr_all
    
@dataclass(frozen = True)
class DummyScene:
    ### a scene that does not contain shapes, but a rasterized representation of what is to be displayed
    rastArrs: List ### list of np.arrays, each holding coordinates, with shape N x 3 (xyz)
    colArrs: List ### list of np.arrays, each holding BGR colors, with shape N x 3 (BGR)
    alphaArrs: List ### List of np.arrays, each holding alpha opacities, with shape N x 1  
        
    def __repr__(self):
        arrays_to_str = lambda r, c, a: f'Raster, colors and alphas of size {r.shape[0]}'
        fsStr = f'{self.__class__.__name__}(['
        sStrs = ['  ' + arrays_to_str(r, c, a) for r, c, a in zip(self.rastArrs, self.colArrs, self.alphaArrs)]
        lsStr = '])'
        if len(sStrs) == 0:
            return 'DummyScene()'
        else:
            return '\n'.join([fsStr, *sStrs, lsStr])

    def __getitem__(self, item):
        return DummyScene([self.rastArrs[item]], [self.colArrs[item]], [self.alphaArrs[item]])

    def __len__(self):
        return len(self.rastArrs)

    def __eq__(self, otherScn):
        if not isinstance(otherScn, DummyScene):
            return False
        else:
            if len(self) != len(otherScn):
                return False
            else:
                for idx in range(len(self)):
                    if not (self.rastArrs[idx] == otherScn.rastArrs[idx]).all():
                        return False
                    if not (self.colArrs[idx] == otherScn.colArrs[idx]).all():
                        return False
                    if not (self.alphaArrs[idx] == otherScn.alphaArrs[idx]).all():
                        return False
                return True

    @staticmethod
    def make_empty():
        return DummyScene([], [], [])

    def copy(self):       
        import copy
        return copy.deepcopy(self)

    def is_empty(self):
        return len(self.rastArrs) == 0
    
    def put(self, rastArr, colArr):
        return DummyScene([*self.rastArrs, rastArr], [*self.colArrs, colArr])

    def recolor(self, new_color):
        new_colArrs = [np.full(colArr.shape, fill_value = new_color) for colArr in self.colArrs]
        return DummyScene(self.rastArrs, new_colArrs, self.alphaArrs)
    
    def realpha(self, new_alpha):
        new_alphaArrs = [np.full(alphaArr.shape, fill_value = new_alpha) for alphaArr in self.alphaArrs]
        return DummyScene(self.rastArrs, self.colArrs, new_alphaArrs)

    def dim(self, dim_factor):
        dim_colArrs = [np.clip(colArr * dim_factor, 0, 255) for colArr in self.colArrs]
        return DummyScene(self.rastArrs, dim_colArrs, self.alphaArrs)
    
    def transparify(self, transparence_factor):
        trans_alphaArrs = [np.clip(alphaArr * transparence_factor, 0, 1) for alphaArr in self.alphaArrs]
        return DummyScene(self.rastArrs, self.colArrs, trans_alphaArrs)

    def rotate_flat(self, angle, origPt = None):         
        s_a, c_a = np.sin(angle), np.cos(angle)

        R = np.array([  [c_a, -s_a, 0],
                        [s_a,  c_a, 0],
                        [0  ,    0, 1]])
        
        if origPt is None:
            origPt = Point3D((0, 0, 0))
        origArr = np.array(origPt.coords)

        rastArrs_rot = []
        for rastArr in self.rastArrs:
            rastArr_norm = rastArr - origArr
            rastArr_norm_rot = np.einsum('ij, nj -> ni', R, rastArr_norm)
            rastArr_rot = rastArr_norm_rot + origArr
            rastArrs_rot.append(rastArr_rot)

        rotScn = DummyScene(rastArrs_rot, self.colArrs, self.alphaArrs)
        return rotScn

    def prune_transparents(self, alpha_threshold = 0.05):
        new_rastArrs, new_colArrs, new_alphaArrs = [], [], []
        for rastArr, colArr, alphaArr in zip(self.rastArrs, self.colArrs, self.alphaArrs):
            is_above = (alphaArr > alpha_threshold).squeeze()

            new_rastArrs.append(rastArr[is_above])
            new_colArrs.append(colArr[is_above])
            new_alphaArrs.append(alphaArr[is_above])
        return DummyScene(new_rastArrs, new_colArrs, new_alphaArrs)

    def combine(self, otherDummyScn, keep_which = 'both'):
        ### keep_which = 'self', 'other', 'both' -> what to keep in case of duplicates
        array_is_in_list = lambda arr, lst: any([(arr == el).all() for el in lst])
        slf_r, slf_c, slf_a = self.rastArrs, self.colArrs, self.alphaArrs
        oth_r, oth_c, oth_a = otherDummyScn.rastArrs, otherDummyScn.colArrs, otherDummyScn.alphaArrs
        if keep_which == 'both':
            cmb_r = [*slf_r, *oth_r]
            cmb_c = [*slf_c, *oth_c]
            cmb_a = [*slf_a, *oth_a]
        elif keep_which == 'self':
            unm_r, unm_c, unm_a = [], [], []
            for r, c, a in zip(oth_r, oth_c, oth_a):
                if not array_is_in_list(r, slf_r):
                    unm_r.append(r)
                    unm_c.append(c)
                    unm_a.append(a)
            cmb_r = [*slf_r, *unm_r]
            cmb_c = [*slf_c, *unm_c]
            cmb_a = [*slf_a, *unm_a]
        elif keep_which == 'other':
            unm_r, unm_c, unm_a = [], [], []
            for r, c, a in zip(slf_r, slf_c, slf_a):
                if not array_is_in_list(r, oth_r):
                    unm_r.append(r)
                    unm_c.append(c)
                    unm_a.append(a)
            cmb_r = [*unm_r, *oth_r]
            cmb_c = [*unm_c, *oth_c]
            cmb_a = [*unm_a, *oth_a]
    
        return DummyScene(cmb_r, cmb_c, cmb_a)
        
    def consolidate(self):
        premul_alpha = np.zeros_like(self.alphaArrs[0])
        premul_color = np.zeros_like(self.colArrs[0])

        for rastArr, colArr, alphaArr in zip(self.rastArrs, self.colArrs, self.alphaArrs):
            premul_alpha = premul_alpha * (1 - alphaArr) + alphaArr
            premul_color = premul_color * (1 - alphaArr) + colArr * alphaArr

        rastArr = self.rastArrs[0]

        return DummyScene([rastArr], [premul_color], [premul_alpha])

    def rasterize(self, resolution, alpha_threshold = 0.05, 
                  camera_point = None, resnorm_point = None, adaptive_resolution_exponent = 0, ### TODO: adaptive resolution -> necessary for dummy scenes?
                  concatenate = True): 
        ### TODO: implement pruning with alpha thresholding (see above) 
        pruned_self = self.prune_transparents()

        rastArr_all = pruned_self.rastArrs
        colArr_all = pruned_self.colArrs
        alphaArr_all = pruned_self.alphaArrs

        if concatenate:
            rastArr_all = np.concatenate(rastArr_all)
            colArr_all = np.concatenate(colArr_all)
            alphaArr_all = np.concatenate(alphaArr_all)
        
        return rastArr_all, colArr_all, alphaArr_all
    

@dataclass(frozen = True)
class Camera:
    M_int: np.array
    M_ext: np.array
    sensor_dims: tuple
    
    def __repr__(self):
        f_x, o_x, f_y, o_y = self.M_int[0, 0], self.M_int[0, 2], self.M_int[1, 1], self.M_int[1, 2]
        pos = self.get_cw()
        return f'Camera(f = ({f_x}, {f_y}), c_w = {pos})'

    def __eq__(self, otherCam):
        int_comp = self.M_int == otherCam.M_int
        ext_comp = self.M_ext == otherCam.M_ext
        sns_comp = self.sensor_dims == otherCam.sensor_dims
        return int_comp.all() and ext_comp.all() and sns_comp
    
    def __neq__(self, otherCam):
        return not self == otherCam

    @staticmethod
    def pinhole_origin_aligned(f_x, f_y, o_x, o_y, s_x, s_y):
        M_int = np.array([[f_x, 0, o_x, 0],
                          [0, f_y, o_y, 0],
                          [0,   0,   1, 0]])
        
        M_ext = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        sensor_dims = (s_x, s_y)
        return Camera(M_int, M_ext, sensor_dims)
    
    @staticmethod
    def calibrated_AO_behind():
        M_int = np.array([[ 3654,  8.3354,  812.59, 0],
                          [    0,    3651, -190.65, 0],
                          [    0,       0,       1, 0]])
        M_ext = np.array([[-0.999098,  0.0423391,  0.003338,  224.399],
                          [ 0.001041, -0.0541579,  0.998532, -162.066],
                          [ 0.042458,  0.9976340,  0.054065, -1735.40],
                          [ 0	    ,  0	    ,  0	   ,  1      ]])
        sensor_dims = (1920, 1080)
        return Camera(M_int, M_ext, sensor_dims)
    
    @staticmethod
    def calibrated_brest_behind():
        M_int = np.array([[ 627.24737689,   -5.45750558,  322.70521678,    0.        ],
                          [   0.        ,  572.64800278, -202.07292776,    0.        ],
                          [   0.        ,    0.        ,    1.        ,    0.        ]])
        M_ext = np.array([[-9.99999954e-01, -2.88825399e-04, -9.50716679e-05, 2.92589667e+02],
                          [-1.46423494e-04,  1.83377177e-01,  9.83042618e-01, -6.16863613e+02],
                          [-2.66493703e-04,  9.83042587e-01, -1.83377211e-01, -1.55133005e+03],
                          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
        sensor_dims = (640, 360)
        return Camera(M_int, M_ext, sensor_dims) 
    
    @staticmethod
    def low_angled_behind_ad():
        baseCam = Camera.calibrated_AO_behind()
        myCam = baseCam.left(20).into(14).pan(-0.06).down(4)
        return myCam
    
    @staticmethod
    def low_angled_behind_deuce():
        baseCam = Camera.calibrated_AO_behind()
        myCam = baseCam.right(20).into(14).pan(0.06).down(4)
        return myCam

    @staticmethod
    def mid_lateral():
        baseCam = Camera.calibrated_AO_behind()
        myCam = baseCam.right(55).pan(0.5).rotate(0.005).right(133).zoom(0.175)
        myCam = myCam.tilt(-0.05).pan(0.008)
        return myCam
    
    @staticmethod
    def birds_eye(cam_height = 100, sensor_dims = (1920, 1080),
                  f = 3650, o_x = None, o_y = None, centerPt = None):
        if o_x is None:
            o_x = sensor_dims[0] / 2
        if o_y is None:
            o_y = sensor_dims[1] / 2

        if centerPt is None:
            centerPt = CourtLines.make().get_center_point()
        x, y = centerPt.coords
        
        coords_cam = np.array([x, y, cam_height])
        
        R = np.array([[-1,  0,  0],
                      [ 0, -1,  0],
                      [ 0,  0,  1]])
        t = - R @ coords_cam
        
        R_t = np.hstack([R, t[:, np.newaxis]])
        M_ext = np.vstack([R_t, np.array([0, 0, 0, 1])])
        
        M_int = np.array([[ f, 0, o_x, 0],
                          [ 0, f, o_y, 0],
                          [ 0, 0,   1, 0]])
        
        return Camera(M_int, M_ext, sensor_dims) 
        
    @staticmethod
    def from_to(origin_point, towards_point, f = 1000, sensor_dims = (1920, 1080)):
        # towards_point, origin_point: Point3D

        ##### EXTERIOR CAMERA PARAMETERS #####
        ### 1) the camera axis (z_c) is described by origin_point -> towards_point
        cam_vector = towards_point - origin_point
        z_c = - cam_vector.normalize()
        z_axis = np.array(z_c.coords)

        ### 2) the x-axis (x_c) can not be tilted at initialization
        # -> it is orthogonal to z_w (and thus co-planar with the world ground)
        # -> it is orthogonal to z_c by virtue of being an axis
        x_axis = np.cross(z_axis, np.array([0, 0, 1]))
        if x_axis.sum() == 0: ### this happens if z_axis is colinear with [0, 0, 1]
            x_axis = np.array([1, 0, 0])
        else:
            x_axis = x_axis / (x_axis ** 2).sum() ** 0.5
        x_axis = - x_axis ### invert the direction of x axis (generally correct?)

        ### 3) the y-axis is orthogonal to x and z
        y_axis = np.cross(x_axis, z_axis)
        y_axis = y_axis / (y_axis ** 2).sum() ** 0.5 ### technically, since x and z are themselves orthogonal, no need to normalize
        y_axis = - y_axis ### invert the direction of x axis (generally correct?)

        ### 4) rotation matrix is just the new coordinate system's axes as collated as column vectors
        R = np.vstack([x_axis, y_axis, z_axis])
        t = - R @ np.array(origin_point.coords)

        R_t = np.hstack([R, t[:, np.newaxis]])
        M_ext = np.vstack([R_t, np.array([0, 0, 0, 1])])
            
        ##### INTERIOR CAMERA PARAMETERS ####
        ## very simple pinhole
        o_x, o_y = sensor_dims[0] / 2, sensor_dims[1] / 2
        M_int = np.array([  [ f, 0, o_x, 0],
                            [ 0, f, o_y, 0],
                            [ 0, 0,   1, 0]])
        
        return Camera(M_int, M_ext, sensor_dims)

    def get_P(self):
        return self.M_int @ self.M_ext
    def get_K(self):
        return self.M_int[:3, :3]
    def get_R(self):
        return self.M_ext[:3, :3]
    def get_R_inv(self):
        return np.linalg.inv(self.get_R())
    def get_t(self): ### center of camera, in rotated coordinates
        return self.M_ext[:3, 3] 
    def get_cw(self): ### center of camera, in world coordinates
        return - self.get_R_inv() @ self.get_t()
    def get_center_point(self):
        return Point3D(tuple(self.get_cw()))
        
    def world_points_to_camera_points(self, ptArr):
        num_points, num_dims = ptArr.shape
        if num_dims == 4: ### --> already in homogenous coordinates!
            homArr = ptArr
        elif num_dims == 3: ### --> not in homogenous coordinates!
            homArr = np.concatenate([ptArr, np.ones((num_points, 1))], axis = 1)
        else:
            raise Exception('Points to be projected need to be (homogenous or inhomogenous) 3D')
        
        homArr_3D = np.einsum('ij, nj -> ni', self.M_ext, homArr)
        ptArr_cam = homArr_3D[:, :-1] / homArr_3D[:, [-1]]
        return ptArr_cam
    
    def project_points(self, ptArr):
        num_points, num_dims = ptArr.shape
        if num_dims == 4: ### --> already in homogenous coordinates!
            homArr = ptArr
        elif num_dims == 3: ### --> not in homogenous coordinates!
            homArr = np.concatenate([ptArr, np.ones((num_points, 1))], axis = 1)
        else:
            raise Exception('Points to be projected need to be (homogenous or inhomogenous) 3D')
        
        M_proj = self.get_P()
        
        homArr_2D = np.einsum('ij, nj -> ni', M_proj, homArr)
        pixArr = homArr_2D[:, :2] / homArr_2D[:, 2][:, np.newaxis]
        return pixArr
    
    def approximate_fov(self, distance_from_pinhole_in_px):
        ### Get the world region that is projected onto the corners of the sensor of a birds eye camera of height z:
        K = self.get_K()
        fx, fy = K[0, 0], K[1, 1]
        ox, oy = K[0, 2], K[1, 2]
        f = (fx + fy) / 2
        z = distance_from_pinhole_in_px
        M_ext_inv = np.linalg.inv(self.M_ext)

        ### one sensor corner point corresponds to the following point in the world, expressed in camera coordinates
        px_c, py_c = ox * z / f, oy * z / f
    
        ### from this point (and from the symmetry of the sensor), calculate two points in the world, expressed in world coordinates
        px_w, py_w, pz_w, _ = M_ext_inv @ np.array([-px_c, -py_c, -z, 1]) 
        qx_w, qy_w, qz_w, _ = M_ext_inv @ np.array([px_c, py_c, -z, 1]) 

        ### unclear what happens when we don't have birds eye here.
        mz_w = (pz_w + qz_w) / 2

        ### these two points define a region
        p1, p2, p3, p4 = Point3D((px_w, py_w, pz_w)), Point3D((qx_w, py_w, mz_w)), Point3D((px_w, qy_w, mz_w)), Point3D((qx_w, qy_w, qz_w))
        return Region3D.from_points3D(p1, p2, p3, p4)


    ############ Internal Matrix Changes ##########
    def zoom(self, factor):
        ### factor > 1 -> zoom in; factor < 1 -> zoom out
        M_int_new = self.M_int.copy()
        M_int_new[0, 0] *= factor
        M_int_new[1, 1] *= factor

        return Camera(M_int_new, self.M_ext, self.sensor_dims)

    def resize(self, factor):
        ### resize the camera sensor
        w, h = self.sensor_dims
        sensor_dims_new = round(w * factor), round(h * factor)
        M_int_new = self.M_int.copy()
        M_int_new[:2] *= factor
        return Camera(M_int_new, self.M_ext, sensor_dims_new)

    ############ External Matrix Changes ##########
    def tilt(self, angle_pi):
        ### Relative to the camera (where the z axis shoots straight out of the pinhole),
        ### the tilt occurs along the x axis
        ### angle_pi specified as multiples of pi
        angle = angle_pi * np.pi
        R = np.array([[1,             0,              0],
                      [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle),  np.cos(angle)]])
        
        M_ext_new = R @ self.M_ext[:3]
        M_ext_new = np.append(M_ext_new, np.array([[0, 0, 0, 1]]), axis = 0)
        
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
    
    def pan(self, angle_pi):
        ### Relative to the camera (where the z axis shoots straight out of the pinhole),
        ### the pan occurs along the y axis
        ### angle_pi specified as multiples of pi
        angle = angle_pi * np.pi
        R = np.array([[ np.cos(angle), 0, np.sin(angle)],
                      [             0, 1,             0],
                      [-np.sin(angle), 0, np.cos(angle)]])
        
        M_ext_new = R @ self.M_ext[:3]
        M_ext_new = np.append(M_ext_new, np.array([[0, 0, 0, 1]]), axis = 0)
        
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
        
    def rotate(self, angle_pi):
        ### Relative to the camera (where the z axis shoots straight out of the pinhole),
        ### the rotation occurs along the z axis.
        ### angle_pi specified as multiples of pi
        angle = angle_pi * np.pi
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle),  np.cos(angle), 0],
                      [            0,              0, 1]])
        M_ext_new = R @ self.M_ext[:3]
        M_ext_new = np.append(M_ext_new, np.array([[0, 0, 0, 1]]), axis = 0)
        
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
        
    def right(self, feet):
        #### left-right motion occurs along the x-axis
        M_ext_new = self.M_ext.copy()
        M_ext_new[0, 3] += feet * const.SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
       
    def left(self, feet):
        #### left-right motion occurs along the x-axis
        M_ext_new = self.M_ext.copy()
        M_ext_new[0, 3] -= feet * const.SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
       
    def down(self, feet):
        #### up-down motion occurs along the y-axis
        M_ext_new = self.M_ext.copy()
        M_ext_new[1, 3] += feet * const.SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
       
    def up(self, feet):
        #### up-down motion occurs along the y-axis
        M_ext_new = self.M_ext.copy()
        M_ext_new[1, 3] -= feet * const.SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
    
    def into(self, feet):
        #### in-out motion occurs along the z-axis
        M_ext_new = self.M_ext.copy()
        M_ext_new[2, 3] += feet * const.SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
       
    def out(self, feet):
        #### in-out motion occurs along the z-axis
        M_ext_new = self.M_ext.copy()
        M_ext_new[2, 3] -= feet * const.SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
    
    def get_camera_ray_params(self, somePt, correct_for_epsilon = True):
        """
        Parameters of the pinhole camera ray going through a point, in camera-coordinate parameters
        ### this is improved by exploiting that M_int[0,1] (named epsilon) is not always zero, 
        ### as it usually is in the pinhole camera model.
        """
        
        u, v = somePt.coords
        o_x, o_y = self.M_int[0, 2], self.M_int[1, 2]
        f_x, f_y = self.M_int[0, 0], self.M_int[1, 1]
        z_norm = self.M_int[2, 2]
        
        coeff_x_c = (z_norm * (u - o_x)) / f_x
        coeff_y_c = (z_norm * (v - o_y)) / f_y
        coeff_z_c = 1       ### since z_c is distance from camera aperture in camera coordinates, 
                            # and we need a parameter for the line, we choose z_c for convenience. 
                            # Note: z_c > 0 -> line_parameter > 0
        
        if correct_for_epsilon:
            epsilon = self.M_int[0, 1]
            correction = epsilon * coeff_y_c / f_x
            coeff_x_c = coeff_x_c - correction
        
        slope = np.array([coeff_x_c, coeff_y_c, coeff_z_c])
        intercept = np.zeros(3)
        return slope, intercept
        
    def get_world_ray_params(self, somePt):
        """
        Parameters of the ray going through a point, in world-coordinate parameters
        """
        m_c, b_c = self.get_camera_ray_params(somePt)
        
        R, t = self.get_R(), self.get_t()
        slope = R.T @ m_c
        intercept = - R.T @ t
    
        return slope, intercept
        
@dataclass(frozen = True)
class View:
    ### a View combines a camera and a scene: 
        # here, we store the Scene's rasterization, 
        # but also the camera projection of it, and the distance vector of all objects to the camera.
    
    ptArr: np.array ### points in 3D space
    clArr: np.array ### colors
    alArr: np.array ### alphas
    pxArr: np.array ### points in 2D space -> 3D points projected onto screen
    dsArr: np.array ### distance of objects to camera

    text_param_dicts: list = None
    
    sensor_dims: tuple = (1920, 1080)
    
    def __getitem__(self, item):
        ptArr_filt = self.ptArr[item]
        clArr_filt = self.clArr[item]
        alArr_filt = self.alArr[item]
        pxArr_filt = self.pxArr[item]
        dsArr_filt = self.dsArr[item]
        return View(ptArr_filt, clArr_filt, alArr_filt, pxArr_filt, dsArr_filt, self.text_param_dicts, self.sensor_dims)

    def __len__(self):
        return len(self.ptArr)

    @staticmethod
    def make_empty(sensor_dims = (1920, 1080), text_param_dicts = None):
        return View(np.zeros((0, 3)), 
                    np.zeros((0, 3)), 
                    np.zeros((0, 1)),
                    np.zeros((0, 2), dtype = int),  ### pixels need to be integer!
                    np.zeros(0),
                    text_param_dicts = text_param_dicts,
                    sensor_dims = sensor_dims)
    
    @staticmethod
    def from_many_views(sub_views):
        if len(sub_views) == 0:
            return View.make_empty()
        
        ptArrs = []
        clArrs = []
        alArrs = []
        pxArrs = []
        dsArrs = []

        for spView in sub_views:
            ptArrs.append(spView.ptArr)
            clArrs.append(spView.clArr)
            alArrs.append(spView.alArr)
            pxArrs.append(spView.pxArr)
            dsArrs.append(spView.dsArr)

        ptArr = np.concatenate(ptArrs, axis = 0)
        clArr = np.concatenate(clArrs, axis = 0)
        alArr = np.concatenate(alArrs, axis = 0)
        pxArr = np.concatenate(pxArrs, axis = 0)
        dsArr = np.concatenate(dsArrs, axis = 0)

        return View(ptArr, clArr, alArr, pxArr, dsArr, 
                    text_param_dicts = sub_views[0].text_param_dicts, sensor_dims = sub_views[0].sensor_dims)

    @staticmethod
    def make_without_single_pixel_pruning(myCamera, myScene, annotatedPoints = None, resolution = 10, 
                                          alpha_threshold = 0.05, adaptive_resolution_exponent = 0):
        ### Deal with annotated points
        if annotatedPoints is not None:
            deduplicated_annotatedPnts = list(set(annotatedPoints))
            text_param_dicts = []
            for anPnt in deduplicated_annotatedPnts:
                ### get parameters like text body, font size, font color, etc.
                pnt_params = anPnt.get_parameters()
                
                ### project the location of the point
                pnt_coordArr = np.array([anPnt.coords])
                pnt_pxArr = myCamera.project_points(pnt_coordArr).astype(int)
                
                ### make a point dictionary
                pnt_tuple = tuple(pnt_pxArr.flatten())
                pnt_params['where'] = pnt_tuple
                text_param_dicts.append(pnt_params)
        else:
            text_param_dicts = None


        if myScene.is_empty(): 
            return View.make_empty(sensor_dims = myCamera.sensor_dims, text_param_dicts = text_param_dicts)
        else:
            if adaptive_resolution_exponent != 0:
                camPt = myCamera.get_center_point()
                cntPt = Point3D.from_point2D(CourtLines.make().get_center_point(), z = 0)
            else:
                camPt = None
                cntPt = None

            ### 1) Collect real world points and their colors
            ptArr, clArr, alArr = myScene.rasterize(resolution, alpha_threshold = alpha_threshold, 
                                                    camera_point = camPt, 
                                                    resnorm_point = cntPt,
                                                    adaptive_resolution_exponent = adaptive_resolution_exponent,
                                                    concatenate = True)
            
            if ptArr is None:
                return View.make_empty(sensor_dims = myCamera.sensor_dims, text_param_dicts = text_param_dicts)

            ### 2A) check if points are in front of camera 
            ptArr_cam = myCamera.world_points_to_camera_points(ptArr)
            px_in_front = ptArr_cam[:, 2] < 0 ### z-coordinate goes straight INTO the camera!
            ptArr = ptArr[px_in_front]
            clArr = clArr[px_in_front]
            alArr = alArr[px_in_front]
        
            ### 3) Project Point Cloud in 3D onto 2D Sensor
            pxArr = myCamera.project_points(ptArr).astype(int)
            px_on_sensor = ((0, 0) <= pxArr) & (pxArr <  myCamera.sensor_dims)
            px_filter = px_on_sensor.all(axis = 1)
            
            ptArr = ptArr[px_filter]
            clArr = clArr[px_filter]
            alArr = alArr[px_filter]
            pxArr = pxArr[px_filter]
            
            ### 4) For every point, get distance from Pinhole
            dsArr = ((ptArr - myCamera.get_cw()) ** 2).sum(axis = 1) ** 0.5
                
            aView = View(ptArr, clArr, alArr, pxArr, dsArr,
                            text_param_dicts = text_param_dicts, 
                            sensor_dims = myCamera.sensor_dims)

            return aView

    @staticmethod
    def make_with_single_pixel_pruning(myCamera, myScene, annotatedPoints = None, resolution = 10, alpha_threshold = 0.05,
                                        adaptive_resolution_exponent = 0, opacity_pruning_threshold = 50):

        ### Deal with annotated points
        if annotatedPoints is not None:
            deduplicated_annotatedPnts = list(set(annotatedPoints))
            text_param_dicts = []
            for anPnt in deduplicated_annotatedPnts:
                ### get parameters like text body, font size, font color, etc.
                pnt_params = anPnt.get_parameters()
                
                ### project the location of the point
                pnt_coordArr = np.array([anPnt.coords])
                pnt_pxArr = myCamera.project_points(pnt_coordArr).astype(int)
                
                ### make a point dictionary
                pnt_tuple = tuple(pnt_pxArr.flatten())
                pnt_params['where'] = pnt_tuple
                text_param_dicts.append(pnt_params)
        else:
            text_param_dicts = None


        if myScene.is_empty(): 
            return View.make_empty(sensor_dims = myCamera.sensor_dims, text_param_dicts = text_param_dicts)
        else:
            if adaptive_resolution_exponent != 0:
                camPt = myCamera.get_center_point()
                cntPt = Point3D.from_point2D(CourtLines.make().get_center_point(), z = 0)
            else:
                camPt = None
                cntPt = None

            ### 1) Collect real world points and their colors
            ptArrs, clArrs, alArrs = myScene.rasterize(resolution, alpha_threshold = alpha_threshold, 
                                                    camera_point = camPt, 
                                                    resnorm_point = cntPt,
                                                    adaptive_resolution_exponent = adaptive_resolution_exponent,
                                                    concatenate = False)
            
            if ptArrs is None:
                return View.make_empty(sensor_dims = myCamera.sensor_dims, text_param_dicts = text_param_dicts)

            sub_views = []
            for ptArr, clArr, alArr in zip(ptArrs, clArrs, alArrs): ### for each shape in the scene, we have our own subView
                ### 2A) check if points are in front of camera 
                ptArr_cam = myCamera.world_points_to_camera_points(ptArr)
                px_in_front = ptArr_cam[:, 2] < 0 ### z-coordinate goes straight INTO the camera!
                ptArr = ptArr[px_in_front]
                clArr = clArr[px_in_front]
                alArr = alArr[px_in_front]
            
                ### 3) Project Point Cloud in 3D onto 2D Sensor
                pxArr = myCamera.project_points(ptArr).astype(int)
                px_on_sensor = ((0, 0) <= pxArr) & (pxArr <  myCamera.sensor_dims)
                px_filter = px_on_sensor.all(axis = 1)
                
                ptArr = ptArr[px_filter]
                clArr = clArr[px_filter]
                alArr = alArr[px_filter]
                pxArr = pxArr[px_filter]
                
                ### 4) For every point, get distance from Pinhole
                dsArr = ((ptArr - myCamera.get_cw()) ** 2).sum(axis = 1) ** 0.5
                    
                aView = View(ptArr, clArr, alArr, pxArr, dsArr,
                                text_param_dicts = text_param_dicts, 
                                sensor_dims = myCamera.sensor_dims)

                ### 5) single pixel pruning (on each shape separately)
                sp_views = aView.make_single_pixel_views()
                if len(sp_views) > 0:
                    sp_views_pruned = [spView.prune_to_singleton() for spView in sp_views]
                    subView = View.from_many_views(sp_views_pruned)

                    sub_views.append(subView)

            jointView = View.from_many_views(sub_views)
            
            if len(sub_views) > opacity_pruning_threshold:
                ### optional: opacity pruning
                ## -> if there are many shapes in the scene, this might speed up the snap, 
                # but the added overhead might not be worth it.
                sp_views = jointView.make_single_pixel_views()
                sp_views_pruned = [spView.prune_behind_opaque() for spView in sp_views]
                outView = View.from_many_views(sp_views_pruned)
            else:
                outView = jointView

            object.__setattr__(outView, 'text_param_dicts', text_param_dicts)
            object.__setattr__(outView, 'sensor_dims', myCamera.sensor_dims)

            return outView

    @staticmethod
    def make(myCamera, myScene, annotatedPoints = None, resolution = 10, alpha_threshold = 0.05,
            adaptive_resolution_exponent = 0, opacity_pruning_threshold = 50):
        
        if isinstance(myScene, DummyScene) and resolution > 6 or resolution > 12:
            return View.make_with_single_pixel_pruning(myCamera, myScene, resolution = resolution, 
                                                       alpha_threshold = alpha_threshold,
                                                       annotatedPoints = annotatedPoints,
                                                       adaptive_resolution_exponent = adaptive_resolution_exponent, 
                                                       opacity_pruning_threshold = opacity_pruning_threshold)
        else:
            return View.make_without_single_pixel_pruning(myCamera, myScene, resolution = resolution, 
                                                          annotatedPoints = annotatedPoints,
                                                          alpha_threshold = alpha_threshold,
                                                          adaptive_resolution_exponent = adaptive_resolution_exponent)

    def sort_by_pixel(self):
        sort_idxs = np.lexsort((self.pxArr[:, 1], self.pxArr[:, 0]))
        return View(self.ptArr[sort_idxs], self.clArr[sort_idxs], 
                    self.alArr[sort_idxs], self.pxArr[sort_idxs], 
                    self.dsArr[sort_idxs],
                    text_param_dicts = self.text_param_dicts,
                    sensor_dims = self.sensor_dims)
    
    def sort_by_distance(self, near_to_far = True):
        ### 2) For every point, get distance from Pinhole, and sort vectors by it!
        sort_idxs = np.argsort(self.dsArr)
        if not near_to_far: ### this is what we need if we want to do alpha blending.
            sort_idxs = np.flip(sort_idxs)
        return View(self.ptArr[sort_idxs], self.clArr[sort_idxs], 
                    self.alArr[sort_idxs], self.pxArr[sort_idxs], 
                    self.dsArr[sort_idxs],
                    text_param_dicts = self.text_param_dicts,
                    sensor_dims = self.sensor_dims)
        
    def filter_duplicates(self, verbose = False):
        def mask_firstocc_with_unique(a):
            mask = np.zeros(len(a), dtype=bool)
            mask[np.unique(a, return_index=True, axis = 0)[1]] = True
            return mask
        
        fs_mask = mask_firstocc_with_unique(self.pxArr)
        
        if verbose:     
            keep_pct = round(fs_mask.sum() / len(fs_mask) * 100)
            print(f'Keeping {keep_pct}% of points!')
        
        return View(self.ptArr[fs_mask], self.clArr[fs_mask], 
                    self.alArr[fs_mask], self.pxArr[fs_mask], 
                    self.dsArr[fs_mask],
                    text_param_dicts = self.text_param_dicts,
                    sensor_dims = self.sensor_dims)

    def make_single_pixel_views(self):
        ### figure out which points are mapped to the same pixel.
        sortView_px = self.sort_by_pixel()
        if len(sortView_px) == 0:
            out_sp_views = []
        elif len(sortView_px) == 1: ### if there's only one element, the rolling will not work
            out_sp_views = [SinglePixelView(sortView_px.ptArr, sortView_px.clArr, sortView_px.alArr, sortView_px.pxArr, sortView_px.dsArr)]
        else:
            pxArr_diff = sortView_px.pxArr - np.roll(sortView_px.pxArr, 1, 0)
            boolArr_diff = (pxArr_diff != 0).any(axis = 1)
            switch_idxs = np.arange(len(boolArr_diff))[boolArr_diff]

            out_sp_views = []
            prev_idx = None
            for idx in switch_idxs:
                if prev_idx is not None:
                    currView = sortView_px[prev_idx:idx]
                    # unique_pxArr = np.unique(currView.pxArr, axis = 0)
                    # assert len(unique_pxArr) == 1
                    currSPView = SinglePixelView(currView.ptArr, currView.clArr, currView.alArr, currView.pxArr, currView.dsArr)
                    out_sp_views.append(currSPView)

                prev_idx = idx

            currView = sortView_px[idx:]
            currSPView = SinglePixelView(currView.ptArr, currView.clArr, currView.alArr, currView.pxArr, currView.dsArr)
            out_sp_views.append(currSPView)

        return out_sp_views

    def snap_sequential(self, over_array = None):
        ### 1) Sort by distance for painters algorithm: far to near
        sortView = self.sort_by_distance(near_to_far = False)
    
        ### 4) Pluck every point onto a frame with the correct value, and show it!
        rows, cols = sortView.sensor_dims[1], sortView.sensor_dims[0] ## note the switch here: u are columns, v are rows, so indexing inverts!
        
        if over_array is None:
            snapArr = np.zeros((rows, cols, 3), dtype = 'uint8')
        else:
            if len(over_array.shape) == 2:
                snapArr = np.stack([over_array] * 3, axis = 2)
            else:
                snapArr = over_array

        pixels = sortView.pxArr
        alphas = sortView.alArr
        premul_colors = sortView.clArr * sortView.alArr

        for pixel, alpha, premul_color in zip(pixels, alphas, premul_colors):
            u, v = pixel
            
            try:
                curr_color = snapArr[v, u] ### note the switch here: u are columns, v are rows, so indexing inverts! 
            except IndexError:
                continue
            else:
                snapArr[v, u] = curr_color * (1 - alpha) + premul_color
        
        snapFrm = BGRFrame(snapArr)

        if self.text_param_dicts is not None:
            for param_dict in self.text_param_dicts:
                text = param_dict['text']
                snapFrm = snapFrm.put(text, **{k: v for k, v in param_dict.items() if k != 'text'})
        
        return snapFrm
    
    def snap_grouped(self, over_array = None):
        ### figure out which points are mapped to the same pixel.
        ### group these, and perform a vectorized calculation of the color on that pixel.
        sp_views = self.make_single_pixel_views()

        rows, cols = self.sensor_dims[1], self.sensor_dims[0]

        ### figure out base array
        if over_array is None:
            snapArr = np.zeros((rows, cols, 3), dtype = 'uint8')
        else:
            if len(over_array.shape) == 2:
                snapArr = np.stack([over_array] * 3, axis = 2)
            else:
                snapArr = over_array

        prev_idx = None
        for currView in sp_views:
            ### get current view, sorted from near to far
            sort_currView = currView.sort_by_distance(near_to_far = True)
            
            try:
                u, v = sort_currView.pxArr[0] ### pixel value is constant within sort_currView -> take first!
            except IndexError:
                continue
            else:
                curr_color = snapArr[v, u]

            ### the cumulative 1 - alphas -> how much of the 
            cum_comp_alphas = np.cumprod(1 - sort_currView.alArr)
            fs, ls = cum_comp_alphas[:-1], cum_comp_alphas[-1]
            fs_shift = np.array([1, *fs])

            premul_colors = sort_currView.clArr * sort_currView.alArr

            final_premul_color = np.sum(premul_colors * fs_shift[:, np.newaxis], axis = 0)
            final_base_color = ls * curr_color
            
            snapArr[v, u] = final_premul_color + final_base_color

        snapFrm = BGRFrame(snapArr)

        if self.text_param_dicts is not None:
            for param_dict in self.text_param_dicts:
                text = param_dict['text']
                snapFrm = snapFrm.put(text, **{k: v for k, v in param_dict.items() if k != 'text'})
        
        return snapFrm

    def snap(self, over_array = None):
        return self.snap_sequential(over_array = over_array) ### the testing code shows that best results are achieved with pruning in make.
        
    def combine(self, otherView):
        assert self.sensor_dims == otherView.sensor_dims

        ptArr = np.concatenate([self.ptArr, otherView.ptArr], axis = 0)
        clArr = np.concatenate([self.clArr, otherView.clArr], axis = 0)
        alArr = np.concatenate([self.alArr, otherView.alArr], axis = 0) 
        pxArr = np.concatenate([self.pxArr, otherView.pxArr], axis = 0)
        dsArr = np.concatenate([self.dsArr, otherView.dsArr], axis = 0)
        
        self_dicts = [] if self.text_param_dicts is None else self.text_param_dicts
        othr_dicts = [] if otherView.text_param_dicts is None else otherView.text_param_dicts
        text_param_dicts = [*self_dicts, *othr_dicts]
        text_param_dicts = None if len(text_param_dicts) == 0 else text_param_dicts
        
        aView = View(ptArr, clArr, alArr, pxArr, dsArr, text_param_dicts = text_param_dicts, sensor_dims = self.sensor_dims)
    
        aView = aView.sort_by_distance(near_to_far = False)
        return aView

@dataclass(frozen = True)
class SinglePixelView(View):
    ### these are because knowing that raster spots are mapped to the same pixel allows for more pruning

    def get_pixel(self):
        return self.pxArr[0]
    
    def prune_behind_opaque(self):
        sortSPView = self.sort_by_distance(near_to_far = True)

        is_opaque = sortSPView.alArr[:, 0] == 1

        if is_opaque.sum() == 0:
            return self
        else:
            nearest_opaque_idx = np.argmax(is_opaque)
            return sortSPView[:nearest_opaque_idx + 1]

    def prune_to_singleton(self):
        ### this is fine if:
            # 1) the view is of a single shape
            # 2) the shape is flat in some dimension
        # if 2) is not met, it's not too bad: a ball for example will lose projected point on the far side, 
        # but its projected onto the camera by some other point on the camera-facing side of the ball anyway.
        return self.sort_by_distance(near_to_far = True)[[0]]
           
    # def prune_close(self, distance_threshold = 1):
    #     ### exploit that points next to each other are either rendered at resolution distance from each other, 
    #     ### or they're far enough apart not to prune.
    #     ### distance threshold -> the distance to the neighbor under which the 3D point is judged to be the same as its neigbor.
    #         ### note: if we had only flat surfaces to render, we wouldn't need this; but a ball is 3D, 
    #         ### so two points on the sphere will be projected onto the same pixel, but these are not the same.
    #     dists = ((self.ptArr[1:] - self.ptArr[:-1]) ** 2).sum(axis = 1) ** 0.5
    #     self.ptArr[:, :, np.newaxis, np.newaxis] - self.ptArr[np.newaxis, np.newaxis, :, :]

@dataclass
class SolidView:
    ### This View exploits the fact that in projective geometry, ...
        # ... straight line remain straight lines
        # ... an area that is filled in in 3D will remain filled in in 2 dimensions
    ### -> for a shape that is passed in, we read out the limiting contour points in 3D 
        # (i.e, the corner points of a rectangle, or the some contour points on the border of a dummy scene)
    ### -> then, when snapping, we project the contour points into 2D, and fill in. 
    # to figure out for DummyScene -> how to find its "outline"?
    pass

if __name__ == '__main__':
    CHECK_SPEED = True
    CHECK_QUATERNIONS = False
    CHECK_FONTS = False

    if CHECK_SPEED:
        import time

        from ATMANpy.types.Shapes3D import Point3D, Ball3D, Lines3D, Region3D, Regions3D, AnnotatedPoint3D
        from ATMANpy.types.ShapesTennis3D import Player
        import ATMANpy.funcs.vis_funcs as vs

        ####### SOME OBJECTS TO SHOW ########
        courtLines = CourtLines.make().add_singles_netline()

        c_w_2D = courtLines['Bottom Baseline'].center().down(150).left(100)
        d_w_2D = courtLines.get_center_point()
    
        c_w = Point3D.from_point2D(c_w_2D, z = 100)
        d_w = Point3D.from_point2D(d_w_2D, z = 0)
        
        cam1 = Camera.from_to(c_w, d_w)

        ### court
        courtScn = Scene.make_court(courtLines)

        ### some region
        botReg = courtLines.get_half_region(side = 'Bottom', width = 'Singles')
        bot_adReg = courtLines.get_service_box(side = 'Bottom', which = 'Deuce')
        botReg3D = Region3D.from_region2D(botReg, z = 0).recolor(const.BGR['grass_green'])
        ad_botReg3D = Region3D.from_region2D(bot_adReg, z = 1).recolor(const.BGR['strong_pink'])

        ### players        
        p1_2D = courtLines['Bottom Baseline'].center().right(100)
        p2_2D = courtLines['Top Baseline'].center().left(100).up(100)

        ply1 = Player.make(p1_2D.coords)
        ply2 = Player.make(p2_2D.coords)

        ### KDE
        from ATMANpy.types.Abstract import Points, read_in_data_tables
        import ATMANpy.scripts.pipeline_brest_tightseg as ppl
        from ATMANpy.funcs.util_funcs import get_folder_struct

        prm_mod = ppl.parameter_files.default
        cnf_mod = ppl.config_files.default
            
        avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
        confs = [cnf_mod.get_conf(vid) for vid in avail_vids]
        cap = confs[0]['cap']

        dataTable = read_in_data_tables(confs)
        dataPnts = Points.from_datatable(dataTable, split_by_match = False)

        import ATMANpy.funcs.filter_points_funcs as filt_p
        filtPnts1 = dataPnts.filter(filt_p.is_returner, name = 'Brandon Nakashima').filter(filt_p.is_longer_than, threshold = 1)

        filtPnts2 = filtPnts1.filter('is_first_serve')
        filtPnts3 = filtPnts2.filter('is_serve_deuce')
        filtPnts4 = filtPnts3.filter(filt_p.is_serve_out_wide)

        filtPnts_before = filtPnts4.filter(filt_p.is_not_server, name = 'Joao Sousa')
        filtPnts_after = filtPnts4.filter(filt_p.is_server, name = 'Joao Sousa')

        def return_is_fair(somePnt):
            try:
                thirdShot = somePnt[2]
            except:
                thirdShot = None
            return somePnt.get_return().is_fair(thirdShot)

        filtPnts_before_fair = filtPnts_before.filter(return_is_fair)
        retShots_before_fair = filtPnts_before_fair.get_returns() ### unrectified!

        def kde_func(someShots, bandwidth = 40, normalize_density = True, display_at_height = False, resolution = 2,
                    heat_bgr = const.BGR['intense_red'], props = [0.2, 0.4, 0.6, 0.8]):
            ### bulls eye look
            
            from ATMANpy.funcs.util_funcs import interpolate_tuples
            background_bgr = tuple(BGRFrame.from_video(cap, 1100).array[200, 350]) ### the color of the court.
            fill_colors = [interpolate_tuples(background_bgr, heat_bgr, itpl_lambda = p) for p in props]

            from functools import reduce
            cont_scenes = [vs.kde_contourmap(someShots, bandwidth = bandwidth,
                                            normalize_density = normalize_density, 
                                            display_at_height = display_at_height,
                                            resolution = resolution, 
                                            fill = True, zero_color = background_bgr,
                                            threshold_proportion = prop, contour_color = col)[0] for prop, col in zip(props, fill_colors)]
            contScn = reduce(DummyScene.combine, cont_scenes, DummyScene.make_empty())

            contScn = contScn.consolidate()

            return contScn
                        
        retShots_before_fair_rectified = retShots_before_fair.enforce_direction('Top')

        ######################################

        ######### TIMING FUNCTION ############
        def view_and_snap_timed(myCam, myScene, resolution,
                                view_func = 'make', snap_func = 'snap'):
            curr_time = time.time()
            myView = getattr(View, view_func)(myCam, myScene, resolution = resolution)
            view_time = time.time()
            myFrm = getattr(myView, snap_func)()
            snap_time = time.time()

            view_duration = view_time - curr_time
            snap_duration = snap_time - view_time
            total_duration = snap_time - curr_time

            print(f'Timing with view_func: {view_func} and snap_func: {snap_func} at res {resolution}...')
            print(f'    View: {view_duration:.2f} seconds')
            print(f'    Snap: {snap_duration:.2f} seconds (on {len(myView)} points)')
            print(f'   Total: {total_duration:.2f} seconds')

            return myFrm
        #####################################

        SHOW = True
        RESOLUTIONS = [0.1, 1, 10]
        for res in RESOLUTIONS:
            myCamera = cam1
            myScene = courtScn.put(ply1).put(ply2)
            myScene = courtScn.put(ply1).put(ply2).put(ad_botReg3D)

            kdeDict = {'props': [0.25, 0.5, 0.75, 0.9], 'heat_bgr': const.BGR['intense_red'], 'bandwidth': 50, 'resolution': res}
            kde_whole = kde_func(retShots_before_fair_rectified, **kdeDict)
            myScene = kde_whole

            ### old school way
            aFrm1 = view_and_snap_timed(myCamera, myScene, resolution = res, 
                                        view_func = 'make_without_single_pixel_pruning', snap_func = 'snap_sequential')

            ### old school with grouping
            aFrm2 = view_and_snap_timed(myCamera, myScene, resolution = res, 
                                        view_func = 'make_without_single_pixel_pruning', snap_func = 'snap_grouped')

            ### single pixel pruning
            aFrm3 = view_and_snap_timed(myCamera, myScene, resolution = res, 
                                        view_func = 'make_with_single_pixel_pruning', snap_func = 'snap_sequential')

            ### single pixel pruning and grouping
            aFrm4 = view_and_snap_timed(myCamera, myScene, resolution = res, 
                                        view_func = 'make_with_single_pixel_pruning', snap_func = 'snap_grouped')
            
            ### automatic decision about which make and snap to take
            aFrm5 = view_and_snap_timed(myCamera, myScene, resolution = res, 
                                        view_func = 'make', snap_func = 'snap')

            if SHOW: 
                aFrm1.show()
                aFrm2.show()
                aFrm3.show()
                aFrm4.show()
                aFrm5.show()


    if CHECK_QUATERNIONS:
        w, x, y, z = 1, 0.5, 1, 2
        myQuat = Quaternion(w, x, y, z)
        R = myQuat.to_rotation_matrix()
        
        ### construct 5 test case vectors
        vecs = [np.random.random(size = (3)) for idx in range(5)]
        print('The following vectors should be the same!')
        for vec in vecs:
            p1 = R @ vec
            p2 = myQuat.rotate(vec)
            print(np.stack([p1, p2], axis = 1))

        ### to_rotation_matrix seems to work!
        print('_______')
        
        print('The following two matrices should be the same')
        print(R)
        print(Quaternion.from_rotation_matrix(R).to_rotation_matrix())
        
        print('The following two quaternions should be the same')
        print(myQuat.unit())
        print(Quaternion.from_rotation_matrix(R))

        print('___________')

        R = np.array([[0.395, 0.362, 0.843],
                      [-0.626, 0.796, -0.056],
                      [-0.677, -0.498, 0.529]])
        
        q = Quaternion.from_rotation_matrix(R) ### as in the paper!!!
        
        print('The following two matrices are close enough.')
        print(q.to_rotation_matrix()) 
        print(R)
        
    if CHECK_FONTS:

        from ATMANpy.types.Shapes3D import Point3D, Ball3D, Lines3D, Region3D, Regions3D, AnnotatedPoint3D
        from ATMANpy.types.ShapesTennis3D import Player
        import cv2 as cv

        RESOLUTION = 1

        courtLines = CourtLines.make().add_singles_netline()

        c_w_2D = courtLines['Bottom Baseline'].center().left(100)
        d_w_2D = courtLines.get_center_point()
    
        c_w = Point3D.from_point2D(c_w_2D, z = 100)
        d_w = Point3D.from_point2D(d_w_2D, z = 0)
            
        botReg = courtLines.get_half_region(side = 'Bottom', width = 'Singles')
        bot_adReg = courtLines.get_service_box(side = 'Bottom', which = 'Deuce')
        botReg3D = Region3D.from_region2D(botReg, z = 0).recolor(const.BGR['grass_green'])
        ad_botReg3D = Region3D.from_region2D(bot_adReg, z = 1).recolor(const.BGR['strong_pink'])
        courtScn = Scene.make_court(courtLines).put(botReg3D).put(ad_botReg3D)
        
        myCam = Camera.from_to(c_w, d_w)

        View.make(myCam, courtScn, resolution = RESOLUTION).snap().show()

        ### TESTING ANNOTATED POINTS ###
        
        l_coords, c_coords, r_coords = courtLines.get_3D_net_points()
        lPt, cPt, rPt = Point3D(l_coords), Point3D(c_coords), Point3D(r_coords)

        cv.FONT_HERSHEY_SIMPLEX
        cv.FONT_HERSHEY_PLAIN
        cv.FONT_HERSHEY_DUPLEX
        cv.FONT_HERSHEY_COMPLEX
        cv.FONT_HERSHEY_TRIPLEX
        cv.FONT_HERSHEY_COMPLEX_SMALL
        cv.FONT_HERSHEY_SCRIPT_SIMPLEX
        cv.FONT_HERSHEY_SCRIPT_COMPLEX
        cv.FONT_ITALIC

        #### SERIF FONTS -> THEY LOOK BAD
        # q4 = rPt.up(30).annotate('FONT_HERSHEY_COMPLEX').refont(cv.FONT_HERSHEY_COMPLEX)
        # p1 = lPt.up(120).annotate('FONT_HERSHEY_TRIPLEX').refont(cv.FONT_HERSHEY_TRIPLEX)
        # p2 = lPt.up(90).annotate('FONT_HERSHEY_COMPLEX_SMALL').refont(cv.FONT_HERSHEY_COMPLEX_SMALL)

        ### SCRIPT FONTS -> TERRIBLE
        # p3 = lPt.up(60).annotate('FONT_HERSHEY_SCRIPT_SIMPLEX').refont(cv.FONT_HERSHEY_SCRIPT_SIMPLEX)
        # p4 = lPt.up(30).annotate('FONT_HERSHEY_SCRIPT_COMPLEX').refont(cv.FONT_HERSHEY_SCRIPT_COMPLEX)

        ### ITALIC
        # p1 = rPt.up(30).annotate('FONT_ITALIC').refont(cv.FONT_ITALIC) ## -> doesn't work?

        ### SANS SERIF -> OK
        # q1 = rPt.up(120).annotate('FONT_HERSHEY_SIMPLEX').refont(cv.FONT_HERSHEY_SIMPLEX) ### ok, but not great
        q1 = rPt.up(20).annotate('FONT HERSHEY PLAIN').refont(cv.FONT_HERSHEY_PLAIN).reline(cv.LINE_AA)
        q2 = rPt.up(50).annotate('FONT HERSHEY PLAIN').refont(cv.FONT_HERSHEY_PLAIN).resize(2).reline(cv.LINE_AA)
        q3 = rPt.up(80).annotate('FONT HERSHEY PLAIN').refont(cv.FONT_HERSHEY_PLAIN).resize(2).rethicken(2).reline(cv.LINE_AA)

        # p1 = lPt.up(40).annotate('FONT HERSHEY DUPLEX').refont(cv.FONT_HERSHEY_DUPLEX) ####### good, but plain is slightly better (and can be smaller!)
        # p2 = lPt.up(70).annotate('FONT HERSHEY DUPLEX').refont(cv.FONT_HERSHEY_DUPLEX).rethicken(2)
        # p3 = lPt.up(100).annotate('FONT HERSHEY DUPLEX').refont(cv.FONT_HERSHEY_DUPLEX).rethicken(3) 
        
        p1 = lPt.up(40).annotate('FONT HERSHEY PLAIN').refont(cv.FONT_HERSHEY_PLAIN).reline(cv.FILLED)
        p2 = lPt.up(70).annotate('FONT HERSHEY PLAIN').refont(cv.FONT_HERSHEY_PLAIN).resize(2).reline(cv.FILLED)
        p3 = lPt.up(100).annotate('FONT HERSHEY PLAIN').refont(cv.FONT_HERSHEY_PLAIN).resize(2).rethicken(2).reline(cv.FILLED)

        #### RESULTS -> LINE AA is best, FONT_HERSHEY_PLAIN is best
        View.make(myCam, courtScn, annotatedPoints = [p1, p2, p3, q1, q2, q3], resolution = RESOLUTION).snap().show()


        #### TESTING ADAPTIVE RESOLUTION ###
        
        c_w_2D = courtLines['Bottom Baseline'].center().down(150).left(100)
        d_w_2D = courtLines.get_center_point()
    
        c_w = Point3D.from_point2D(c_w_2D, z = 100)
        d_w = Point3D.from_point2D(d_w_2D, z = 0)
        
        myCam = Camera.from_to(c_w, d_w)

        p1_2D = courtLines['Bottom Baseline'].center().right(100)
        p2_2D = courtLines['Top Baseline'].center().left(100).up(100)

        ply1 = Player.make(p1_2D.coords)
        ply2 = Player.make(p2_2D.coords)
        courtScn = Scene.make_court(courtLines).put(ply1).put(ply2)

        
        View.make(myCam, courtScn, adaptive_resolution_exponent = 0, resolution = RESOLUTION).snap().show()
        View.make(myCam, courtScn, adaptive_resolution_exponent = 1, resolution = RESOLUTION).snap().show()
        View.make(myCam, courtScn, adaptive_resolution_exponent = 2, resolution = RESOLUTION).snap().show()
        View.make(myCam, courtScn, adaptive_resolution_exponent = 3, resolution = RESOLUTION).snap().show()

        # Player.make()

        emptyScn = Scene.make_empty() 
        courtRegions3D = Court.make()
        myBall = Ball3D((30, 30, 30))
        myNet = Net.make_singles()

        backScn = emptyScn.put(courtRegions3D).put(myNet)
        ballScn = emptyScn.put(myBall)

        myCam = Camera.calibrated_AO_behind()

        backView = View.make(myCam, backScn)
        frntView = View.make(myCam, ballScn)

        backView.snap().show()
        frntView.snap().show()

        bothView = backView.combine(frntView)
        bothView.snap().show()

