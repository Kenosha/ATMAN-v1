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
class Scene:
    shapes: List ### a collection of various 3D shapes
        
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

    @staticmethod
    def make_empty():
        return Scene([])
    
    @staticmethod
    def make_court(courtLines = None, line_color = const.BGR['bright_grey'], net_color = const.BGR['smokey_grey']):
        courtRegions3D = Court.make(courtLines = courtLines, color = line_color)
        net3D = Net.make_singles(color = net_color)

        return Scene.make_empty().put(courtRegions3D).put(net3D)
    
    def is_empty(self):
        return len(self.shapes) == 0
    
    def put(self, someShape):
        if someShape is None:
            return self
        elif isinstance(someShape, list):
            return Scene([*self.shapes, *someShape])
        else:
            return Scene([*self.shapes, someShape])
    
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
                  camera_point = None, resnorm_point = None, adaptive_resolution_exponent = 0): ### these factors are relevant if we want adaptive resolution
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

    @staticmethod
    def make_empty():
        return DummyScene([], [], [])
    
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
        

    def rasterize(self, resolution, alpha_threshold = 0.05, 
                  camera_point = None, resnorm_point = None, adaptive_resolution_exponent = 0): ### TODO: adaptive resolution -> necessary for dummy scenes?
        ### TODO: implement pruning with alpha thresholding (see above) 
        pruned_self = self.prune_transparents()

        rastArr_all = np.concatenate(pruned_self.rastArrs)
        colArr_all = np.concatenate(pruned_self.colArrs)
        alphaArr_all = np.concatenate(pruned_self.alphaArrs)
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
    
    def project_points(self, ptArr):
        num_points, num_dims = ptArr.shape
        if num_dims == 4: ### --> already in homogenous coordinates!
            homArr = ptArr
        elif num_dims == 3: ### --> not in homogenous coordinates!
            homArr = np.concatenate([ptArr, np.ones((num_points, 1))], axis = 1)
        else:
            raise Exception('Points to be projected need to be (homogenous or inhomogenous) 3D')
        
        M_proj = self.M_int @ self.M_ext
        
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
    def make(myCamera, myScene, annotatedPoints = None, resolution = 10, alpha_threshold = 0.05,
             adaptive_resolution_exponent = 0):

        ### Deal with annotated points
        if annotatedPoints is not None:
            text_param_dicts = []
            for anPnt in annotatedPoints:
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
                                                    adaptive_resolution_exponent = adaptive_resolution_exponent)
            
            if ptArr is None:
                return View.make_empty(sensor_dims = myCamera.sensor_dims, text_param_dicts = text_param_dicts)

            ### 2) Suppress negative coordinates
            all_positive_indices = (ptArr >= 0).all(axis = 1)
            ptArr = ptArr[all_positive_indices]
            clArr = clArr[all_positive_indices]
            alArr = alArr[all_positive_indices]
        
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

    def snap(self):
        ### 1) Sort by distance for painters algorithm: far to near
        sortView = self.sort_by_distance(near_to_far = False)
    
        ### 4) Pluck every point onto a frame with the correct value, and show it!
        rows, cols = sortView.sensor_dims[1], sortView.sensor_dims[0] ## note the switch here: u are columns, v are rows, so indexing inverts!
        snapArr = np.zeros((rows, cols, 3), dtype = 'uint8')

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
                text = param_dict.pop('text')
                snapFrm = snapFrm.put(text, **param_dict)
        
        return snapFrm
        
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

if __name__ == '__main__':
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

    # emptyScn = Scene.empty() 
    # courtRegions3D = Regions3D.make_court()
    # myBall = Ball3D((30, 30, 30))
    # myNet = Regions3D.make_singles_net()

    # backScn = emptyScn.put(courtRegions3D).put(myNet)
    # ballScn = emptyScn.put(myBall)

    # myCam = Camera.calibrated_AO_behind()

    # backView = View.make(myCam, backScn)
    # frntView = View.make(myCam, ballScn)

    # backView.snap().show()
    # frntView.snap().show()

    # bothView = backView.combine(frntView)
    # bothView.show()

