# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:26:23 2023

@author: gianl
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from ATMANpy.types.Frames import BGRFrame
from ATMANpy.types.ShapesTennis3D import Court, Net
from ATMANpy.types.Lines import CourtLines

import ATMANpy.auxil.constants as const

@dataclass(frozen = True)
class Scene:
    shapes: List ### a collection of various 3D shapes
    
    @staticmethod
    def make_empty():
        return Scene([])
    
    @staticmethod
    def make_court(courtLines = None):
        courtRegions3D = Court.make(courtLines = courtLines)
        net3D = Net.make_singles()

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

    ### METHODS ###
    def rasterize(self, resolution, alpha_threshold = 0.05):
        ### yields a point-cloud representation of the objects in the scene, 
        # with points stored in rastArr_all and their colors stored in colArr_all
        pruned_self = self.prune_transparents(alpha_threshold = alpha_threshold)

        if pruned_self.is_empty():
            return None, None
        else:
            rastArr_all = []
            colArr_all = []
            alphaArr_all = []
            for shape3D in pruned_self.shapes:
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
    
    @staticmethod
    def make_empty():
        return DummyScene([], [])
    
    def is_empty(self):
        return len(self.rastArrs) == 0
    
    def put(self, rastArr, colArr):
        return DummyScene([*self.rastArrs, rastArr], [*self.colArrs, colArr])

    def rasterize(self, resolution, alpha_threshold = 0.05):
        ### TODO: implement pruning with alpha thresholding (see above) 
        rastArr_all = np.concatenate(self.rastArrs)
        colArr_all = np.concatenate(self.colArrs)
        alphaArr_all = np.concatenate(self.alphaArrs)
        return rastArr_all, colArr_all, alphaArr_all
    

@dataclass(frozen = True)
class Camera:
    M_int: np.array
    M_ext: np.array
    sensor_dims: tuple
    
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
    def low_angled_behind():
        baseCam = Camera.calibrated_AO_behind()
        myCam = baseCam.left(20).into(14).pan(-0.06).down(4)
        return myCam

    @staticmethod
    def mid_lateral():
        baseCam = Camera.calibrated_AO_behind()
        myCam = baseCam.right(55).pan(0.5).rotate(0.005).right(133).zoom(0.175)
        myCam = myCam.tilt(-0.05).pan(0.008)
        return myCam
    
    @staticmethod
    def birds_eye(cam_height = 100, sensor_dims = (1920, 1080),
                  f = 3650, o_x = None, o_y = None):
        if o_x is None:
            o_x = sensor_dims[0] / 2
        if o_y is None:
            o_y = sensor_dims[1] / 2

        ctPt = CourtLines.make().get_center_point()
        x, y = ctPt.coords
        
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

    ############ Internal Matrix Changes ##########
    
    def zoom(self, factor):
        ### factor > 1 -> zoom in; factor < 1 -> zoom out
        M_int_new = self.M_int
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
        M_ext_new = self.M_ext
        M_ext_new[0, 3] += feet * const.SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
       
    def left(self, feet):
        #### left-right motion occurs along the x-axis
        M_ext_new = self.M_ext
        M_ext_new[0, 3] -= feet * const.SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
       
    def down(self, feet):
        #### up-down motion occurs along the y-axis
        M_ext_new = self.M_ext
        M_ext_new[1, 3] += feet * const.SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
       
    def up(self, feet):
        #### up-down motion occurs along the y-axis
        M_ext_new = self.M_ext
        M_ext_new[1, 3] -= feet * const.SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
    
    def into(self, feet):
        #### in-out motion occurs along the z-axis
        M_ext_new = self.M_ext
        M_ext_new[2, 3] += feet * const.SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
       
    def out(self, feet):
        #### in-out motion occurs along the z-axis
        M_ext_new = self.M_ext
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
    
    sensor_dims: tuple = (1920, 1080)
    
    @staticmethod
    def make_empty(sensor_dims = (1920, 1080)):
        return View(np.zeros((0, 3)), 
                    np.zeros((0, 3)), 
                    np.zeros((0, 1)),
                    np.zeros((0, 2), dtype = int),  ### pixels need to be integer!
                    np.zeros(0),
                    sensor_dims = sensor_dims)
    
    @staticmethod
    def make(myCamera, myScene, resolution = 10, alpha_threshold = 0.05):
        
        if myScene.is_empty(): 
            return View.make_empty()
        else:
            ### 1) Collect real world points and their colors
            ptArr, clArr, alArr = myScene.rasterize(resolution, alpha_threshold = alpha_threshold)
            
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
                         sensor_dims = myCamera.sensor_dims)
        
            return aView

    def sort_by_pixel(self):
        sort_idxs = np.lexsort((self.pxArr[:, 1], self.pxArr[:, 0]))
        return View(self.ptArr[sort_idxs], self.clArr[sort_idxs], 
                    self.alArr[sort_idxs], self.pxArr[sort_idxs], 
                    self.dsArr[sort_idxs],
                    sensor_dims = self.sensor_dims)
    
    def sort_by_distance(self, near_to_far = True):
        ### 2) For every point, get distance from Pinhole, and sort vectors by it!
        sort_idxs = np.argsort(self.dsArr)
        if not near_to_far: ### this is what we need if we want to do alpha blending.
            sort_idxs = np.flip(sort_idxs)
        return View(self.ptArr[sort_idxs], self.clArr[sort_idxs], 
                    self.alArr[sort_idxs], self.pxArr[sort_idxs], 
                    self.dsArr[sort_idxs],
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
        
        return snapFrm
        
    def combine(self, otherView):
        ptArr = np.concatenate([self.ptArr, otherView.ptArr], axis = 0)
        clArr = np.concatenate([self.clArr, otherView.clArr], axis = 0)
        alArr = np.concatenate([self.alArr, otherView.alArr], axis = 0) 
        pxArr = np.concatenate([self.pxArr, otherView.pxArr], axis = 0)
        dsArr = np.concatenate([self.dsArr, otherView.dsArr], axis = 0)
        
        aView = View(ptArr, clArr, alArr, pxArr, dsArr)
    
        aView = aView.sort_by_distance(near_to_far = False)
        return aView

    
# from ATMANpy.types.Shapes3D import Ball3D, Lines3D, Region3D, Regions3D

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

