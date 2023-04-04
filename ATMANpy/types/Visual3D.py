# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:26:23 2023

@author: gianl
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from ATMANpy.types.Frames import BGRFrame

@dataclass(frozen = True)
class Scene:
    shapes: List ### a collection of various 3D shapes
    
    @staticmethod
    def make():
        return Scene([])
    
    def put(self, someShape):
        return Scene([*self.shapes, someShape])
    
    def rasterize(self, resolution):
        ### yields a point-cloud representation of the objects in the scene, 
        # with points stored in rastArr_all and their colors stored in colArr_all
        
        
        rastArr_all = []
        colArr_all = []
        for shape3D in self.shapes:
            rastArr = shape3D.rasterize(resolution)
            
            colTup = shape3D.color
            colArr = np.tile(np.array(colTup), (len(rastArr), 1)) ### make same shape as rastArr
            
            rastArr_all.append(rastArr)
            colArr_all.append(colArr)
            
        rastArr_all = np.concatenate(rastArr_all)
        colArr_all = np.concatenate(colArr_all)
        
        return rastArr_all, colArr_all
    

@dataclass(frozen = True)
class Camera:
    M_int: np.array
    M_ext: np.array
    sensor_dims: tuple
    
    @staticmethod
    def pinhole_origin_aligned(f_x, f_y, o_x, o_y, s_x, s_y):
        M_int = np.array([[f_x, 0, o_x, 0],
                          [0, f_y, o_y, 0],
                          [0,   1,   1, 0]])
        
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
    
    def get_K(self):
        return self.M_int[:3, :3]
    def get_t(self):
        return self.M_ext[:3, 3]
    def get_R(self):
        return self.M_ext[:3, :3]
        
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
    
    def snap_scene(self, myScn, resolution = 10):
        
        ### 1) Project Point Cloud in 3D onto 2D Sensor
        ptArr, clArr = myScn.rasterize(resolution)
        pxArr = self.project_points(ptArr).astype(int)
        
        ### 2) For every point, get distance from Pinhole, and sort vectors by it!
        distVec = ((ptArr - self.get_t()) ** 2).sum(axis = 1) ** 0.5
        sort_idxs = np.argsort(distVec) ### indices, sorted by distance!
        sort_pxArr, sort_clArr = pxArr[sort_idxs], clArr[sort_idxs]
        
        ### 3) Filter out all duplicates now: the only one we keep is the one closest to the camera!
        def mask_firstocc_with_unique(a):
            mask = np.ones(len(a), dtype=bool)
            mask[np.unique(a, return_index=True, axis = 0)[1]] = False
            return mask
        
        fs_mask = mask_firstocc_with_unique(sort_pxArr)
        mask_pxArr, mask_clArr = sort_pxArr[fs_mask], sort_clArr[fs_mask]
        
        ### 4) Pluck every point onto a frame with the correct value, and show it!
        rows, cols = self.sensor_dims[1], self.sensor_dims[0]
        snapArr = np.zeros((rows, cols, 3), dtype = 'uint8')

        for pixel, color in zip(mask_pxArr, mask_clArr):
            u, v = pixel
            snapArr[u, v] = color
        
        snapFrm = BGRFrame(snapArr)
        
        return snapFrm
    
from ATMANpy.types.Shapes3D import Ball3D, Lines3D, Region3D, Regions3D


myScn = Scene.make() 

courtRegions3D = Regions3D.make_court()
myBall = Ball3D((30, 30, 30))


myScn = myScn.put(myBall).put(courtRegions3D['Top Baseline'])

ptArr, clArr = myScn.rasterize(10)

myCam = Camera.calibrated_AO_behind()

prArr = myCam.project_points(ptArr)

snapFrm = myCam.snap_scene(myScn, resolution = 10)
snapFrm.show()