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

from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
SCALE = 2 * SCHEMA_SIZE  ## times two, because there is a minimum scale factor... 


@dataclass(frozen = True)
class Scene:
    shapes: List ### a collection of various 3D shapes
    
    @staticmethod
    def make_empty():
        return Scene([])
    
    @staticmethod
    def make_court():
        courtRegions3D = Court.make()
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
    
    def rasterize(self, resolution):
        ### yields a point-cloud representation of the objects in the scene, 
        # with points stored in rastArr_all and their colors stored in colArr_all
        if self.is_empty():
            return None, None
        else:
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
        M_ext_new[0, 3] += feet * SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
       
    def left(self, feet):
        #### left-right motion occurs along the x-axis
        M_ext_new = self.M_ext
        M_ext_new[0, 3] -= feet * SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
       
    def down(self, feet):
        #### up-down motion occurs along the y-axis
        M_ext_new = self.M_ext
        M_ext_new[1, 3] += feet * SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
       
    def up(self, feet):
        #### up-down motion occurs along the y-axis
        M_ext_new = self.M_ext
        M_ext_new[1, 3] -= feet * SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
    
    def into(self, feet):
        #### in-out motion occurs along the z-axis
        M_ext_new = self.M_ext
        M_ext_new[2, 3] += feet * SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
       
    def out(self, feet):
        #### in-out motion occurs along the z-axis
        M_ext_new = self.M_ext
        M_ext_new[2, 3] -= feet * SCALE
        return Camera(self.M_int, M_ext_new, self.sensor_dims)
    
    
    
@dataclass(frozen = True)
class View:
    ### a View combines a camera and a scene: 
        # here, we store the Scene's rasterization, 
        # but also the camera projection of it, and the distance vector of all objects to the camera.
    
    ptArr: np.array ### points in 3D space
    clArr: np.array ### colors
    pxArr: np.array ### points in 2D space -> 3D points projected onto screen
    dsArr: np.array ### distance of objects to camera
    
    @staticmethod
    def make_empty():
        return View(np.zeros((0, 3)), 
                    np.zeros((0, 3)), 
                    np.zeros((0, 2), dtype = int),  ### pixels need to be integer!
                    np.zeros(0))
    
    @staticmethod
    def make(myCamera, myScene, resolution = 10, do_sort = True, do_filter = True):
        if myScene.is_empty(): 
            return View.make_empty()
        else:
            ### 1) Project Point Cloud in 3D onto 2D Sensor
            ptArr, clArr = myScene.rasterize(resolution)
            pxArr = myCamera.project_points(ptArr).astype(int)
            
            ### 2) For every point, get distance from Pinhole
            dsArr = ((ptArr - myCamera.get_t()) ** 2).sum(axis = 1) ** 0.5
            aView = View(ptArr, clArr, pxArr, dsArr)
            
            if do_sort:
                ### 3) Sort view by distance from pinhole
                aView = aView.sort_by_distance()
            
            if do_filter:
                ### 4) Filter out duplicate pixel
                aView = aView.filter_duplicates()
            
            return aView

    def sort_by_distance(self):
        ### 2) For every point, get distance from Pinhole, and sort vectors by it!
        sort_idxs = np.argsort(self.dsArr)
        return View(self.ptArr[sort_idxs], self.clArr[sort_idxs], self.pxArr[sort_idxs], self.dsArr[sort_idxs])
        
    def filter_duplicates(self, verbose = False):
        def mask_firstocc_with_unique(a):
            mask = np.zeros(len(a), dtype=bool)
            mask[np.unique(a, return_index=True, axis = 0)[1]] = True
            return mask
        
        fs_mask = mask_firstocc_with_unique(self.pxArr)
        
        if verbose:     
            keep_pct = round(fs_mask.sum() / len(fs_mask) * 100)
            print(f'Keeping {keep_pct}% of points!')
        
        return View(self.ptArr[fs_mask], self.clArr[fs_mask], self.pxArr[fs_mask], self.dsArr[fs_mask])

    def snap(self, sensor_dims = (1920, 1080)):
        
        ### 4) Pluck every point onto a frame with the correct value, and show it!
        rows, cols = sensor_dims[1], sensor_dims[0] ## note the switch here: u are columns, v are rows, so indexing inverts!
        snapArr = np.zeros((rows, cols, 3), dtype = 'uint8')

        for pixel, color in zip(self.pxArr, self.clArr):
            u, v = pixel
            try:
                snapArr[v, u] = color ### note the switch here: u are columns, v are rows, so indexing inverts! 
            except IndexError:
                pass
        
        snapFrm = BGRFrame(snapArr)
        
        return snapFrm
        
    def combine(self, otherView):
        ptArr = np.concatenate([self.ptArr, otherView.ptArr], axis = 0)
        clArr = np.concatenate([self.clArr, otherView.clArr], axis = 0)
        pxArr = np.concatenate([self.pxArr, otherView.pxArr], axis = 0)
        dsArr = np.concatenate([self.dsArr, otherView.dsArr], axis = 0)
        
        return View(ptArr, clArr, pxArr, dsArr)#.sort_by_distance()#.filter_duplicates()

    
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

