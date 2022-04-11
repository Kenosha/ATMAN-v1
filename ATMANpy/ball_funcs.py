# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 18:26:12 2022

@author: gianl
"""

import warnings
import cv2 as cv
import numpy as np

from ATMANpy.Frames import Frame, BoolFrame

####### color orientation versus spatial orientation of ball candidate ellipse #####

#### Get the Ellipse ###    
def get_ellipse(cnt):
    
    cntEllipse = cnt.fit_ellipse()
    centerCoords, majminAxis, rotAngle = cntEllipse
    
    x, y = round(centerCoords[0]), round(centerCoords[1])
    a, b = round(majminAxis[0] / 2), round(majminAxis[1] / 2)
    theta = round(rotAngle)
    
    return (x, y), (a, b), theta

def get_ellipse_mask(greyBlob, centerCoords, semiAxes, theta):
    zeroArr = greyBlob.zero().array.copy()
    
    ellipseArr = cv.ellipse(zeroArr, centerCoords, semiAxes, theta, 0, 360, 255, thickness = 1)
    ellipseMask = BoolFrame(ellipseArr)
    
    # ellipseArray = cv.ellipse(foreBlob_large.array.copy(), (x, y), (a, b), theta, 0, 360, 255, thickness = 2)
    # ellipseFrm = GreyFrame(ellipseArray)
    # ellipseFrm.show()
    
    # ellipseArray = cv.ellipse(foreBool_large.array.copy(), (x, y), (a, b), theta, 0, 360, 255, thickness = 2)
    # ellipseFrm = BoolFrame(ellipseArray)
    # ellipseFrm.show()
    
    return ellipseMask


### Get the Sobel Gradients ###
def get_sobel_grads(greyBlob, unit = 'deg'):
    warnings.filterwarnings("default", category = RuntimeWarning)
    sobGrads = Frame(greyBlob.sobel_dirs(toward_high_intensity = False))
    ### we're turning around the sobel gradients directions to point out and away from the ellipse,
    # (given that the center is light and outside is dark)
    # this way, we're close to the formulation in yan x-mas kittler (2005).
    
    if unit == 'deg':
        ### if we downscale, we turn the sobel gradients (which are now 0 to 2*pi)
        ### into angles that go from 0 to 360, to have only uint16, which is faster
        ### than calculating with floats
        degArray = (sobGrads.array * 180 / np.pi).astype('uint16') 
        sobGrads = Frame(degArray)
    
    # sobChk = sobGrads.mask(ellipseMask)
    # sobChk.show()
    
    # self_copy = foreBlob_large.array.copy()
    # grad_x = cv.Sobel(self_copy, cv.CV_64F, 1, 0)
    # grad_y = cv.Sobel(self_copy, cv.CV_64F, 0, 1)
    
    # xChk = Frame(grad_x).mask(ellipseMask)
    # yChk = Frame(grad_y).mask(ellipseMask)
    
    return sobGrads


### Get the Normal Gradients ###
def get_normal_grads(greyBlob, ellipse_centercoords, unit = 'deg'):
    
    # construct a grid that contains a coordinate system, essentially
    # rows = np.array(list(range(foreBlob_large.rows, 0, -1))) # note that in index terms, low numbers are high. We want low numbers to be at the bottom of coord system.
    ### above is unnecessary! sobel gradient coords are also top-left origin.
    rows = np.array(list(range(greyBlob.rows)))
    cols = np.array(list(range(greyBlob.cols)))
    
    idxGrid = np.dstack(np.meshgrid(rows, cols, indexing = 'ij')) ### a grid which contains as values of a cell the index of a cell
    assert idxGrid.shape[:2] == greyBlob.shape[:2]
    
    idxGrid_rel = idxGrid - ellipse_centercoords
    normGrads = np.arctan2(idxGrid_rel[:, :, 0], idxGrid_rel[:, :, 1])
    normGrads = Frame(normGrads)
    
    if unit == 'deg':
        degArray = (normGrads.array + 2 * np.pi) % (2 * np.pi) # map [-pi, pi] to [0, 2pi]
        degArray = (degArray * 180 / np.pi).astype('uint16') 
        normGrads = Frame(degArray)
    
    # normGrads.show()
    
    return normGrads

def calculate_alpha(sobGrads, normGrads, ellipseMask, unit = 'deg'):
    
    #### putting it all together
    absdiffFrm = (normGrads - sobGrads)
    
    half_circle = 180 if unit == 'deg' else np.pi  
    anglFrm = (absdiffFrm + half_circle) % (2 * half_circle) - half_circle
    ### ensure that we take the smaller difference between two angles
    # i.e: d(360, 5) != 355, but 10
    
    # # chk = diffFrm.mask(ellipseMask)
    # chk2 = anglFrm.mask(ellipseMask)
    # # diffFrm.mask(ellipseMask).show()
    
    mean, std = anglFrm.mean_std(mask = ellipseMask)
    return mean, std