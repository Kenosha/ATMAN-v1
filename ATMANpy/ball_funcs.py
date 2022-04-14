# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 18:26:12 2022

@author: gianl
"""

import warnings
import cv2 as cv
import numpy as np

from ATMANpy.Frames import Frame, BoolFrame, GreyFrame
from ATMANpy.Contours import Contour

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


def find_alpha_and_std_and_semiaxes(cnt, gFrm, maskFrm):
    PIXEL_MARGIN = 3
    ZOOM_FACTOR = 10
    
    cntRegion = cnt.enclosing_region(margin = PIXEL_MARGIN)
    
    foreBlob_large = gFrm.crop(cntRegion).zoom(ZOOM_FACTOR)
    foreBool_large = maskFrm.crop(cntRegion).zoom(ZOOM_FACTOR)
    
    cnts_large, _ = foreBool_large.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    if len(cnts_large) == 1:
        cnt_large = Contour(cnts_large[0])
    else:
        cnt_large = sorted([Contour(c) for c in cnts_large], key = lambda c: c.area())[-1] #take largest
    
    from ATMANpy.ball_funcs import get_ellipse, get_ellipse_mask, get_sobel_grads, get_normal_grads, calculate_alpha
    
    centerCoords, semiAxes, theta = get_ellipse(cnt_large)
    ellipseMask = get_ellipse_mask(foreBlob_large, centerCoords, semiAxes, theta)
    
    sobGrads = get_sobel_grads(foreBlob_large)
    normGrads = get_normal_grads(foreBlob_large, centerCoords)
    
    alpha, std = calculate_alpha(sobGrads, normGrads, ellipseMask)
    return alpha, std, semiAxes ### we also return the semiAxes, since they will be params


##### Motion Mask, Ball Candidates ####

def get_motion_mask(cap, frameNum, diffList = [4, 6, 8]):
    myFrame = GreyFrame.from_video(cap, frameNum)
    
    frameList = [frameNum - d for d in diffList] + [frameNum + d for d in diffList]
    gFrames = [GreyFrame.from_video(cap, fnum) for fnum in frameList]
    
    def isFG(myFrame, diffFrame):
        THRESH = 10
        return myFrame - diffFrame > THRESH
    
    outFrame = BoolFrame(myFrame.zero().array + 1)
    for diffFrame in gFrames:
        outFrame = outFrame & isFG(myFrame, diffFrame)
   
    return outFrame

def get_motion_mask_from_frames(gFrm, diffFrames):
    
    def isFG(myFrame, diffFrame):
        THRESH = 10
        return myFrame - diffFrame > THRESH
    
    outFrame = BoolFrame(gFrm.zero().array + 1)
    for diffFrame in diffFrames:
        outFrame = outFrame & isFG(gFrm, diffFrame)
   
    return outFrame

def find_ball_candidate_mask(cFrm, motMask, topPlayerContour, botPlayerContour):
    # cFrm.mask(motMask).show()
    
    def cntMask(frm, cnt):
        ### Dilated player mask
        dil_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 10))
        return frm.within_contour(cnt).dilate(dil_kernel)
    playerMask = cntMask(cFrm, topPlayerContour) | cntMask(cFrm, botPlayerContour)
    # cFrm.mask(playerMask).show()
    
    ### Opened Motion Mask
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    motMask = motMask.open(open_kernel)
    # cFrm.mask(motMask).show()
    
    ### Player and Motion Mask
    intMask = ~playerMask & motMask
    # cFrm.mask(intMask).show()
    
    return intMask

#### SVM Feature collection for each ball candidate ####
import pandas as pd
def make_feature_frame(cnts, hFrm, gFrm, intMask):
    
    cntSers = []
    
    for cnt_index, cnt in enumerate(cnts):

        ### COLLECT SVM FEATURES
        cntDict = {}
            
        # Alpha (and maybe add std?) and semiaxes
        try:
            alpha, std, semiAxes = find_alpha_and_std_and_semiaxes(cnt, gFrm, intMask)
            semiMaj, semiMin = semiAxes
        except:
            alpha, std, semiMaj, semiMin = np.nan, np.nan, np.nan, np.nan #occasionally, a contour has no ellipse that can be fitted.
        finally:
            cntDict['Alpha'] = alpha
            cntDict['AlphaSTD'] = std
            cntDict['SemiMaj'] = semiMaj
            cntDict['SemiMin'] = semiMin
        
        # Blob Center Location
        try:
            centPt = cnt.center()
            x, y = centPt.x, centPt.y
        except:
            x, y = np.nan, np.nan ### this happens if the contour does not have an area (if the points are colinear)
            
        cntDict['Cx'] = x
        cntDict['Cy'] = y
        
        # HSV Channels
        mean, std = hFrm.mean_std(mask = hFrm.within_contour(cnt))
        h, s, v = mean
        cntDict['H'] = h
        cntDict['S'] = s
        cntDict['V'] = v
        
        cntSer = pd.Series(cntDict, name = cnt_index)
        cntSers.append(cntSer)
        
    cntFrm = pd.concat(cntSers, axis = 1).T
    return cntFrm
