# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 18:26:12 2022

@author: gianl
"""

import warnings
import cv2 as cv
import numpy as np

from ATMANpy.types.Frames import Frame, BoolFrame, GreyFrame
from ATMANpy.types.Contours import Contour

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
def get_sobel_grads(foreBlob_large, ksize = 3):
    warnings.filterwarnings("default", category = RuntimeWarning)
    sobArr = foreBlob_large.sobel_dirs(toward_high_intensity = False, ksize = ksize)
    ### we're turning around the sobel gradients directions to point out and away from the ellipse,
    # (given that the center is light and outside is dark)
    # this way, we're close to the formulation in yan x-mas kittler (2005).

    ### if we downscale, we turn the sobel gradients (which are now 0 to 2*pi)
    ### into angles that go from 0 to 360, to have only uint16, which is faster
    ### than calculating with floats
    degArr = (sobArr * 180 / np.pi)
    
    return degArr


### Get the Normal Gradients ###
def get_normal_grads(foreBlob_large, centerCoords, theta):
    
    # construct a grid that contains a coordinate system, essentially
    # rows = np.array(list(range(foreBlob_large.rows, 0, -1))) # note that in index terms, low numbers are high. We want low numbers to be at the bottom of coord system.
    ### above is unnecessary! sobel gradient coords are also top-left origin.
    rows, cols = np.arange(foreBlob_large.rows), np.arange(foreBlob_large.cols)
    center_row, center_col = centerCoords[1], centerCoords[0] ### first element of centerCoords is x -> the column. second is y -> the row.
    
    idxGrid = np.dstack(np.meshgrid(rows, cols, indexing = 'ij')) ### a grid which contains as values of a cell the index of a cell
    assert idxGrid.shape[:2] == foreBlob_large.shape[:2]
    
    ### Get the angle of the normal vectors, from 0 at 3 o clock, 90 at 6 o clock, etc.
    idxGrid_rel = idxGrid - (center_row, center_col)
    angle_of_an_upright_ellipse = np.arctan2(idxGrid_rel[:, :, 0], idxGrid_rel[:, :, 1]) ### basically, arctan2(y, x)
    inclination_correction = theta * np.pi / 180
    normArr = angle_of_an_upright_ellipse - inclination_correction
    ## why minus? see notes, 22.07.2023
    
    ### remap [-pi, pi] to [0, 2pi]
    normArr = (normArr + 4 * np.pi) % (2 * np.pi) ### x 4pi instead of x 2pi 
                                                    # just in case the inclination correction goes up to 2pi.
                                                    # (in which case we could get as low as -3pi, in normArr.)
    
    ### Turn into Degrees
    normArr = (normArr * 180 / np.pi)
    
    return normArr

def calculate_alpha(sobArr, normArr, ellipseMask):
    
    #### putting it all together
    absdiffFrm = np.abs(normArr - sobArr)
    
    # half_circle = 180 if unit == 'deg' else np.pi  
    half_circle = 180
    anglArr = np.abs((absdiffFrm + half_circle) % (2 * half_circle) - half_circle)
    ### ensure that we take the smaller difference between two angles
    # i.e: d(360, 5) != 355, but 10
    
    mean, std = Frame(anglArr).mean_std(mask = ellipseMask)
    return mean, std


def find_alpha_and_std_and_semiaxes(cnt, gFrm,
                                    ksize = 3,
                                    pixel_margin = 3, zoom_factor = 10):

    cntRegion = cnt.enclosing_region(margin = pixel_margin)
    foreBlob_large = gFrm.crop(cntRegion).zoom(zoom_factor)
    foreBool_large = gFrm.within_contour(cnt).crop(cntRegion).zoom(zoom_factor)
    
    ### technically, we could just scale up the contour, 
    ### but then the contour would be much chunkier than finding a new contour in the interpolated image.
    cnts_large, _ = foreBool_large.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
    assert cnts_large.length() == 1
    cnt_large = cnts_large[0]
        
    from ATMANpy.funcs.ball_funcs import get_ellipse, get_ellipse_mask, get_sobel_grads, get_normal_grads, calculate_alpha
    
    centerCoords, semiAxes, theta = get_ellipse(cnt_large)
    ellipseMask = get_ellipse_mask(foreBlob_large, centerCoords, semiAxes, theta)
    
    sobArr = get_sobel_grads(foreBlob_large, ksize = ksize)
    normArr = get_normal_grads(foreBlob_large, centerCoords, theta)
    
    alpha, std = calculate_alpha(sobArr, normArr, ellipseMask)
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
    
    if not topPlayerContour is None:
        topMask = cntMask(cFrm, topPlayerContour)
    else:
        topMask = BoolFrame(cFrm.to_grey().zero().array)
        print('Lost Top Player Tracking!')
        
    if not botPlayerContour is None:
        botMask = cntMask(cFrm, botPlayerContour)
    else:
        botMask = BoolFrame(cFrm.to_grey().zero().array)
        print('Lost Bottom Player Tracking!')
    
    playerMask = topMask | botMask
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
def make_feature_frame(cnts, hFrm, gFrm,
                       ksize = 3,
                       zoom_factor = 10, pixel_margin = 3):
    
    cntSers = []
    
    for cnt_index, cnt in enumerate(cnts):

        ### COLLECT SVM FEATURES
        cntDict = {}
            
        # Alpha (and maybe add std?) and semiaxes
        try:
            alpha, std, semiAxes = find_alpha_and_std_and_semiaxes(cnt, gFrm,
                                                                   ksize = ksize,
                                                                   zoom_factor = zoom_factor, 
                                                                   pixel_margin = pixel_margin)
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
