# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:04:54 2022

@author: gianl
"""


import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.utils as atu
from ATMANpy.Points import Point, SafePoint
from ATMANpy.Lines import Line, Lines, CourtLines, Region
from ATMANpy.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame

# image operation
import cv2 as cv
import numpy as np


##### VIDEO FILE READ IN ####
video_id = 'pqxAmUWIoKc'
video_full_name = video_id + '.mp4'

videoDict = atu.get_folder_struct('video_files')


videoPath = videoDict[video_full_name]

cap = cv.VideoCapture(videoPath)

#### GET ARBITRARY FRAME ####
FRAME_NUM = 11800
gFrm = GreyFrame.from_video(cap, FRAME_NUM)
gFrm.show()

cFrm = BGRFrame.from_video(cap, FRAME_NUM)
cFrm.show()
##########################


### PARAMETERS ###
WHITE_PIXEL_THRESHOLD = 150 #this depends heavily on brightness of picture... lines are white, but they might not appear at 255 intensity.
COURT_LINE_WIDTH = 8 ### according to own rough calcs at 1920 x 1080, and also according to vinyes (2018), court line width is equal to 5 pixels
INTENSITY_DIFFERENCE_THRESHOLD = 30

############ DETECTING WHITE LINES #############
def get_vicinity(aPt: Point, 
                 radius: int = 5, direction: str = None):
    radius = round(radius)
    if direction == 'vertical':
        return(aPt, aPt.down(radius), aPt.up(radius))
    elif direction == 'horizontal':
        return(aPt, aPt.left(radius), aPt.right(radius))
    else:
        print('Bad Keyword, enter "Horizontal" or "Vertical"')

def compare_intensity(refInt: int, newInt: int, 
                      int_diff_thresh: int = 10):
    """
    Compares a reference intensity with a new intensity, and returns a boolean
    If the reference intensity is smaller than the new intensity, 
        then the reference point is darker than the new point, and we have no white line candidate
    If the reference intensity is larger than the new intensity by a small amount,
        then the whites extend beyond the reference point, and we have no line again
    If the reference intensity is larger than the new intensity by a sufficiently large amount, 
        then we have a white line candidate
    """
    if refInt >= newInt:
        return refInt - newInt > int_diff_thresh
    else:
        return False
    
def check_vicinity_of_point(aPt: Point, aFrm: Frame, 
                            radius: int = 5, direction: str = None, int_diff_thresh: int = 10):
    refPt, newPt1, newPt2 = get_vicinity(aPt, radius = radius, direction = direction)
    refInt, newInt1, newInt2 = aFrm.eval_at(refPt), aFrm.eval_at(newPt1), aFrm.eval_at(newPt2)
    Bool1 = compare_intensity(refInt, newInt1, int_diff_thresh = int_diff_thresh)
    Bool2 = compare_intensity(refInt, newInt2, int_diff_thresh = int_diff_thresh)
    return (Bool1 and Bool2)

def get_white_lines(gFrm: GreyFrame, width_of_line: int = 5,
                    white_thresh: int = 200, int_diff_thresh: int = 10,
                    verbose = False):
    whiteBFrm = gFrm > white_thresh
    
    if verbose:
        gFrm.mask(whiteBFrm).show()
    
    ## since frames are immutable, but arrays are not, we will build the new array dynamically, then turn it into a frame
    horiArr = gFrm.zero().array.astype("bool")
    vertArr = gFrm.zero().array.astype("bool")

    for bPt in whiteBFrm:
        if bPt.value:
            horiArr[bPt.y, bPt.x] = check_vicinity_of_point(bPt, gFrm, 
                                                direction = 'horizontal',
                                                radius = width_of_line / 2, 
                                                int_diff_thresh = int_diff_thresh)
            
            vertArr[bPt.y, bPt.x] = check_vicinity_of_point(bPt, gFrm, 
                                                direction = 'vertical',
                                                radius = width_of_line / 2, 
                                                int_diff_thresh = int_diff_thresh)
    
    horiBFrm = BoolFrame(horiArr)
    vertBFrm = BoolFrame(vertArr)
   
    if verbose:
        gFrm.mask(horiBFrm | vertBFrm).show()
    
    return horiBFrm, vertBFrm






### new strategy: create 'rolled' frames, 
    # which are gFrm but shifted so that a given pixel in a rolled frame corresponds to a neighbour 
    # of the pixel in the original frame

axis = 0
shift = int(COURT_LINE_WIDTH / 2)
int_diff_thresh = 10    
white_thresh = 200

def get_white_lines_fast(gFrm: GreyFrame, width_of_line: int = 5, 
                         white_thresh: int = 200, int_diff_thresh: int = 10,
                         verbose = False):
    shift = round(width_of_line / 2)
    
    upFrm = gFrm.roll(how = 'up', shift = shift)
    dnFrm = gFrm.roll(how = 'down', shift = shift)
    lfFrm = gFrm.roll(how = 'left', shift = shift)
    rtFrm = gFrm.roll(how = 'right', shift = shift)
 
    whiteMask = (gFrm > white_thresh)
    horiMask = (gFrm - upFrm > int_diff_thresh) & (gFrm - dnFrm > int_diff_thresh)
    vertMask = (gFrm - lfFrm > int_diff_thresh) & (gFrm - rtFrm > int_diff_thresh)
    
    line_candidateMask = whiteMask & (horiMask | vertMask)
    
    if verbose:
        gFrm.mask(whiteMask).show()
        gFrm.mask(horiMask).show()
        gFrm.mask(vertMask).show()
        gFrm.mask(line_candidateMask)
    
    return line_candidateMask
    
hmask, vmask = get_white_lines(gFrm, width_of_line = COURT_LINE_WIDTH, white_thresh = WHITE_PIXEL_THRESHOLD,
                        int_diff_thresh = INTENSITY_DIFFERENCE_THRESHOLD, verbose = False)
mask1 = hmask | vmask

mask2 = get_white_lines_fast(gFrm, width_of_line = COURT_LINE_WIDTH, white_thresh = WHITE_PIXEL_THRESHOLD,
                             int_diff_thresh = INTENSITY_DIFFERENCE_THRESHOLD, verbose = False)

mask1.array == mask2.array


