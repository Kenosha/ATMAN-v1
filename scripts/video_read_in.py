# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:43:22 2021

@author: gianl
"""

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.utils as atu
import ATMANpy.cv_funcs as acv

from ATMANpy.types import Frame, BGRFrame, GreyFrame, BoolFrame, Point, Line, Lines


# misc
import os
import shutil
import math
import datetime
# plots
import matplotlib.pyplot as plt
# image operation
import cv2 as cv
import numpy as np


##### VIDEO FILE READ IN ####
video_id = 'pqxAmUWIoKc'
video_full_name = video_id + '.mp4'

videoDict = atu.get_folder_struct('video_files')


videoPath = videoDict[video_full_name]

cap = cv.VideoCapture(videoPath)
############################

#### GET ARBITRARY FRAME ####
FRAME_NUM = 11500
gFrm = GreyFrame.from_video(cap, FRAME_NUM)
gFrm.show()

cFrm = BGRFrame.from_video(cap, FRAME_NUM)
cFrm.show()
############################# 


##### FIND LINES #######


#### 1) find best court line width
aPt = Point((400, 802))
pointFrm = gFrm.to_BGR().put(aPt, radius = 4, thickness = 1)  
pointFrm.show()
### radius 4 looks good... so COURT_LINE_WIDTH is 8!

#### 2) get white line candidates 
from ATMANpy.line_funcs import get_white_lines

WHITE_PIXEL_THRESHOLD = 150 #this depends heavily on brightness of picture... lines are white, but they might not appear at 255 intensity.
COURT_LINE_WIDTH = 8 ### according to own rough calcs at 1920 x 1080, and also according to vinyes (2018), court line width is equal to 5 pixels
INTENSITY_DIFFERENCE_THRESHOLD = 30

horiBFrm, vertBFrm = get_white_lines(gFrm, 
                                     width_of_line = COURT_LINE_WIDTH,
                                     int_diff_thresh = INTENSITY_DIFFERENCE_THRESHOLD,
                                     white_thresh = WHITE_PIXEL_THRESHOLD)
allBFrm = horiBFrm | vertBFrm
        
#### 3) Apply Probabilistic Hough
houghLines = allBFrm.prob_hough(rho = 3, theta = np.pi/180, min_votes = 500,
                                minLineLength = COURT_LINE_WIDTH + 1, maxLineGap = 1)
gFrm.zero().put(houghLines).show()


#### 4) Post-Process Hough Lines
from ATMANpy.line_funcs import split_court_line_candidates, sort_court_line_candidates

l_v_Lines, r_v_Lines, h_Lines = split_court_line_candidates(houghLines, 
                                                            gFrm.cols / 3,
                                                            2 * gFrm.cols / 3)                   
l_v_Lines, r_v_Lines, h_Lines = sort_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines)                   

gFrm.zero().put(l_v_Lines).show()
gFrm.zero().put(r_v_Lines).show()
gFrm.zero().put(h_Lines).show()

from ATMANpy.line_funcs import merge_court_line_candidates, filter_court_line_candidates

threshold = 2 * COURT_LINE_WIDTH # instantiate parameters as in appendix C vinyes:
l_v_Lines, r_v_Lines, h_Lines = merge_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                            threshold = threshold)   

margin = 40
l_v_Lines, r_v_Lines, h_Lines = filter_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                             margin = margin)   


gFrm.zero().put(l_v_Lines).show()
gFrm.zero().put(r_v_Lines).show()
gFrm.zero().put(h_Lines).show()





