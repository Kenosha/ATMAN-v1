# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:43:22 2021

@author: gianl
"""

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.utils as atu
from ATMANpy.Points import Point2D, SafePoint2D
from ATMANpy.Lines import Line, Lines, CourtLines
from ATMANpy.Frames import BGRFrame, GreyFrame

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

####### 5) Court Correspondences

### i) a theoretical model of the court 

from ATMANpy.line_funcs import fit_frame_to_lines
# court display parameters, not important
SIZE = 4 
MARGIN = 10
schemaLines = CourtLines.make(size = SIZE, margin = MARGIN).add_singles_netline()

courtFrm = fit_frame_to_lines(schemaLines)
courtFrm.show()

### ii) create the 11 theoretical 4-point configurations on the court schema so created.
# from ATMANpy.line_funcs import show_configs
# show_configs(courtFrm, court_configs, thickness = 6, radius = 5, color = (0, 0, 255))

### iii) by comparing these configs with all possible 4-line intersections of our hough lines, 
# we can derive a homography. 
# The best homography is selected via goodness of fit, using the comparing the implied court with the white in the image.
from ATMANpy.line_funcs import get_homography 

v_Lines = l_v_Lines.append(r_v_Lines).sort('left_point', 'l2r', 't2b')
H = get_homography(v_Lines, h_Lines, allBFrm, verbose = False)

courtLines = schemaLines.transform('apply_homography', H).transform('round_points')
gFrm.to_BGR().put(courtLines, thickness = 3).show()

###### 6) Find the top of the net #####
FACTOR = 0.9
POINTS_TO_SEARCH = 100

from ATMANpy.net_funcs import find_net_points, remove_outlier_net_points
relPoints = find_net_points(courtLines, gFrm, 
                            factor = FACTOR, points_to_search = POINTS_TO_SEARCH)
relPoints = remove_outlier_net_points(relPoints)

from ATMANpy.net_funcs import cubic_regress, make_net_cord_lines
b_0, b_1 = cubic_regress(relPoints)
netLines = make_net_cord_lines(courtLines, b_0, b_1)


gFrm.to_BGR().put(courtLines, thickness = 3).put(netLines, color = (255, 0, 0), thickness = 3).show()











