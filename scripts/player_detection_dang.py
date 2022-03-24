# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:28:50 2022

@author: gianl
"""


import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.utils as atu
from ATMANpy.Points import Point, SafePoint
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

# show_video(cap, fps_counter = True, start_frame = 11500, stop_frame = 11900)

############################

from ATMANpy.cv_funcs import show_video

#### GET ARBITRARY FRAME ####
FRAME_NUM = 11800
gFrm = GreyFrame.from_video(cap, FRAME_NUM)
gFrm.show()

cFrm = BGRFrame.from_video(cap, FRAME_NUM)
cFrm.show()
############################# 

#### FIND COURT LINES AND HOMOGRAPHY

### PARAMETERS ###
WHITE_PIXEL_THRESHOLD = 150 #this depends heavily on brightness of picture... lines are white, but they might not appear at 255 intensity.
COURT_LINE_WIDTH = 8 ### according to own rough calcs at 1920 x 1080, and also according to vinyes (2018), court line width is equal to 5 pixels
INTENSITY_DIFFERENCE_THRESHOLD = 30

MERGE_THRESHOLD = 2 * COURT_LINE_WIDTH 
MARGIN_TO_CUT_HORIZONTAL = 40

SCHEMA_SIZE = 4 
SCHEMA_MARGIN = 40

NET_FACTOR = 0.9
NET_POINTS_TO_SEARCH = 100
#####

from ATMANpy.line_funcs import get_white_lines

horiBFrm, vertBFrm = get_white_lines(gFrm, 
                                     width_of_line = COURT_LINE_WIDTH,
                                     int_diff_thresh = INTENSITY_DIFFERENCE_THRESHOLD,
                                     white_thresh = WHITE_PIXEL_THRESHOLD)
allBFrm = horiBFrm | vertBFrm
houghLines = allBFrm.prob_hough(rho = 3, theta = np.pi/180, min_votes = 500, 
                                minLineLength = COURT_LINE_WIDTH + 1, maxLineGap = 1)

from ATMANpy.line_funcs import split_court_line_candidates, sort_court_line_candidates
from ATMANpy.line_funcs import merge_court_line_candidates, filter_court_line_candidates

l_v_Lines, r_v_Lines, h_Lines = split_court_line_candidates(houghLines, 
                                                            gFrm.cols / 3, 2 * gFrm.cols / 3) 
                  
l_v_Lines, r_v_Lines, h_Lines = sort_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines)    
               
l_v_Lines, r_v_Lines, h_Lines = merge_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                            threshold = MERGE_THRESHOLD)   

l_v_Lines, r_v_Lines, h_Lines = filter_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                             margin = MARGIN_TO_CUT_HORIZONTAL)  
 
from ATMANpy.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
courtFrm = fit_frame_to_lines(schemaLines)

from ATMANpy.line_funcs import get_homography 

v_Lines = l_v_Lines.append(r_v_Lines).sort('left_point', 'l2r', 't2b')
H = get_homography(v_Lines, h_Lines, allBFrm, verbose = False, size = SCHEMA_SIZE, margin = SCHEMA_MARGIN)
courtLines = schemaLines.transform('apply_homography', H).transform('round_points')

from ATMANpy.net_funcs import find_net_points, remove_outlier_net_points
relPoints = find_net_points(courtLines, gFrm, 
                            factor = NET_FACTOR, points_to_search = NET_POINTS_TO_SEARCH)
relPoints = remove_outlier_net_points(relPoints)

from ATMANpy.net_funcs import cubic_regress, make_net_cord_lines
b_0, b_1 = cubic_regress(relPoints)
netLines = make_net_cord_lines(courtLines, b_0, b_1)

gFrm.to_BGR().put(courtLines, thickness = 3).put(netLines, color = (255, 0, 0), thickness = 3).show()
#################################

#### SEPARATE BACKGROUND AND FOREGROUND ####

ALPHA = 4.5 #maximum acceptable number of standard deviations a value can be away from
            #the average color until it is assumed to be foreground

### i) find average court color and deviation

def within_court(courtLines):
    # court mask - what is inside the court?
    
    # bounding lines of court
    l, r = courtLines['Top Doubles Deuce Line'], courtLines['Top Doubles Ad Line']
    t, b = courtLines['Top Baseline'], courtLines['Bottom Baseline']
    
    # bool frame - right of left line, left of right line, etc...
    courtboolFrame = gFrm.right_of(l) & gFrm.left_of(r) & gFrm.below(t) & gFrm.above(b)
    return courtboolFrame

courtboolFrame = within_court(courtLines)

# non_white elements
nonwhiteboolFrame = gFrm <= WHITE_PIXEL_THRESHOLD

boolFrame_in = nonwhiteboolFrame & courtboolFrame
cFrm.mask(boolFrame_in).show()

def get_avg_std_bgr(cFrm, boolFrame):
    avg_bgr = cFrm.mask(boolFrame).sum((0, 1)) / boolFrame.sum((0, 1))
    
    devFrm = (cFrm - avg_bgr) ** 2
    var_bgr = devFrm.mask(boolFrame).sum((0, 1)) / boolFrame.sum((0, 1))
    std_bgr = var_bgr ** 0.5
    
    return avg_bgr, std_bgr

avg_bgr_in, std_bgr_in = get_avg_std_bgr(cFrm, boolFrame_in)

### ii) find average court color outside of court
# -> for this, we'll use average (or median) color just outside of the court lines,
# this way, we will not include weirdness.
# idea: construct a schematic court that is slightly larger than our used schema

FACTOR_SCALE = 1.2
FACTOR_MARGIN = 0.3

schemaLines_large = CourtLines.make(size = FACTOR_SCALE * SCHEMA_SIZE, 
                                    margin = FACTOR_MARGIN * SCHEMA_MARGIN,
                                    force_dimensions = True)
courtLines_large = schemaLines_large.transform('apply_homography', H).transform('round_points')

fit_frame_to_lines(schemaLines_large).put(schemaLines).show()

gFrm.to_BGR().put(courtLines_large, thickness = 3).show()

courtboolFrame_large = within_court(courtLines_large)

boolFrame_out = courtboolFrame_large & ~courtboolFrame & nonwhiteboolFrame
cFrm.mask(boolFrame_out).show()

avg_bgr_out, std_bgr_out = get_avg_std_bgr(cFrm, boolFrame_out)

## ii) compare pixels with average colour

dev_in = abs(cFrm - avg_bgr_in) / std_bgr_in
dev_out = abs(cFrm - avg_bgr_out) / std_bgr_out

dev = dev_in.mask(boolFrame_in) + dev_in.mask(~ boolFrame_in)

foreground_mask = (dev > ALPHA).any(2)

cFrm.mask(foreground_mask).show()

import cv2 as cv
contours, hierarchy = cv.findContours(foreground_mask.array.astype('uint8'), 
                           cv.RETR_EXTERNAL, 
                           cv.CHAIN_APPROX_SIMPLE)

contourArr = cFrm.array.copy()
contourFrm = BGRFrame(cv.drawContours(contourArr, contours, -1, (0, 255, 0), 3))

contourFrm.show()

