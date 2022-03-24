# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:28:50 2022

Replication of Lai and Hsieh (2009)

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

# show_video(cap, fps_counter = True, start_frame = 11500, stop_frame = 15900)

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
SCHEMA_MARGIN = 150

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

#### Find ROI ####

# as opposed to the paper, we'll define the ROI on the court schema, and 
# translate it to projection coordinates later

# courtFrm.show()

WIDTH_FACTOR = 2 ### how much of the doubles alley width does a player maximally next to court
DEPTH_FACTOR = 0.2 ### how much of side length does a player maximally stand behind baseline
ADD_DEPTH_FACTOR = 1.25


### top side depth adjustment - player figure is tall, so looks to be outside of ROI.
# we could adjust this with the camera matrix, but let's do something eyeballed here, for the moment.

from ATMANpy.player_funcs import get_initial_ROIs, find_player

ROI_lower, ROI_upper = get_initial_ROIs(schemaLines,
                                         width_factor = WIDTH_FACTOR, 
                                         depth_factor = DEPTH_FACTOR, 
                                         add_depth_factor_top = ADD_DEPTH_FACTOR)
courtFrm.to_BGR().put(ROI_upper, color = (0, 0, 255), thickness = 2).put(ROI_lower, color = (0, 255, 0), thickness = 2).show()

projROI_lower = ROI_lower.transform('apply_homography', H).transform('round_points')
projROI_upper = ROI_upper.transform('apply_homography', H).transform('round_points')
cFrm.put(projROI_upper, color = (0, 0, 255), thickness = 2).put(projROI_lower, color = (0, 255, 0), thickness = 2).show()

COURT_DILATION = 1.5
NET_DILATION = 2.5
ROI_DILATION = 1
    
playerRegion_bottom = find_player(cFrm, projROI_lower, courtLines, netLines, 
                               line_width = COURT_LINE_WIDTH,
                               court_dilation_factor = COURT_DILATION, 
                               net_dilation_factor = NET_DILATION, 
                               roi_dilation_factor = ROI_DILATION, 
                               which = 'Bottom',
                               verbose = True)

playerRegion_top = find_player(cFrm, projROI_upper, courtLines, netLines, 
                               line_width = COURT_LINE_WIDTH,
                               court_dilation_factor = COURT_DILATION, 
                               net_dilation_factor = NET_DILATION, 
                               roi_dilation_factor = ROI_DILATION, 
                               which = 'Top',
                               verbose = True)

cFrm.put(playerRegion_bottom, thickness = 2).show()
cFrm.put(playerRegion_top, thickness = 2).show()







