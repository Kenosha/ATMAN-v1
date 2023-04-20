# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:45:04 2022

@author: gianl
"""

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.funcs.util_funcs as atu
from ATMANpy.types.Points import Point2D, SafePoint2D
from ATMANpy.types.Lines import Line, Lines, CourtLines, Region
from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame
from ATMANpy.types.Contours import Contour

# image operation
import cv2 as cv
import numpy as np

import pandas as pd

import os
from random import randrange

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################


########## READ-IN PASSAGES OF PLAY ##########
evtDict = atu.get_folder_struct('ball_event_hmm')

try:
    readPath = evtDict[VIDEO_ID]
except KeyError:
    print('Data not found.')
    
playseqPath = os.path.join(readPath, 'play_sequences')    
start_stop_timestamp = '2022-06-11-02-42-30'
file = f'play_start_stop_{start_stop_timestamp}.csv'

seqFrm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)


###############################################
        
######## GET SCHEMA ########
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)

######## 

sample_frames = []
for idx, c in seqFrm.iterrows():
    start, stop = c
    randFrm = randrange(start, stop)
    sample_frames.append(randFrm)



#### 
frmIdx = 2
myFrame = sample_frames[frmIdx]

cFrm = BGRFrame.from_video(cap, myFrame)
gFrm = GreyFrame.from_video(cap, myFrame)

def construct_scoreboard_prior(cols, rows):
    x_share = 0.33
    y_share = 0.2

    x_offset = int(x_share * cols)
    y_offset = int(y_share * rows)
    
    bot_left = Point2D((5, rows))
    top_right = bot_left.right(x_offset).up(y_offset)
    
    return Region.from_points(bot_left, top_right)
    
scoreReg = construct_scoreboard_prior(cFrm.cols, cFrm.rows)

gFrm.put(scoreReg).show()

gFrm.sobel()

gFrm.sobel().show()
gFrm.sobel_x().show()
gFrm.sobel_y().show()

SOBEL_THRESHOLD = 30
myFrm = gFrm.crop(scoreReg)

blur_myFrm = myFrm.gaussian_blur((3, 3))

# (gFrm.sobel_x() > SOBEL_THRESHOLD).show()
# (gFrm.sobel_y() > SOBEL_THRESHOLD).show()

vertBool = blur_myFrm.sobel_x() > SOBEL_THRESHOLD
horiBool = blur_myFrm.sobel_y() > SOBEL_THRESHOLD

vertBool.show()
horiBool.show()

lns = vertBool.prob_hough(minLineLength = 0, maxLineGap = 100)
lns2 = horiBool.prob_hough(minLineLength = 0, maxLineGap = 100)

myFrm.to_BGR().put(lns).show()
myFrm.to_BGR().put(lns2).show()

### for the moment, let's take only the horizontal lines.

myLn = lns2[19]
myLn.slope()

### virtually, we could take everything. 
# for all lines with near-0 slope, take lowest and highest x and y. This is the box.

### but what if there is interference? a leg sticking into the scoreboard?

from ATMANpy.funcs.cv_funcs import show_video
show_video(cap, frame_counter = True, start_frame = 11000, ms_between_frames = 40)

weird_frames = [11300, 11950, 12200, 12300, 14100]

myFrm = weird_frames[0]
cFrm = BGRFrame.from_video(cap, myFrm)
cFrm.show()

gFrm = GreyFrame.from_video(cap, myFrm)

SOBEL_THRESHOLD = 30
myFrm = gFrm.crop(scoreReg)

blur_myFrm = myFrm.gaussian_blur((3, 3))

# (gFrm.sobel_x() > SOBEL_THRESHOLD).show()
# (gFrm.sobel_y() > SOBEL_THRESHOLD).show()

vertBool = blur_myFrm.sobel_x() > SOBEL_THRESHOLD
horiBool = blur_myFrm.sobel_y() > SOBEL_THRESHOLD

vertBool.show()
horiBool.show()

lns = vertBool.prob_hough(minLineLength = 0, maxLineGap = 100)
lns2 = horiBool.prob_hough(minLineLength = 0, maxLineGap = 100)

myFrm.to_BGR().put(lns).show()
myFrm.to_BGR().put(lns2).show()

SLOPE_THRESHOLD = 0.05

lns3 = [ln for ln in lns2 if -SLOPE_THRESHOLD < ln.slope() < SLOPE_THRESHOLD]
myFrm.to_BGR().put(lns3).show()

def get_horizontal_lines(gFrm, scoreReg):
    
    SOBEL_THRESHOLD = 30
    SLOPE_THRESHOLD = 0.0001
    
    myFrm = gFrm.crop(scoreReg)
    blur_myFrm = myFrm.gaussian_blur((3, 3))
    
    # (gFrm.sobel_x() > SOBEL_THRESHOLD).show()
    # (gFrm.sobel_y() > SOBEL_THRESHOLD).show()
    
    vertBool = blur_myFrm.sobel_x() > SOBEL_THRESHOLD
    horiBool = blur_myFrm.sobel_y() > SOBEL_THRESHOLD
    
    # vertBool.show()
    # horiBool.show()
    
    vertLns = vertBool.prob_hough(minLineLength = 5, maxLineGap = 10)
    horiLns = horiBool.prob_hough(minLineLength = 5, maxLineGap = 10)
    
    # myFrm.to_BGR().put(vertLns).show() ### strange...
    # myFrm.to_BGR().put(lns2).show() ### ok! 
    
    straightLns = [ln for ln in horiLns if -SLOPE_THRESHOLD < ln.slope() < SLOPE_THRESHOLD]
    myFrm.to_BGR().put(straightLns).show()
    
    # adjust for score region cropping
    tlPt = scoreReg.to_points()[0]
    straightLns = [ln.right(tlPt.x).down(tlPt.y) for ln in straightLns]
    
    return straightLns



myFrm = weird_frames[2]
gFrm = GreyFrame.from_video(cap, myFrm)
gFrm.show()
lns = get_horizontal_lines(gFrm, scoreReg)

gFrm.to_BGR().put(lns).show()


def lines_to_reg(lns):
    left_xs = [ln.leftPt.x for ln in lns]
    left_ys = [ln.leftPt.y for ln in lns]
    right_xs = [ln.rightPt.x for ln in lns]
    right_ys = [ln.rightPt.y for ln in lns]
    
    min_x, min_y = min(left_xs), min(left_ys)
    leftPt = Point2D((min_x, min_y))
    
    max_x, max_y = max(right_xs), max(right_ys)
    rightPt = Point2D((max_x, max_y))
    
    maskReg = Region.from_points(leftPt, rightPt)
    

def find_scoreboard(gFrm, verbose = False):
    
    scoreROI = construct_scoreboard_prior(cFrm.cols, cFrm.rows)
    
    scoreLns = get_horizontal_lines(gFrm, scoreROI)

    scoreReg = lines_to_reg(scoreLns)

    if verbose: gFrm.to_BGR().put(scoreReg, thickness = 2).show()
        
    return scoreReg



