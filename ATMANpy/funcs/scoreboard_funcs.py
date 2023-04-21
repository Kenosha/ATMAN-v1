# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 23:17:50 2022

@author: gianl
"""


from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line, Lines, Region

from ATMANpy.funcs.line_funcs import merge_horizontal_lines
from functools import partial



def construct_scoreboard_prior(cols, rows):
    x_share = 0.33
    y_share = 0.2

    x_offset = int(x_share * cols)
    y_offset = int(y_share * rows)
    
    bot_left = Point2D((5, rows))
    top_right = bot_left.right(x_offset).up(y_offset)
    
    return Region.from_points(bot_left, top_right)
    

def merge_horizontal_lines(aLn, bLn):

    THRESHOLD = 3 ### want to merge only lines which are very close!    

    if aLn.left_point().is_left(bLn.left_point()):
        if aLn.right_point().is_right(bLn.right_point()):
            return Lines([aLn])  ### bLn is redundant!
        else:
            lLn, rLn = aLn, bLn
    else:
        if aLn.right_point().is_right(bLn.right_point()):
            lLn, rLn = bLn, aLn
        else:
            return Lines([bLn])  ### aLn is redundant!

    d = abs(lLn.right_point().y - rLn.left_point().y)
    
    if d < THRESHOLD:
        newLine = Line(lLn.left_point(), rLn.right_point())
        return Lines([newLine])
    else:
        return Lines([aLn, bLn])


def get_horizontal_lines(gFrm, scoreROI,
                         sobel_threshold = 30, slope_threshold = 0.0001,
                         merge_lines = True, min_line_length_after_merge = 50):
    
    myFrm = gFrm.crop(scoreROI)
    blur_myFrm = myFrm.gaussian_blur((3, 3))
    
    # (gFrm.sobel_x() > SOBEL_THRESHOLD).show()
    # (gFrm.sobel_y() > SOBEL_THRESHOLD).show()
    
    # vertBool = blur_myFrm.sobel_x() > SOBEL_THRESHOLD
    horiBool = blur_myFrm.sobel_y() > sobel_threshold
    
    # vertBool.show()
    # horiBool.show()
    
    # vertLns = vertBool.prob_hough(minLineLength = 5, maxLineGap = 10)
    horiLns = horiBool.prob_hough(minLineLength = 5, maxLineGap = 5)
    
    # myFrm.to_BGR().put(vertLns).show() ### strange...
    # myFrm.to_BGR().put(lns2).show() ### ok! 
    
    straightLns = [ln for ln in horiLns if -slope_threshold < ln.slope() < slope_threshold]
    # myFrm.to_BGR().put(straightLns).show()
    
    # adjust for score region cropping
    tlPt = scoreROI.to_points()[0]
    straightLns = Lines([ln.right(tlPt.x).down(tlPt.y) for ln in straightLns])
    
    # gFrm.to_BGR().put(straightLns).show()
    
    ### merge horizontal lines
    if merge_lines:
        straightLns = straightLns.sort('left_point', 't2b', 'l2r').merge(merge_horizontal_lines)
        
        straightLns = [l for l in straightLns if l.length() > min_line_length_after_merge]
    
    # gFrm.to_BGR().put(straightLns).show()
    
    return straightLns



def lines_to_reg(lns):
    left_xs = [ln.leftPt.x for ln in lns]
    left_ys = [ln.leftPt.y for ln in lns]
    right_xs = [ln.rightPt.x for ln in lns]
    right_ys = [ln.rightPt.y for ln in lns]
    
    min_x, min_y = min(left_xs), min(left_ys)
    leftPt = Point((min_x, min_y))
    
    max_x, max_y = max(right_xs), max(right_ys)
    rightPt = Point((max_x, max_y))
    
    maskReg = Region.from_points(leftPt, rightPt)
    return maskReg

def find_scoreboard(gFrm, 
                    sobel_threshold = 30, slope_threshold = 0.0001,
                    merge_lines = True, min_line_length_after_merge = 50, 
                    min_lines = 2, 
                    verbose = False):
    
    scoreROI = construct_scoreboard_prior(gFrm.cols, gFrm.rows)
    
    scoreLns = get_horizontal_lines(gFrm, scoreROI, 
                                    sobel_threshold = sobel_threshold, 
                                    slope_threshold = slope_threshold,
                                    merge_lines = merge_lines,
                                    min_line_length_after_merge = min_line_length_after_merge)

    if len(scoreLns) >= min_lines:
        scoreReg = lines_to_reg(scoreLns)
    else:
        return None

    if verbose: gFrm.to_BGR().put(scoreReg, thickness = 2).show()
        
    return scoreReg