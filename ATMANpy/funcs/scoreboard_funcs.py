# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 23:17:50 2022

@author: gianl
"""


from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line, Lines, Region

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
                         # merge_lines = True, min_line_length_after_merge = 50
                         ):
    
    myFrm = gFrm.crop(scoreROI)
    blur_myFrm = myFrm.gaussian_blur((3, 3))
    
    ### apply sobel gradients to find edges, taking into account the normalisation
    # (sobelFrm fits into 0 to 255, by construction!!)
    sobelFrm, normalizer = blur_myFrm.sobel_y()
    horiBool = sobelFrm > sobel_threshold / normalizer
    
    ### create hough lines
    horiLns = horiBool.prob_hough(minLineLength = 5, maxLineGap = 5)
    
    ### keep only lines which are parallel to edge!
    straightLns = []
    for ln in horiLns:
        if ln.slope() is not None:
            if -slope_threshold < ln.slope() < slope_threshold:
                straightLns.append(ln)
    
    # adjust for score region cropping
    tlPt = scoreROI.to_points()[0]
    straightLns = Lines([ln.right(tlPt.x).down(tlPt.y) for ln in straightLns])
    
    # ### merge horizontal lines
    # if merge_lines:
    #     straightLns = straightLns.sort('left_point', 't2b', 'l2r').merge(merge_horizontal_lines)
        
    #     straightLns = [l for l in straightLns if l.length() > min_line_length_after_merge]
    
    return straightLns

def filter_horizontal_lines(straightLns, outlier_percentile = 0.8,
                            min_lines_for_filtering = 4):
    ### idea: throw away all lines whose left points are 
        # to the right of outlier_percentile the right points ...
        # of all right points, seen from left-to-right
        # and vice versa, ("whose right points are to the left...")
    rightLns = straightLns.sort('right_point', 'l2r', 't2b')
    leftLns = straightLns.sort('left_point', 'r2l', 't2b')
    
    critical_value = round(outlier_percentile * len(straightLns))
    crit_rightLn = rightLns[critical_value]
    crit_leftLn = leftLns[critical_value]
    
    rightmost_acceptable = crit_rightLn.rightPt.x
    leftmost_acceptable = crit_leftLn.leftPt.x
    
    is_outlier_func = lambda someLn: someLn.is_strictly_right(rightmost_acceptable) or someLn.is_strictly_left(leftmost_acceptable)
    
    filterLns = Lines([someLn for someLn in straightLns if not is_outlier_func(someLn)], name = straightLns.name)
    return filterLns

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
    return maskReg

def find_scoreboard(gFrm, 
                    sobel_threshold = 30, slope_threshold = 0.0001,
                    outlier_percentile = 0.8, min_lines_for_filtering = 4,
                    verbose = False):
    
    scoreROI = construct_scoreboard_prior(gFrm.cols, gFrm.rows)
    
    scoreLns = get_horizontal_lines(gFrm, scoreROI, 
                                    sobel_threshold = sobel_threshold, 
                                    slope_threshold = slope_threshold,
                                    # merge_lines = merge_lines,
                                    # min_line_length_after_merge = min_line_length_after_merge
                                    )
    
    if len(scoreLns) >= min_lines_for_filtering:
        filterLns = filter_horizontal_lines(scoreLns, outlier_percentile = outlier_percentile)
    else:
        filterLns = scoreLns

    if len(scoreLns) >= 2:
        scoreBoard = lines_to_reg(filterLns)
    else:
        return None

    if verbose: gFrm.to_BGR().put(scoreBoard, thickness = 2).show()
        
    return scoreBoard
