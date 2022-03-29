# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:57:49 2022

@author: gianl
"""

import numpy as np

from ATMANpy.Points import Point, SafePoint
from ATMANpy.Lines import Line, Lines, CourtLines, Region
from ATMANpy.Frames import Frame, BGRFrame, GreyFrame, BoolFrame


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

def find_homography_225_only(gFrm, verbose = False):
    from ATMANpy.line_funcs import get_white_lines_fast
    
    allBFrm = get_white_lines_fast(gFrm, 
                                   width_of_line = COURT_LINE_WIDTH,
                                   int_diff_thresh = INTENSITY_DIFFERENCE_THRESHOLD,
                                   white_thresh = WHITE_PIXEL_THRESHOLD)
    
    houghLines = allBFrm.prob_hough(rho = 3, theta = np.pi/180, min_votes = 500, 
                                    minLineLength = COURT_LINE_WIDTH + 1, maxLineGap = 1)
    
    if len(houghLines) < 9:
        return None
    
    if verbose: gFrm.zero().to_BGR().put(houghLines).show()
    
    from ATMANpy.line_funcs import split_court_line_candidates, sort_court_line_candidates
    from ATMANpy.line_funcs import merge_court_line_candidates, filter_court_line_candidates
    
    l_v_Lines, r_v_Lines, h_Lines = split_court_line_candidates(houghLines, 
                                                                gFrm.cols / 3, 2 * gFrm.cols / 3) 
                      
    if len(l_v_Lines) < 2 or len(r_v_Lines) < 2:
        return None
    
    l_v_Lines, r_v_Lines, h_Lines = sort_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines)    
    
    l_v_Lines, r_v_Lines, h_Lines = merge_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                                threshold = MERGE_THRESHOLD)   
    
    l_v_Lines, r_v_Lines, h_Lines = filter_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                                 margin = MARGIN_TO_CUT_HORIZONTAL)  
    if verbose: 
        gFrm.zero().to_BGR().put(r_v_Lines).show()
        gFrm.zero().to_BGR().put(l_v_Lines).show()
        gFrm.zero().to_BGR().put(h_Lines).show()
    
    if len(l_v_Lines) == 2 and len(r_v_Lines) == 2 and len(h_Lines) == 5:
        # print('Fast Homography Algorithm...')
        from ATMANpy.line_funcs import get_homography_225
        H, config = get_homography_225(l_v_Lines, r_v_Lines, h_Lines, allBFrm, verbose = verbose, size = SCHEMA_SIZE, margin = SCHEMA_MARGIN)
        return H, config
    else:
        return None
    
    


def find_homography(gFrm, verbose = False):
    
    from ATMANpy.line_funcs import get_white_lines_fast
    
    allBFrm = get_white_lines_fast(gFrm, 
                                   width_of_line = COURT_LINE_WIDTH,
                                   int_diff_thresh = INTENSITY_DIFFERENCE_THRESHOLD,
                                   white_thresh = WHITE_PIXEL_THRESHOLD)
    
    houghLines = allBFrm.prob_hough(rho = 3, theta = np.pi/180, min_votes = 500, 
                                    minLineLength = COURT_LINE_WIDTH + 1, maxLineGap = 1)
    
    if len(houghLines) < 4:
        return None
    
    if verbose: gFrm.zero().to_BGR().put(houghLines).show()
    
    from ATMANpy.line_funcs import split_court_line_candidates, sort_court_line_candidates
    from ATMANpy.line_funcs import merge_court_line_candidates, filter_court_line_candidates
    
    l_v_Lines, r_v_Lines, h_Lines = split_court_line_candidates(houghLines, 
                                                                gFrm.cols / 3, 2 * gFrm.cols / 3) 
                      
    if len(l_v_Lines) == 0 and len(r_v_Lines) == 0:
        return None
    
    l_v_Lines, r_v_Lines, h_Lines = sort_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines)    
    
    l_v_Lines, r_v_Lines, h_Lines = merge_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                                threshold = MERGE_THRESHOLD)   
    
    l_v_Lines, r_v_Lines, h_Lines = filter_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                                 margin = MARGIN_TO_CUT_HORIZONTAL)  
     
    if verbose: 
        gFrm.zero().to_BGR().put(r_v_Lines).show()
        gFrm.zero().to_BGR().put(l_v_Lines).show()
        gFrm.zero().to_BGR().put(h_Lines).show()
    
    # from ATMANpy.line_funcs import fit_frame_to_lines
    # courtFrm = fit_frame_to_lines(schemaLines)
    if len(l_v_Lines) == 2 and len(r_v_Lines) == 2 and len(h_Lines) == 5:
        print('Fast Homography Algorithm...')
        from ATMANpy.line_funcs import get_homography_225
        H, config = get_homography_225(l_v_Lines, r_v_Lines, h_Lines, allBFrm, verbose = verbose, size = SCHEMA_SIZE, margin = SCHEMA_MARGIN)

    else:
        print('General Homography Algorithm...')
        from ATMANpy.line_funcs import get_homography 
        v_Lines = l_v_Lines.append(r_v_Lines).sort('left_point', 'l2r', 't2b')
        H, config = get_homography(v_Lines, h_Lines, allBFrm, verbose = verbose, size = SCHEMA_SIZE, margin = SCHEMA_MARGIN)

    return H, config

def find_courtlines(H, schemaLines):
    courtLines = schemaLines.transform('apply_homography', H).transform('round_points')
    return courtLines
    
def find_netlines(gFrm, courtLines):

    from ATMANpy.net_funcs import find_net_points, remove_outlier_net_points
    relPoints = find_net_points(courtLines, gFrm, 
                                factor = NET_FACTOR, points_to_search = NET_POINTS_TO_SEARCH)
    relPoints = remove_outlier_net_points(relPoints)
    
    from ATMANpy.net_funcs import cubic_regress, make_net_cord_lines
    b_0, b_1 = cubic_regress(relPoints)
    netLines = make_net_cord_lines(courtLines, b_0, b_1)

    return netLines

#####################
#####################

    
from ATMANpy.line_funcs import linear_regress, calculate_homography, find_brightest_neighbour

def update_homography(gFrm, H, config, white_pixel_threshold = WHITE_PIXEL_THRESHOLD,
                      points_on_line = 100, acceptable_failure_rate = 0.5,
                      verbose = False):
    """
    The 'Court Tracking' step in Vinyes (2018).
    Idea: Using the best homography so far, and the best configuration that produced it,
    we get the four projected court lines that correspond to the config, 
    then, for each of these court lines,
        we try to track them by finding the brightest direct neighbour pixels
            (if we find too few bright neighbours, tracking fails!)
        and construct a least squares new line.
    Then intersect these four new lines, get the four updated corner points,
    and recalculate the homography using the correspondence to the four schema config points.
    """
    failure_threshold = acceptable_failure_rate * points_on_line
    
    schemaRegion = Region.from_points(*config)
    courtRegion = schemaRegion.transform('apply_homography', H).transform('round_points')

    newLines = []
    for aLn in courtRegion:
        
        coordList = []
        lPts = aLn.subdivide(points_on_line)
        
        for aPt in lPts:
            maxPt, maxVal = find_brightest_neighbour(gFrm, aPt)
            if maxVal > white_pixel_threshold:
                coordList.append(maxPt.coords)
        
        if len(coordList) < failure_threshold:
            ### failure to track!!!
            # print(f'Failed to track: {aLn.name}')
            # gFrm.to_BGR().put(courtRegion, thickness = 2).show()
            # print(lPts)
            print('Lost Tracking...')
            return None
        else:
            b0, b1 = linear_regress(coordList)
            oldPt_1, oldPt_2 = coordList[0],coordList[-1]
            
            newPt_1 = Point((oldPt_1[0], b0 + b1 * oldPt_1[0]))
            newPt_2 = Point((oldPt_2[0], b0 + b1 * oldPt_2[0]))
            
            newLn = Line(newPt_1, newPt_2, name = aLn.name)

            newLines.append(newLn)
    
    newRegion = Region(newLines).join_corners()

    H = calculate_homography(config, newRegion.to_points())
    
    return H, config









