# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:57:49 2022

@author: gianl
"""

import numpy as np

from ATMANpy.types.Points import Point2D, SafePoint2D
from ATMANpy.types.Lines import Line, Lines, CourtLines, Region
from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, BoolFrame

import ATMANpy.auxil.constants as const

### PARAMETERS ###
WHITE_PIXEL_THRESHOLD = 150 #this depends heavily on brightness of picture... lines are white, but they might not appear at 255 intensity.
COURT_LINE_WIDTH = 8 ### according to own rough calcs at 1920 x 1080, and also according to vinyes (2018), court line width is equal to 5 pixels
INTENSITY_DIFFERENCE_THRESHOLD = 30

MERGE_THRESHOLD = 2 * COURT_LINE_WIDTH 
MARGIN_TO_CUT_HORIZONTAL = 40

RHO = 3
THETA = np.pi / 180
MIN_VOTES = 500
MIN_LINE_LENGTH = COURT_LINE_WIDTH + 1
MAX_LINE_GAP = 1

NET_FACTOR = 0.9
NET_POINTS_TO_SEARCH = 100
#####

def find_linesets(gFrm,
                  width_of_line = COURT_LINE_WIDTH, 
                  int_diff_thresh = INTENSITY_DIFFERENCE_THRESHOLD,
                  white_thresh = WHITE_PIXEL_THRESHOLD,
                  rho = RHO, theta = THETA, min_votes = MIN_VOTES,
                  minLineLength = MIN_LINE_LENGTH, maxLineGap = MAX_LINE_GAP,
                  merge_threshold = MERGE_THRESHOLD, 
                  horizontal_filter_margin = MARGIN_TO_CUT_HORIZONTAL,
                  vertical_filter_margin = None,
                  verbose = False):
    from ATMANpy.funcs.line_funcs import get_white_lines_fast
    
    allBFrm = get_white_lines_fast(gFrm, 
                                   width_of_line = width_of_line,
                                   int_diff_thresh = int_diff_thresh,
                                   white_thresh = white_thresh)
    
    if verbose: allBFrm.show()
    
    houghLines = allBFrm.prob_hough(rho = rho, theta = theta, min_votes = min_votes, 
                                    minLineLength = minLineLength, maxLineGap = maxLineGap)
    
    if len(houghLines) < 9:
        return None, None, None
    
    if verbose: gFrm.zero().to_BGR().put(houghLines).show()
    
    from ATMANpy.funcs.line_funcs import split_court_line_candidates, sort_court_line_candidates
    from ATMANpy.funcs.line_funcs import merge_court_line_candidates
    from ATMANpy.funcs.line_funcs import filter_horizontal_line_candidates, filter_vertical_line_candidates
    from ATMANpy.funcs.line_funcs import keep_only_central_horizontal_lines
    
    l_v_Lines, r_v_Lines, h_Lines = split_court_line_candidates(houghLines, 
                                                                2 * gFrm.cols / 5, 3 * gFrm.cols / 5) 
    h_Lines = keep_only_central_horizontal_lines(h_Lines, gFrm.cols / 3, 2 * gFrm.cols / 3)  
    
    if len(l_v_Lines) < 2 or len(r_v_Lines) < 2:
        return None, None, None
    
    l_v_Lines, r_v_Lines, h_Lines = sort_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines)    

    l_v_Lines, r_v_Lines, h_Lines = merge_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                                threshold = merge_threshold)   
    
    if horizontal_filter_margin is not None:
        h_Lines = filter_horizontal_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                    margin = horizontal_filter_margin)
    if vertical_filter_margin is not None:
        l_v_Lines, r_v_Lines = filter_vertical_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                               margin = vertical_filter_margin)
        
    if verbose: 
        gFrm.zero().to_BGR().put(r_v_Lines).show()
        gFrm.zero().to_BGR().put(l_v_Lines).show()
        gFrm.zero().to_BGR().put(h_Lines).show()
    
    return l_v_Lines, r_v_Lines, h_Lines

def find_homography_225_only(gFrm,
                             width_of_line = COURT_LINE_WIDTH, 
                             int_diff_thresh = INTENSITY_DIFFERENCE_THRESHOLD,
                             white_thresh = WHITE_PIXEL_THRESHOLD,
                             rho = RHO, theta = THETA, min_votes = MIN_VOTES,
                             minLineLength = MIN_LINE_LENGTH, maxLineGap = MAX_LINE_GAP,
                             merge_threshold = MERGE_THRESHOLD, 
                             horizontal_filter_margin = MARGIN_TO_CUT_HORIZONTAL,
                             vertical_filter_margin = None,
                             verbose = False):
    from ATMANpy.funcs.line_funcs import get_white_lines_fast
    
    allBFrm = get_white_lines_fast(gFrm, 
                                   width_of_line = width_of_line,
                                   int_diff_thresh = int_diff_thresh,
                                   white_thresh = white_thresh)
    
    houghLines = allBFrm.prob_hough(rho = rho, theta = theta, min_votes = min_votes, 
                                    minLineLength = minLineLength, maxLineGap = maxLineGap)
    
    if len(houghLines) < 9:
        return None, None
    
    if verbose: gFrm.zero().to_BGR().put(houghLines).show()
    
    from ATMANpy.funcs.line_funcs import split_court_line_candidates, sort_court_line_candidates
    from ATMANpy.funcs.line_funcs import merge_court_line_candidates
    from ATMANpy.funcs.line_funcs import filter_horizontal_line_candidates, filter_vertical_line_candidates
    from ATMANpy.funcs.line_funcs import keep_only_central_horizontal_lines
    
    l_v_Lines, r_v_Lines, h_Lines = split_court_line_candidates(houghLines, 
                                                                2 * gFrm.cols / 5, 3 * gFrm.cols / 5) 
    h_Lines = keep_only_central_horizontal_lines(h_Lines, gFrm.cols / 3, 2 * gFrm.cols / 3)  
    
    if len(l_v_Lines) < 2 or len(r_v_Lines) < 2:
        return None, None
    
    l_v_Lines, r_v_Lines, h_Lines = sort_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines)    

    l_v_Lines, r_v_Lines, h_Lines = merge_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                                threshold = merge_threshold)   
    
    if horizontal_filter_margin is not None:
        h_Lines = filter_horizontal_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                    margin = horizontal_filter_margin)
    if vertical_filter_margin is not None:
        l_v_Lines, r_v_Lines = filter_vertical_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                               margin = vertical_filter_margin)
    
    if verbose: 
        gFrm.zero().to_BGR().put(r_v_Lines).show()
        gFrm.zero().to_BGR().put(l_v_Lines).show()
        gFrm.zero().to_BGR().put(h_Lines).show()
    
    if len(l_v_Lines) == 2 and len(r_v_Lines) == 2 and len(h_Lines) == 5:
        # print('Fast Homography Algorithm...')
        from ATMANpy.funcs.line_funcs import get_homography_225
        H, config = get_homography_225(l_v_Lines, r_v_Lines, h_Lines, allBFrm, verbose = verbose)
        return H, config
    else:
        return None, None
    
    
def find_homography(gFrm, width_of_line = COURT_LINE_WIDTH, 
                    int_diff_thresh = INTENSITY_DIFFERENCE_THRESHOLD,
                    white_thresh = WHITE_PIXEL_THRESHOLD,
                    rho = RHO, theta = THETA, min_votes = MIN_VOTES,
                    minLineLength = MIN_LINE_LENGTH, maxLineGap = MAX_LINE_GAP,
                    merge_threshold = MERGE_THRESHOLD, 
                    horizontal_filter_margin = MARGIN_TO_CUT_HORIZONTAL,
                    vertical_filter_margin = None,
                    verbose = False):
    from ATMANpy.funcs.line_funcs import get_white_lines_fast
    
    allBFrm = get_white_lines_fast(gFrm, 
                                   width_of_line = width_of_line,
                                   int_diff_thresh = int_diff_thresh,
                                   white_thresh = white_thresh)
    
    houghLines = allBFrm.prob_hough(rho = rho, theta = theta, min_votes = min_votes, 
                                    minLineLength = minLineLength, maxLineGap = maxLineGap)
    
    if len(houghLines) < 4:
        return None, None
    
    if verbose: gFrm.zero().to_BGR().put(houghLines).show()
    
    from ATMANpy.funcs.line_funcs import split_court_line_candidates, sort_court_line_candidates
    from ATMANpy.funcs.line_funcs import merge_court_line_candidates
    from ATMANpy.funcs.line_funcs import filter_horizontal_line_candidates, filter_vertical_line_candidates
    from ATMANpy.funcs.line_funcs import keep_only_central_horizontal_lines
    
    l_v_Lines, r_v_Lines, h_Lines = split_court_line_candidates(houghLines, 
                                                                2 * gFrm.cols / 5, 3 * gFrm.cols / 5) 
    h_Lines = keep_only_central_horizontal_lines(h_Lines, gFrm.cols / 3, 2 * gFrm.cols / 3)                  
    
    if len(l_v_Lines) < 2 or len(r_v_Lines) < 2:
        return None, None
    
    l_v_Lines, r_v_Lines, h_Lines = sort_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines)    
    
    l_v_Lines, r_v_Lines, h_Lines = merge_court_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                                threshold = merge_threshold)   
    
    if horizontal_filter_margin is not None:
        h_Lines = filter_horizontal_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                    margin = horizontal_filter_margin)
    if vertical_filter_margin is not None:
        l_v_Lines, r_v_Lines = filter_vertical_line_candidates(l_v_Lines, r_v_Lines, h_Lines,
                                                               margin = vertical_filter_margin)
     
    if verbose: 
        gFrm.zero().to_BGR().put(r_v_Lines).show()
        gFrm.zero().to_BGR().put(l_v_Lines).show()
        gFrm.zero().to_BGR().put(h_Lines).show()
    
    # from ATMANpy.line_funcs import fit_frame_to_lines
    # courtFrm = fit_frame_to_lines(schemaLines)
    if len(l_v_Lines) == 2 and len(r_v_Lines) == 2 and len(h_Lines) == 5:
        # print('Fast Homography Algorithm...')
        from ATMANpy.funcs.line_funcs import get_homography_225
        H, config = get_homography_225(l_v_Lines, r_v_Lines, h_Lines, allBFrm, verbose = verbose)

    else:
        # print('General Homography Algorithm...')
        from ATMANpy.funcs.line_funcs import get_homography 
        if verbose: print(f'Constellation: Left lines: {len(l_v_Lines)}, right lines: {len(r_v_Lines)}, horizontal lines: {len(h_Lines)}')
        v_Lines = l_v_Lines.append(r_v_Lines).sort('left_point', 'l2r', 't2b')
        H, config = get_homography(v_Lines, h_Lines, allBFrm, verbose = verbose)

    return H, config

def find_courtlines(H, schemaLines):
    courtLines = schemaLines.transform('apply_homography', H).transform('round_points')
    return courtLines
    
def find_netlines(gFrm, courtLines):

    from ATMANpy.funcs.net_funcs import find_net_points, remove_outlier_net_points
    relPoints = find_net_points(courtLines, gFrm, 
                                factor = NET_FACTOR, points_to_search = NET_POINTS_TO_SEARCH)
    relPoints = remove_outlier_net_points(relPoints)
    
    from ATMANpy.funcs.net_funcs import cubic_regress, make_net_cord_lines
    b_0, b_1 = cubic_regress(relPoints)
    netLines = make_net_cord_lines(courtLines, b_0, b_1)

    return netLines

def calibrate_camera(courtLines, netLines, 
                     width = 'Doubles',
                     schemaLines = None, 
                     add_coords_world = None, add_coords_image = None,
                     verbose = False):
    ### Based on youtube: Camera Calibration by Shree Nayar. 
    
    if add_coords_world is None:
        add_coords_world = []
    if add_coords_image is None:
        add_coords_image = []
        
    # courtLines = find_courtlines(H, schemaLines)
    # netLines = find_netlines(gFrm, courtLines)
    
    # b) gather points in world and on image
    court_points_3D = schemaLines.get_3D_corner_points(width = width)
    net_points_3D = schemaLines.get_3D_net_points()
    court_points_3D = [p for p in court_points_3D]
    net_points_3D = [p for p in net_points_3D]
    
    court_points_2D = courtLines.get_court_region(width = width).to_points()
    court_points_2D = [np.array(p.coords, dtype = np.float32) for p in court_points_2D]
        
    net_l = netLines['Left Net']
    net_r = netLines['Right Net']
    net_points_2D = [net_l.leftPt, net_l.rightPt, net_r.rightPt]
    net_points_2D = [np.array(p.coords, dtype = np.float32) for p in net_points_2D]

    objpoints = np.array([*court_points_3D, *net_points_3D, *add_coords_world], dtype = np.float32)
    imgpoints =  np.array([*court_points_2D, *net_points_2D, *add_coords_image], dtype = np.float32)

    # if verbose:
    #     from ATMANpy.types.Points import Point2D
    #     from ATMANpy.funcs.line_funcs import fit_frame_to_lines
    #     schemaFrm = fit_frame_to_lines(schemaLines)
    #     for objP, imgP in zip(objpoints, imgpoints):
    #         pt_3D = Point2D(tuple(objP[:2]))
    #         pt_2D = Point2D(tuple(imgP))
            
    #         schemaFrm.to_BGR().put(pt_3D, thickness = 2, radius = 4).show()
    #         gFrm.to_BGR().put(pt_2D, thickness = 2, radius = 4).show()

    # c) create matrix A
    def make_A_entry(objpoint, imgpoint):
        u, v = imgpoint
        objpoint_homog = np.append(objpoint, 1)
        
        u_entry = np.hstack([objpoint_homog, np.zeros(4), - u * objpoint_homog])
        v_entry = np.hstack([np.zeros(4), objpoint_homog, - v * objpoint_homog]) 
        
        entry = np.vstack([u_entry, v_entry])
    
        return entry
    
    def make_A(objpoints, imgpoints):
        entryList = []
        for objP, imgP in zip(objpoints, imgpoints):
            entry = make_A_entry(objP, imgP)
            entryList.append(entry)
        return np.vstack(entryList)    
        
    A = make_A(objpoints, imgpoints)
    
    # d) Find Projection Matrix
    vals, vecs = np.linalg.eig(A.T @ A)
    smallest_value_index = vals.argmin()
    
    p = vecs[:, smallest_value_index]
    P = p.reshape(3, 4)

    if verbose:
        ### check
        print(f'True Value: {court_points_2D[0]}')
        proj = P @ np.array([*court_points_3D[0], 1])
        proj = proj[:2] / proj[2]
        print(proj)
        print('______')

    return P
   
def make_P_cond_inv(P, cond_val, cond_dim):
    #### get P, conditional on the height (or longitude or latitude) of the object observed
    unknown_cols = [d for d in range(3) if d != cond_dim]
    projection_for_unknowns = P[:, unknown_cols] ### 2 x 3 matrix -> how do x and y (world) contribute to u and v (pixels)
    
    conditional_column = P[:, cond_dim] * cond_val + P[:,3] ### go from 2 x 3 to 1 x 3 -> join known effect of z on u and v with constant P[:,3]
    conditional_column = conditional_column[:, np.newaxis]
    
    P_cond = np.append(projection_for_unknowns, conditional_column, axis = 1)
    return np.linalg.inv(P_cond) ### this takes us from a point on the image plane to a point in 3D, at height z
    
#####################
#####################

    
from ATMANpy.funcs.line_funcs import linear_regress, calculate_homography, find_brightest_neighbour

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
            # print('Lost Tracking...')
            return None, None
        else:
            b0, b1 = linear_regress(coordList)
            oldPt_1, oldPt_2 = coordList[0],coordList[-1]
            
            newPt_1 = Point2D((oldPt_1[0], b0 + b1 * oldPt_1[0]))
            newPt_2 = Point2D((oldPt_2[0], b0 + b1 * oldPt_2[0]))
            
            newLn = Line(newPt_1, newPt_2, name = aLn.name)

            newLines.append(newLn)
    
    newRegion = Region(newLines).join_corners()

    H = calculate_homography(config, newRegion.to_points())
    
    return H, config









