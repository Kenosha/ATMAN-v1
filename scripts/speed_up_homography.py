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

MERGE_THRESHOLD = 2 * COURT_LINE_WIDTH 
MARGIN_TO_CUT_HORIZONTAL = 40

from ATMANpy.line_funcs import get_white_lines_fast
allBFrm = get_white_lines_fast(gFrm, width_of_line = COURT_LINE_WIDTH, white_thresh = WHITE_PIXEL_THRESHOLD,
                             int_diff_thresh = INTENSITY_DIFFERENCE_THRESHOLD, verbose = False)

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


from ATMANpy.line_funcs import get_court_schema_intersection_points, calculate_homography, fit_frame_to_lines
def get_homography(v_Lines, h_Lines, white_BoolFrm, 
                   verbose = False, show_threshold_score = 1000, displayFrame = None,
                   size = 4, margin = 10):
    ## scores for matched and unmatched
    MATCH_SCORE = 1
    UNMATCH_SCORE = -0.5
    
    ### prepare court lines and point configs
    courtLines = CourtLines.make(size = size, margin = margin)
    court_configs = get_court_schema_intersection_points(courtLines)
    if displayFrame is None:
        displayFrame = white_BoolFrm.to_grey()
    
    from itertools import combinations
    vert_combos = list(combinations(v_Lines, 2))
    hori_combos = list(combinations(h_Lines, 2))
    
    max_score = 0
    best_config = None
    for v1, v2 in vert_combos:
        for h1, h2 in hori_combos:
            
            pts_prj = (v1.weak_intersect(h1), 
                       v2.weak_intersect(h1), 
                       v1.weak_intersect(h2), 
                       v2.weak_intersect(h2)) ### order here is important! has to be lexicographic!
        
            for pts_crt in court_configs:
                
                H = calculate_homography(pts_crt, pts_prj)
                
                ### iii) given a homography, project court schema onto frame.
                projLines = courtLines.transform('apply_homography', H).transform('round_points')
                
                try:
                    projFrm = (displayFrame.zero().put(projLines) > 0)
                except OverflowError:
                    continue
                
                projected_matches_white = projFrm & white_BoolFrm
                good_score = MATCH_SCORE * projected_matches_white.array.sum()
    
                projected_does_not_match_white = projFrm & ~white_BoolFrm
                bad_score = UNMATCH_SCORE * projected_does_not_match_white.array.sum()
    
                score = good_score + bad_score
                
                if verbose:
                    if score > show_threshold_score:
                        t1 = f'Match Score: {good_score}'
                        t2 = f'Non-Match Penalty: {-bad_score}'
                        t3 = f'TOTAL: {score}'
                        t4 = f'Best so far: {max_score}'
                        scoreFrm = displayFrame.to_BGR().put(projLines, thickness = 3) ### THICKNESS: HALF, OR FULL COURT LINE WIDTH?
                        scoreFrm = scoreFrm.put(t1, where = (50, 50), thickness = 3)
                        scoreFrm = scoreFrm.put(t2, where = (50, 80), thickness = 3)
                        scoreFrm = scoreFrm.put(t3, where = (50, 110), thickness = 3)
                        scoreFrm = scoreFrm.put(t4, where = (50, 140), thickness = 3)
                        scoreFrm.show()
    
                if score > max_score:
                    max_score = score
                    best_homography = H.copy()
                    best_config = pts_crt

    return best_homography, best_config


def get_homography_225(l_v_Lines, r_v_Lines, h_Lines, white_BoolFrm, 
                       verbose = False, show_threshold_score = 1000, displayFrame = None,
                       size = 4, margin = 10):
    """
    This is a fast version of get homography if we encounter a 2-2-5 line config:
        2 left vertical, 2 right vertical, 5 horizontal lines. 
    In this case, we can guess the court configuration immediately, and don't need to try all combinations.
    """
    
    ## scores for matched and unmatched
    MATCH_SCORE = 1
    UNMATCH_SCORE = -0.5
    
    ### prepare court lines and point configs
    courtLines = CourtLines.make(size = size, margin = margin)
    fit_frame_to_lines(courtLines).put(courtLines).show()
    court_configs = get_court_schema_intersection_points(courtLines)
    if displayFrame is None:
        displayFrame = white_BoolFrm.to_grey()

    ### initialize guesses ###
    # note: vertical lines are sorted so that the singles line is the first element!
        # (left vertical: right to left, right vertical, left to right)
    lsLine, ldLine = l_v_Lines
    rsLine, rdLine = r_v_Lines
    tbLine, tsLine, _, bsLine, bbLine = h_Lines
    
    def lines_to_points(vLines, hLines):
        return [v.weak_intersect(h) for h in hLines for v in vLines]
    
    combos_to_check = [(lsLine, rsLine, tbLine, bbLine),
                       (lsLine, rsLine, tbLine, bsLine),
                       (lsLine, rsLine, tsLine, bbLine),
                       (lsLine, rsLine, tsLine, bsLine), ### if the single lines are chosen, 
                                                           # then we can intersect them with the baselines and the service lines.
                                                           # also, the two h-Lines have to be on opposing sides
                       (ldLine, rdLine, tbLine, bbLine),   # if any vertical line is a doubles line, then we can only use the opposing two baselines
                       (lsLine, rdLine, tbLine, bbLine),
                       (ldLine, rsLine, tbLine, bbLine)]
    combos_to_check = [lines_to_points((lv, rv), (th, bh)) for lv, rv, th, bh in combos_to_check]

    # corresponding_configs = [2, 7, 6, 5, 1, 3, 4] ### indexed as in Vinyes (2018) Appendix F
    corresponding_configs = [1, 6, 5, 4, 0, 2, 3] ### 0-indexed.
    corresponding_configs = [court_configs[c] for c in corresponding_configs]
    
    max_score = 0
    best_config = None
    for pts_prj, pts_crt in zip(combos_to_check, corresponding_configs):
        H = calculate_homography(pts_crt, pts_prj)
        
        ### iii) given a homography, project court schema onto frame.
        projLines = courtLines.transform('apply_homography', H).transform('round_points')
        
        try:
            projFrm = (displayFrame.zero().put(projLines) > 0)
        except OverflowError:
            continue
        
        projected_matches_white = projFrm & white_BoolFrm
        good_score = MATCH_SCORE * projected_matches_white.array.sum()

        projected_does_not_match_white = projFrm & ~white_BoolFrm
        bad_score = UNMATCH_SCORE * projected_does_not_match_white.array.sum()

        score = good_score + bad_score
        
        if verbose:
            if score > show_threshold_score:
                t1 = f'Match Score: {good_score}'
                t2 = f'Non-Match Penalty: {-bad_score}'
                t3 = f'TOTAL: {score}'
                t4 = f'Best so far: {max_score}'
                scoreFrm = displayFrame.to_BGR().put(projLines, thickness = 3) ### THICKNESS: HALF, OR FULL COURT LINE WIDTH?
                scoreFrm = scoreFrm.put(t1, where = (50, 50), thickness = 3)
                scoreFrm = scoreFrm.put(t2, where = (50, 80), thickness = 3)
                scoreFrm = scoreFrm.put(t3, where = (50, 110), thickness = 3)
                scoreFrm = scoreFrm.put(t4, where = (50, 140), thickness = 3)
                scoreFrm.show()

        if score > max_score:
            max_score = score
            best_homography = H.copy()
            best_config = pts_crt

    return best_homography, best_config












