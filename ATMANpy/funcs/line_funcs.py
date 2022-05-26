# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:58:08 2022

@author: gianl
"""

import numpy as np
import cv2 as cv

from ATMANpy.types.Points import Point, SafePoint
from ATMANpy.types.Lines import Line, Lines, CourtLines
from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, BoolFrame


############ DETECTING WHITE LINES #############
def get_vicinity(aPt: Point, 
                 radius: int = 5, direction: str = None):
    radius = round(radius)
    if direction == 'vertical':
        return(aPt, aPt.down(radius), aPt.up(radius))
    elif direction == 'horizontal':
        return(aPt, aPt.left(radius), aPt.right(radius))
    else:
        print('Bad Keyword, enter "Horizontal" or "Vertical"')

def compare_intensity(refInt: int, newInt: int, 
                      int_diff_thresh: int = 10):
    """
    Compares a reference intensity with a new intensity, and returns a boolean
    If the reference intensity is smaller than the new intensity, 
        then the reference point is darker than the new point, and we have no white line candidate
    If the reference intensity is larger than the new intensity by a small amount,
        then the whites extend beyond the reference point, and we have no line again
    If the reference intensity is larger than the new intensity by a sufficiently large amount, 
        then we have a white line candidate
    """
    if refInt >= newInt:
        return refInt - newInt > int_diff_thresh
    else:
        return False
    
def check_vicinity_of_point(aPt: Point, aFrm: Frame, 
                            radius: int = 5, direction: str = None, int_diff_thresh: int = 10):
    refPt, newPt1, newPt2 = get_vicinity(aPt, radius = radius, direction = direction)
    refInt, newInt1, newInt2 = aFrm.eval_at(refPt), aFrm.eval_at(newPt1), aFrm.eval_at(newPt2)
    Bool1 = compare_intensity(refInt, newInt1, int_diff_thresh = int_diff_thresh)
    Bool2 = compare_intensity(refInt, newInt2, int_diff_thresh = int_diff_thresh)
    return (Bool1 and Bool2)


def find_brightest_neighbour(gFrm, aPt):
    c, l, r = get_vicinity(aPt, radius = 1, direction = 'horizontal')
    _, b, t = get_vicinity(aPt, radius = 1, direction = 'vertical')
    
    neighbours = [c, l, r, b, t] 
    vals = sorted([(pt, gFrm.eval_at(pt)) for pt in neighbours], key = lambda tup: tup[1])
    maxPt, maxVal = vals[-1]
    return maxPt, maxVal


def get_white_lines(gFrm: GreyFrame, width_of_line: int = 5,
                    white_thresh: int = 200, int_diff_thresh: int = 10,
                    verbose = False):
    whiteBFrm = gFrm > white_thresh
    
    if verbose:
        gFrm.mask(whiteBFrm).show()
    
    ## since frames are immutable, but arrays are not, we will build the new array dynamically, then turn it into a frame
    horiArr = gFrm.zero().array.astype("bool")
    vertArr = gFrm.zero().array.astype("bool")

    for bPt in whiteBFrm:
        if bPt.value:
            horiArr[bPt.y, bPt.x] = check_vicinity_of_point(bPt, gFrm, 
                                                direction = 'horizontal',
                                                radius = width_of_line / 2, 
                                                int_diff_thresh = int_diff_thresh)
            
            vertArr[bPt.y, bPt.x] = check_vicinity_of_point(bPt, gFrm, 
                                                direction = 'vertical',
                                                radius = width_of_line / 2, 
                                                int_diff_thresh = int_diff_thresh)
    
    horiBFrm = BoolFrame(horiArr)
    vertBFrm = BoolFrame(vertArr)
   
    if verbose:
        gFrm.mask(horiBFrm | vertBFrm).show()
    
    return horiBFrm, vertBFrm
      

def get_white_lines_fast(gFrm: GreyFrame, width_of_line: int = 5, 
                         white_thresh: int = 200, int_diff_thresh: int = 10,
                         verbose = False):
    
    shift = round(width_of_line / 2)
    
    NAN_SENTINEL = 255  # nans become completely white: this way, we can preserve the uint8 type, 
                        # and have the same behaviour as if we had had np.nan
    upFrm = gFrm.roll(how = 'up', shift = shift, nan_sentinel = NAN_SENTINEL)
    dnFrm = gFrm.roll(how = 'down', shift = shift, nan_sentinel = NAN_SENTINEL)
    lfFrm = gFrm.roll(how = 'left', shift = shift, nan_sentinel = NAN_SENTINEL)
    rtFrm = gFrm.roll(how = 'right', shift = shift, nan_sentinel = NAN_SENTINEL)
 
    whiteMask = (gFrm > white_thresh)
    horiMask = (gFrm - upFrm > int_diff_thresh) & (gFrm - dnFrm > int_diff_thresh) & (gFrm > upFrm) & (gFrm > dnFrm)
    vertMask = (gFrm - lfFrm > int_diff_thresh) & (gFrm - rtFrm > int_diff_thresh) & (gFrm > lfFrm) & (gFrm > rtFrm)
    
    line_candidateMask = whiteMask & (horiMask | vertMask)
    
    if verbose:
        gFrm.mask(whiteMask).show()
        gFrm.mask(horiMask).show()
        gFrm.mask(vertMask).show()
        gFrm.mask(line_candidateMask)
    
    return line_candidateMask

#####################

######## LINE SPLIT, FILTER, MERGE ########

def split_court_line_candidates(aLines, left_thresh: int, right_thresh: int):
    left_vert_list = []
    right_vert_list = []
    horizontal_list = []
    
    for line in aLines:
        if line.is_horizontal():
            horizontal_list.append(line)
        else:
            if line.is_weakly_left(left_thresh):
                left_vert_list.append(line)
            elif line.is_weakly_right(right_thresh):
                right_vert_list.append(line)

    return Lines(left_vert_list), Lines(right_vert_list), Lines(horizontal_list)

def keep_only_central_horizontal_lines(hLines, left_thresh: int, right_thresh: int):
    ### this step is added because the merge criterion for hoorizontal lines,
    ### while being fast due to sorting and not requiring every line to be compared with every other,
    ### fails 
    
    def keep_criterion(line):
        left_within = left_thresh < line.left_point().x < right_thresh
        right_within = left_thresh < line.right_point().x < right_thresh
        return left_within | right_within
        
    keepList = []
    for line in hLines:
        if keep_criterion(line):
            keepList.append(line)
    
    return Lines(keepList)

def merge_horizontal_lines(aLn, bLn, threshold = 10):
    
    if aLn.left_point().is_left(bLn.left_point()):
        lLn, rLn = aLn, bLn
    else:
        lLn, rLn = bLn, aLn
    
    d = abs(lLn.right_point().y - rLn.left_point().y)
    
    if d < threshold:
        newLine = Line(lLn.left_point(), rLn.right_point())
        return Lines([newLine])
    else:
        return Lines([aLn, bLn])

def merge_vertical_lines(aLn, bLn, fixed_horizontal_line = None, threshold = 10):
    aPt = aLn.weak_intersect(fixed_horizontal_line)
    bPt = bLn.weak_intersect(fixed_horizontal_line)

    d = abs(aPt.x - bPt.x)
    
    outLines = Lines([aLn, bLn])
    if d < threshold:
        loPt = outLines.lowest()
        hiPt = outLines.highest()
        return Lines([Line(loPt, hiPt)])
    else:
        return outLines

def horizontal_filter(horiLn, lo_thresh = 1000, hi_thresh = 0):
    return not (horiLn.is_strictly_above(hi_thresh) or horiLn.is_strictly_below(lo_thresh))

# def sort_court_line_candidates(left_vert_Lines, right_vert_Lines, horizontal_Lines):
#     return (left_vert_Lines.sort('left_point', 'r2l', 't2b'), 
#             right_vert_Lines.sort('left_point', 'l2r', 't2b'),
#             horizontal_Lines.sort('left_point', 't2b', 'l2r'))

def get_high_line(vertLines):
    hiPt = vertLines.highest()
    return Line(hiPt, hiPt.right(20))

def sort_court_line_candidates(left_vert_Lines, right_vert_Lines, horizontal_Lines):
    
    ### get the horizontal line used for vertical sort
    ### vertical lines are sorted by where they intersect the respective line!!!
    left_vert_hori_line = get_high_line(left_vert_Lines)
    right_vert_hori_line = get_high_line(right_vert_Lines)
    
    return (left_vert_Lines.sort(left_vert_hori_line, 'r2l', 't2b'),
            right_vert_Lines.sort(right_vert_hori_line, 'l2r', 't2b'),
            horizontal_Lines.sort('left_point', 't2b', 'l2r'))

def merge_court_line_candidates(left_vert_Lines, right_vert_Lines, horizontal_Lines,
                                threshold = 10):
    
    ### get the horizontal line used for vertical merge
    left_vert_hori_line = get_high_line(left_vert_Lines)
    right_vert_hori_line = get_high_line(right_vert_Lines)
    
    ### define the partial merge functions
    from functools import partial
    my_left_vertical_merge = partial(merge_vertical_lines, threshold = threshold,
                                     fixed_horizontal_line = left_vert_hori_line)
    my_right_vertical_merge = partial(merge_vertical_lines, threshold = threshold,
                                     fixed_horizontal_line = right_vert_hori_line)
    my_horizontal_merge = partial(merge_horizontal_lines, threshold = threshold)
    
    merged_left_vert_Lines = left_vert_Lines.merge(my_left_vertical_merge)    
    merged_right_vert_Lines = right_vert_Lines.merge(my_right_vertical_merge)    
    merged_hori_Lines = horizontal_Lines.merge(my_horizontal_merge)

    return merged_left_vert_Lines, merged_right_vert_Lines, merged_hori_Lines

def filter_court_line_candidates(left_vert_Lines, right_vert_Lines, horizontal_Lines,
                                 margin = 40):
    ### get highest, lowest points for each part of picture
    hi_left, hi_right = left_vert_Lines.highest(), right_vert_Lines.highest()
    lo_left, lo_right = left_vert_Lines.lowest(), right_vert_Lines.lowest()
    
    ### get highest of both parts, add a margin
    hiPt = Line(hi_left, hi_right).high_point().up(margin)
    loPt = Line(lo_left, lo_right).low_point().down(margin)
    
    ### construct filter function
    from functools import partial
    my_filter = partial(horizontal_filter, lo_thresh = loPt.y, hi_thresh = hiPt.y)
    
    new_horizontal_Lines = horizontal_Lines.filt(my_filter)
    
    return left_vert_Lines, right_vert_Lines, new_horizontal_Lines


########## COURT CORRESPONDENCES #######


def fit_frame_to_lines(aLns):
    t, l = aLns.highest(), aLns.leftest()
    b, r = aLns.lowest(), aLns.rightest()
    gFrm = GreyFrame(np.zeros((b.y + t.y, r.x + l.x), dtype = 'uint8')) 
    lnFrm = gFrm.put(aLns)
    return lnFrm

def get_court_schema_intersection_points(courtLines: CourtLines):
    # court schema intersection points
    p11 = courtLines['Top Baseline'].weak_intersect(courtLines['Top Doubles Deuce Line'])
    p12 = courtLines['Top Baseline'].weak_intersect(courtLines['Top Singles Deuce Line'])
    p13 = courtLines['Top Baseline'].weak_intersect(courtLines['Top Singles Ad Line'])
    p14 = courtLines['Top Baseline'].weak_intersect(courtLines['Top Doubles Ad Line'])
    p21 = courtLines['Bottom Baseline'].weak_intersect(courtLines['Bottom Doubles Ad Line'])
    p22 = courtLines['Bottom Baseline'].weak_intersect(courtLines['Bottom Singles Ad Line'])
    p23 = courtLines['Bottom Baseline'].weak_intersect(courtLines['Bottom Singles Deuce Line'])
    p24 = courtLines['Bottom Baseline'].weak_intersect(courtLines['Bottom Doubles Deuce Line'])
    p31 = courtLines['Top Service Line'].weak_intersect(courtLines['Top Singles Deuce Line'])
    p32 = courtLines['Top Service Line'].weak_intersect(courtLines['Top Center Line'])
    p33 = courtLines['Top Service Line'].weak_intersect(courtLines['Top Singles Ad Line'])
    p41 = courtLines['Bottom Service Line'].weak_intersect(courtLines['Bottom Singles Ad Line'])
    p42 = courtLines['Bottom Service Line'].weak_intersect(courtLines['Bottom Center Line'])
    p43 = courtLines['Bottom Service Line'].weak_intersect(courtLines['Bottom Singles Deuce Line'])
    
    # replicate 11 four-point court configs given in appendix f
    court_configs = [(p11, p14, p21, p24),
                     (p12, p13, p22, p23),
                     (p12, p14, p22, p24),
                     (p11, p13, p21, p23),
                     (p31, p33, p41, p43),
                     (p31, p33, p22, p23),
                     (p12, p13, p41, p43),
                     (p13, p14, p23, p24),
                     (p11, p12, p21, p22),
                     (p31, p32, p41, p42),
                     (p32, p33, p42, p43)]
    return court_configs

def show_configs(courtFrm, court_configs, **kwargs):
    for a, b, c, d in court_configs:
        pFrm = courtFrm.to_BGR().put(a, **kwargs).put(b, **kwargs).put(c, **kwargs).put(d, **kwargs)
        pFrm.show()

def calculate_homography(schemaPts: tuple, projectedPts: tuple):
    schemaPtsArray = np.array([list(pt.coords) for pt in schemaPts])
    projectedPtsArray = np.array([list(pt.coords) for pt in projectedPts])    
    H, status = cv.findHomography(schemaPtsArray, projectedPtsArray)
    return H

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
    best_homography = None
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
    # fit_frame_to_lines(courtLines).put(courtLines).show()
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
    best_homography = None
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

####### COURT TRACKING #############

def linear_regress(relPoints):
    ### iv) lets now try to estimate a parabola fitting these points
    import statsmodels.api as sm
    
    Y = [y for x, y in relPoints]
    X = [x for x, y in relPoints]
    X = sm.add_constant(X)
    
    model = sm.OLS(Y, X)
    results = model.fit()
    b_0, b_1 = results.params
    return b_0, b_1
