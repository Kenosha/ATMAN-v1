# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:58:08 2022

@author: gianl
"""

from ATMANpy.types import Frame, BGRFrame, GreyFrame, BoolFrame, Point, Line, Lines

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


def get_white_lines(gFrm: GreyFrame, width_of_line: int = 5,
                    white_thresh: int = 200, int_diff_thresh: int = 10):
    whiteBFrm = gFrm > white_thresh
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
   
    gFrm.mask(horiBFrm | vertBFrm).show()
    
    return horiBFrm, vertBFrm
      
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

def sort_court_line_candidates(left_vert_Lines, right_vert_Lines, horizontal_Lines):
    return (left_vert_Lines.sort('left_point', 'r2l', 't2b'), 
            right_vert_Lines.sort('left_point', 'l2r', 't2b'),
            horizontal_Lines.sort('left_point', 't2b', 'l2r'))

def merge_court_line_candidates(left_vert_Lines, right_vert_Lines, horizontal_Lines,
                                threshold = 10):
    ### instantiate parameters as in appendix C vinyes:
    hi_left, hi_right = left_vert_Lines.highest(), right_vert_Lines.highest()
    
    ### get the horizontal line used for vertical merge
    left_vert_hori_line = Line(hi_left, hi_left.right(20))
    right_vert_hori_line = Line(hi_right, hi_right.right(20))
    
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


