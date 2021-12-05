# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:43:22 2021

@author: gianl
"""

import ATMANpy.utils as atu

# misc
import os
import shutil
import math
import datetime
# plots
import matplotlib.pyplot as plt
# image operation
import cv2 as cv
import numpy as np
videoDict = atu.get_folder_struct('video_files')


video_id = 'pqxAmUWIoKc'


    
    


#### Get a frame ###
videoDict = atu.get_folder_struct('video_files')
videoPath = videoDict[video_id + '.mp4']

def get_frame(frame_num, videoPath):
    
    cap = cv.VideoCapture(videoPath)
    total_frames = cap.get(7)  ### total number of frames
    
    if frame_num < 0 or frame_num > total_frames:
        print(f'No such frame, choose number between 0 and {total_frames}')
        return None
    
    cap.set(1, frame_num)
    ret, frame = cap.read()
    
    return frame

def show(frame, name = 'frame'):
    cv.imshow(name, frame)
    cv.waitKey(0)
    #closing all open windows  
    cv.destroyAllWindows()  
    
    
frame = get_frame(11500, videoPath)
show(frame)

grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# inv = grey_frame[110:145,360:380]

# show(inv)

# rectangle = cv.crop(frame, (360, 120), (380, 145))
# show(rectangle)
# cv.bitwise_and()


## FIND LINES #######

def get_white_lines(frame, threshold = 200, verbose = False,
                    width_of_line = 5, int_diff_thresh = 10):

    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    whiteBool = np.array(grey_frame > threshold, dtype = 'uint8')
    
    if verbose:
        masked_white = cv.bitwise_and(grey_frame, grey_frame, mask = whiteBool)
        show(masked_white, name = 'White Pixels')

    def check_single_point(pixel, direction = 'vertical', 
                           width_of_line = width_of_line, 
                           int_diff_thresh = int_diff_thresh):
        x, y = pixel
        
        intensity = grey_frame[x, y]
        
        if direction == 'horizontal':
            coords_lo = (x - width_of_line, y)
            coords_hi = (x + width_of_line, y)
        else:
            coords_lo = (x, y - width_of_line)
            coords_hi = (x, y + width_of_line)
        
        try:
            intensity_lo = grey_frame[coords_lo]
        except IndexError:
            condition_lo = True
        else:
            condition_lo = intensity - intensity_lo > int_diff_thresh
        
        if condition_lo:
            try:
                intensity_hi = grey_frame[coords_hi]
            except IndexError:
                condition_hi = True
            else:
                condition_hi = intensity - intensity_hi > int_diff_thresh
            finally:
                return condition_hi
        else:
            return False
        
    hori_array = np.zeros(grey_frame.shape, dtype = 'uint8')
    vert_array = np.zeros(grey_frame.shape, dtype = 'uint8')
    
    ### This loop is tremendously ugly.
    for row_idx, row in enumerate(whiteBool): 
        for col_idx, myBool in enumerate(row):
            if myBool:
                pixel = (row_idx, col_idx)
                hori_array[row_idx, col_idx] = check_single_point(pixel,
                                                                  direction = 'horizontal')
                vert_array[row_idx, col_idx] = check_single_point(pixel,
                                                                  direction = 'vertical')
    
    if verbose:
        lineBool = np.bitwise_or(hori_array, vert_array)
        masked_lines = cv.bitwise_and(grey_frame, grey_frame, mask = lineBool)
        show(masked_lines, name = 'Line Pixels')
    
    return hori_array, vert_array
    
WHITE_PIXEL_THRESHOLD = 150 #this depends heavily on brightness of picture... lines are white, but they might not appear at 255 intensity.
COURT_LINE_WIDTH = 5 ### according to own rough calcs at 1920 x 1080, and also according to vinyes (2018), court line width is equal to 5 pixels
INTENSITY_DIFFERENCE_THRESHOLD = 30

h_lines, v_lines = get_white_lines(frame, verbose = True, 
                                   threshold = WHITE_PIXEL_THRESHOLD,
                                   width_of_line = COURT_LINE_WIDTH, 
                                   int_diff_thresh = INTENSITY_DIFFERENCE_THRESHOLD)

show(h_lines * 255)
#### HOUGH TRANSFORM
all_lines = cv.bitwise_or(h_lines, v_lines)

### Q: do we do hough lines on h_lines and v_lines separately, or do we join them?
# For a start, we join them, since we re-classify verticals and horizontals anyway.

def prob_hough(binFrm, rho = 3, theta = np.pi/180, min_votes = 100, 
               minLineLength = 10, maxLineGap = 1,
               verbose = True):
    hough_lines = cv.HoughLinesP(binFrm, rho, theta, min_votes, 
                                 minLineLength = minLineLength)
    if verbose:
        colFrm = cv.cvtColor(binFrm, cv.COLOR_GRAY2RGB) 
        
        for v in hough_lines:
            x1, y1, x2, y2 = v[0]
            cv.line(colFrm,(x1,y1),(x2,y2),(0,255,0),2)
        
        show(colFrm)
        print(f'{hough_lines.shape[0]} hough lines found.')

    return hough_lines

hough_lines = prob_hough(all_lines, rho = 3, theta = np.pi/180, min_votes = 500,
                         minLineLength = COURT_LINE_WIDTH + 1, maxLineGap = 1)


"""
Selected lines classication: The lines that have been detected are classified into horizontal
and vertical lines on the left and right-hand-side of the image separately. Next, the lines
are sorted within each of the categories: horizontal lines are ordered by y-coordinate from
top to bottom (left to right if the y coordinate is the same), left vertical lines are ordered
by x-coordinate from right to left, and right vertical lines from left to right (from top to
bottom if the x-coordinate is the same). A more detailed description of this process is
provided in Appendix B
"""

def classify_line(line_points, verbose = False):
    """
    Takes two points (x1, y1) and (x2, y2) and figures out whether the points
    form a horizontal or vertical line.
    """
    x1, y1, x2, y2 = line_points
    
    if verbose:
        max_coord = max(x1, y1, x2, y2)
        blank = np.zeros((max_coord, max_coord))
        blank_with_line = cv.line(blank.copy(),(x1,y1),(x2,y2),255,2)
        show(blank_with_line)
        
    if abs(x2 - x1) > abs(y2 - y1): 
        return 'horizontal'
    else:
        return 'vertical'
    
def find_line(line_points, border_left, border_right):
    x1, y1, x2, y2 = line_points    
    
    if x1 < border_left or x2 < border_left:
        return 'left'
    elif x1 > border_right or x2 > border_right:
        return 'right'
    else:
        return 'center'
    
    
line_points = hough_lines[0][0]
classify_line(line_points, verbose = True)

def group_and_sort_lines(hough_lines, frame):
    
    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    ### divide the frame up into thirds, with three vertical borders marking
    ### a left, centre and right section (like the italian flag, or smth.)
    v_pix, h_pix = grey_frame.shape
    border_left, border_right = h_pix // 3, 2 * h_pix // 3 
    
    ### Initialize Line Dict
    lDict = {'h': [], 'v_l': [], 'v_r': []}
    
    ### Categorize Lines
    for h in hough_lines:
        line_points = h[0] #have to do this because the hough lines are doubly wrapped in array.
    
        if classify_line(line_points) == 'horizontal':
            lDict['h'].append(line_points)
        else:
            line_location = find_line(line_points, border_left, border_right)
            if line_location == 'left':
                lDict['v_l'].append(line_points)
            elif line_location == 'right':
                lDict['v_r'].append(line_points)
            ### ignore vertical lines in the center!
    
    ### Sort 
    def sort_criterion(line_points, precedence = 'x', 
                       x_ordering = 'l2r', y_ordering = 't2b'):
        """
        According to a logic provided in the keyword arguments, 
        gives a sorting score, expressed as a tuple, to a line expressed in terms
        of two endpoints.
        The sorting score is ascending; low is better. 
        """
        x1, y1, x2, y2 = line_points
        
        x_score = min(x1, x2) if x_ordering == 'l2r' else -max(x1, x2)
        y_score = min(y1, y2) if y_ordering == 't2b' else -max(y1, y2)
        
        scoreTup = (x_score, y_score) if precedence == 'x' else (y_score, x_score)
        return scoreTup
    
    from functools import partial
    h_func   = partial(sort_criterion, 
                       precedence = 'y', x_ordering = 'l2r', y_ordering = 't2b')
    v_l_func = partial(sort_criterion, 
                       precedence = 'x', x_ordering = 'r2l', y_ordering = 't2b')
    v_r_func = partial(sort_criterion, 
                       precedence = 'x', x_ordering = 'l2r', y_ordering = 't2b')
    
    lDict['h']   = sorted(lDict['h'], key = h_func)
    lDict['v_l'] = sorted(lDict['v_l'], key = v_l_func)
    lDict['v_r'] = sorted(lDict['v_r'], key = v_r_func) 
    
    return lDict
    
lineDict = group_and_sort_lines(hough_lines, frame)
    
### merging of lines
def sort_line(line_points, direction = 'l2r'):
    ### rearranges a line to go from left to right
    x1, y1, x2, y2 = line_points
    
    direction_condition = (x1 <= x2) if direction == 'l2r' else (y1 <= y2) ### else t2b
    if direction_condition:
        return [x1, y1, x2, y2]
    else:
        return [x2, y2, x1, y1]
    
def sort_two_lines(line_points1, line_points2, direction = 'l2r'):
    """
    Input two lines, and return two lines, but in order: Left, Right or Top and Bottom  
    Also, this function enforces the direction of the lines: they go from left to right
    (See Vinyes, Appendix C: create points Ll, Lr, Rl, Rr)
    """
    x1_1, y1_1, x2_1, y2_1 = sort_line(line_points1, direction = direction)
    x1_2, y1_2, x2_2, y2_2 = sort_line(line_points2, direction = direction)
    
    direction_condition = (x1_1 <= x1_2) if direction == 'l2r' else (y1_1 <= y1_2) ### else t2b
    if direction_condition:
        return line_points1, line_points2
    else:
        print(f'Sorting inverted! Please check {line_points1}, {line_points2}, which have been switched.')
        return line_points2, line_points1

### In principle, due to our previous sorting, lines should already be sorted 
# from left to right and or top to bottom, in the way we want.
# however, just to be sure, let's check with this function.

def merge_two_horizontal_lines(left_line_points, right_line_points, 
                               mergeThresh = 10):
    ### mergeThresh: preferably set to twice COURT_LINE_WIDTH
    xl_L, yl_L, xr_L, yr_L = left_line_points
    xl_R, yl_R, xr_R, yr_R = right_line_points
    
    d = abs(yr_L - yl_R)
    if d < mergeThresh:
        return [xl_L, yl_L, xr_R, yr_R], [] ### return merged line (plus empty line)
    else:
        return left_line_points, right_line_points ### return original lines

def merge_two_vertical_lines(top_line_points, bottom_line_points,
                             mergeThresh = 10):
    ### mergeThresh: preferably set to twice COURT_LINE_WIDTH
    xt_T, yt_T, xb_T, yb_T = top_line_points
    xt_B, yt_B, xb_B, yb_B = bottom_line_points

    d = abs(xb_T - xt_B)
    if d < mergeThresh:
        return [xb_B, yb_B, xt_T, yt_T], [] ### return merged line (plus empty line)
    else:
        return top_line_points, bottom_line_points ### return original lines

#### I know these two functions could be the same... however, for the sake
# of clarity of notation (L, R or T, B), we split these functions

def merge_lines(lines, 
                direction = 'horizontal', mergeThresh = 10):
    ### 
    if len(lines) == 0:
        return []
    elif len(lines) == 1:
        return [lines[0]]
    else:        
        l1, l2, ls = lines[0], lines[1], lines[2:]

    sorted1, sorted2 = sort_two_lines(l1, l2, direction = direction)
    if direction == 'horizontal':
        out1, out2 = merge_two_horizontal_lines(sorted1, sorted2)
    else:
        out1, out2 = merge_two_vertical_lines(sorted1, sorted2)

    if len(out2) == 0: ### the empty line, in case of merger
        return [out1, *merge_lines(ls, direction = direction, mergeThresh = mergeThresh)]
    else:
        return [out1, *merge_lines([out2, *ls], direction = direction, mergeThresh = mergeThresh)]
    
h_lines = merge_lines(lineDict['h'])

def merge_lines_multi_pass(lines, 
                           direction = 'horizontal', mergeThresh = 10):
    print(f'Lines remaining: {len(lines)}')
    lines_out = merge_lines(lines,
                            direction = direction, mergeThresh = mergeThresh)
    if len(lines) != len(lines_out):
        return merge_lines_multi_pass(lines_out, 
                                      direction = direction, mergeThresh = mergeThresh)
    else:
        print(f'Merging converged; Distinct lines found: {len(lines_out)}')
        return lines_out

mergeDict = {}
mergeDict['h'] = merge_lines_multi_pass(lineDict['h'],
                                        direction = 'horizontal', mergeThresh = 10)
mergeDict['v_l'] = merge_lines_multi_pass(lineDict['v_l'],
                                        direction = 'vertical', mergeThresh = 10)
mergeDict['v_r'] = merge_lines_multi_pass(lineDict['v_r'],
                                        direction = 'vertical', mergeThresh = 10)

grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
grey_frame2 = cv.cvtColor(grey_frame, cv.COLOR_GRAY2BGR)

for l in mergeDict['h']:
    x1, y1, x2, y2 = l
    cv.line(grey_frame2,(x1,y1),(x2,y2),(255,0,0),2)

for l in mergeDict['v_l']:
    x1, y1, x2, y2 = l
    cv.line(grey_frame2,(x1,y1),(x2,y2),(0,255,0),2)

for l in mergeDict['v_r']:
    x1, y1, x2, y2 = l
    cv.line(grey_frame2,(x1,y1),(x2,y2),(0,0,255),2)
        
    
show(grey_frame2)

e = lineDict['h']
len(e)
len(h_lines)

np.float32([[1, 0, 3], [0, 1, 3]])

dir(cv)

blank = np.zeros((600, 400), dtype = 'uint8')
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 100, 100, -1)

obj = cv.bitwise_and(rectangle, circle)

# show(rectangle)
# show(circle)
show(obj)

show(cv.bitwise_not(circle))

dir(frame)





