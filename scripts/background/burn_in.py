# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:56:05 2022

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

#%% COURTLINE SCHEMA INIT

from ATMANpy.high_level_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)

#%% FUNCTIONS #####

###### DEFINE TRACK PROCEDURE ########
from ATMANpy.high_level_funcs import find_homography_225_only, update_homography
from ATMANpy.high_level_funcs import find_courtlines, find_netlines

def track_courtlines(gFrm, prev_H, prev_config, schemaLines = schemaLines):
    if prev_H is None:
        res = find_homography_225_only(gFrm)
    else:
        # print('Courtlines found, updating homography...')
        res = update_homography(gFrm, prev_H, prev_config)
        
    if not res is None:
        H, config = res
        courtLines = find_courtlines(H, schemaLines)
        netLines = find_netlines(gFrm, courtLines)       
    
        # outFrm = gFrm.to_BGR().put(courtLines, thickness = 3)
        # outFrm = outFrm.put(netLines, color = (255, 0, 0), thickness = 3)
    else:
        courtLines, netLines = None, None
        H, config = None, None
        
    return courtLines, netLines, H, config
#########################################

###### LOOK SEMI-RANDOMLY FOR COURT LINES #####
def collect_courtlines(cap, target_number = 1000, verbose = False):
    courtlineList = []
    
    H = None
    config = None
        
    ### init video stream ###
    from ATMANpy.cv_funcs import random_or_next_stream
    gen = random_or_next_stream(cap, verbose = verbose)
    frmNum, rawArr = next(gen) ### this is needed to "prime" the generator
    ######
    
    while len(courtlineList) < TARGET_COURTLINE_NUMBER:
        cFrm = BGRFrame(rawArr)
        gFrm = cFrm.to_grey()
        
        try:
            courtLines, netLines, H, config = track_courtlines(gFrm, H, config)
        except Exception as e:
            print('Exception in track_courtlines:')
            print(e)
            courtLines = None
        
        if courtLines is None:
            
            if verbose:
                sendFrm = cFrm.put('No Lines!', thickness = 2, color = (0, 0, 255))
                sendArr = sendFrm.array
            else:
                sendArr = None
            sendMessage = 'random'
    
        else:
            courtlineList.append((frmNum, courtLines, H))
            
            if verbose:
                sendFrm = cFrm.put(f'Lines found! {len(courtlineList)} so far', thickness = 2)
                sendFrm = sendFrm.put(courtLines, thickness = 2, color = (0, 255, 0))
                sendFrm = sendFrm.put(netLines, thickness = 2, color = (255, 0, 0))
                sendArr = sendFrm.array
            else:
                sendArr = None
                
            sendMessage = 'next'
        
        frmNum, rawArr = gen.send((sendMessage, sendArr))
    else:
        if verbose: cv.destroyAllWindows()
    
    return courtlineList

####### FIND NORMAL COURT CONFIGS #######

def get_normal_bool(courtlineList, normalized_deviation_threshold = 1, verbose = False):
    
    def court_points(courtLines):
        p1, p2 = courtLines['Top Baseline'].leftPt, courtLines['Top Baseline'].rightPt
        p3, p4 = courtLines['Bottom Baseline'].leftPt, courtLines['Bottom Baseline'].rightPt
        return [p1.coords, p2.coords, p3.coords, p4.coords]
    
    courtpoints = np.array([court_points(cLns) for frmNum, cLns, H in courtlineList])
    
    # avg_points = np.average(courtpoints, axis = 0)
    med_points = np.median(courtpoints, axis = 0)
    std_points = np.std(courtpoints, axis = 0)
    
    if verbose:
        # avgRegion = Region.from_points(*[Point(tuple(c)) for c in avg_points])
        medRegion = Region.from_points(*[Point(tuple(c)) for c in med_points])
        # avgFrm = fit_frame_to_lines(medRegion.transform('round_points'))
        medFrm = fit_frame_to_lines(medRegion.transform('round_points'))
        # avgFrm.show()
        medFrm.show()

    normalized_deviation = (courtpoints - med_points) / std_points
    within_bool = np.abs(normalized_deviation) < normalized_deviation_threshold
    all_corners_within = np.all(within_bool, axis = (1, 2))
    
    print(f'   Number of Court Lines Found:   {len(courtpoints)}')
    print(f'    ... of which judged normal:   {all_corners_within.sum()}')
    print(f'    ... of which judged abnormal: {(~all_corners_within).sum()}')

    return all_corners_within

def courtlines_to_schema(courtlineList, normalboolList, schemaFrm = schemaFrm, color_encoding = 'BGR'):
    ######### APPLY INVERSE H TO FIT FRAME COLORS RELATIVE TO COURTLINES BACK TO SCHEMA #######
    courtlines_to_check = len(courtlineList)
    wrpList = []
    
    if color_encoding == 'BGR':
        frmType = BGRFrame
    elif color_encoding == 'HSV':
        frmType = HSVFrame
    else:
        raise Exception('color_encoding: select one of BGR or HSV')
    
    for i in range(courtlines_to_check):
        
        if i % 100 == 0:
            print(f'   checked {i} of {courtlines_to_check} sets of court lines...')
        
        frmNum, aLns, H = courtlineList[i]
        normBool = normalboolList[i]
        
        if normBool:
            H_inv = np.linalg.inv(H)
            cFrm = frmType.from_video(cap, frmNum)
            # cFrm.show()
            cols_rows = (schemaFrm.cols, schemaFrm.rows)
            wrpArr = cv.warpPerspective(cFrm.array, H_inv, cols_rows)
            
            # wrpFrm = BGRFrame(wrpArr)
            # wrpFrm.show()
            wrpList.append(wrpArr)
    
    return np.stack(wrpList, axis = 0)

def get_median_and_deviation_frame(wrpArrs, max_arrays = 1000, verbose = False, color_encoding = 'BGR'):
    ## GET MEDIAN AND STD OF PIXEL COLORS REL. TO COURTLINES, ON THE SCHEMA ##
    
    if color_encoding == 'BGR':
        frmType = BGRFrame
    elif color_encoding == 'HSV':
        frmType = HSVFrame
    else:
        raise Exception('color_encoding: select one of BGR or HSV')
        
    ### prune away parts that are superfluous because they are completely dark
    ## in the unimaginable case that we have black rows within the picture, (not at the edges), 
    ## we'd be fucked.
    row_is_not_dark = wrpArrs.any(axis = (0, 2, 3))
    wrpArrs = wrpArrs[:, row_is_not_dark]
    
    ## due to high memory requirements for std, limit number of arrays
    wrpArrs = wrpArrs[:MAX_NUMBER_OF_ARRAYS]
    
    # avgArr = np.average(wrpArrs, axis = 0)
    medArr = np.median(wrpArrs, axis = 0)
    stdArr = np.std(wrpArrs, axis = 0) ### memory requirements for standard deviation are quite high.....
                                         # check task manager while it runs; RAM usage increases a lot.
    
    # avgFrm = BGRFrame(avgArr.astype('uint8'))
    medFrm = frmType(medArr.astype('uint8'))
    stdFrm = frmType(stdArr.astype('uint8'))
    
    if verbose:
        # avgFrm.show()
        medFrm.show() ### BEAUTIFUL! 
        stdFrm.show()
        
    return medFrm, stdFrm

def save_burn_in_data(medFrm, stdFrm, video_id, color_encoding = 'BGR'):
    ####### SAVE DATA ##############
    
    if color_encoding == 'BGR':
        colMap = {0: 'blue', 1: 'green', 2: 'red'}
    elif color_encoding == 'HSV':
        colMap = {0: 'hue', 1: 'saturation', 2: 'value'}
    else:
        raise Exception('color_encoding: select one of BGR or HSV')
    
    import os
    burnDict = atu.get_folder_struct('burn_in')
    savePath = os.path.join(burnDict['main'], video_id)
    
    try:
        os.mkdir(savePath)
    except FileExistsError:
        print('   Burn-In Data already exists... Overwriting Results...')
    
    def to_csv(frm, path, name = 'frame'):
        
        for idx, col in colMap.items():
            np.savetxt(os.path.join(path, f'{name}_{col}.csv'), frm.array[:, :, idx], delimiter = ',')
    
    # to_csv(avgFrm, savePath, name = 'average')
    to_csv(medFrm, savePath, name = 'median')
    to_csv(stdFrm, savePath, name = 'std')

#%% SCRIPT ####

### SET PARAMETERS ###
TARGET_COURTLINE_NUMBER = 3000
NORM_DEV_THRESHOLD = 1
MAX_NUMBER_OF_ARRAYS = 1000
COLOR_ENCODING = 'HSV'
verbose = True
######

##### VIDEO FILE READ IN ####
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
####################


if __name__ == '__main__':
    
    print('Collecting Courtlines ...')
    courtlineList = collect_courtlines(cap, target_number = TARGET_COURTLINE_NUMBER, 
                                       verbose = verbose)
    
    print('Separating Normal from Weird Courtline Configurations...')
    all_corners_within = get_normal_bool(courtlineList, 
                                         normalized_deviation_threshold = NORM_DEV_THRESHOLD, 
                                         verbose = verbose)
    
    print('Warping back courtlines to schema...')
    wrpArrs = courtlines_to_schema(courtlineList, all_corners_within, 
                                   color_encoding = COLOR_ENCODING)
    
    print('Calculating median and deviation pixel frames...')
    medFrm, stdFrm = get_median_and_deviation_frame(wrpArrs, 
                                                    max_arrays = MAX_NUMBER_OF_ARRAYS,
                                                    color_encoding = COLOR_ENCODING,
                                                    verbose = verbose)
    
    print('Saving...')
    save_burn_in_data(medFrm, stdFrm, VIDEO_ID, color_encoding = COLOR_ENCODING)

    print('Done!')




