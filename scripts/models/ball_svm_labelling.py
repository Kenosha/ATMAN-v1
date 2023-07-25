# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 21:37:13 2023

@author: gianl
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 12:28:19 2023

@author: gianl
"""


import os
import time

import cv2 as cv
import numpy as np
import pandas as pd

### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video
from ATMANpy.funcs.cv_funcs import join_video_captures

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines, NetLines

import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCRandStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region, Regions
from ATMANpy.types.Contours import Contour, Contours

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

START_FRAME = 0
STOP_FRAME = 100000
VIDEO_ID = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
STORE_NAME = 'svm_run'
NUM_WORKERS = 24
CHUNKSIZE = 5000

cap = get_video(VIDEO_ID)

width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
NATURAL_SCALE = width / 1920
ADD_SCALE = 1 
SCALE_FACTOR = NATURAL_SCALE * ADD_SCALE

####### AUXILIARY FUNCTIONS ##################

        
def label_ball_candidate(clickPt, contourFrm: pd.DataFrame, max_range_threshold = 100):
    outFrm = contourFrm.copy()
    outFrm['isBall'] = 0

    if clickPt is None:
        print('No click detected, no ball.')
    else:
        dFrm = (contourFrm[['Cx', 'Cy']] - clickPt) ** 2
        distance = (dFrm['Cx'] + dFrm['Cy']) ** 0.5
        if distance.min() < max_range_threshold:
            identified_ball = distance.argmin()
            outFrm.loc[identified_ball, 'isBall'] = 1
            print(f'Contour Nr.{identified_ball} identified as ball.')
        else:
            print('Click was too far away, no ball.')
            
    return outFrm

def show_and_callback(showFrm1, showFrm2):
    global clickPt
    
    def select_point(event, x, y, flags, param):
        global clickPt
    	# grab references to the global variables	
    	# if the left mouse button was clicked, record the (x, y) coordinates
        if event == cv.EVENT_LBUTTONDOWN:
            clickPt = (x, y)
    
    # set up the window
    cv.namedWindow("image")
    cv.setMouseCallback("image", select_point)
    # display the image and wait for a keypress
    cv.imshow("image", showFrm1.array)
    key = cv.waitKey(0) & 0xFF
    
    if key == ord('s'):
        return show_and_callback(showFrm2, showFrm1)
    else:
        return key
    
    
#################################################


#### TODO: Important 
    ### -> Save contours instead of frame -> in case the algorithm for get_feature_frame changes, 
    ### we can always recreate the training data! 

if __name__ == '__main__':
    frameList = list(range(START_FRAME, STOP_FRAME + 1))
    
    cntSF = StreamFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['movCnts']).squeeze()
    cntStm = NCRandStream.from_SF(cntSF, cap, number_of_frames = 400)
           
    clickList = []
    frameList = []
    
    for frmNum, myFrm, myFeats in cntStm.generator():
        
        cnts = myFeats['movCnts']
        hFrm = myFrm.to_HSV()
        gFrm = myFrm.to_grey()
        
        cntFrm = ex.get_feature_frames(hFrm, gFrm, cnts)
    
        showFrm = myFrm.put(cnts, enumeration = True, thickness = 0).zoom(1 / SCALE_FACTOR)
        showFrm_with_contours = myFrm.put(cnts, enumeration = True, thickness = 1).zoom(1 / SCALE_FACTOR)
        
        clickPt = None
        key = show_and_callback(showFrm, showFrm_with_contours)
        
        # ######### CLICK STUFF ########
        # # reset the mouse click location
        # clickPt = None
        # # set up the window
        # cv.namedWindow("image")
        # cv.setMouseCallback("image", select_point)
        # # display the image and wait for a keypress
        # cv.imshow("image", showFrm.array)
        # key = cv.waitKey(0) & 0xFF
        # showing_contours = False
        
        # if key == ord('s'): ### switch between contours painted on or not.
        #     if showing_contours:
        #         cv.imshow("image", showFrm.array)
        #         showing_contours = False
        #     else:
        #         cv.imshow("image", showFrm_with_contours.array)
        #         showing_contours = True
        
        if key == ord('q'): ## press q to quit
            cv.destroyAllWindows()
            break
        
        if key == ord(' '): ## press confirm click and go to next picture
            clickPt = clickPt[0] * SCALE_FACTOR, clickPt[1] * SCALE_FACTOR
            clickList.append(clickPt)
            print(f'Logged {clickPt}')
            outFrm = label_ball_candidate(clickPt, cntFrm)
            
            outFrm = outFrm.reset_index().rename({'index': 'idx_within_image'}, axis = 1)
            outFrm['frameNum'] = frmNum
            
            frameList.append(outFrm)
        
        if key == ord('x'): ## press x to indicate there is nothing to click here
            pass
            # frameNum = randrange(0, total_frames - 1)
            # cFrm = BGRFrame.from_video(cap, frameNum)
        
            
    ##### CONCATENATE AND SAVE ######
    
    ballDict = ut.get_folder_struct('ball_svm')
    
    vidPath = os.path.join(ballDict['main'], VIDEO_ID)
    
    try:
        os.mkdir(vidPath)
    except FileExistsError:
        print('   Adding to existing ball_svm data')
        
    savePath = os.path.join(vidPath, 'raw_data')
    
    try:
        os.mkdir(savePath)
    except FileExistsError:
        print('   Adding to existing ball_svm data')
    
    import datetime as dt
    timestamp = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    filePath = os.path.join(savePath, f'labelled_balls_{timestamp}.csv')
    
    # frameList2 = [frm.reset_index().rename({'index': 'idx_within_image'}, axis = 1) for frm in frameList]
    bigFrm = pd.concat(frameList, ignore_index = True)    
    
    bigFrm.to_csv(filePath)   
        

    
    
    
