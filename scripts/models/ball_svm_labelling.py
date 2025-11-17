# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 21:37:13 2023

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

conf = ut.read_config('brest')
vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
nwo, nch = conf['num_workers'], conf['chunksize']
cap = conf['cap']

NUM_FRAMES = 1000
    
####### AUXILIARY FUNCTIONS ##################

def label_ball_candidate(clickPt, cnts: Contours, max_range_threshold = 100):
    if clickPt is None:
        print('No click detected, no ball.')
    else:
        d = lambda aPt, bPt: ((aPt[0] - bPt[0]) ** 2 + (aPt[1] - bPt[1]) ** 2) ** 0.5
        distances = [d(centerPt.coords, clickPt) for centerPt in cnts.center()]
        if min(distances) < max_range_threshold:
            identified_ball = np.argmin(distances)
            idCnt = cnts[identified_ball]
            idCnts = Contours([idCnt]) ### to make it easier for us: turn into Contours of length 1
            otherCnts = cnts.drop(identified_ball)
            print(f'Contour Nr.{identified_ball} identified as ball.')
            return idCnts, otherCnts
        else:
            print('Click was too far away, no ball.')

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
    
##################

if __name__ == '__main__':
    frameList = list(range(conf['start_frame'], conf['stop_frame'] + 1))
    
    mySF = StreamFeatures.from_store(vid, sto, featnameList = ['large_dilation_movCnts', 'topPlayer', 'botPlayer']).squeeze()
    myStm = NCRandStream.from_SF(mySF, cap, number_of_frames = NUM_FRAMES)
           
    # clickList = []
    idDict = {} ### for ball contours
    otherDict = {} ### for non-ball contours
    
    for frmNum, myFrm, myFeats in myStm.generator():
        
        cnts = myFeats['large_dilation_movCnts']
        topPlayer, botPlayer = myFeats['topPlayer'], myFeats['botPlayer']
        hFrm = myFrm.to_HSV()
        gFrm = myFrm.to_grey()
        
        # cntFrm = ex.get_feature_frames(hFrm, gFrm, cnts)
    
        showFrm = myFrm.put(cnts, enumeration = True, thickness = 0)
        showFrm = showFrm.zoom(1 / conf['scale_factor'])
        
        showFrm_with_contours = myFrm.put(cnts, enumeration = True, thickness = 1)
        showFrm_with_contours = showFrm_with_contours.put(topPlayer, color = (255, 0, 0)).put(botPlayer, color = (255, 0, 0))
        showFrm_with_contours = showFrm_with_contours.zoom(1 / conf['scale_factor'])
        
        clickPt = None
        key = show_and_callback(showFrm, showFrm_with_contours)
        
        if key == ord('q'): ## press q to quit
            cv.destroyAllWindows()
            break
        
        if key == ord(' '): ## press confirm click and go to next picture
            clickPt = clickPt[0] * conf['scale_factor'], clickPt[1] * conf['scale_factor']
            # clickList.append(clickPt)
            print(f'Logged {clickPt}')
            
            try:
                idCnts, otherCnts = label_ball_candidate(clickPt, cnts)
            except:
                continue
            else:
                idDict[frmNum] = idCnts
                otherDict[frmNum] = otherCnts
                
        if key == ord('x'): ## press x to indicate there is nothing to click here
            pass
            # frameNum = randrange(0, total_frames - 1)
            # cFrm = BGRFrame.from_video(cap, frameNum)
        
    #### CREATE CONTOUR(S) STREAM FEATURES ####
    ballSF = StreamFeature(idDict, name = 'svm_labelled_ballCnts')
    otherSF = StreamFeature(otherDict, name = 'svm_labelled_nonballCnts')

    ### SAVE ###
    ballSF.save(vid, sto, compression = conf['compression_ball'])    
    otherSF.save(vid, sto, compression = conf['compression_ball'])    

        

    
    
    
