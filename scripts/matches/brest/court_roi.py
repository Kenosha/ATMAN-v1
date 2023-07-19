# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:45:37 2023

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

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, StreamList, MulStream, RandStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region, Regions

from ATMANpy.funcs.player_funcs import get_service_ROIs
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

START_FRAME = 0
STOP_FRAME = 10000
VIDEO_ID = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
STORE_NAME = 'test'
NUM_WORKERS = 24
CHUNKSIZE = 1000

cap = get_video(VIDEO_ID)

width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
NATURAL_SCALE = width / 1920
ADD_SCALE = 1 
SCALE_FACTOR = NATURAL_SCALE * ADD_SCALE

if __name__ == '__main__':

    ### Make folders for feature storage
    featPath = ut.get_folder_struct('features_stored')['main']
    feat_vidPath = os.path.join(featPath, VIDEO_ID)
    try:
        os.mkdir(feat_vidPath)
    except FileExistsError:
        pass
    
    storePath = os.path.join(feat_vidPath, STORE_NAME)
    try:
        os.mkdir(storePath)
    except FileExistsError:
        pass
    
    randStm = RandStream(cap, START_FRAME, STOP_FRAME, number_of_frames = 1000,
                         cap_color_space = 'bgr', target_color_space = 'grey',
                         scale_factor = ADD_SCALE)
    
    ### Set up a region of interest to focus attention on court. 
    # -> region of interest will be saved as a StillFrame of Mask type
    some_frame = 300
    someFrm = randStm[some_frame]
    # someFrm.show()
    
    p1, p2, p3, p4 = Point2D((200, 0)), Point2D((440, 0)), Point2D((0, 295)), Point2D((width, 295))
    ROI = Region.from_points(p1, p2, p3, p4)
    someFrm.put(ROI).show()
    
    savePath = ff.get_and_make_save_path('courtroiReg', VIDEO_ID, STORE_NAME)
    fileName = f'from_{START_FRAME}_to_{STOP_FRAME}.npy'
    fullPath = os.path.join(savePath, fileName)
    roiArr = ff.extract_region_values(ROI)
    np.save(fullPath, roiArr)
    
    #### Ball Kid ROIs
    myRegs = []
    currPts = []
    
    def draw_ballkid_rois(myFrm):
        imageArr = myFrm.array
        cloneArr = imageArr.copy()
        
        def click_and_crop(event, x, y, flags, param):
        	# grab references to the global variables
            global myRegs, currPts
            
            # if the left mouse button was clicked, 
            # record the starting (x, y) coordinates 
            if event == cv.EVENT_LBUTTONDOWN:
                currPts = [Point2D((x, y)).conform_to(myFrm)]
        	# check to see if the left mouse button was released
            elif event == cv.EVENT_LBUTTONUP:
                currPts.append(Point2D((x, y)).conform_to(myFrm))
                myRegs.append(Region.from_points(*currPts))

                cv.rectangle(imageArr, currPts[0].coords, currPts[1].coords, (0, 255, 0), 2)
                cv.imshow("image", imageArr)

                currPts = []
        
        ### Prepare CallBack
        cv.namedWindow('image')
        cv.setMouseCallback('image', click_and_crop)
        
        ### Show
        while True:
            cv.imshow('image', imageArr)
            
            key = cv.waitKey(1) & 0xFF
            
        	# if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                imageArr = cloneArr.copy()
            # if the 'q' key is pressed, exit
            elif key == ord("q"):
                cv.destroyAllWindows()
                break
            
        return Regions(myRegs)

    kidFrm = someFrm.to_BGR()
    kidRegs = draw_ballkid_rois(kidFrm)
    
    assert kidRegs.length() == 6
    
    fileName = f'from_{START_FRAME}_to_{STOP_FRAME}.npy'    
    savePath = ff.get_and_make_save_path(f'ballkidROIs', VIDEO_ID, STORE_NAME)
    fullPath = os.path.join(savePath, fileName)
    
    roiArr = ff.extract_regions_values(kidRegs)
    np.save(fullPath, roiArr)
    
    
    
    