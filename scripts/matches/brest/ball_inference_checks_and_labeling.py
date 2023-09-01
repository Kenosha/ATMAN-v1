# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:26:01 2023

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

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region, Regions
from ATMANpy.types.Contours import Contour, Contours

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)


if __name__ == '__main__':

    conf = ut.read_config('brest')
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']

    frameList = list(range(conf['start_frame'], conf['stop_frame'] + 1))
    
    ### Load Courtlines ###
    H = ConstFeatures.from_store(vid, sto, featnameList = ['H'], frameList = frameList).squeeze()
    courtLines = sa.find_courtlines(H.feature)
    base_length = courtLines['Bottom Baseline'].length() ### take the larger of the two (bot side) so that our maximum ratio is larger!
    side_length = courtLines['Bottom Singles Ad Line'].length() + courtLines['Top Singles Deuce Line'].length()
    
    ### Load Candidate Contours ###
    cntSF = StreamFeatures.from_store(vid, sto, featnameList = ['candCnts']).squeeze()
    chamfSF = StreamFeatures.from_store(vid, sto, featnameList = ['chamferCnts'])
    
    ### Load Players
    playerSF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayer', 'botPlayer'])

    ### Load Chosen Contours ###
    choseSF = StreamFeatures.from_store(vid, sto, featnameList = ['ball_choicePt'])
    
    ### Load Interpolated Contours ###
    itplSF = StreamFeatures.from_store(vid, sto, featnameList = ['ball_itplPt'])
    
    ### Load Previously Deleted or Added Contours ###
    editSF = StreamFeatures.from_store(vid, sto, featnameList = ['addCnts', 'delCnts'])
    
    ### Load Play Segmenter
    playBF = BoolFeatures.from_store(vid, sto, featnameList = ['isPlay']).squeeze()
    
    ### Join SFs and Split
    allSF = playerSF.merge(cntSF, how = 'outer').merge(choseSF, how = 'outer').merge(itplSF, how = 'outer').merge(editSF, how = 'outer')
    allSFs = allSF.conform(playBF.filter(playBF)).split()
    
    
    #%% ### Check Sequences and Add/Remove Candidate Contours as Needed
    
    def pick_nearest_candidate(clickPt, candCnts, max_range_threshold = 100):
        if candCnts.is_empty():
            return None
        
        distances = [centerPt.distance_to(clickPt) for centerPt in candCnts.center()]
        if min(distances) < max_range_threshold:
            identified_ball = np.argmin(distances)
            idCnt = candCnts[identified_ball]
            print(f'Identified contour at {idCnt.center()} as nearest Contour!')
            return idCnt
        else:
            print('Click was too far away, no ball.')

    def show_and_callback(showFrm, zoom_factor = 3):
        global clickPts
        
        def select_point(event, x, y, flags, param):
            global clickPts
        	# grab references to the global variables	
        	# if the left mouse button was clicked, record the (x, y) coordinates
            if event == cv.EVENT_LBUTTONDOWN:
                clickPts.append(Point2D((x // zoom_factor, y // zoom_factor)))
        
        showFrm = showFrm.zoom(zoom_factor, interpolation = cv.INTER_CUBIC)

        # set up the window
        cv.namedWindow("image")
        cv.setMouseCallback("image", select_point)
        # display the image and wait for a keypress
        cv.imshow("image", showFrm.array)
        keypress = cv.waitKey(0) & 0xFF
        
        return keypress
        
    def record_edited_contours(mySF,
                               resolution = 6, zoom_factor = 3):
        global clickPts
        
        PLAYER_COLOR = const.BGR['marine_blue']
        CANDIDATE_COLOR = const.BGR['rose_orange']
        CHOICE_COLOR = const.BGR['fair_yellow']
        ITPL_COLOR = const.BGR['ball_yellow']
        
        DEL_COLOR = const.BGR['intense_red']
        ADD_COLOR = const.BGR['fair_yellow']
        
        start_frame, stop_frame = mySF.first()[0], mySF.last()[0]
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        
        recorded_event_dict = {}
        breakBool = False
        
        show_players = False
        show_candidates = False
        show_ball_choice = False
        show_ball_itpl = False
        
        toggle_add = True
        
        def empty_if_none(someCnts):
            if someCnts is None:
                return []
            else:
                return someCnts.contourList
            
        addDict = {frame: empty_if_none(myCnts) for frame, myCnts in mySF['addCnts'].items()}
        delDict = {frame: empty_if_none(myCnts) for frame, myCnts in mySF['delCnts'].items()}
        
        while cap.isOpened():
            frameNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            
            ### Collect Data
            featDict = mySF[frameNum]
            addCnts = addDict[frameNum]
            delCnts = delDict[frameNum]
            
            ### Create Base Frame
            showFrm = BGRFrame(frame)
            
            ### Add Features
            if show_players:
                showFrm = showFrm.put(featDict['topPlayer'], color = PLAYER_COLOR, thickness = 2).put(featDict['botPlayer'], color = PLAYER_COLOR, thickness = 2)
            if show_candidates:
                showFrm = showFrm.put(featDict['candCnts'], color = CANDIDATE_COLOR, thickness = 2)
            if show_ball_choice:
                showFrm = showFrm.put(featDict['ball_choicePt'], color = CHOICE_COLOR, thickness = 2)
            if show_ball_itpl:
                showFrm = showFrm.put(featDict['ball_itplPt'], color = ITPL_COLOR, thickness = 2)
            
            ### Add Markings for New/Deleted Contours...
            for addCnt in addCnts:
                if addCnt not in delCnts: ### we don't show deleted additions!
                    showFrm = showFrm.put(addCnt.center(), marker = cv.MARKER_TILTED_CROSS, color = ADD_COLOR, size = 6, thickness = 1)
            for delCnt in delCnts:
                if delCnt not in addCnts: ### we don't show deleted additions!
                    showFrm = showFrm.put(delCnt.center(), marker = cv.MARKER_TILTED_CROSS, color = DEL_COLOR, size = 6, thickness = 1)
                
            ### Add Toggle Text
            if toggle_add:
                showFrm = showFrm.put('Adding...')
            else:
                showFrm = showFrm.put('Deleting...')
        
            ### Show, register clicks, and collect key
            keypress = show_and_callback(showFrm, zoom_factor = zoom_factor)
        
            ### Deal with clickpoints, and empty them.
            if clickPts != []:
                if toggle_add:
                    for cPt in clickPts:
                        addCnts.append(Contour.from_point(cPt, radius = 1, resolution = resolution))
                else:
                    compareCnts = Contours([]) if featDict['candCnts'] is None else featDict['candCnts']
                    compareCnts = compareCnts.extend(Contours(addCnts)) ### extend by additional cnts, so we can remove these too!
                    
                    for cPt in clickPts:
                        nearCnt = pick_nearest_candidate(cPt, compareCnts)
                        if nearCnt is not None:
                            delCnts.append(nearCnt)
                clickPts = []
                    
            #### EXIT ALL
            if keypress == 27: ### ESC key!
                breakBool = True ### break the outer loop of all sequences
                break
        
            #### EXIT    
            if keypress == ord('q'): 
                break
            
            ### GO AHEAD
            if keypress == ord(' '):
                go_to_frame = min(frameNum + 1, stop_frame) ### can't go farther
                                                                ## than start_frame
                cap.set(cv.CAP_PROP_POS_FRAMES, go_to_frame) 
                continue
            
            ### REWIND
            if keypress == ord('b'): 
                back_to_frame = max(start_frame, frameNum - 1) ### can't go farther
                                                                ## than start_frame
                cap.set(cv.CAP_PROP_POS_FRAMES, back_to_frame) 
                continue
            
            ### RECORD
            if keypress == 13: ### return key
                recorded_event_dict[frameNum] = ''
               
            ### DELETE
            if keypress == 8: ### backspace key
                recorded_event_dict.pop(frameNum, None)
             
            ### TOGGLE DELETION/ADDITION
            if keypress == ord('d'):
                toggle_add = not toggle_add
            
            ### TOGGLE SHOW
            if keypress == ord('p'):
                show_players = not show_players
            if keypress == ord('o'):
                show_candidates = not show_candidates
            if keypress == ord('u'):
                show_ball_choice = not show_ball_choice
            if keypress == ord('i'):
                show_ball_itpl = not show_ball_itpl
             
            cap.set(cv.CAP_PROP_POS_FRAMES, frameNum) 

        # cap.release()
        cv.destroyAllWindows()
        
        ### turn addDict and delDict into SFs
        def none_if_empty(myCnts):
            if myCnts.is_empty():
                return None
            else:
                return myCnts
        
        addSF = StreamFeature({frame: none_if_empty(Contours(myCnts)) for frame, myCnts in addDict.items()}, name = 'addCnts')
        delSF = StreamFeature({frame: none_if_empty(Contours(myCnts)) for frame, myCnts in delDict.items()}, name = 'delCnts')
        
        return addSF, delSF, breakBool
    
    
    #######################
    #######################
    #######################
    
    ZOOM_FACTOR = 2 ## zoom before displaying
    RESOLUTION = 6 ### ball contours have this many points
    clickPts = []
    
    ### check how many have been done...
    add_and_del_are_empty = [mySF['addCnts'].dropna().is_empty() and mySF['delCnts'].dropna().is_empty() for mySF in allSFs]
    last_sf_done = 0
    for idx, emptyBool in enumerate(add_and_del_are_empty):
        if not emptyBool:
            last_sf_done = idx

    for mySF in allSFs[last_sf_done + 1:]:
        addSF, delSF, breakBool = record_edited_contours(mySF, zoom_factor = ZOOM_FACTOR, resolution = RESOLUTION)
        
        addSF.save(vid, sto, compression = RESOLUTION)
        delSF.save(vid, sto, compression = conf['compression_ball'])
        
        if breakBool:
            break
        
    
    
    
    