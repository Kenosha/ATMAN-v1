# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 21:05:27 2023

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

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region, Regions
from ATMANpy.types.Contours import Contour

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

START_FRAME = 0
STOP_FRAME = 10000
VIDEO_ID = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
STORE_NAME = 'test'
NUM_WORKERS = 24
CHUNKSIZE = 5000

cap = get_video(VIDEO_ID)

width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
NATURAL_SCALE = width / 1920
ADD_SCALE = 1 
SCALE_FACTOR = NATURAL_SCALE * ADD_SCALE

if __name__ == '__main__':
    frameList = list(range(START_FRAME, STOP_FRAME + 1))
    
    ### Read constant Homography and Projection Matrix
    HPSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['H', 'P'], frameList = list(range(START_FRAME, STOP_FRAME + 1)))
    courtSF = HPSF['H'].apply(sa.find_courtlines)

    ### Read Segmentation Boolean
    courtBF = BoolFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['isCourt']).squeeze()    
    
    ### Read ROIs
    courtroiSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['courtroiReg'], 
                                          featTypeList = [Region], frameList = frameList).squeeze()
    courtROI = courtroiSF.feature
    
    kidroiSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['ballkidROIs'], 
                                         featTypeList = [Regions], frameList = frameList).squeeze()
    kidROIs = kidroiSF.feature
    
    ### Label the ROIs depending on position
    namedROIs = []
    cntPnts = kidROIs.centroids()
    
    court = courtSF.feature
    ordered_labelDict = {'Mid Left': court['Net Line'].leftPt,
                         'Mid Right': court['Net Line'].rightPt,
                         'Top Left': court['Top Baseline'].leftPt,
                         'Top Right': court['Top Baseline'].rightPt,
                         'Bot Left': court['Bottom Baseline'].leftPt,
                         'Bot Right': court['Bottom Baseline'].rightPt}
    for label, myPt in ordered_labelDict.items():
        nearest_idx = np.argmin([cntPt.distance_to(myPt) for cntPt in cntPnts])
        nearest_centroid = cntPnts[nearest_idx]
        nearest_region = [roi for roi in kidROIs if roi.centroid() == nearest_centroid][0]
        namedROIs.append(nearest_region.rename(label))
        cntPnts.pop(nearest_idx)
        
    namedRois = Regions(namedROIs)
        
    ### Read Relevant Masks
    burnStms = StreamList.from_store('foregroundMask', VIDEO_ID, STORE_NAME, cap_color_space = 'bool', target_color_space = 'mask')
    shadStms = StreamList.from_store('shadowMask', VIDEO_ID, STORE_NAME, cap_color_space = 'bool', target_color_space = 'mask') 
    motiStms = StreamList.from_store('rough_motMask', VIDEO_ID, STORE_NAME, cap_color_space = 'bool', target_color_space = 'mask') 
    
    ### Find motion contours
    params = {'min_area_of_contours': 10,
              'ROI': None, ### 6x slower when specifying a ROI!
              'max_number_of_contours': None}
    cntSFs = motiStms.rename('intMask').extract_chunked_mp(ex.get_contours_from_mask, NUM_WORKERS, CHUNKSIZE, **params)
    
    ### Find ROI Intersection Proportions
    
    motion_contour_SF = cntSFs[0]
    def contour_proportion_in_ROI(motion_contour_SF, kidROIs):
        streamDict = {frame: {} for frame in motion_contour_SF.get_frames()}
        
        for idx, kidROI in enumerate(kidROIs):
            for frame, motCnts in motion_contour_SF.items():
                
                motRegs = motCnts.enclosing_regions()
                propList = []
                for motReg in motRegs:
                    intReg = kidROI.intersection(motReg)
                    if intReg is not None:
                        subset_proportion = intReg.area() / motReg.area() ### two what extent is the motion region within the ROI?
                        propList.append(subset_proportion)
                max_prop = None if propList == [] else max(propList)
                streamDict[frame].update({kidROI.name: max_prop})
        
        return StreamFeatures(streamDict)
    
    intSF = contour_proportion_in_ROI(cntSFs[0], namedRois)
    
    intDF = intSF.to_dataframe()
    
    ####### Work with DataFrame to find blocks of movement
    fill_limit = 3
    
    ### interpolate over small range, without extrapolation.
    intDF = ut.fill_with_hard_limit(intDF, fill_limit, limit_area = 'inside')
    
    ### find areas with incidences of short movement (= noise)
    nanDF = intDF.isna().replace(False, np.nan).replace(True, 1)
    chkDF = ut.fill_with_hard_limit(nanDF, fill_limit, limit_area = 'inside')
    noiseDF = nanDF.fillna(0) != chkDF.fillna(0)
    intDF = intDF[~noiseDF]
    
    ### Make sense of the proportions -> are we moving out of a ROI, or moving into, or moving around in?
    ## -> ballkid sections begin with ballkids moving out of a ROI, or moving around in.
    ## -> ballkid sections end with ballkids moving back into a ROI and stopping, or moving around in, and stopping.
    motSer = ut.split(intDF['Mid Left'].dropna())[0]
    def understand_motion(motSer, threshold_length = 12, 
                          I = 0.95, ### minimum proportion needed for a contour to be judged inside of ROI
                          O = 0.5, ### maximum proportion that a contour can have to be judged outside of ROI
                          D = 0.02, ### typical change in intensity needed per frame to assess that a contour is moving into or out of.
                          ):
        long_length = min(len(motSer), threshold_length) ### used for checking change in contour overlap proportion
        short_length = round(long_length / 2) ### used for checking the immediate beginning
        
        ### understand if a contour spawned inside of ROI, or came from outside, or unclear
        spawn_inside = motSer[:short_length].median() > I
        from_outside = (motSer[:short_length].median() < O) & (motSer[:long_length].diff().median() > D) ### low initial value, and increasing
        ### understand if a contour ends inside, or moves out of.
        end_inside = (motSer[-short_length:].median() > I)
        moves_outside = (motSer[-short_length:].median() < O) & (motSer[-long_length:].diff().median() < -D)
        
        if spawn_inside and moves_outside:
            return 'Motion out of'
        if spawn_inside and end_inside:
            return 'Motion around in'
        if from_outside and moves_outside:
            return 'Motion through'
        if from_outside and end_inside:
            return 'Motion into'
        else:
            return 'Motion unclear'
            
    def prop_frame_to_motion(intDF):
        outList = []
        for col in intDF:
            intSers = ut.split(intDF[col].dropna())
            strSers = []
            for intSer in intSers:
                strSer = pd.Series(data = understand_motion(intSer), index = intSer.index, name = intSer.name)
                strSers.append(strSer)
            outSer = pd.concat(strSers, axis = 0)
            outList.append(outSer)
        return pd.concat(outList, axis = 1).reindex(intDF.index)
    
    strDF = prop_frame_to_motion(intDF)
    
    ### remove nan rows and split into contiguous chunks
    strDF = strDF.dropna(axis = 0, how = 'all')
    strDFs = ut.split(strDF)
            
    ### Evaluate blocks: 
        # are they potential beginnings of a ballkid sequence (sudden movement out of, or around in, ROIs?, without initial movement into.)
        # are they potential endings of a ballkid sequence (movement back into, or around in, ROIs; without movement out of.)
    blockDF = strDFs[0]
    
    def evaluate_resolution(blockDF):
        """
        blockDF : A DataFrame of contiguous movement strings
        
        This function evaluates a block as to its meaning:
            # is blockDF a potential beginning of a ballkid sequence (sudden movement out of, or around in, ROIs?, with no or little initial movement into.)
            # is blockDF a potential ending of a ballkid sequence (movement back into, or around in, ROIs; with no or little movement out of.)
        
        The big idea is: We only need to keep track of what happens 
            # at the very beginning (who spawns in a ROI), 
            # and the end (which ROI is resolved, in the sense that a kid is back inside the ROI)
        """

        serList = []
        protDF = blockDF.dropna(axis = 1, how = 'all') ### protagonist ROIs -> drop those where nothing happens
        
        for col in protDF:
            protSers = ut.split(protDF[col].dropna()) ### split each protagonist ROI into blocks of contiguous movement
            initSer, endSer = protSers[0], protSers[-1]
            init_movement, end_movement = initSer.unique()[0], endSer.unique()[0]
            
            begins_in = init_movement in ['Motion out of', 'Motion around in']
            ends_in = end_movement in ['Motion around in', 'Motion into']
            
            colSer = pd.Series({'begins_in': begins_in, 'ends_in': ends_in}, name = col)
            serList.append(colSer)
        
        evalFrm = pd.concat(serList, axis = 1).astype(int)
        diffFrm = evalFrm.loc['ends_in'] - evalFrm.loc['begins_in']               
        diffFrm = diffFrm.replace({1: 'resolves', 0: 'is resolved', -1: 'needs resolving'})
        return diffFrm
        
    evaluate_resolution(blockDF)
    
    def evaluate_relevance(blockDF, threshold_length = 12):
        """
        blockDF : A DataFrame of contiguous movement strings
        threshold_length -> the length in frames at which the length score is 0.5
        
        Evaluates the relevance of this block -> how likely does it represent ballkid motion, versus erratic that do not matter?
        A function of length and width of the block:
            # if it is short, then likely the motion is just random movement, potentially adjustments to posture during play
            # if it is limited to one ROI, the same holds. For two ROIs, we have to distinguish:
                # two ROIs on the same side are very relevant, because at least one has to be a ballkid -> only one player per side
                # two ROIs on opposite sides are potentially both players in ROIs simultaneously. Very unlikely, but possible.
        """
        
        protDF = blockDF.dropna(axis = 1, how = 'all') ### protagonist ROIs -> drop those where nothing happens
        length, width = protDF.shape
        
        ### first, deal with width
        if width <= 1:
            width_score = 0
        elif width > 2: 
            width_score = 1
        else: ### width == 2 -> here, we have to discriminate!
            aROI, bROI = protDF.columns
            a_side, b_side = aROI.split(' ')[0], bROI.split(' ')[0] ### top, mid or bottom
            if (a_side == b_side == 'Top') or (a_side == b_side == 'Bottom'):
                width_score = 1
            elif a_side == b_side == 'Mid':
                width_score = 0.8 ### unlikely that both players are on opposite sides of the court laterally, but possible in principle
            else:
                width_score = 0.5
        
        ### now, deal with length
        norm_length = length / threshold_length
        length_score = 1 / (1 + np.exp(-norm_length + 1))
        
        return width_score * length_score
    
    
    evaluate_relevance(blockDF)     
        
        

        
        
    def segment_play(strDFs):
        """
        Relevance Levels: Unequivocal (clearly a ballkid block), Conditional (potentially a ballkid block), Irrelevant (not a ballkid block)
        Distance Levels: Distant (between is clearly play), Near (between might be play), Adjacent (between is never play)
        
        if the current segment is empty:
            if curr_block spawns exclusively:
                if relevance >= conditional:
                    --> current segment = [block]
                else:
                    --> current segment = []
            else:
                if relevance >= unequivocal:
                    --> current segment = [block]
                else:
                    --> current segment = []
        else:
            if prev_block requires resolution:
                if curr_block offers congruent resolution:
                    if distance <= near:
                        --> current segment.append(block)
                    else:
                        --> outList.append(current_segment)
                        --> current_segment = []
                        --> proceed as if current_segment is empty (see above)
                else:
                    if distance <= adjacent:
                        --> current segment.append(block)
                    else:
                        --> outList.append(current_segment)
                        --> current_segment = []
                        --> proceed as if current_segment is empty (see above)
            else:
                if distance <= adjacent:
                    if relevance >= unequivocal:
                        --> current_segment.append(block)
                    else:
                        --> outList.append(current_segment)
                        --> current_segment = []
                        --> proceed as if current_segment is empty (see above)
                else:
                    --> outList.append(current_segment)
                    --> current_segment = []
                    --> proceed as if current_segment is empty (see above)
        """
        
        ballkid_segments = []
        prevDF, currDFs = strDFs[0], strDFs[1:]
        
        if len(prevDF) > discard_threshold:
            prev_eval = evaluate_block(prevDF)
            need_resolving = prev_eval[prev_eval == 'needs resolving'].index
            offer_resolving = prev_eval[prev_eval == 'resolves'].index
            current_segment = prevDF.index.tolist()
        else:
            need_resolving = []
            offer_resolving = []
            current_segment = []
            
        
        for currDF in currDFs:
            curr_eval = evaluate_block(currDF)
            
            if need_resolving == []:
                ### the previous segment 
            
            need_resolving = curr_eval[curr_eval == 'needs resolving'].index
            offer_resolving = curr_eval[curr_eval == 'resolves'].index
            
            statcurr_eval.loc['resolve'] - curr_eval.loc['spawn']
            
            if needs_resolution:
    
    
    [len(df) for df in strDFs]
    strDFs[1]
    
    motSer.transform(understand_motion)
    understand_motion(motSer)
            
           
        
    intDF
    
    
    ### do we join blocks that are "close enough?"
    
    ### construct a boolean -> does a block represent ballkid motion?
    # -> it might, if there are enough blocks with registered movement.
    
    [~df.isna().sum()]
    
    ### TODO:
        # interpolate the proportions
        # count them in a general region
        # threshold, to find if play is happening (noting that movement in up to two opposite ballkid regions can happen if players move into them!)
    
    
    Stream(cap, 350, 600).show(ms_between_frames = 50, frame_counter = True)
    