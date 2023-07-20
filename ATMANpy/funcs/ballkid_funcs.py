# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 22:34:24 2023

@author: gianl
"""

import pandas as pd
import numpy as np

import ATMANpy.funcs.util_funcs as ut

from ATMANpy.types.Features import StreamFeatures, BoolFeature

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

def prop_sf_to_dataframe(intSF, fill_limit = 3):
    intDF = intSF.to_dataframe()
    
    ####### Work with DataFrame to find blocks of movement
    ### interpolate over small range, without extrapolation.
    intDF = ut.fill_with_hard_limit(intDF, fill_limit, limit_area = 'inside')
    
    ### find areas with incidences of short movement (= noise)
    nanDF = intDF.isna().replace(False, np.nan).replace(True, 1)
    chkDF = ut.fill_with_hard_limit(nanDF, fill_limit, limit_area = 'inside')
    noiseDF = nanDF.fillna(0) != chkDF.fillna(0)
    intDF = intDF[~noiseDF]

    ### since the shadow mask does not extend to first and last frame (TODO: ... fix this!)
    # fill the first and last entries with second and penultimate, resp.
    intDF.iloc[0] = intDF.iloc[1]
    intDF.iloc[-1] = intDF.iloc[-2]

    return intDF

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
    
    evalFrm = pd.concat(serList, axis = 1)
    return evalFrm

def evaluate_relevance(blockDF, threshold_length = 12,
                       both_same_side = 1, both_mid = 0.8, both_different = 0.5):
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
            width_score = both_same_side
        elif a_side == b_side == 'Mid':
            width_score = both_mid ### unlikely that both players are on opposite sides of the court laterally, but possible in principle
        else:
            width_score = both_different
    
    ### now, deal with length
    norm_length = length / threshold_length
    length_score = 1 / (1 + np.exp(-norm_length + 1))
    
    return width_score * length_score
    

def segment_play(strDFs, 
                 distance_adjacent = 25, distance_near = 100,
                 relevance_irrelevant = 0.25, relevance_conditional = 0.8):
    """
    The Big One: 
        Looks at block dataframes with string data describing motion of contours in ROIs and joins them,
        to produce a list of lists. These lists contain blocks, each list constituting a single interruption of play.
    
    
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
    
    Questions: What to do when resolution carries over? 
        Say we have two ROIs in need of resolution; next block resolves one; block after that resolves the second.
        --> need to remember open resolutions!
    
    Idea: Only flush to outList whenever we are about to begin a new segment, and current_segment is not empty.
    
    """
    
    def require_resolution(blockDF):
        ### list of ROIs in which a contour begins moving, but does not come to rest.
        resolutionDF = evaluate_resolution(blockDF)
        endSer = resolutionDF.loc['ends_in']
        resList = endSer[~endSer].index.tolist()
        return resList
    
    def offer_resolution(blockDF):
        ### right now, basically the inverse of require resolution: list of ROIs with movement, in which a contour comes to rest.
        resolutionDF = evaluate_resolution(blockDF)
        endSer = resolutionDF.loc['ends_in']
        resList = endSer[endSer].index.tolist()
        return resList
    
    def spawns_exclusively(blockDF):
        resolutionDF = evaluate_resolution(blockDF)
        all_begin = resolutionDF.loc['begins_in'].all()
        return all_begin
    
    def distance(blockA, blockB):
        return blockB.index.min() - blockA.index.max() 
    
    ### Initialize Result List
    outList = []
    ### Initialize Dynamic Variables
    prevDF = None
    current_segment = []
    open_resolutions = []
    
    for currDF in strDFs:
        
        ### initialize some flags for the current dataframe ###
        spawns = spawns_exclusively(currDF)
        relevance = evaluate_relevance(currDF)
        required_resolutions = require_resolution(currDF)
        offered_resolutions = offer_resolution(currDF)
        
        useful_resolutions = [roi for roi in offered_resolutions if roi in open_resolutions]
        residual_resolutions = [roi for roi in open_resolutions if roi not in useful_resolutions]
        
        ### decision tree ###
        
        # a) the previous segment is finished!
        if current_segment == []: 
            if spawns:
                if relevance > relevance_irrelevant:
                    current_segment = [currDF]
                    prevDF = currDF.copy()
                    open_resolutions = required_resolutions
            else:
                if relevance > relevance_conditional:
                    current_segment = [currDF]
                    prevDF = currDF.copy()
                    open_resolutions = required_resolutions

        # b) the previous segment is not finished, and requires resolution
        elif len(open_resolutions) > 0:
            if len(useful_resolutions) > 0: ### (partial) congruence!
                if distance(prevDF, currDF) < distance_near:
                    current_segment.append(currDF)
                    prevDF = currDF.copy()
                    open_resolutions = list(set(residual_resolutions + required_resolutions)) ### remaining resolutions, plus whatever comes up new
                else:
                    if spawns:
                        if relevance > relevance_irrelevant:
                            outList.append(current_segment)
                            current_segment = [currDF]
                            prevDF = currDF.copy()
                            open_resolutions = required_resolutions
                    else:
                        if relevance > relevance_conditional:
                            outList.append(current_segment)
                            current_segment = [currDF]
                            prevDF = currDF.copy()
                            open_resolutions = required_resolutions
            else:
                if distance(prevDF, currDF) < distance_adjacent:
                    current_segment.append(currDF)
                    prevDF = currDF.copy()
                    open_resolutions = list(set(residual_resolutions + required_resolutions)) ### remaining resolutions, plus whatever comes up new
                else:
                    if spawns:
                        if relevance > relevance_irrelevant:
                            outList.append(current_segment)
                            current_segment = [currDF]
                            prevDF = currDF.copy()
                            open_resolutions = required_resolutions
                    else:
                        if relevance > relevance_conditional:
                            outList.append(current_segment)
                            current_segment = [currDF]
                            prevDF = currDF.copy()
                            open_resolutions = required_resolutions
                    
        else: ### the previous segment is not finished, but there are no open resolutions!
            if distance(prevDF, currDF) < distance_adjacent:
                if relevance > relevance_irrelevant:
                    current_segment.append(currDF)
                    prevDF = currDF.copy()
                    open_resolutions = required_resolutions
      
            else:
                if spawns:
                    if relevance > relevance_irrelevant:
                        outList.append(current_segment)
                        current_segment = [currDF]
                        prevDF = currDF.copy()
                        open_resolutions = required_resolutions
                else:
                    if relevance > relevance_conditional:
                        outList.append(current_segment)
                        current_segment = [currDF]
                        prevDF = currDF.copy()
                        open_resolutions = required_resolutions
            
    ### append one last time
    if not current_segment == []:
        outList.append(current_segment)
     
    return outList


def outlist_to_bool(outList):
    ##### Now, join indices of elements of outlist, to find a segmentation
    ballkid_frames = []
    for subList in outList:
        start, stop = subList[0].index.min(), subList[-1].index.max()
        ballkid_frames.extend(list(range(start, stop + 1)))
    
    return BoolFeature({frame: True for frame in ballkid_frames}, name = 'ballkidBool')