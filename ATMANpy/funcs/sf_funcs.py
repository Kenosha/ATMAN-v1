# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:00:53 2023

Here we write functions here that take a stream feature and return a stream feature, 
if the feature in question lends itself to vectorization or other speed ups 
that call for treating all features as a whole, across frames 

@author: gianl
"""

import numpy as np

import ATMANpy.auxil.constants as const

from ATMANpy.types.Features import StreamFeatures, StreamFeature, BoolFeatures, BoolFeature
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Region
from ATMANpy.types.Contours import Contours

import ATMANpy.funcs.sf_apply_funcs as sf

from ATMANpy.types.Lines import CourtLines
schemaLines = CourtLines.make().add_singles_netline()

def get_redo_frames(mySF, redo_threshold = 24): 
    ### below this many consequent frames that were not found,
    ### we will retry all of them.
    
    def list_to_ranges(inList):
        outList = []
        left_elem = None
        last_elem = None
        for i in inList:
            if last_elem is None:
                last_elem = i
                left_elem = i
                continue
                
            if left_elem is None:
                left_elem = i
                continue
            
            if i == last_elem + 1:
                last_elem = i
                continue
            elif i > last_elem + 1:
                outList.append((left_elem, last_elem))
                left_elem = i
                last_elem = i
                continue
            
        return outList
            
    def flatten(list_of_lists):
        if list_of_lists == []:
            return []
        else:
            return [*list_of_lists[0], *flatten(list_of_lists[1:])]
    
    nfSF = mySF.filter(mySF['trackFlag'] == 'Not Found')
    nfList = nfSF.keys()
    
    contiguous = list_to_ranges(nfList)      
    to_retry = [list(range(start, end + 1)) for start, end in contiguous if end - start < redo_threshold]
    to_retry_flat = flatten(to_retry)
    
    streamDict = {frame: frame in to_retry_flat for frame in mySF.keys()}
    
    return BoolFeature(streamDict, 'to_retry')

def get_hole_frames(mySF):
    ### similar to get_redo_frames above    
    
    DISTANCE_THRESHOLD = 15
    inList = mySF.split()
    end_and_begin = [(max(sf_prev), min(sf_succ)) for sf_prev, sf_succ in zip(inList[:-1], inList[1:])]
    end_and_begin = [(end, beg) for end, beg in end_and_begin if beg - end < DISTANCE_THRESHOLD]
    
    to_retry = []
    for end, beg in end_and_begin:
        frames_missing_between = list(range(end + 1, beg))
        to_retry.extend(frames_missing_between)
        
    # streamDict = {frame: frame in to_retry for frame in mySF.keys()}
    streamDict = {frame: True for frame in to_retry}
    
    return BoolFeature(streamDict, name = 'to_retry')

def median_corner_regions(inSF, how_many_unilateral = 8):
    
    required_features = ['tl', 'tr', 'bl', 'br']
    mySF = inSF[required_features]
    myArr = mySF.apply(sf.points_to_coords).to_array()
    numFrames = mySF.length()
    
    ### Auxiliary function for finding outlier courtlines    
    def get_point_bounds(myArr, 
                         lower_q = 0.2, upper_q = 0.8, acceptable_multiple = 3,
                         min_box_diameter = 4):
        
        qs = [lower_q, 0.5, upper_q]
        loPts, mdPts, hiPts = np.round(np.quantile(myArr, qs, axis = 0)).astype('int')
    
        ### interquantile distance
        iq_up, iq_lo = hiPts - mdPts, mdPts - loPts
        
        ### lower bind our interquantile distances, so we don't have tiny boxes.
        iq_up = iq_up.clip(min = min_box_diameter // 2)
        iq_lo = iq_lo.clip(min = min_box_diameter // 2)
    
        lower_bound = mdPts - iq_lo * acceptable_multiple - 1 ### subtract/add one because checking whether
        upper_bound = mdPts + iq_up * acceptable_multiple + 1 ### a point is inside region is checked STRONGLY.
    
        return lower_bound, upper_bound
    
    ### Identify 'normal' corner regions and whether a corner point is an outlier
    dictList = []
    for idx, ptArr in enumerate(myArr):
        featDict = {}
        
        ### calculate quantile court mask
        # a) get the surrounding frames court corner points
        left_slice = max(0, idx - how_many_unilateral) ### frames before!
        right_slice = min(idx + how_many_unilateral + 1, numFrames) ### frames after
        sliceArr = myArr[left_slice:right_slice]
        
        ### establish lower and upper bounds based on IQR for each point
        lower_bounds, upper_bounds = get_point_bounds(sliceArr) ### shape of each: 4 x 2
        
        tl_lo, tr_lo, bl_lo, br_lo = lower_bounds
        tl_hi, tr_hi, bl_hi, br_hi = upper_bounds
        
        featDict['tlReg'] = Region.from_points(Point2D(tl_lo), Point2D(tl_hi))
        featDict['trReg'] = Region.from_points(Point2D(tr_lo), Point2D(tr_hi))
        featDict['blReg'] = Region.from_points(Point2D(bl_lo), Point2D(bl_hi))
        featDict['brReg'] = Region.from_points(Point2D(br_lo), Point2D(br_hi))
        
        # tl, tr, bl, br = ptArr
        
        # featDict['within_tlReg'] = Point2D(tl).within(featDict['tlReg'])
        # featDict['within_trReg'] = Point2D(tr).within(featDict['trReg'])
        # featDict['within_blReg'] = Point2D(bl).within(featDict['blReg'])
        # featDict['within_brReg'] = Point2D(br).within(featDict['brReg'])
    
        dictList.append(featDict)
    
    return StreamFeatures(dict(zip(mySF.get_frames(), dictList)), name = 'median_corner_regions')
    
def get_normal_bool(courtSF, normalized_deviation_threshold = 1, verbose = False):
    
    def court_points(courtLines):
        p1, p2 = courtLines['Top Baseline'].leftPt, courtLines['Top Baseline'].rightPt
        p3, p4 = courtLines['Bottom Baseline'].leftPt, courtLines['Bottom Baseline'].rightPt
        p1, p2, p3, p4 = p1.coords, p2.coords, p3.coords, p4.coords
        # return [p1.coords, p2.coords, p3.coords, p4.coords]
        return p1, p2, p3, p4
    
    pointSF = courtSF['courtLines'].apply(court_points)
    courtpoints = pointSF.to_array()
    
    med_points = np.median(courtpoints, axis = 0)
    std_points = np.std(courtpoints, axis = 0)
    
    normalized_deviation = (courtpoints - med_points) / std_points
    within_bool = np.abs(normalized_deviation) < normalized_deviation_threshold
    all_corners_within = np.all(within_bool, axis = (1, 2))
    
    if verbose:
        print(f'   Number of Court Lines Found:   {len(courtpoints)}')
        print(f'    ... of which judged normal:   {all_corners_within.sum()}')
        print(f'    ... of which judged abnormal: {(~all_corners_within).sum()}')

    boolSF = BoolFeature(dict(zip(courtSF.keys(), all_corners_within)), name = 'all_corners_within')
    return boolSF

def get_nonserve_bool(courtSF, first_seconds_to_disqualify = 5):
    frames = courtSF.get_frames()
    disq_num = 25 * first_seconds_to_disqualify
    disq_frames = frames[0: disq_num]
    
    boolDict = {frame: frame not in disq_frames for frame in courtSF.get_frames()}
    return BoolFeature(boolDict, name = f'not_in_first_{first_seconds_to_disqualify}_seconds')

def get_play_bool(courtSF, last_seconds_to_take = 3, min_length_of_sequence = 13):
    frames = courtSF.get_frames()
    take_num, length_num = 25 * last_seconds_to_take, 25 * min_length_of_sequence
    if len(frames) < length_num:
        take_frames = []
    else:
        take_frames = frames[-take_num:]
    
    boolDict = {frame: frame in take_frames for frame in frames}
    return BoolFeature(boolDict, name = f'last_{last_seconds_to_take}_seconds_of_long_sequence')

def get_median_and_deviation_frame(wrpSF):
    
    from ATMANpy.types.Frames import HSVFrame
    
    wrpArrs = np.array([frm.array for frm in wrpSF.values()])
    
    ### prune away parts that are superfluous because they are completely dark
    ## in the unimaginable case that we have black rows within the picture, (not at the edges), 
    ## we'd be fucked.
    row_is_not_dark = wrpArrs.any(axis = (0, 2, 3))
    wrpArrs = wrpArrs[:, row_is_not_dark]
    
    
    # avgArr = np.average(wrpArrs, axis = 0)
    medArr = np.median(wrpArrs, axis = 0)
    stdArr = np.std(wrpArrs, axis = 0) ### memory requirements for standard deviation are quite high.....
                                         # check task manager while it runs; RAM usage increases a lot.
    
    # avgFrm = BGRFrame(avgArr.astype('uint8'))
    medFrm = HSVFrame(medArr.astype('uint8'), name = 'median_backFrm')
    stdFrm = HSVFrame(stdArr.astype('uint8'), name = 'std_backFrm')
    
    return medFrm, stdFrm

def find_ball_candidates(contSF, svmDict = None, use_training_moments = False,
                         probability_threshold = None):
    ballTable = contSF['ballTable'].to_frame()
    
    ### get the SVM and the features used
    svm = svmDict['svm']
    features = svmDict['features']
    
    ### figure out normalising moments
    if use_training_moments:
        mean = svmDict['mean']
        std = svmDict['std']
    else:
        mean = ballTable.mean()
        std = ballTable.std()
        
    normTable = (ballTable - mean) / std
    
    ### Using the SVM estimated, predict whether a contour is a ball.
    normTable['isBall_proba'] = svm.predict_proba(normTable[features].values)[:,1]
    if probability_threshold is None:
        normTable['isBall'] = svm.predict(normTable[features].values)
    else:
        normTable['isBall'] = (normTable['isBall_proba'] > probability_threshold).astype(int)
    
    ### get all True predictions
    trueTable = normTable[normTable.isBall == 1]
    
    trueDict = {}
    for frmNum, cnts in contSF['movCnts'].items():
        try:
            myTable = trueTable.loc[frmNum]
        except KeyError:
            trueDict[frmNum] = None
        else:
            true_contours = []
            for true_idx in myTable.index:
                true_contours.append(cnts[true_idx])
            trueDict[frmNum] = Contours(true_contours)

    return StreamFeature(trueDict, name = 'candCnts')

def find_particles(mySF, 
                   dims = (1920, 1080), pixel_in_meter = (200, 100), N = 1000, fps = 25,
                   chamfer_threshold = None,
                   normal_propagation = None, close_propagation = None, switch_probability = None,):

    import ATMANpy.funcs.particle_funcs_vectorized as pf
    
    indexHistories = None
    closeHistories = None
    modelHistories = None
    particles = pf.initialize_particles(dims, pixel_in_meter = pixel_in_meter, N = N, fps = fps)
    
    partDict = {}
    for frame, featDict in mySF.items():
        ### write to frame!
        partDict[frame] = particles
        
        chamferCnts = featDict['chamferCnts']
        ballContours = featDict['candCnts']
        
        propagation_models, closeBools, modelBools = pf.get_motion_model_with_contours(particles, chamferCnts,
                                                                         closeHistories = closeHistories,
                                                                         modelHistories = modelHistories,
                                                                         normal_propagation = normal_propagation, 
                                                                         close_propagation = close_propagation, 
                                                                         switch_probability = switch_probability)    
    
        ########### UPDATE PARTICLES #########
        mDict, wDict = pf.get_means_and_weights(particles, propagation_models, ballContours) ### means_and_weights updates last periods particles, and compares to evidence in this period!
        particles, particleIdxs, _ = pf.resample(mDict, wDict)
        
        ########### UPDATE HISTORIES #########
        indexHistories, closeHistories, modelHistories = pf.update_histories(indexHistories, closeHistories, modelHistories,
                                                                             particleIdxs, closeBools, modelBools) 

    partDict[frame + 1] = particles
    
    return StreamFeature(partDict, name = 'particles')
    

