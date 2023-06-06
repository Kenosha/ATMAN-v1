# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:17:12 2021

Utility Functions

@author: gianl
"""

import numpy as np
import cv2 as cv
import ATMANpy.funcs.util_funcs as atu

def show(frame, name = 'frame'):
    cv.imshow(name, frame)
    cv.waitKey(0)
    #closing all open windows  
    cv.destroyAllWindows()  
    
def show_continuous(frame, name = 'frame', ms_between_frames = 30):
    cv.imshow('frame', frame)
    if cv.waitKey(ms_between_frames) & 0xFF == ord('q'):
        cv.destroyAllWindows()  
        return False ### needed for signalling a possible break condition
    return True

def get_frame(cap, frame_num):
    
    total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)  ### total number of frames
    
    if frame_num < 0 or frame_num > total_frames:
        print(f'No such frame, choose number between 0 and {total_frames}')
        return None
    
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    return frame

def get_frames(cap, start_frame, stop_frame):
    
    outFrames = []
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    frmNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    
    while cap.isOpened() and (start_frame <= frmNum <= stop_frame):
        ret, frame = cap.read()
        
        # if color_space == 'hsv':
        #     frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # if color_space == 'grey':
        #     frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
        outFrames.append(frame)
        
        frmNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        
    return np.array(outFrames)

def get_framelist(cap, frameList):
    
    outFrames = []
    
    for frmNum in frameList:
        cap.set(cv.CAP_PROP_POS_FRAMES, frmNum)
        ret, frame = cap.read()
        
        # if color_space == 'hsv':
        #     frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # if color_space == 'grey':
        #     frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
        outFrames.append(frame)
        
    return np.array(outFrames)

def show_video(cap, start_frame = None, stop_frame = None, ms_between_frames = None,
               frame_counter = False):
       
    if ms_between_frames is None:
        ms_between_frames = 25
    if start_frame is None:
        start_frame = 0
    if stop_frame is None:
        stop_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
    
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    frame_num = start_frame
    
    while cap.isOpened() and (start_frame <= frame_num <= stop_frame):
        ret, frame = cap.read()
        
        if frame_counter:
            cv.putText(frame, str(frame_num), org = (40, 40), 
                       fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 2,
                       color = (0, 255, 0), thickness = 3)
        
        if ret:
            cv.imshow('frame', frame)
            frame_num += 1
            if cv.waitKey(ms_between_frames) & 0xFF == ord('q'):
                break
        else:
            break
        
    # cap.release()
    cv.destroyAllWindows()
    
def video_stream(cap, start_frame = None, stop_frame = None, ms_between_frames = None):
    if ms_between_frames is None:
        ms_between_frames = 25
    if start_frame is None:
        start_frame = 0
    if stop_frame is None:
        stop_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
    
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    frmNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    
    while cap.isOpened() and (start_frame <= frmNum <= stop_frame):
        ret, frame = cap.read()
        
        frame = yield frmNum, frame # take in frame, possibly altered, to show
        # print(frame)
        
        if frame is not None:
            cv.imshow('frame', frame)
            if cv.waitKey(ms_between_frames) & 0xFF == ord('q'):
                cv.destroyAllWindows()  
                yield None
                
        frmNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    
    else:
        cv.destroyAllWindows()
                
       

def random_or_next_stream(cap, ms_between_frames = None, verbose = False):
    from random import randrange
    
    if ms_between_frames is None:
        ms_between_frames = 25
    total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
        
    doFlag = 'random' ### this flag signals to the generator what to do.
                        # it takes three values. 'random', 'next', 'stop'
                   
    while doFlag != 'stop':
        
        if doFlag == 'random':
            
            randFrame = randrange(0, total_frames - 1)
            cap.set(cv.CAP_PROP_POS_FRAMES, randFrame)
            ret, frame = cap.read()
            
            doFlag, frame = yield randFrame, frame 
        
        elif doFlag == 'next':
            
            frmNum = cap.get(cv.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
            
            doFlag, frame = yield frmNum, frame
        
        if verbose:
            cv.imshow('frame', frame)
            if cv.waitKey(ms_between_frames) & 0xFF == ord('q'):
                cv.destroyAllWindows()  


def get_stream_features_from_generator(gen,
                                       existing_features = None,
                                       cache_length = None,
                                       init_feat_func = None, init_frame_func = None,
                                       cached_line_func = None,
                                       court_line_func = None, court_feature_func = None,
                                       draw_func = None,
                                       features_of_interest = None,
                                       verbose = True):
    
     ### CACHING ###
    if cache_length is None:
        cache_length = 1 ### cache holds only the current frame.
    curr_frame_idx = cache_length // 2 ### the offset with which features are entered.
                                        ## note that due to caching of past and future frames, 
                                        ## the video generator is ahead by curr_frame_idx frames, 
                                        ## so we adjust frmNum (which comes from the video generator) 
                                        ## by that amount curr_frame_idx.
    
    ### init outputList
    streamFeatures = {}
    
    #### init feature dictionary
    featDict = None
    
    #### init cache ###
    frmCache = []
    
    from ATMANpy.types.Frames import BGRFrame
    from copy import deepcopy
    
    frmNum, rawArr = next(gen) ### this is needed to "prime" the generator
    ######
    
    while True:
        
        ### prepare and feed the cache ###
        # cache frames, so we can frame difference 
        # without having to re-generate the frames from the video capture
        latestFrame = BGRFrame(rawArr)    
        frmCache.append(latestFrame)
        if len(frmCache) < cache_length:
            frmNum, rawArr = gen.send(None)
            continue
        if len(frmCache) > cache_length:
            print(f'Cache FUBAR, {len(frmCache)} items') ### this happens if the cache drop at the end fails
            break
        
        ### prepare frame objects of interest
        frmDict = init_frame_func(frmCache)
        frmNum_on_display = frmNum - curr_frame_idx
        
        ### initialize the feature dictionary with possible stored features
        if existing_features is not None:
            try:
                stored_featDict = existing_features[frmNum_on_display]
            except KeyError:
                stored_featDict = None
        else:
            stored_featDict = None
            
        featDict = init_feat_func(current_featDict = featDict, 
                                  stored_featDict = stored_featDict)
        
        ### Get the features of interest on the frame
        if not court_line_func is None:
            featDict = court_line_func(frmDict, featDict)
        
        if not court_feature_func is None:
            featDict = court_feature_func(frmDict, featDict)
            
            
        ### extract out stream features of interest
        if features_of_interest is None:
            streamFeatures[frmNum_on_display] = deepcopy(featDict)
        else:
            saveDict = {k: v for k, v in featDict.items() if k in features_of_interest}
            streamFeatures[frmNum_on_display] = deepcopy(saveDict)
    
        if verbose:
            ### Send Frame Back to Generator
            sendFrm = draw_func(frmDict, featDict, frameNum = frmNum_on_display)
            sendArr = sendFrm.array  
        else:
            sendArr = None
            
        try:
            frmNum, rawArr = gen.send(sendArr)
        except:
            cv.destroyAllWindows()
            break 
        
        ### drop oldest element from cache ### 
        frmCache = frmCache[1:] # drop oldest item
        
    else:
        cv.destroyAllWindows()

    return streamFeatures
    
def get_stream_features(cap, start_frame, stop_frame, 
                        existing_features = None,
                        cache_length = None,
                        init_feat_func = None, init_frame_func = None,
                        cached_line_func = None,
                        court_line_func = None, court_feature_func = None,
                        draw_func = None,
                        features_of_interest = None,
                        verbose = True):
    
    if isinstance(cap, str):
        cap = atu.get_video(cap) ### this is needed for multiprocessing
            # video captures can not be pickled, therefore, we create it inside of the function
     
    from copy import deepcopy
    from ATMANpy.types.Frames import BGRFrame
    
    ### CACHING ###
    if cache_length is None:
        cache_length = 1 ### cache holds only the current frame.
    curr_frame_idx = cache_length // 2 ### the offset with which features are entered.
                                        ## note that due to caching of past and future frames, 
                                        ## the video generator is ahead by curr_frame_idx frames, 
                                        ## so we adjust frmNum (which comes from the video generator) 
                                        ## by that amount curr_frame_idx.
    
    if start_frame is not None:
        start_frame_with_caching = start_frame - curr_frame_idx
    else:
        start_frame_with_caching = 0
        
    if stop_frame is not None:
        stop_frame_with_caching = stop_frame + curr_frame_idx
    else:
        stop_frame_with_caching = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    ### init outputList
    streamFeatures = {}
    
    #### init feature dictionary
    featDict = None
    
    #### init cache ###
    frmCache = []
    
    ### init video stream ###
    from ATMANpy.funcs.cv_funcs import video_stream
    gen = video_stream(cap, 
                       start_frame = start_frame_with_caching, 
                       stop_frame = stop_frame_with_caching)
    frmNum, rawArr = next(gen) ### this is needed to "prime" the generator
    ######
    
    while True:
        
        ### prepare and feed the cache ###
        # cache frames, so we can frame difference 
        # without having to re-generate the frames from the video capture
        latestFrame = BGRFrame(rawArr)    
        frmCache.append(latestFrame)
        if len(frmCache) < cache_length:
            frmNum, rawArr = gen.send(None)
            continue
        if len(frmCache) > cache_length:
            print(f'Cache FUBAR, {len(frmCache)} items') ### this happens if the cache drop at the end fails
            break
        
        ### prepare frame objects of interest
        frmDict = init_frame_func(frmCache)
        frmNum_on_display = frmNum - curr_frame_idx
        
        ### initialize the feature dictionary with possible stored features
        if existing_features is not None:
            try:
                stored_featDict = existing_features[frmNum_on_display]
            except KeyError:
                stored_featDict = None
        else:
            stored_featDict = None
            
        featDict = init_feat_func(current_featDict = featDict, 
                                  stored_featDict = stored_featDict)
        
        ### Get the features of interest on the frame
        if not court_line_func is None:
            featDict = court_line_func(frmDict, featDict)
        
        if not court_feature_func is None:
            featDict = court_feature_func(frmDict, featDict)
            
            
        ### extract out stream features of interest
        if features_of_interest is None:
            streamFeatures[frmNum_on_display] = deepcopy(featDict)
        else:
            saveDict = {k: v for k, v in featDict.items() if k in features_of_interest}
            streamFeatures[frmNum_on_display] = deepcopy(saveDict)
    
        if verbose:
            ### Send Frame Back to Generator
            sendFrm = draw_func(frmDict, featDict, frameNum = frmNum_on_display)
            sendArr = sendFrm.array  
        else:
            sendArr = None
            
        try:
            frmNum, rawArr = gen.send(sendArr)
        except:
            cv.destroyAllWindows()
            break 
        
        ### drop oldest element from cache ### 
        frmCache = frmCache[1:] # drop oldest item
        
    else:
        cv.destroyAllWindows()

    return streamFeatures
    

def replay(cap, stream_features, 
           start_frame = None, stop_frame = None,
           ms_between_frames = None,
           drawFunc = None):
    
    if start_frame is None:
        start_frame = min(stream_features.keys())
        
    if stop_frame is None:
        stop_frame = max(stream_features.keys())
    
    from ATMANpy.types.Frames import BGRFrame
    
    ### init video stream ###
    from ATMANpy.funcs.cv_funcs import video_stream
    gen = video_stream(cap, 
                       start_frame = start_frame, 
                       stop_frame = stop_frame,
                       ms_between_frames = ms_between_frames)
    frmNum, rawArr = next(gen) ### this is needed to "prime" the generator
    ######
    
    while True:
        
        cFrm = BGRFrame(rawArr)
        
        if not stream_features is None:
            featDict = stream_features[frmNum]
        else:
            featDict = {}
        
        ### Send Frame Back to Generator
        if drawFunc is not None:
            sendFrm = drawFunc({'cFrm': cFrm}, featDict, frameNum = frmNum)
            sendArr = sendFrm.array  
        else:
            sendArr = None
        
        try:
            frmNum, rawArr = gen.send(sendArr)
        except:
            cv.destroyAllWindows()
            break 
            
        
        
        
        
        
        
        
        
        