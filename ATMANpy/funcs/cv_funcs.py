# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:17:12 2021

Utility Functions

@author: gianl
"""

import cv2 as cv
import ATMANpy.funcs.util_funcs as atu

def show(frame, name = 'frame'):
    cv.imshow(name, frame)
    cv.waitKey(0)
    #closing all open windows  
    cv.destroyAllWindows()  

def get_frame(cap, frame_num):
    
    total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)  ### total number of frames
    
    if frame_num < 0 or frame_num > total_frames:
        print(f'No such frame, choose number between 0 and {total_frames}')
        return None
    
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    return frame

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
        
    cap.release()
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
    
    while cap.isOpened() and (start_frame <= frmNum < stop_frame):
        
        frmNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        
        frame = yield frmNum, frame # take in frame, possibly altered, to show
        # print(frame)
        
        if frame is not None:
            cv.imshow('frame', frame)
            if cv.waitKey(ms_between_frames) & 0xFF == ord('q'):
                cv.destroyAllWindows()  
                yield None
    
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


    
def get_stream_features(cap, start_frame, stop_frame, 
                        cache_length = None,
                        init_feat_func = None, init_frame_func = None,
                        court_line_func = None, court_feature_func = None,
                        draw_func = None,
                        verbose = True):
    
    if isinstance(cap, str):
        cap = atu.get_video(cap) ### this is needed for multiprocessing
                                        # video captures can not be pickled, therefore, we create it inside of the function
     
    from copy import deepcopy
    from ATMANpy.types.Frames import BGRFrame
    
    if cache_length is None:
        cache_length = 1 ### cache holds only the current frame.
        
    if start_frame is not None:
        start_frame_with_caching = start_frame - cache_length + 1
    else:
        start_frame_with_caching = 0
    
    ### init outputList
    streamFeatures = {}
    
    #### init feature dictionary
    featDict = init_feat_func()
    
    #### init cache ###
    frmCache = []
    
    ### init video stream ###
    from ATMANpy.funcs.cv_funcs import video_stream
    gen = video_stream(cap, 
                       start_frame = start_frame_with_caching, 
                       stop_frame = stop_frame)
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
        
        ### Get the features of interest on the frame
        if not court_line_func is None:
            featDict = court_line_func(frmDict, featDict)
        
        if not court_feature_func is None:
            featDict = court_feature_func(frmDict, featDict)
            
        streamFeatures[frmNum] = deepcopy(featDict)
    
        if verbose:
            ### Send Frame Back to Generator
            sendFrm = draw_func(frmDict, featDict)
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
           cache_frame_offset = None,
           drawFunc = None):
    
    from ATMANpy.types.Frames import BGRFrame
    
    START_FRAME = min(stream_features.keys())
    STOP_FRAME = max(stream_features.keys())
    
    ### init video stream ###
    from ATMANpy.funcs.cv_funcs import video_stream
    gen = video_stream(cap, 
                       start_frame = START_FRAME - cache_frame_offset, 
                       stop_frame = STOP_FRAME - cache_frame_offset)
    frmNum, rawArr = next(gen) ### this is needed to "prime" the generator
    ######
    
    while True:
        
        cFrm = BGRFrame(rawArr)
        featDict = stream_features[frmNum + cache_frame_offset]
        
        ### Send Frame Back to Generator
        if drawFunc is not None:
            sendFrm = drawFunc({'cFrm': cFrm}, featDict)
            sendArr = sendFrm.array  
        else:
            sendArr = None
        
        try:
            frmNum, rawArr = gen.send(sendArr)
        except:
            cv.destroyAllWindows()
            break 
            
        
        
        
        
        
        
        
        
        